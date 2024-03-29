/* Copyright (c) 2021 jiangcheng Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef SINGLE_TEST_ERNIEDOC_SOFTMAX_FORWARD_D1_H_
#define SINGLE_TEST_ERNIEDOC_SOFTMAX_FORWARD_D1_H_

// C system file
#include <stdio.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Library file
#include "../common.h"

template <typename T, int N> struct GetPackType;

template <typename T> struct GetPackType<T, 1> { using type = T; };

template <> struct GetPackType<half, 2> { using type = half2; };

template <> struct GetPackType<float, 2> { using type = float2; };

template <typename T, int N> using PackType = typename GetPackType<T, N>::type;

template <typename T, int N> union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template <typename SRC, typename DST, int N> struct MultiFetch {
  __device__ void operator()(DST *dst, const SRC *src) const {
    Pack<SRC, N> pack;
    pack.storage = *reinterpret_cast<const PackType<SRC, N> *>(src);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }
};

template <typename SRC, typename DST, int N> struct MultiStore {
  __device__ void operator()(DST *dst, const SRC *src) const {
    Pack<DST, N> pack;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pack.elem[i] = static_cast<DST>(src[i]);
    }
    *reinterpret_cast<PackType<DST, N> *>(dst) = pack.storage;
  }
};

template <typename T, typename AccT, int VECSIZE, int COLS, bool padding>
__global__ void KeOneflowD1WarpSoftmaxForward(T *__restrict__ dst,
                                              const T *__restrict__ src,
                                              const int N, const int dim) {
  assert(COLS % VECSIZE == 0);
  constexpr int num_vec = COLS / VECSIZE;
  assert(dim <= COLS * WARP_SIZE);
  AccT buf[COLS];
  const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int warp_num = gridDim.x * blockDim.y;
  const int tid = threadIdx.x;
  for (int row = warp_id; row < N; row += warp_num) {
    const int offset = row * dim;
    const T *src_row = src + offset;
    T *dst_row = dst + offset;
    AccT max_val = -std::numeric_limits<AccT>::infinity();
#pragma unroll
    for (int vec_id = 0; vec_id < num_vec; vec_id++) {
      // Vectorize read data from global memory
      const int col = (vec_id * WARP_SIZE + tid) * VECSIZE;
      if (!padding || col < dim) {
        // do not need padding
        // read VECSIZE data from global memory
        MultiFetch<T, AccT, VECSIZE>()(buf + vec_id * VECSIZE, src_row + col);
#pragma unroll
        for (int i = 0; i < VECSIZE; i++) {
          max_val = max(buf[vec_id * VECSIZE + i], max_val);
        }
      } else {
// col not enough, need padding
#pragma unroll
        for (int i = 0; i < VECSIZE; i++) {
          buf[vec_id * VECSIZE + i] = -std::numeric_limits<AccT>::infinity();
        }
      }
    }
    max_val = warpReduceMax(max_val, 0xffffffff);

    AccT sum_val(0);
#pragma unroll
    for (int i = 0; i < COLS; i++) {
      buf[i] = Exp(buf[i] - max_val);
      sum_val += buf[i];
    }
    sum_val = warpReduceSum(sum_val, 0xffffffff);

#pragma unroll
    for (int i = 0; i < COLS; i++) {
      buf[i] = buf[i] / sum_val;
    }
#pragma unroll
    for (int i = 0; i < num_vec; i++) {
      const int col = (i * WARP_SIZE + tid) * VECSIZE;
      if (!padding || col < dim) {
        MultiFetch<T, AccT, VECSIZE>()(dst_row + col, buf + i * VECSIZE);
      }
    }
  }
}

template <typename T, typename AccT, int VECSIZE, int COLS, bool padding>
inline void LaunchOneflowD1WarpSoftmax(const CUDAStream &context,
                                       const T *in_data, T *out_data,
                                       const int rows, const int cols) {
  constexpr int block_size = 128;
  static_assert(block_size % WARP_SIZE == 0, "");
  constexpr int rows_per_block = block_size / WARP_SIZE;
  dim3 block_dim(WARP_SIZE, rows_per_block);
  const int64_t grid_dim_x = (rows + rows_per_block - 1) / rows_per_block;
  KeOneflowD1WarpSoftmaxForward<
      T, AccT, VECSIZE, COLS,
      padding><<<grid_dim_x, block_dim, 0, context.stream()>>>(
      out_data, in_data, rows, cols);
}

template <typename T, typename AccT, int VECSIZE, int COLS>
inline void DispatchOneflowD1WarpSoftmaxPadding(const CUDAStream &context,
                                                const T *in_data, T *out_data,
                                                const int rows,
                                                const int cols) {
  if (cols == COLS * WARP_SIZE) {
    LaunchOneflowD1WarpSoftmax<T, AccT, VECSIZE, COLS, false>(
        context, in_data, out_data, rows, cols);
  } else {
    LaunchOneflowD1WarpSoftmax<T, AccT, VECSIZE, COLS, true>(
        context, in_data, out_data, rows, cols);
  }
}

template <typename T, typename AccT, int VECSIZE>
typename std::enable_if<VECSIZE == 1, void>::type
DispatchOneflowD1WarpSoftmaxCols(const CUDAStream &context, const T *in_data,
                                 T *out_data, const int rows, const int cols) {
#define DEFINE_ONE_ELIF(col)                                                   \
  } else if (cols <= (col)*WARP_SIZE) {                                        \
    DispatchOneflowD1WarpSoftmaxPadding<T, AccT, VECSIZE, col>(                \
        context, in_data, out_data, rows, cols);
  if (cols <= 0) {
    assert(0);
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(7)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(9)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(11)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(13)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(15)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(17)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(19)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(21)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(23)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(25)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(27)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(29)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(31)
    DEFINE_ONE_ELIF(32)
  } else {
    assert(0);
  }
#undef DEFINE_ONE_ELIF
}

template <typename T, typename AccT, int VECSIZE>
typename std::enable_if<VECSIZE == 2, void>::type
DispatchOneflowD1WarpSoftmaxCols(const CUDAStream &context, const T *in_data,
                                 T *out_data, const int rows, const int cols) {
  if (cols <= 0) {
    assert(0);
  }
#define DEFINE_ONE_ELIF(col)                                                   \
  else if (cols <= (col)*WARP_SIZE) {                                          \
    DispatchOneflowD1WarpSoftmaxPadding<T, AccT, VECSIZE, col>(                \
        context, in_data, out_data, rows, cols);                               \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    assert(0);
  }
}
/************************************************************************/

template <typename T, typename AccT, int COLS>
__global__ void NoVec_KeD1WarpSoftmaxForward(T *dst, const T *src, const int N,
                                             const int dim) {
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int tid = threadIdx.x;

  for (int row = warp_id; row < N; row += gridDim.x * blockDim.y) {
    int offset = row * dim;
    const T *src_row = src + offset;
    T *dst_row = dst + offset;
    // Load src data from global memory to register,
    // and compute max value
    AccT buf[COLS];
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    int real_cols = 0;
#pragma unroll
    for (int col = 0; col < COLS; col++) {
      int src_col = tid + col * WARP_SIZE;
      if (src_col >= dim)
        break;
      buf[col] = static_cast<AccT>(src_row[src_col]);
      max_val = max(buf[col], max_val);
      real_cols++;
    }
    max_val = warpReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
#pragma unroll
    for (int i = 0; i < COLS; i++) {
      if (i >= real_cols)
        break;
      buf[i] = Exp(buf[i] - max_val);
      sum_val += buf[i];
    }
    sum_val = warpReduceSum(sum_val, 0xffffffff);
// compute softmax result
#pragma unroll
    for (int col = 0; col < COLS; col++) {
      int dst_col = tid + col * WARP_SIZE;
      if (dst_col >= dim)
        break;
      dst_row[dst_col] = static_cast<T>(buf[col] / sum_val);
    }
  }
}

template <typename T> struct GetAccType { using type = T; };
template <> struct GetAccType<half> { using type = float; };

template <typename T, int N> struct GetVecType;
template <typename T> struct GetVecType<T, 1> { using type = T; };
template <> struct GetVecType<half, 2> { using type = half2; };
template <> struct GetVecType<half, 4> { using type = float2; };
template <> struct GetVecType<float, 2> { using type = float2; };
template <> struct GetVecType<float, 4> { using type = float4; };
template <> struct GetVecType<double, 2> { using type = double2; };
template <> struct GetVecType<double, 4> { using type = double4; };

/*****************************************************************/
// when D == 1 && 320 <= dim <= 1024, using KeD1WarpSoftmaxForward faster,
// each warp compute one row's element,
// each thread compute COLS element of dim and store in register
template <typename T, typename AccT, int COLS, int VECSIZE>
__global__ void KeD1WarpSoftmaxForward(T *__restrict__ dst,
                                       const T *__restrict__ src, const int N,
                                       const int dim) {
  static_assert(COLS % VECSIZE == 0);
  constexpr int num_vec = COLS / VECSIZE;
  const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  for (int row = warp_id; row < N; row += gridDim.x * blockDim.y) {
    const int offset = row * dim;
    const T *__restrict__ src_row = src + offset;
    T *__restrict__ dst_row = dst + offset;

    using VecT = typename GetVecType<T, VECSIZE>::type;
    // vectorization for global memory coalescing
    VecT vec;
    // Load src data from global memory to register,
    // and compute max value
    AccT buf[COLS]{0};
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    int real_cols = 0;
#pragma unroll
    for (int col = 0; col < num_vec; col++) {
      int src_col = (tid + col * WARP_SIZE) * VECSIZE;
      if (src_col >= dim)
        break;
      vec = reinterpret_cast<const VecT *>(&src_row[src_col])[0];
      T *buf_src = reinterpret_cast<T *>(&vec);

      AccT *buf_acc = buf + real_cols;
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        buf_acc[i] = static_cast<AccT>(buf_src[i]);
        max_val = max(buf_acc[i], max_val);
      }
      real_cols += VECSIZE;
    }
    max_val = warpReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
#pragma unroll
    for (int i = 0; i < COLS; i++) {
      // "break" set in "for loop" aims to avoid local memory
      if (i >= real_cols)
        break;
      buf[i] = Exp(buf[i] - max_val);
      sum_val += buf[i];
    }
    sum_val = warpReduceSum(sum_val, 0xffffffff);
// compute softmax result
#pragma unroll
    for (int col = 0; col < num_vec; col++) {
      int dst_col = (tid + col * WARP_SIZE) * VECSIZE;
      if (dst_col >= dim)
        break;
      T *buf_dst = reinterpret_cast<T *>(&vec);
      AccT *buf_acc = buf + col * VECSIZE;
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        buf_dst[i] = static_cast<T>(buf_acc[i] / sum_val);
      }
      reinterpret_cast<VecT *>(&dst_row[dst_col])[0] = vec;
    }
  }
}

template <typename T, int COLS, int VECSIZE>
inline void LaunchD1WarpSoftmaxForwardKernel(const cudaStream_t &stream,
                                             const T *in_data, T *out_data,
                                             const int N, const int dim) {
  int N_b = std::min(8, N);
  dim3 threads(WARP_SIZE, N_b);
  int grids = (N + N_b - 1) / N_b;
  using AccT = typename GetAccType<T>::type;

  KeD1WarpSoftmaxForward<T, AccT, COLS, VECSIZE><<<grids, threads, 0, stream>>>(
      out_data, in_data, N, dim);
}

#define LAUNCH_D1WARP_COLS(COLS)                                               \
  case COLS:                                                                   \
    LaunchD1WarpSoftmaxForwardKernel<T, COLS, VECSIZE>(stream, in_data,        \
                                                       out_data, N, dim);      \
    break;

template <typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 1, void>::type
DispatchD1WarpSoftmaxForward(const cudaStream_t &stream, const T *in_data,
                             T *out_data, const int N, const int dim,
                             const int cols_per_thread) {
  switch (cols_per_thread) {
    LAUNCH_D1WARP_COLS(1)
    LAUNCH_D1WARP_COLS(2)
    LAUNCH_D1WARP_COLS(3)
    LAUNCH_D1WARP_COLS(4)
    LAUNCH_D1WARP_COLS(5)
    LAUNCH_D1WARP_COLS(6)
    LAUNCH_D1WARP_COLS(7)
    LAUNCH_D1WARP_COLS(8)
    LAUNCH_D1WARP_COLS(9)
    LAUNCH_D1WARP_COLS(10)
    LAUNCH_D1WARP_COLS(11)
    LAUNCH_D1WARP_COLS(12)
    LAUNCH_D1WARP_COLS(13)
    LAUNCH_D1WARP_COLS(14)
    LAUNCH_D1WARP_COLS(15)
    LAUNCH_D1WARP_COLS(16)
    LAUNCH_D1WARP_COLS(17)
    LAUNCH_D1WARP_COLS(18)
    LAUNCH_D1WARP_COLS(19)
    LAUNCH_D1WARP_COLS(20)
    LAUNCH_D1WARP_COLS(21)
    LAUNCH_D1WARP_COLS(22)
    LAUNCH_D1WARP_COLS(23)
    LAUNCH_D1WARP_COLS(24)
    LAUNCH_D1WARP_COLS(25)
    LAUNCH_D1WARP_COLS(26)
    LAUNCH_D1WARP_COLS(27)
    LAUNCH_D1WARP_COLS(28)
    LAUNCH_D1WARP_COLS(29)
    LAUNCH_D1WARP_COLS(30)
    LAUNCH_D1WARP_COLS(31)
    LAUNCH_D1WARP_COLS(32)
  default:
    break;
  }
}

template <typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 2, void>::type
DispatchD1WarpSoftmaxForward(const cudaStream_t &stream, const T *in_data,
                             T *out_data, const int N, const int dim,
                             const int cols_per_thread) {
  switch (cols_per_thread) {
    LAUNCH_D1WARP_COLS(2)
    LAUNCH_D1WARP_COLS(4)
    LAUNCH_D1WARP_COLS(6)
    LAUNCH_D1WARP_COLS(8)
    LAUNCH_D1WARP_COLS(10)
    LAUNCH_D1WARP_COLS(12)
    LAUNCH_D1WARP_COLS(14)
    LAUNCH_D1WARP_COLS(16)
    LAUNCH_D1WARP_COLS(18)
    LAUNCH_D1WARP_COLS(20)
    LAUNCH_D1WARP_COLS(22)
    LAUNCH_D1WARP_COLS(24)
    LAUNCH_D1WARP_COLS(26)
    LAUNCH_D1WARP_COLS(28)
    LAUNCH_D1WARP_COLS(30)
    LAUNCH_D1WARP_COLS(32)
  default:
    break;
  }
}

template <typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 4, void>::type
DispatchD1WarpSoftmaxForward(const cudaStream_t &stream, const T *in_data,
                             T *out_data, const int N, const int dim,
                             const int cols_per_thread) {
  switch (cols_per_thread) {
    LAUNCH_D1WARP_COLS(4)
    LAUNCH_D1WARP_COLS(8)
    LAUNCH_D1WARP_COLS(12)
    LAUNCH_D1WARP_COLS(16)
    LAUNCH_D1WARP_COLS(20)
    LAUNCH_D1WARP_COLS(24)
    LAUNCH_D1WARP_COLS(28)
    LAUNCH_D1WARP_COLS(32)
  default:
    break;
  }
}
#undef LAUNCH_D1WARP_COLS

template <typename T>
inline void LaunchD1WarpSoftmaxForward(const cudaStream_t &stream,
                                       const T *in_data, T *out_data,
                                       const int N, const int dim) {
  const int cols_per_thread = (dim + WARP_SIZE - 1) / WARP_SIZE;

  if (dim % 4 == 0 && cols_per_thread % 4 == 0) {
    DispatchD1WarpSoftmaxForward<T, 4>(stream, in_data, out_data, N, dim,
                                       cols_per_thread);
  } else if (dim % 2 == 0 && cols_per_thread % 2 == 0) {
    DispatchD1WarpSoftmaxForward<T, 2>(stream, in_data, out_data, N, dim,
                                       cols_per_thread);
  } else {
    DispatchD1WarpSoftmaxForward<T, 1>(stream, in_data, out_data, N, dim,
                                       cols_per_thread);
  }
}

/************************************************************************/

template <typename T, typename AccT>
__global__ void NoVec_KeD1BlockSharedSoftmaxForward(T *__restrict__ dst,
                                                    const T *__restrict__ src,
                                                    const int N,
                                                    const int dim) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT *s_data = reinterpret_cast<AccT *>(s_mem);

  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T *__restrict__ src_row = src + offset;
    T *__restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for (int col = tid; col < dim; col += blockDim.x) {
      AccT tmp_val = static_cast<AccT>(src_row[col]);
      s_data[col] = tmp_val;
      max_val = max(tmp_val, max_val);
    }
    max_val = blockReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
    for (int col = tid; col < dim; col += blockDim.x) {
      AccT tmp_val = Exp(s_data[col] - max_val);
      s_data[col] = tmp_val;
      sum_val += tmp_val;
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for (int col = tid; col < dim; col += blockDim.x) {
      dst_row[col] = static_cast<T>(s_data[col] / sum_val);
    }
  }
}

// when D == 1 && 1024 < dim <= 4096, using KeD1BlockSharedSoftmaxForward,
// each block compute a row, and synchronization by blockReduce,
// each thread compute VECSIZE elements of dim, and store in shared memory
template <typename T, typename AccT, int VECSIZE>
__global__ void KeD1BlockSharedSoftmaxForward(T *__restrict__ dst,
                                              const T *__restrict__ src,
                                              const int N, const int dim) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT *s_data = reinterpret_cast<AccT *>(s_mem);

  const int tid = threadIdx.x;
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec;
  T *buf_src = reinterpret_cast<T *>(&vec);

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T *__restrict__ src_row = src + offset;
    T *__restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for (int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec = reinterpret_cast<const VecT *>(&src_row[col])[0];
      AccT *buf_s = s_data + col;
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        buf_s[i] = static_cast<AccT>(buf_src[i]);
        max_val = max(buf_s[i], max_val);
      }
    }
    max_val = blockReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
    for (int col = tid; col < dim; col += blockDim.x) {
      AccT tmp_val = Exp(s_data[col] - max_val);
      s_data[col] = tmp_val;
      sum_val += tmp_val;
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for (int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      T *buf_dst = reinterpret_cast<T *>(&vec);
      AccT *buf_s = s_data + col;
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        buf_dst[i] = static_cast<T>(buf_s[i] / sum_val);
      }
      reinterpret_cast<VecT *>(&dst_row[col])[0] = vec;
    }
  }
}

template <typename T, int VECSIZE>
inline void LaunchD1BlockSharedSoftmaxForwardKernel(const cudaStream_t &stream,
                                                    const T *in_data,
                                                    T *out_data, const int N,
                                                    const int dim) {
  const int threads = std::min(dim, 256);
  const int grids = N;
  using AccT = typename GetAccType<T>::type;

  KeD1BlockSharedSoftmaxForward<
      T, AccT, VECSIZE><<<grids, threads, dim * sizeof(AccT), stream>>>(
      out_data, in_data, N, dim);
}

template <typename T>
inline void LaunchD1BlockSharedSoftmaxForward(const cudaStream_t &stream,
                                              const T *in_data, T *out_data,
                                              const int N, const int dim) {
  if (dim % 4 == 0) {
    LaunchD1BlockSharedSoftmaxForwardKernel<T, 4>(stream, in_data, out_data, N,
                                                  dim);
  } else if (dim % 2 == 0) {
    LaunchD1BlockSharedSoftmaxForwardKernel<T, 2>(stream, in_data, out_data, N,
                                                  dim);
  } else {
    LaunchD1BlockSharedSoftmaxForwardKernel<T, 1>(stream, in_data, out_data, N,
                                                  dim);
  }
}

/************************************************************************/

template <typename T, typename AccT>
__global__ void NoVec_KeD1BlockSoftmaxForward(T *__restrict__ dst,
                                              const T *__restrict__ src,
                                              const int N, const int dim) {
  const int tid = threadIdx.x;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T *__restrict__ src_row = src + offset;
    T *__restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for (int col = tid; col < dim; col += blockDim.x) {
      max_val = max(static_cast<AccT>(src_row[col]), max_val);
    }
    max_val = blockReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
    for (int col = tid; col < dim; col += blockDim.x) {
      sum_val += Exp(static_cast<AccT>(src_row[col]) - max_val);
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for (int col = tid; col < dim; col += blockDim.x) {
      dst_row[col] = static_cast<T>(
          Exp(static_cast<AccT>(src_row[col]) - max_val) / sum_val);
    }
  }
}

// when D == 1 && 4096 < dim, using KeD1BlockSoftmaxForward,
// each block compute a row, and synchronization by blockReduce,
// each thread compute VECSIZE elements of dim
template <typename T, typename AccT, int VECSIZE>
__global__ void KeD1BlockSoftmaxForward(T *__restrict__ dst,
                                        const T *__restrict__ src, const int N,
                                        const int dim) {
  const int tid = threadIdx.x;

  using VecT = typename GetVecType<T, VECSIZE>::type;
  // vectorization for global memory coalescing
  VecT vec_src, vec_dst;
  T *buf_src = reinterpret_cast<T *>(&vec_src);
  T *buf_dst = reinterpret_cast<T *>(&vec_dst);

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T *__restrict__ src_row = src + offset;
    T *__restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for (int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_src = reinterpret_cast<const VecT *>(&src_row[col])[0];
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        max_val = max(static_cast<AccT>(buf_src[i]), max_val);
      }
    }
    max_val = blockReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
    for (int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_src = reinterpret_cast<const VecT *>(&src_row[col])[0];
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        sum_val += Exp(static_cast<AccT>(buf_src[i]) - max_val);
      }
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for (int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_src = reinterpret_cast<const VecT *>(&src_row[col])[0];
#pragma unroll
      for (int i = 0; i < VECSIZE; i++) {
        buf_dst[i] =
            static_cast<T>(Exp(static_cast<AccT>(buf_src[i]) - max_val) /
                           (sum_val + std::numeric_limits<AccT>::min()));
      }
      reinterpret_cast<VecT *>(&dst_row[col])[0] = vec_dst;
    }
  }
}

template <typename T, int VECSIZE>
inline void LaunchD1BlockSoftmaxForwardKernel(const cudaStream_t &stream,
                                              const T *in_data, T *out_data,
                                              const int N, const int dim) {
  const int threads = std::min(dim, MAX_BLOCK_DIM);
  const int grids = N;
  using AccT = typename GetAccType<T>::type;

  KeD1BlockSoftmaxForward<T, AccT, VECSIZE><<<grids, threads, 0, stream>>>(
      out_data, in_data, N, dim);
}

template <typename T>
inline void LaunchD1BlockSoftmaxForward(const cudaStream_t &stream,
                                        const T *in_data, T *out_data,
                                        const int N, const int dim) {
  if (dim % 4 == 0) {
    LaunchD1BlockSoftmaxForwardKernel<T, 4>(stream, in_data, out_data, N, dim);
  } else if (dim % 2 == 0) {
    LaunchD1BlockSoftmaxForwardKernel<T, 2>(stream, in_data, out_data, N, dim);
  } else {
    LaunchD1BlockSoftmaxForwardKernel<T, 1>(stream, in_data, out_data, N, dim);
  }
}

#endif // SINGLE_TEST_ERNIEDOC_SOFTMAX_FORWARD_D1_H_

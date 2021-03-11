/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

#define LAUNCH_SOFTMAX_WARP_BACKWARD(Log2Elements)                 \
  case Log2Elements:                                               \
    softmax_warp_backward<T, float, Log2Elements><<<               \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dx_data, mul_grad.data<T>(), out->data<T>(), N, dim, dim); \
    break;

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T, int VLEN>
union vec_t {
  static_assert(sizeof(T) == -1, "vec_t is only available by specialization.");
};

template <>
union vec_t<float, 4> {
  float4 s;
  float v[4];
};

template <>
union vec_t<platform::float16, 4> {
  int2 s;
  platform::float16 v[4];
};

template<typename T> struct GetAccType {using type = T;};
template<> struct GetAccType<paddle::platform::float16> {using type = float;};

template<typename T, int N> struct GetVecType;
template<typename T> struct GetVecType<T, 1> {using type = T;};
template<> struct GetVecType<paddle::platform::float16, 2> {using type = half2;};
template<> struct GetVecType<paddle::platform::float16, 4> {using type = float2;};
template<> struct GetVecType<float, 2> {using type = float2;};
template<> struct GetVecType<float, 4> {using type = float4;};
template<> struct GetVecType<double, 2> {using type = double2;};
template<> struct GetVecType<double, 4> {using type = double4;};

template<typename T>
__forceinline__ __device__ T Exp(const T val) {return exp(val);}
template<>
__forceinline__ __device__ float Exp<float>(const float val) {return __expf(val);}

/*****************************************************************/
// when D == 1 && dim < 320, using WarpSoftmaxForward faster
template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_sum(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_max(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForward(T* dst, const T* src, const int batch_size,
                                   const int stride, const int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // load data from global memory
  AccT elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        elements[i][it] =
            static_cast<float>(src[i * element_count + it * warp_size_softmax]);
      } else {
        elements[i][it] = -std::numeric_limits<AccT>::infinity();
      }
    }
  }

  // compute max_value
  AccT max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce_max<AccT, WARP_BATCH, warp_size_softmax>(max_value);

  AccT sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = (std::exp((elements[i][it] - max_value[i])));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        dst[i * element_count + it * warp_size_softmax] =
            elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template<typename T>
void LaunchWarpSoftmaxForward(cudaStream_t &stream, const T* in_data, T* out_data,
                        const int N, const int dim) {
  int log2_elements = static_cast<int>(log2_ceil(dim));
  const int next_power_of_two = 1 << log2_elements;
  int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
  int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  // use 128 threads per block to maximimize gpu utilization
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (N + batches_per_block - 1) / batches_per_block;
  dim3 threads(warp_size, warps_per_block, 1);

#define LAUNCH_SOFTMAX_WARP_FORWARD(Log2Elements)                  \
  case Log2Elements:                                               \
    WarpSoftmaxForward<T, float, Log2Elements><<<                  \
        blocks, threads, 0, stream>>>(                             \
        out_data, in_data, N, dim, dim);                      \
    break;

  switch (log2_elements) {
    LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
    LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
    LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
    LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
    LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
    LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
    LAUNCH_SOFTMAX_WARP_FORWARD(6);  // 64
    LAUNCH_SOFTMAX_WARP_FORWARD(7);  // 128
    LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
    LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
    default:
      break;
  }
#undef LAUNCH_SOFTMAX_WARP_FORWARD
}

/*****************************************************************/
// when D == 1 && 320 <= dim <= 1024, using KeD1WarpSoftmaxForward faster,
// each warp compute one row's element,
// each thread compute COLS element of dim and store in register
template<typename T, typename AccT, int COLS, int VECSIZE>
__global__ void KeD1WarpSoftmaxForward(T* __restrict__ dst,
            const T* __restrict__ src,const int N, const int dim) {
  static_assert(COLS % VECSIZE == 0);
  constexpr int num_vec = COLS / VECSIZE;
  const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  for(int row = warp_id; row < N; row += gridDim.x * blockDim.y) {
    const int offset = row * dim;
    const T* __restrict__ src_row = src + offset;
    T* __restrict__ dst_row = dst + offset;

    using VecT = typename GetVecType<T, VECSIZE>::type;
    VecT vec; // vectorization for global memory coalescing 
    // Load src data from global memory to register,
    // and compute max value
    AccT buf[COLS]{0};
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    int real_cols = 0;
  #pragma unroll
    for(int col = 0; col < num_vec; col ++) {
      int src_col =  (tid + col * WARP_SIZE) * VECSIZE;
      if(src_col >= dim) break;
      vec = reinterpret_cast<const VecT*>(&src_row[src_col])[0];
      T* buf_src = reinterpret_cast<T*>(&vec);

      AccT* buf_acc = buf + real_cols;
  #pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_acc[i] = static_cast<AccT>(buf_src[i]);
        max_val = max(buf_acc[i], max_val);
      }
      real_cols += VECSIZE;
    }
    max_val = math::warpReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
  #pragma unroll
    for(int i = 0; i < COLS; i ++) {
      // "break" set in "for loop" aims to avoid local memory
      if(i >= real_cols) break;
      buf[i] = Exp(buf[i] - max_val);
      sum_val += buf[i];
    }
    sum_val = math::warpReduceSum(sum_val, 0xffffffff);
    // compute softmax result
  #pragma unroll
    for(int col = 0; col < num_vec; col ++) {
      int dst_col =  (tid + col * WARP_SIZE) * VECSIZE;
      if(dst_col >= dim) break;
      T *buf_dst = reinterpret_cast<T*>(&vec);
      AccT* buf_acc = buf + col * VECSIZE;
  #pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_dst[i] = static_cast<T>(buf_acc[i] / sum_val);
      }
      reinterpret_cast<VecT*>(&dst_row[dst_col])[0] = vec;
    }
  }
}

template<typename T, int COLS, int VECSIZE>
inline void LaunchD1WarpSoftmaxForwardKernel(cudaStream_t &stream,
          const T* in_data, T* out_data, const int N, const int dim) {
  int N_b = std::min(8, N);
  dim3 threads(WARP_SIZE, N_b);
  int grids = (N + N_b - 1) / N_b;
  using AccT = typename GetAccType<T>::type;

  KeD1WarpSoftmaxForward<T, AccT, COLS, VECSIZE>
    <<<grids, threads, 0, stream>>>(
      out_data, in_data, N, dim);
}

#define LAUNCH_D1WARP_COLS(COLS)                          \
  case COLS:                                              \
    LaunchD1WarpSoftmaxForwardKernel<T, COLS, VECSIZE>(   \
            stream, in_data, out_data, N, dim);           \
    break;

template<typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 1, void>::type DispatchD1WarpSoftmaxForward(
                        cudaStream_t &stream, const T* in_data, T* out_data,
                        const int N, const int dim, const int cols_per_thread) {
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

template<typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 2, void>::type DispatchD1WarpSoftmaxForward(
                        cudaStream_t &stream, const T* in_data, T* out_data,
                        const int N, const int dim, const int cols_per_thread) {
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

template<typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 4, void>::type DispatchD1WarpSoftmaxForward(
                        cudaStream_t &stream, const T* in_data, T* out_data,
                        const int N, const int dim, const int cols_per_thread) {
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

template<typename T>
inline void LaunchD1WarpSoftmaxForward(cudaStream_t &stream, const T* in_data,
                      T* out_data, const int N, const int dim) {
  const int cols_per_thread = (dim + WARP_SIZE - 1) / WARP_SIZE;

  if(dim % 4 == 0 && cols_per_thread % 4 == 0) {
    DispatchD1WarpSoftmaxForward<T, 4>(
      stream, in_data, out_data, N, dim, cols_per_thread);
  } else if(dim % 2 == 0 && cols_per_thread % 2 == 0) {
    DispatchD1WarpSoftmaxForward<T, 2>(
      stream, in_data, out_data, N, dim, cols_per_thread);
  } else {
    DispatchD1WarpSoftmaxForward<T, 1>(
      stream, in_data, out_data, N, dim, cols_per_thread);
  }
}

/*****************************************************************/
// when D == 1 && 1024 < dim <= 4096, using KeD1BlockSharedSoftmaxForward,
// each block compute a row, and synchronization by blockReduce,
// each thread compute VECSIZE elements of dim, and store in shared memory
template<typename T, typename AccT, int VECSIZE>
__global__ void KeD1BlockSharedSoftmaxForward(T* __restrict__ dst,
            const T* __restrict__ src,const int N, const int dim) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  const int tid = threadIdx.x;
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec;
  T* buf_src = reinterpret_cast<T*>(&vec);

  for(int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T* __restrict__ src_row = src + offset;
    T* __restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec = reinterpret_cast<const VecT*>(&src_row[col])[0];
      AccT* buf_s = s_data + col;
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_s[i] = static_cast<AccT>(buf_src[i]);
        max_val = max(buf_s[i], max_val);
      }
    }
    max_val = math::blockReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
    for(int col = tid; col < dim; col += blockDim.x) {
      AccT tmp_val = Exp(s_data[col] - max_val);
      s_data[col] = tmp_val;
      sum_val += tmp_val;
    }
    sum_val = math::blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      T* buf_dst = reinterpret_cast<T*>(&vec);
      AccT* buf_s = s_data + col;
  #pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_dst[i] = static_cast<T>(buf_s[i] / sum_val);
      }
      reinterpret_cast<VecT*>(&dst_row[col])[0] = vec;
    }
  }
}

template<typename T, int VECSIZE>
inline void LaunchD1BlockSharedSoftmaxForwardKernel(cudaStream_t &stream,
                const T* in_data, T* out_data, const int N, const int dim) {
  const int threads = std::min(dim, 256);
  const int grids = N;
  using AccT = typename GetAccType<T>::type;

  KeD1BlockSharedSoftmaxForward<T, AccT, VECSIZE>
    <<<grids, threads, dim * sizeof(AccT), stream>>>(
    out_data, in_data, N, dim);
}

template<typename T>
inline void LaunchD1BlockSharedSoftmaxForward(cudaStream_t &stream, const T* in_data,
                        T* out_data, const int N, const int dim) {
  if(dim % 4 == 0) {
    LaunchD1BlockSharedSoftmaxForwardKernel<T, 4>(
      stream, in_data, out_data, N, dim);
  } else if(dim % 2 == 0) {
    LaunchD1BlockSharedSoftmaxForwardKernel<T, 2>(
      stream, in_data, out_data, N, dim);
  } else {
    LaunchD1BlockSharedSoftmaxForwardKernel<T, 1>(
      stream, in_data, out_data, N, dim);
  }
}

/*****************************************************************/
// when D == 1 && 4096 < dim, using KeD1BlockSoftmaxForward,
// each block compute a row, and synchronization by blockReduce,
// each thread compute VECSIZE elements of dim
template<typename T, typename AccT, int VECSIZE>
__global__ void KeD1BlockSoftmaxForward(T* __restrict__ dst,
            const T* __restrict__ src,const int N, const int dim) {
  const int tid = threadIdx.x;

  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec_src, vec_dst;// vectorization for global memory coalescing
  T* buf_src = reinterpret_cast<T*>(&vec_src);
  T* buf_dst = reinterpret_cast<T*>(&vec_dst);

  for(int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T* __restrict__ src_row = src + offset;
    T* __restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[col])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        max_val = max(static_cast<AccT>(buf_src[i]), max_val);
      }
    }
    max_val = math::blockReduceMax(max_val, 0xffffffff);
    // compute sum value
    AccT sum_val(0);
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[col])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        sum_val += Exp(static_cast<AccT>(buf_src[i]) - max_val);
      }
    }
    sum_val = math::blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[col])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_dst[i] = static_cast<T>(
              Exp(static_cast<AccT>(buf_src[i]) - max_val) / (sum_val + 1e-6f));
      }
      reinterpret_cast<VecT*>(&dst_row[col])[0] = vec_dst;
    }
  }
}

template<typename T, int VECSIZE>
inline void LaunchD1BlockSoftmaxForwardKernel(cudaStream_t &stream, const T* in_data,
                          T* out_data, const int N, const int dim) {
  const int threads = std::min(dim, 1024);
  const int grids = N;
  using AccT = typename GetAccType<T>::type;

  KeD1BlockSoftmaxForward<T, AccT, VECSIZE>
    <<<grids, threads, 0, stream>>>(
    out_data, in_data, N, dim);
}

template<typename T>
inline void LaunchD1BlockSoftmaxForward(cudaStream_t &stream, const T* in_data,
                        T* out_data, const int N, const int dim) {
  if(dim % 4 == 0) {
    LaunchD1BlockSoftmaxForwardKernel<T, 4>(
      stream, in_data, out_data, N, dim);
  } else if(dim % 2 == 0) {
    LaunchD1BlockSoftmaxForwardKernel<T, 2>(
      stream, in_data, out_data, N, dim);
  } else {
    LaunchD1BlockSoftmaxForwardKernel<T, 1>(
      stream, in_data, out_data, N, dim);
  }
}

/*****************************************************************/
// When D is larg and dim is small:
// Each block arranged by N，each thread arranged by D
// each thread compute dim * VECSIZE number's softmax
template<typename T, typename AccT, int VECSIZE>
__global__ void KeLoopDimSoftmaxForward(T* __restrict__ dst,
      const T* __restrict__ src, const int N, const int dim, const int D) {
  assert(D % VECSIZE == 0);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int vec_id = tid * VECSIZE;
  const int out_id = vec_id / D;
  if(out_id >= N) return;
  const int in_id = vec_id - out_id * D;
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec_src, vec_dst;
  T* buf_src = reinterpret_cast<T*>(&vec_src);
  T* buf_dst = reinterpret_cast<T*>(&vec_dst);

  const T* __restrict__ src_row = src + out_id * dim * D + in_id;
  T* __restrict__ dst_row = dst + out_id * dim * D + in_id;
  // compute max value
  AccT max_val[VECSIZE];
#pragma unroll
  for(int i = 0; i < VECSIZE; i ++) {
    max_val[i] = -std::numeric_limits<AccT>::infinity();
  }
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    vec_src = reinterpret_cast<const VecT*>(&src_row[dim_id * D])[0];
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      max_val[i] = max(static_cast<AccT>(buf_src[i]), max_val[i]);
    }
  }
  // compute exponent value and sum value
  AccT sum_val[VECSIZE]{0};
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    vec_src = reinterpret_cast<const VecT*>(&src_row[dim_id * D])[0];
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      sum_val[i] += Exp(static_cast<AccT>(buf_src[i]) - max_val[i]);
    }
  }

  // compute softmax value
  // TODO(jiangcheng): how to eliminate twice Exp
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    vec_src = reinterpret_cast<const VecT*>(&src_row[dim_id * D])[0];
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      buf_dst[i] = static_cast<T>(
          Exp(static_cast<AccT>(buf_src[i]) - max_val[i]) /
          (sum_val[i] + 1e-6f));
    }
    reinterpret_cast<VecT*>(&dst_row[dim_id * D])[0] = vec_dst;
  }
}

template<typename T, int VECSIZE>
inline void LaunchLoopDimSoftmaxForwardKernel(cudaStream_t &stream, const T* in_data,
                  T* out_data, const int N, const int dim, const int D) {
  int loop_num = N * D / VECSIZE;
  int threads = std::min(loop_num, 1024);
  int grids = (loop_num + threads - 1) / threads;
  using AccT = typename GetAccType<T>::type;

  KeLoopDimSoftmaxForward<T, AccT, VECSIZE>
      <<<grids, threads, 0, stream>>>(
      out_data, in_data, N, dim, D);
}

template<typename T>
inline void LaunchLoopDimSoftmaxForward(cudaStream_t &stream, const T* in_data,
                  T* out_data, const int N, const int dim, const int D) {
  if(D % 4 == 0) {
    LaunchLoopDimSoftmaxForwardKernel<T, 4>(
      stream, in_data, out_data, N, dim, D);
  } else if (D % 2 == 0) {
    LaunchLoopDimSoftmaxForwardKernel<T, 2>(
      stream, in_data, out_data, N, dim, D);
  } else {
    LaunchLoopDimSoftmaxForwardKernel<T, 1>(
      stream, in_data, out_data, N, dim, D);
  }
}

/*****************************************************************/

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT, int VECSIZE>
__global__ void KeSpandDimDSoftmaxForward(T* __restrict__ dst,
      const T* __restrict__ src, const int N, const int dim, const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec_src, vec_dst;
  T* buf_src = reinterpret_cast<T*>(&vec_src);
  T* buf_dst = reinterpret_cast<T*>(&vec_dst);

  const int tid = threadIdx.x;
  const int vec_id = tid * VECSIZE;
  const int BlockDim = blockDim.x;
  const int vec_num = BlockDim * VECSIZE;
  for(int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const T* __restrict__ src_row = src + out_id * dim * D;
    T* __restrict__ dst_row = dst + out_id * dim * D;

    // Compute each thread's max value
    AccT max_val[VECSIZE];
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      max_val[i] = -std::numeric_limits<AccT>::infinity();
    }
    for(int id = vec_id; id < dim * D; id += vec_num) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[id])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        max_val[i] = max(static_cast<AccT>(buf_src[i]), max_val[i]);
      }
    }
    // write to shared memory
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      s_data[vec_id + i] = max_val[i];
    }
    __syncthreads();
    // compute total max value
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      for(int k = (vec_id + i) % D; k < vec_num; k += D) {
        max_val[i] = max(s_data[k], max_val[i]);
      }
    }
    // Compute each thread's sum value
    AccT sum_val[VECSIZE]{0};
    for(int id = vec_id; id < dim * D; id += vec_num) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[id])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        sum_val[i] += Exp(static_cast<AccT>(buf_src[i]) - max_val[i]);
      }
    }
    // write to shared memory
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      s_data[vec_id + i] = sum_val[i];
    }
    __syncthreads();
    // compute total sum value
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      sum_val[i] = 0;
      for(int k = (vec_id + i) % D; k < vec_num; k += D) {
        sum_val[i] += s_data[k];
      }
    }
    // Compute finally softmax result
    // TODO(jiangcheng): how to eliminate twice Exp
    for(int id = vec_id; id < dim * D; id += vec_num) {
      vec_src = reinterpret_cast<const VecT*>(&src_row[id])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_dst[i] = static_cast<T>(
          Exp(static_cast<AccT>(buf_src[i]) - max_val[i]) /
          (sum_val[i] + 1e-6f));
      }
      reinterpret_cast<VecT*>(&dst_row[id])[0] = vec_dst;
    }
  }
}

template<typename T, int VECSIZE>
inline void LaunchSpandDimDSoftmaxForwardKernel(cudaStream_t &stream,
                        const T* in_data, T* out_data,
                        const int N, const int dim, const int D) {
  const int grids = N;
  const int threads = std::min(dim * D, 256);
  using AccT = typename GetAccType<T>::type;

  KeSpandDimDSoftmaxForward<T, AccT, VECSIZE>
    <<<grids, threads, threads * VECSIZE * sizeof(AccT), stream>>>(
    out_data, in_data, N, dim, D);
}

template<typename T>
inline void LaunchSpandDimDSoftmaxForward(cudaStream_t &stream,
                        const T* in_data, T* out_data,
                        const int N, const int dim, const int D) {
  const int cols = dim * D;
  if(cols % 4 == 0) {
    LaunchSpandDimDSoftmaxForwardKernel<T, 4>(
      stream, in_data, out_data, N, dim, D);
  } else if(cols % 2 == 0) {
    LaunchSpandDimDSoftmaxForwardKernel<T, 2>(
      stream, in_data, out_data, N, dim, D);
  } else {
    LaunchSpandDimDSoftmaxForwardKernel<T, 1>(
      stream, in_data, out_data, N, dim, D);
  }
}

template <typename T, typename AccT, int Log2Elements>
__global__ void softmax_warp_backward(T* gradInput, const T* grad,
                                      const T* output, int batch_size,
                                      int stride, int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x % warp_size_softmax;

  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // load data from global memory
  AccT grad_reg[WARP_BATCH][WARP_ITERATIONS];
  AccT output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        grad_reg[i][it] =
            static_cast<AccT>(grad[i * element_count + it * warp_size_softmax]);
        output_reg[i][it] = static_cast<AccT>(
            output[i * element_count + it * warp_size_softmax]);
      } else {
        grad_reg[i][it] = AccT(0);
        output_reg[i][it] = AccT(0);
      }
    }
  }

  AccT sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        // compute gradients
        gradInput[i * element_count + it * warp_size_softmax] =
            (grad_reg[i][it] - output_reg[i][it] * sum[i]);
      }
    }
  }
}

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT>
__global__ void KeSpandDimDSoftmaxBackward(T *dx, const T *out, const T *dout,
                        const int N, const int dim, const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  const int tid = threadIdx.x;
  const int BlockDim = blockDim.x;
  for(int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const T *src_out = out + out_id * dim * D;
    const T *src_dout = dout + out_id * dim * D;
    T *dst_dx = dx + out_id * dim * D;

    // Compute each thread's sum value
    AccT sum_val(0);
    for(int id = tid; id < dim * D; id += BlockDim)
      sum_val += static_cast<AccT>(src_out[id]) *
                 static_cast<AccT>(src_dout[id]);
    // write to shared memory
    s_data[tid] = sum_val;
    __syncthreads();
    // compute total sum value
    sum_val = 0;
    for(int id = tid % D; id < BlockDim; id += D)
      sum_val += s_data[id];

    // Compute finally softmax result
    for(int id = tid; id < dim * D; id += BlockDim)
      dst_dx[id] =
        static_cast<T>(static_cast<AccT>(src_out[id]) *
        (static_cast<AccT>(src_dout[id]) - sum_val));
  }
}

template <typename T>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto dims = x->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int max_dim = 320;
    bool optimize = false;
    auto stream = ctx.cuda_device_context().stream();
    if (D == 1) {
      if (dim < max_dim && sizeof(T) <= 4) {
        optimize = true;
        LaunchWarpSoftmaxForward(stream, x->data<T>(), out_data, N, dim);
      } else if(dim <= 1024) {
        optimize = true;
        LaunchD1WarpSoftmaxForward(stream, x->data<T>(), out_data, N, dim);
      } else if(dim <= 4096) {
        optimize = true;
        LaunchD1BlockSharedSoftmaxForward(stream, x->data<T>(), out_data, N, dim);
      } else {
        optimize = true;
        LaunchD1BlockSoftmaxForward(stream, x->data<T>(), out_data, N, dim);
      }
    } else {
      if(D <= 256) {
        optimize = true;
        LaunchSpandDimDSoftmaxForward(stream, x->data<T>(), out_data, N, dim, D);
      } else if(dim <= 512) {
        optimize = true;
        LaunchLoopDimSoftmaxForward(stream, x->data<T>(), out_data, N, dim, D);
      }
    }
    if (!optimize) {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward(
          handle, platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
#endif
    }
  }
};

template <typename T>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->data<T>();

    auto dims = out->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr bool AccT_use_float =
        std::is_same<T, float>::value ||
        std::is_same<T, platform::float16>::value;
    bool optimize = false;
    if(D <= 1024) {
      optimize = true;
      const int grids = N;
      const int threads = D * (1024 / D);

      if(AccT_use_float) {
        KeSpandDimDSoftmaxBackward<T, float>
          <<<grids, threads, threads * sizeof(float),
          ctx.cuda_device_context().stream()>>>(
          dx_data, out->data<T>(), dout->data<T>(), N, dim, D);
      } else {
        KeSpandDimDSoftmaxBackward<T, double>
          <<<grids, threads, threads * sizeof(double),
          ctx.cuda_device_context().stream()>>>(
          dx_data, out->data<T>(), dout->data<T>(), N, dim, D);
      }
    }
    if (!optimize) {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxBackward(
          handle, platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(),
          desc_, dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(), desc_,
          dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#else
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#endif

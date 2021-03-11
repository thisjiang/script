// C system file
#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Library file
#include "../common.h"

/************************************************************************/

template<typename T, typename AccT, int COLS>
__global__ void NoVec_KeD1WarpSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim) {
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int tid = threadIdx.x;

  for(int row = warp_id; row < N; row += gridDim.x * blockDim.y) {
    int offset = row * dim;
    const T* __restrict__ out_row = out + offset;
    const T* __restrict__ dout_row = dout + offset;
    T* dx_row = dx + offset;
    // Load src data from global memory to register,
    // and compute max value
    AccT buf_out[COLS], buf_dout[COLS];
    AccT sum_val(0);
    int real_cols = 0;
  #pragma unroll
    for(int col = 0; col < COLS; col ++) {
      int src_col =  tid + col * WARP_SIZE;
      if(src_col >= dim) break;
      buf_out[col] = static_cast<AccT>(out_row[src_col]);
      buf_dout[col] = static_cast<AccT>(dout_row[src_col]);

      sum_val += buf_out[col] * buf_dout[col];
      real_cols ++;
    }
    sum_val = warpReduceSum(sum_val, 0xffffffff);
    // compute softmax result
  #pragma unroll
    for(int col = 0; col < COLS; col ++) {
      int dst_col =  tid + col * WARP_SIZE;
      if(dst_col >= dim) break;
      dx_row[dst_col] =
          static_cast<T>(buf_out[col] * (buf_dout[col] - sum_val));
    }
  }
}

template<typename T> struct GetAccType {using type = T;};
template<> struct GetAccType<half> {using type = float;};

template<typename T, int N> struct GetVecType;
template<typename T> struct GetVecType<T, 1> {using type = T;};
template<> struct GetVecType<half, 2> {using type = half2;};
template<> struct GetVecType<half, 4> {using type = float2;};
template<> struct GetVecType<float, 2> {using type = float2;};
template<> struct GetVecType<float, 4> {using type = float4;};
template<> struct GetVecType<double, 2> {using type = double2;};
template<> struct GetVecType<double, 4> {using type = double4;};

// when D == 1 && 320 <= dim <= 512, using KeD1WarpSoftmaxBackward faster,
// each warp compute one row's element,
// each thread compute COLS element of dim and store in register
template<typename T, typename AccT, int COLS, int VECSIZE>
__global__ void KeD1WarpSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim) {
  static_assert(COLS % VECSIZE == 0);
  constexpr int num_vec = COLS / VECSIZE;
  const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  for(int row = warp_id; row < N; row += gridDim.x * blockDim.y) {
    const int offset = row * dim;
    const T* __restrict__ out_row = out + offset;
    const T* __restrict__ dout_row = dout + offset;
    T* __restrict__ dx_row = dx + offset;
    // vectorization for global memory coalescing 
    using VecT = typename GetVecType<T, VECSIZE>::type;
    VecT vec;
    T* buf_src = reinterpret_cast<T*>(&vec);
    // Load src data from global memory to register,
    // and compute max value
    AccT buf_out[COLS], buf_dout[COLS];
    int real_cols = 0;
    AccT sum_val(0);
#pragma unroll
    for(int col = 0; col < num_vec; col ++) {
      int src_col =  (tid + col * WARP_SIZE) * VECSIZE;
      if(src_col >= dim) break;
      // Vectorize read out data 
      vec = reinterpret_cast<const VecT*>(&out_row[src_col])[0];
      AccT* out_acc = buf_out + real_cols;
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        out_acc[i] = static_cast<AccT>(buf_src[i]);
      }
      // Vectorize read dout data 
      vec = reinterpret_cast<const VecT*>(&dout_row[src_col])[0];
      AccT* dout_acc = buf_dout + real_cols;
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        dout_acc[i] = static_cast<AccT>(buf_src[i]);
        sum_val += out_acc[i] * dout_acc[i];
      }
      real_cols += VECSIZE;
    }
    sum_val = warpReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    real_cols = 0;
#pragma unroll
    for(int col = 0; col < num_vec; col ++) {
      int dst_col =  (tid + col * WARP_SIZE) * VECSIZE;
      if(dst_col >= dim) break;
      AccT* out_acc = buf_out + real_cols;
      AccT* dout_acc = buf_dout + real_cols;
  #pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {        
        buf_src[i] = 
            static_cast<T>(out_acc[i] * (dout_acc[i] - sum_val));
      }
      reinterpret_cast<VecT*>(&dx_row[dst_col])[0] = vec;
      real_cols += VECSIZE;
    }
  }
}

template<typename T, int COLS, int VECSIZE>
void LaunchD1WarpSoftmaxBackwardKernel(const cudaStream_t &stream, T *dx_data,
      const T* out_data, const T* dout_data, const int N, const int dim) {
  int N_b = std::min(8, N);
  dim3 threads(WARP_SIZE, N_b);
  int grids = (N + N_b - 1) / N_b;
  using AccT = typename GetAccType<T>::type;

  KeD1WarpSoftmaxBackward<T, AccT, COLS, VECSIZE>
    <<<grids, threads, 0, stream>>>(
      dx_data, out_data, dout_data, N, dim);
}

#define LAUNCH_D1WARP_BACKWARD_COLS(COLS)               \
  case COLS:                                            \
    LaunchD1WarpSoftmaxBackwardKernel<T, COLS, VECSIZE>(\
        stream, dx_data, out_data, dout_data, N, dim);  \
    break;

template<typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 1, void>::type DispatchD1WarpSoftmaxBackward(
        const cudaStream_t &stream, T *dx_data, const T* out_data, const T* dout_data,
        const int N, const int dim, const int cols_per_thread) {
  switch (cols_per_thread) {
    LAUNCH_D1WARP_BACKWARD_COLS(1)
    LAUNCH_D1WARP_BACKWARD_COLS(2)
    LAUNCH_D1WARP_BACKWARD_COLS(3)
    LAUNCH_D1WARP_BACKWARD_COLS(4)
    LAUNCH_D1WARP_BACKWARD_COLS(5)
    LAUNCH_D1WARP_BACKWARD_COLS(6)
    LAUNCH_D1WARP_BACKWARD_COLS(7)
    LAUNCH_D1WARP_BACKWARD_COLS(8)
    LAUNCH_D1WARP_BACKWARD_COLS(9)
    LAUNCH_D1WARP_BACKWARD_COLS(10)
    LAUNCH_D1WARP_BACKWARD_COLS(11)
    LAUNCH_D1WARP_BACKWARD_COLS(12)
    LAUNCH_D1WARP_BACKWARD_COLS(13)
    LAUNCH_D1WARP_BACKWARD_COLS(14)
    LAUNCH_D1WARP_BACKWARD_COLS(15)
    LAUNCH_D1WARP_BACKWARD_COLS(16)
    default:
      fprintf(stderr, "[DispatchD1WarpSoftmaxBackward::VECSIZE==1] "
                      "BAD PARAM (%d, %d) with %d\n",
              N, dim, cols_per_thread);
      // (32 for out + 32 for dout) * 4 byte for float = 256 > 255
      // so only support max(COLS) == 16
      break;
  }
}

template<typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 2, void>::type DispatchD1WarpSoftmaxBackward(
        const cudaStream_t &stream, T *dx_data, const T* out_data, const T* dout_data,
        const int N, const int dim, const int cols_per_thread) {
  switch (cols_per_thread) {
    LAUNCH_D1WARP_BACKWARD_COLS(2)
    LAUNCH_D1WARP_BACKWARD_COLS(4)
    LAUNCH_D1WARP_BACKWARD_COLS(6)
    LAUNCH_D1WARP_BACKWARD_COLS(8)
    LAUNCH_D1WARP_BACKWARD_COLS(10)
    LAUNCH_D1WARP_BACKWARD_COLS(12)
    LAUNCH_D1WARP_BACKWARD_COLS(14)
    LAUNCH_D1WARP_BACKWARD_COLS(16)
    default:
      fprintf(stderr, "[DispatchD1WarpSoftmaxBackward::VECSIZE==2] "
                      "BAD PARAM (%d, %d) with %d\n",
              N, dim, cols_per_thread);
      break;
  }
}

template<typename T, int VECSIZE>
typename std::enable_if<VECSIZE == 4, void>::type DispatchD1WarpSoftmaxBackward(
        const cudaStream_t &stream, T *dx_data, const T* out_data, const T* dout_data,
        const int N, const int dim, const int cols_per_thread) {
  switch (cols_per_thread) {
    LAUNCH_D1WARP_BACKWARD_COLS(4)
    LAUNCH_D1WARP_BACKWARD_COLS(8)
    LAUNCH_D1WARP_BACKWARD_COLS(12)
    LAUNCH_D1WARP_BACKWARD_COLS(16)
    default:
      fprintf(stderr, "[DispatchD1WarpSoftmaxBackward::VECSIZE==4] "
                      "BAD PARAM (%d, %d) with %d\n",
              N, dim, cols_per_thread);
      break;
  }
}
#undef LAUNCH_D1WARP_BACKWARD_COLS

template<typename T>
inline void LaunchD1WarpSoftmaxBackward(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim) {
  const int cols_per_thread = (dim + WARP_SIZE - 1) / WARP_SIZE;

  if(dim % 4 == 0 && cols_per_thread % 4 == 0) {
    DispatchD1WarpSoftmaxBackward<T, 4>(
      stream, dx_data, out_data, dout_data, N, dim, cols_per_thread);
  } else if(dim % 2 == 0 && cols_per_thread % 2 == 0) {
    DispatchD1WarpSoftmaxBackward<T, 2>(
      stream, dx_data, out_data, dout_data, N, dim, cols_per_thread);
  } else {
    DispatchD1WarpSoftmaxBackward<T, 1>(
      stream, dx_data, out_data, dout_data, N, dim, cols_per_thread);
  }
}

/************************************************************************/

// when D == 1 && 512 < dim <= 2048, using KeD1BlockSharedSoftmaxBackward
// each block compute a row, and synchronization by shared memory
// each thread compute VECSIZE elements of dim, and store in shared memory
template<typename T, typename AccT, int VECSIZE>
__global__ void KeD1BlockSharedSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* buf_out = reinterpret_cast<AccT*>(s_mem);
  AccT* buf_dout = buf_out + dim;

  const int tid = threadIdx.x;
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec;
  T* buf_src = reinterpret_cast<T*>(&vec);

  for(int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T* __restrict__ out_row = out + offset;
    const T* __restrict__ dout_row = dout + offset;
    T* __restrict__ dx_row = dx + offset;

    // compute max value
    AccT sum_val(0);
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      // Vectorize read out data 
      vec = reinterpret_cast<const VecT*>(&out_row[col])[0];
      AccT* out_acc = buf_out + col;
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        out_acc[i] = static_cast<AccT>(buf_src[i]);
      }
      // Vectorize read dout data 
      vec = reinterpret_cast<const VecT*>(&dout_row[col])[0];
      AccT* dout_acc = buf_dout + col;
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        dout_acc[i] = static_cast<AccT>(buf_src[i]);
        sum_val += out_acc[i] * dout_acc[i];
      }
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      AccT* out_acc = buf_out + col;
      AccT* dout_acc = buf_dout + col;
  #pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_src[i] = static_cast<T>(out_acc[i] * (dout_acc[i] - sum_val));
      }
      reinterpret_cast<VecT*>(&dx_row[col])[0] = vec;
    }
  }
}

template<typename T, int VECSIZE>
void LaunchD1BlockSharedSoftmaxBackwardKernel(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim) {
  const int threads = std::min(dim, MAX_BLOCK_DIM);
  const int grids = N;
  using AccT = typename GetAccType<T>::type;

  KeD1BlockSharedSoftmaxBackward<T, AccT, VECSIZE>
    <<<grids, threads, dim * 2 * sizeof(AccT), stream>>>(
    dx_data, out_data, dout_data, N, dim);
}

template<typename T>
inline void LaunchD1BlockSharedSoftmaxBackward(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim) {
  if(dim % 4 == 0) {
    LaunchD1BlockSharedSoftmaxBackwardKernel<T, 4>(
      stream, dx_data, out_data, dout_data, N, dim);
  } else if(dim % 2 == 0) {
    LaunchD1BlockSharedSoftmaxBackwardKernel<T, 2>(
      stream, dx_data, out_data, dout_data, N, dim);
  } else {
    LaunchD1BlockSharedSoftmaxBackwardKernel<T, 1>(
      stream, dx_data, out_data, dout_data, N, dim);
  }
}

/************************************************************************/

// when D == 1 && 2048 < dim, using KeD1BlockSoftmaxBackward,
// each block compute a row, and synchronization by blockReduce,
// each thread compute VECSIZE elements of dim
template<typename T, typename AccT, int VECSIZE>
__global__ void KeD1BlockSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim) {
  const int tid = threadIdx.x;
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec_out, vec_dout, vec_dx;
  T* buf_out = reinterpret_cast<T*>(&vec_out);
  T* buf_dout = reinterpret_cast<T*>(&vec_dout);
  T* buf_dx = reinterpret_cast<T*>(&vec_dx);

  for(int row = blockIdx.x; row < N; row += gridDim.x) {
    const int offset = row * dim;
    const T* __restrict__ out_row = out + offset;
    const T* __restrict__ dout_row = dout + offset;
    T* __restrict__ dx_row = dx + offset;
    // compute sum value
    AccT sum_val(0);
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_out = reinterpret_cast<const VecT*>(&out_row[col])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout_row[col])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        sum_val += static_cast<AccT>(buf_out[i]) *
                   static_cast<AccT>(buf_dout[i]);
      }
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);
    // compute softmax result
    for(int col = tid * VECSIZE; col < dim; col += blockDim.x * VECSIZE) {
      vec_out = reinterpret_cast<const VecT*>(&out_row[col])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout_row[col])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_dx[i] = static_cast<T>(static_cast<AccT>(buf_out[i]) *
                    (static_cast<AccT>(buf_dout[i]) - sum_val));
      }
      reinterpret_cast<VecT*>(&dx_row[col])[0] = vec_dx;
    }
  }
}

template<typename T, int VECSIZE>
void LaunchD1BlockSoftmaxBackwardKernel(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim) {
  const int threads = std::min(dim, MAX_BLOCK_DIM);
  const int grids = N;
  using AccT = typename GetAccType<T>::type;

  KeD1BlockSoftmaxBackward<T, AccT, VECSIZE>
    <<<grids, threads, 0, stream>>>(
    dx_data, out_data, dout_data, N, dim);
}

template<typename T>
inline void LaunchD1BlockSoftmaxBackward(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim) {
  if(dim % 4 == 0) {
    LaunchD1BlockSoftmaxBackwardKernel<T, 4>(
      stream, dx_data, out_data, dout_data, N, dim);
  } else if(dim % 2 == 0) {
    LaunchD1BlockSoftmaxBackwardKernel<T, 2>(
      stream, dx_data, out_data, dout_data, N, dim);
  } else {
    LaunchD1BlockSoftmaxBackwardKernel<T, 1>(
      stream, dx_data, out_data, dout_data, N, dim);
  }
}

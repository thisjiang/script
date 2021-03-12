// class header file
#include "softmax_backward_D1.h"
// C system file
#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// C++ system file
#include <iostream>
#include <vector>
// Library file
#include "../common.h"
#include "../cudnn_helper.h"

constexpr int LOOPNUM = 10;

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

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
union vec_t<double, 4> {
  double4 s;
  float v[4];
};

template <>
union vec_t<float, 4> {
  float4 s;
  float v[4];
};

template <>
union vec_t<float16, 4> {
  int2 s;
  float16 v[4];
};

/************************************************************************/

template <typename T, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxBackward(T* dst, const T* grad, const T* src,
                                   const int batch_size,
                                   const int softmax_ele) {
  const int offset =
      blockIdx.x * softmax_ele * WARP_PER_BLOCK + threadIdx.x * VPT;

  float local_sum_gy = 0.f;
  vec_t<T, VPT> local_grad;
  vec_t<T, VPT> local_src;

  local_grad.s =
      reinterpret_cast<const decltype(local_grad.s)*>(&grad[offset])[0];
  local_src.s = reinterpret_cast<const decltype(local_src.s)*>(&src[offset])[0];

  for (int i = 0; i < VPT; ++i) {
    local_sum_gy += static_cast<float>(local_grad.v[i]) *
                    static_cast<float>(local_src.v[i]);
  }
  float sum_gy = warpReduceSum<float>(local_sum_gy, 0xffffffff);

  vec_t<T, VPT> local_dst;
  for (int i = 0; i < VPT; ++i) {
    local_dst.v[i] =
        static_cast<T>(static_cast<float>(local_src.v[i]) *
                       (static_cast<float>(local_grad.v[i]) - sum_gy));
  }
  reinterpret_cast<decltype(local_dst.s)*>(&dst[offset])[0] = local_dst.s;
}


template<typename T>
float TimeOfOldVecSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  CUDNNHandle handle(context.stream());

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr int warps_per_block = 4;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    if (std::is_same<T, float16>::value) {
      VecSoftmaxBackward<T, 4, warps_per_block><<<
          N / warps_per_block, warps_per_block * WARP_SIZE, 0,
          context.stream()>>>(dx_data, dout_data, out_data, N, dim);
    } else if (std::is_same<T, float>::value) {
      VecSoftmaxBackward<T, 4, warps_per_block><<<
          N / warps_per_block, warps_per_block * WARP_SIZE, 0,
          context.stream()>>>(dx_data, dout_data, out_data, N, dim);
    } else {
      assert(false && "not support");
    }
  }

  float cost = clock->stop();

  return cost;
}

/************************************************************************/

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_sum(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T sum_val = __shfl_xor_sync(0xFFFFFFFF, sum[i], offset, WARP_SIZE);
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
      T max_val = __shfl_xor_sync(0xFFFFFFFF, sum[i], offset, WARP_SIZE);
      sum[i] = max(sum[i], max_val);
    }
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

#define CUDA_KERNEL_LOOP(i, num)                             \
  int64_t __index__ = blockIdx.x * blockDim.x + threadIdx.x; \
  for (int i = __index__; __index__ < (num);                 \
       __index__ += blockDim.x * gridDim.x, i = __index__)

template <typename T>
__global__ void MultiplyCUDAKernel(T* C, const T* A, const T* B, int N) {
  CUDA_KERNEL_LOOP(i, N) {
    C[i] = static_cast<T>(static_cast<float>(A[i]) * static_cast<float>(B[i]));
  }
}

template<typename T>
void LaunchOldWarpSoftmaxBackward(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                T *mul_grad, const int N, const int dim) {
  int log2_elements = log2_ceil(dim);
  const int next_power_of_two = 1 << log2_elements;
  int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;
  int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int threads_per_block = 128;
  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (N + batches_per_block - 1) / batches_per_block;
  dim3 threads(warp_size, warps_per_block, 1);

  const int grids = (N * dim + threads_per_block - 1) / threads_per_block;
  MultiplyCUDAKernel<<<grids, threads_per_block, 0, stream>>>(
    mul_grad, out_data, dout_data, N * dim);

#define LAUNCH_SOFTMAX_WARP_BACKWARD(Log2Elements)                 \
  case Log2Elements:                                               \
    softmax_warp_backward<T, float, Log2Elements><<<               \
        blocks, threads, 0, stream>>>(                   \
        dx_data, mul_grad, out_data, N, dim, dim); \
    break;

  switch (log2_elements) {
    LAUNCH_SOFTMAX_WARP_BACKWARD(0);  // 1
    LAUNCH_SOFTMAX_WARP_BACKWARD(1);  // 2
    LAUNCH_SOFTMAX_WARP_BACKWARD(2);  // 4
    LAUNCH_SOFTMAX_WARP_BACKWARD(3);  // 8
    LAUNCH_SOFTMAX_WARP_BACKWARD(4);  // 16
    LAUNCH_SOFTMAX_WARP_BACKWARD(5);  // 32
    LAUNCH_SOFTMAX_WARP_BACKWARD(6);  // 64
    LAUNCH_SOFTMAX_WARP_BACKWARD(7);  // 128
    LAUNCH_SOFTMAX_WARP_BACKWARD(8);  // 256
    LAUNCH_SOFTMAX_WARP_BACKWARD(9);  // 512
    default:
      break;
  }
#undef LAUNCH_SOFTMAX_WARP_BACKWARD
}

template<typename T>
float TimeOfOldWarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  if(sizeof(T) <= 4) return 0.0f;

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  MallocDevice<T> mul_grad(N * dim, context);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchOldWarpSoftmaxBackward<T>(
      context.stream(), dx_data, out_data, dout_data, mul_grad.data(), N, dim);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfCudnnSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  CUDNNHandle handle(context.stream());

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  // the format is not used now, will add later
  std::vector<int> tensor_dims = {N, dim, D, 1};
  int tensor_size = tensor_dims.size();
  std::vector<int> strides(tensor_size);
  strides[tensor_size - 1] = 1;
  for (int i = tensor_size - 2; i >= 0; i--) {
    strides[i] = tensor_dims[i + 1] * strides[i + 1];
  }
  // cudnnTensorFormat_t layout = CUDNN_TENSOR_NCHW;
  cudnnDataType_t type = GetCudnnDataType<T>();

  cudnnTensorDescriptor_t desc_;
  cudnnCreateTensorDescriptor(&desc_);
  cudnnSetTensorNdDescriptor(desc_, type, tensor_size, tensor_dims.data(), strides.data());

  auto cudnn_hd = handle.cudnn_handle();
  auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                : CUDNN_SOFTMAX_MODE_CHANNEL;
  const T one = static_cast<T>(1), zero = static_cast<T>(0);
  cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    status = cudnnSoftmaxBackward(
        cudnn_hd, CUDNN_SOFTMAX_ACCURATE, mode,
        &one, desc_, out_data, desc_, dout_data,
        &zero, desc_, dx_data);
  }
  float cost = clock->stop();

  cudnnDestroyTensorDescriptor(desc_);
  auto err = handle.CheckError(status);
  if(err != EMPTY_STRING) {
    fprintf(stderr, "%s Cudnn ERROR: %s\n",
            ToString(dims).c_str(), err);
    return 0.0f;
  }

  return cost;
}

/************************************************************************/
// When D is larg and dim is small:
// Each block arranged by N，each thread arranged by D
// each thread compute dim number's softmax
template<typename T, typename AccT>
__global__ void NoVec_KeLoopDimSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim, const int D) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_id = tid / D;
  if(out_id >= N) return;
  const int in_id = tid - out_id * D;

  const T *src_out = out + out_id * dim * D + in_id;
  const T *src_dout = dout + out_id * dim * D + in_id;
  T *dst_dx = dx + out_id * dim * D + in_id;

  // compute multiply then sum value
  AccT sum_val(0);
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    sum_val += static_cast<AccT>(src_out[dim_id * D]) *
               static_cast<AccT>(src_dout[dim_id * D]);
  }

  // compute softmax gradient value
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    dst_dx[dim_id * D] =
        static_cast<T>(static_cast<AccT>(src_out[dim_id * D]) *
        (static_cast<AccT>(src_dout[dim_id * D]) - sum_val));
}

template<typename T>
float TimeOfNoVecLoopDimSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  CUDNNHandle handle(context.stream());

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  using AccT = typename GetAccType<T>::type;
  int threads = std::min(N * D, 256);
  int grids = (N * D + threads - 1) / threads;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    NoVec_KeLoopDimSoftmaxBackward<T, AccT>
      <<<grids, threads, 0, context.stream()>>>(
      dx_data, out_data, dout_data, N, dim, D);
  }

  float cost = clock->stop();

  return cost;
}

// When D is larg and dim is small:
// Each block arranged by N，each thread arranged by D
// each thread compute dim number's softmax
template<typename T, typename AccT, int VECSIZE>
__global__ void KeLoopDimSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim, const int D) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int vec_id = tid * VECSIZE;
  const int out_id = vec_id / D;
  if(out_id >= N) return;
  const int in_id = vec_id - out_id * D;
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec_out, vec_dout, vec_dx;
  T* buf_out = reinterpret_cast<T*>(&vec_out);
  T* buf_dout = reinterpret_cast<T*>(&vec_dout);
  T* buf_dx = reinterpret_cast<T*>(&vec_dx);

  const int offset = out_id * dim * D + in_id;
  const T* __restrict__ out_row = out + offset;
  const T* __restrict__ dout_row = dout + offset;
  T* __restrict__ dx_row = dx + offset;
  // compute multiply then sum value
  AccT sum_val[VECSIZE]{0};
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    vec_out = reinterpret_cast<const VecT*>(&out_row[dim_id * D])[0];
    vec_dout = reinterpret_cast<const VecT*>(&dout_row[dim_id * D])[0];
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      sum_val[i] += static_cast<AccT>(buf_out[i]) *
                    static_cast<AccT>(buf_dout[i]);
    }
  }
  // compute softmax gradient value
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    vec_out = reinterpret_cast<const VecT*>(&out_row[dim_id * D])[0];
    vec_dout = reinterpret_cast<const VecT*>(&dout_row[dim_id * D])[0];
#pragma unroll
    for(int i = 0; i < VECSIZE; i ++) {
      buf_dx[i] = static_cast<T>(static_cast<AccT>(buf_out[i]) *
          (static_cast<AccT>(buf_dout[i]) - sum_val[i]));
    }
    reinterpret_cast<VecT*>(&dx_row[dim_id * D])[0] = vec_dx;
  }
}

template<typename T, int VECSIZE>
inline void LaunchLoopDimSoftmaxBackwardKernel(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim, const int D) {
  int loop_num = N * D / VECSIZE;
  int threads = std::min(loop_num, MAX_BLOCK_DIM);
  int grids = (loop_num + threads - 1) / threads;
  using AccT = typename GetAccType<T>::type;

  KeLoopDimSoftmaxBackward<T, AccT, VECSIZE>
      <<<grids, threads, 0, stream>>>(
      dx_data, out_data, dout_data, N, dim, D);
}

template<typename T>
inline void LaunchLoopDimSoftmaxBackward(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim, const int D) {
  const int num = N * D / MAX_BLOCK_DIM;
  const int occupy_count = CUDAStream::GetSMCount() * 4;
  if(D % 4 == 0 && num / 4 >= occupy_count) {
    LaunchLoopDimSoftmaxBackwardKernel<T, 4>(
      stream, dx_data, out_data, dout_data, N, dim, D);
  } else if (D % 2 == 0 && num / 2 >= occupy_count) {
    LaunchLoopDimSoftmaxBackwardKernel<T, 2>(
      stream, dx_data, out_data, dout_data, N, dim, D);
  } else {
    LaunchLoopDimSoftmaxBackwardKernel<T, 1>(
      stream, dx_data, out_data, dout_data, N, dim, D);
  }
}

template<typename T>
float TimeOfLoopDimSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  CUDNNHandle handle(context.stream());

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchLoopDimSoftmaxBackward<T>(
      context.stream(), dx_data, out_data, dout_data, N, dim, D);
  }

  float cost = clock->stop();

  return cost;
}

/************************************************************************/

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT>
__global__ void NoVec_KeSpandDimDSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
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

template<typename T>
float TimeOfNoVecSpandDimDSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  if(D > 1024) return 0.0f;
  using AccT = typename GetAccType<T>::type;

  const int grids = N;
  const int threads = D * (1024 / D);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    NoVec_KeSpandDimDSoftmaxBackward<T, AccT>
      <<<grids, threads, threads * sizeof(AccT), context.stream()>>>(
      dx_data, out_data, dout_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT, int VECSIZE>
__global__ void KeSpandDimDSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim, const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);
  // vectorization for global memory coalescing
  using VecT = typename GetVecType<T, VECSIZE>::type;
  VecT vec_out, vec_dout, vec_dx;
  T* buf_out = reinterpret_cast<T*>(&vec_out);
  T* buf_dout = reinterpret_cast<T*>(&vec_dout);
  T* buf_dx = reinterpret_cast<T*>(&vec_dx);

  const int vec_id = threadIdx.x * VECSIZE;
  const int vec_num = blockDim.x * VECSIZE;
  for(int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const int offset = out_id * dim * D;
    const T* __restrict__ out_row = out + offset;
    const T* __restrict__ dout_row = dout + offset;
    T* __restrict__ dx_row = dx + offset;
    // Compute each thread's sum value
    AccT sum_val[VECSIZE]{0};
    for(int id = vec_id; id < dim * D; id += vec_num) {
      vec_out = reinterpret_cast<const VecT*>(&out_row[id])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout_row[id])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        sum_val[i] += static_cast<AccT>(buf_out[i]) *
                      static_cast<AccT>(buf_dout[i]);
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
    for(int id = vec_id; id < dim * D; id += vec_num) {
      vec_out = reinterpret_cast<const VecT*>(&out_row[id])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout_row[id])[0];
#pragma unroll
      for(int i = 0; i < VECSIZE; i ++) {
        buf_dx[i] = static_cast<T>(static_cast<AccT>(buf_out[i]) *
          (static_cast<AccT>(buf_dout[i]) - sum_val[i]));
      }
      reinterpret_cast<VecT*>(&dx_row[id])[0] = vec_dx;
    }
  }
}

template<typename T, int VECSIZE>
inline void LaunchSpandDimDSoftmaxBackwardKernel(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim, const int D) {
  int D_size = 128;
  if(D <= 256) D_size = 256;
  else if(D <= 512) D_size = 512;
  else if(D <= MAX_BLOCK_DIM) D_size = MAX_BLOCK_DIM;
  else assert(false && "not support");

  const int grids = N;
  const int threads = D * (D_size / D);
  using AccT = typename GetAccType<T>::type;

  KeSpandDimDSoftmaxBackward<T, AccT, VECSIZE>
    <<<grids, threads, threads * VECSIZE * sizeof(AccT), stream>>>(
    dx_data, out_data, dout_data, N, dim, D);
}

template<typename T>
inline void LaunchSpandDimDSoftmaxBackward(const cudaStream_t &stream,
                T *dx_data, const T* out_data, const T* dout_data,
                const int N, const int dim, const int D) {
  if(D % 4 == 0) {
    LaunchSpandDimDSoftmaxBackwardKernel<T, 4>(
      stream, dx_data, out_data, dout_data, N, dim, D);
  } else if(D % 2 == 0) {
    LaunchSpandDimDSoftmaxBackwardKernel<T, 2>(
      stream, dx_data, out_data, dout_data, N, dim, D);
  } else {
    LaunchSpandDimDSoftmaxBackwardKernel<T, 1>(
      stream, dx_data, out_data, dout_data, N, dim, D);
  }
}

template<typename T>
float TimeOfSpandDimDSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  if(D > MAX_BLOCK_DIM) return 0.0f;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchSpandDimDSoftmaxBackward<T>(
      context.stream(), dx_data, out_data, dout_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfD1WarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim <= 512);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
    LaunchD1WarpSoftmaxBackward<T>(
      context.stream(), dx_data, out_data, dout_data, N, dim);
  float cost = clock->stop();

  return cost;
}

/************************************************************************/

template<typename T>
float TimeOfD1BlockSharedSoftmax(CUDAStream &context, const DDim &dims,
      const int in_axis, const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim > 512);
  assert(dim <= 2048);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchD1BlockSharedSoftmaxBackward<T>(
      context.stream(), dx_data, out_data, dout_data, N, dim);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfD1BlockSoftmax(CUDAStream &context, const DDim &dims,
      const int in_axis, const T* out_data, const T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim > 2048);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchD1BlockSharedSoftmaxBackward<T>(
      context.stream(), dx_data, out_data, dout_data, N, dim);
  }
  float cost = clock->stop();

  return cost;
}
/************************************************************************/


/************************************************************************/
template<typename T>
int TestSoftmax(CUDAStream &context, const DDim &dims, const int in_axis) {
  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);
  const size_t num = GetSize(dims);

  MallocHost<T> out_h(num, context);
  MallocDevice<T> out(num, context);
  MallocHost<T> dout_h(num, context);
  MallocDevice<T> dout(num, context);

  MallocDevice<T> dx_cudnn(num, context);

  size_t n_oldvec(0), n_oldwarp(0), n_loop(0), n_spand(0), n_D1(0);

  if(D == 1) {
    if(dim <= 320 && sizeof(T) <= 4){
      if(dim == 128 && N % 4 == 0) {
        n_oldvec = num;
      } else if(dim < 320) {
        n_oldwarp = num;
      }
    }
    n_D1 = num;
  } else {
    if(dim <= 1024) {
      n_loop = num;
    }
    if(D <= 1024) {
      n_spand = num;
    }
  }

  // D1 : condition 1
  MallocDevice<T> dx_oldvec(n_oldvec, context);
  MallocDevice<T> dx_oldwarp(n_oldwarp, context);
  // D1 : condition 2
  MallocDevice<T> dx_D1(n_D1, context);
  // Dn : condition 1
  MallocDevice<T> dx_Loop(n_loop, context);
  MallocDevice<T> dx_LoopNoVec(n_loop, context);
  // Dn : condition 2
  MallocDevice<T> dx_Spand(n_spand, context);
  MallocDevice<T> dx_SpandNoVec(n_spand, context);

  out_h.Random(static_cast<T>(1e-7), static_cast<T>(1e-6));
  out.CopyFrom(out_h);
  dout_h.Random(static_cast<T>(100), static_cast<T>(1000));
  dout.CopyFrom(dout_h);
  T* out_data = out.data();
  T* dout_data = dout.data();

  std::vector<std::string> methods;
  std::vector<float> costs;
  std::vector<MallocDevice<T>*> results;

  float cost_cudnn = TimeOfCudnnSoftmax(
        context, dims, in_axis, out_data, dout_data, dx_cudnn.data());
  printf("Cudnn cost %f\n", cost_cudnn);
  methods.push_back("Cudnn");
  costs.push_back(cost_cudnn);
  results.push_back(&dx_cudnn);

  float cost;
  if(D == 1) {
    // D1 : condition 1
    if(dim <= 320 && sizeof(T) <= 4){
      if(dim == 128 && N % 4 == 0) {
        // Vec和warp版本都只支持(axis == -1)，否则计算结果不正确
        cost = TimeOfOldVecSoftmax(
            context, dims, in_axis, out_data, dout_data, dx_oldvec.data());
        printf("Oldvec cost %f\n", cost);
        methods.push_back("Oldvec");
        costs.push_back(cost);
        results.push_back(&dx_oldvec);
      } else if(dim < 40 && dim % 32 != 0) {
        cost = TimeOfOldWarpSoftmax(
            context, dims, in_axis, out_data, dout_data, dx_oldwarp.data());
        printf("Oldwarp cost %f\n", cost);
        methods.push_back("Oldwarp");
        costs.push_back(cost);
        results.push_back(&dx_oldwarp);
      }
    }
    // D1 : condition 2
    if(dim <= 512) {
      cost = TimeOfD1WarpSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_D1.data());
      printf("D1Warp cost %f\n", cost);
      methods.push_back("D1Warp");
      costs.push_back(cost);
      results.push_back(&dx_D1);
    } else if(dim <= 2048) {
      cost = TimeOfD1BlockSharedSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_D1.data());
      printf("D1BlockShared cost %f\n", cost);
      methods.push_back("D1BlockShared");
      costs.push_back(cost);
      results.push_back(&dx_D1);
    } else {
      cost = TimeOfD1BlockSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_D1.data());
      printf("D1Block cost %f\n", cost);
      methods.push_back("D1Block");
      costs.push_back(cost);
      results.push_back(&dx_D1);
    }
  } else {
    // Dn : condition 1
    if(dim <= 512) {
      cost = TimeOfLoopDimSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_Loop.data());
      printf("LoopDim cost %f\n", cost);
      methods.push_back("LoopDim");
      costs.push_back(cost);
      results.push_back(&dx_Loop);

      cost = TimeOfNoVecLoopDimSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_LoopNoVec.data());
      printf("LoopDimNoVec cost %f\n", cost);
      methods.push_back("LoopDimNoVec");
      costs.push_back(cost);
      results.push_back(&dx_LoopNoVec);
    }
    // Dn : condition 2
    if(D <= 1024) {
      cost = TimeOfSpandDimDSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_Spand.data());
      printf("SpandDimD cost %f\n", cost);
      methods.push_back("SpandDimD");
      costs.push_back(cost);
      results.push_back(&dx_Spand);

      cost = TimeOfNoVecSpandDimDSoftmax(
              context, dims, in_axis, out_data, dout_data, dx_SpandNoVec.data());
      printf("SpandDimDNoVec cost %f\n", cost);
      methods.push_back("SpandDimDNoVec");
      costs.push_back(cost);
      results.push_back(&dx_SpandNoVec);
    }
  }

  printf("*******************\n");
  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "CUDA ERROR: %s\n", err);
    return CUDA_FAILED;
  }
  int min_idex = 0;
  for(int i = 1; i < costs.size(); i ++) {
    if(costs[i] != 0 && costs[i] < costs[min_idex]) min_idex = i;
  }
  fprintf(stderr, "%s, axis = %d, %s faster, which cost %f ms\n",
          ToString(dims).c_str(), in_axis, methods[min_idex].c_str(),
          costs[min_idex]);

  printf("Cudnn\n");
  results[0]->Print(1, 20);
  float max_err(0);
  int large_one = 0;
  std::vector<float> vec_err(1, 0);
  for(int i = 1; i < results.size(); i ++) {
    printf("%s\n", methods[i].c_str());
    results[i]->Print(1, 20);
    float err_val = type2type<T, float>(results[0]->MaxError(*results[1]));
    vec_err.push_back(err_val);
    if(err_val > max_err) {
      max_err = err_val;
      large_one = i;
    }
  }
  if(max_err > 0) {
    std::cout << "\nThe max diff is " << methods[large_one]
              << ", where " << max_err << "\n";
    for(int i = 1; i < vec_err.size(); i ++) {
      std::cout << methods[i] << " Diff: " << vec_err[i] << "; ";
    }
  }

  for(int i = 0; i < results.size(); i ++) {
    results[i]->~MallocDevice<T>();
  }
  return SUCCESS;
}

/************************************************************************/

int main() {
  srand(time(0));
  CUDAStream context;
  typedef double T;
  do {
    DDim dims = {512, 896, 48};
    int in_axis = 1;
    printf("Please Input Backward Dim [x, y, z]:\n");
    std::cin >> dims[0] >> dims[1] >> dims[2];
    printf("Please Input axis\n");
    std::cin >> in_axis;
    // dims[0] = rand() % 1024 + 1;
    // dims[1] = rand() % 1024 + 1;
    // dims[2] = rand() % 1024 + 1;
    printf("Shape = %s, axis = %d\n", ToString(dims).c_str(), in_axis);
    if(TestSoftmax<T>(context, dims, in_axis) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
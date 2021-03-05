// class header file
#include "softmax_forward_D1.h"

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

/************************************************************************/
template <typename T, typename VECT, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxForward(T* dst, const T* src, const int batch_size,
                                  const int softmax_ele) {
  int offset = blockIdx.x * softmax_ele * WARP_PER_BLOCK;
  int idx = threadIdx.x * VPT;

  VECT buf = reinterpret_cast<const VECT*>(&src[offset + idx])[0];
  T* bufp = reinterpret_cast<T*>(&buf);
  float4 val4;
  float* val4p = reinterpret_cast<float*>(&val4);
  for (int i = 0; i < VPT; ++i) {
    val4p[i] = static_cast<float>(bufp[i]);
  }
  // float val = val4.x + val4.y + val4.z + val4.w;
  float max_val = warpReduceMax<float>(
      max(max(val4.x, val4.y), max(val4.z, val4.w)), 0xffffffff);
  float4 tmp4 = make_float4(__expf(val4.x - max_val), __expf(val4.y - max_val),
                            __expf(val4.z - max_val), __expf(val4.w - max_val));
  float* tmp4p = reinterpret_cast<float*>(&tmp4);
  float invsum = 1.f / (warpReduceSum<float>(
                            tmp4.x + tmp4.y + tmp4.z + tmp4.w, 0xffffffff) +
                        1e-6f);
  for (int i = 0; i < VPT; ++i) {
    bufp[i] = static_cast<T>(tmp4p[i] * invsum);
  }
  reinterpret_cast<VECT*>(&dst[offset + idx])[0] = buf;
}

template<typename T>
float TimeOfOldVecSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                       const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr int warps_per_block = 4;
  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    if (sizeof(T) == 2) {
      VecSoftmaxForward<T, int2, 4, warps_per_block><<<
          N / warps_per_block, warps_per_block * WARP_SIZE, 0,
          context.stream()>>>(out_data, in_data, N, dim);
    } else if (sizeof(T) == 4) {
      VecSoftmaxForward<T, int4, 4, warps_per_block><<<
          N / warps_per_block, warps_per_block * WARP_SIZE, 0,
          context.stream()>>>(out_data, in_data, N, dim);
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

#define LAUNCH_SOFTMAX_WARP_FORWARD(Log2Elements)                  \
  case Log2Elements:                                               \
    WarpSoftmaxForward<T, float, Log2Elements><<<                  \
        blocks, threads, 0, context.stream()>>>( \
        out_data, in_data, N, dim, dim);                      \
    break;

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template<typename T>
float TimeOfOldWarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                       const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

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

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
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
  }
  float cost = clock->stop();

  return cost;
}
/************************************************************************/

template<typename T, typename AccT>
__global__ void KeBlockSoftmaxForward(T* __restrict__ dst, const T* __restrict__ src,
                          const int N, const int dim, const int D) {
  for(int bid = blockIdx.x; bid < N * D; bid += gridDim.x) {
    const int out_id = bid / D;
    const int in_id = bid - out_id * D;
    const int tid = threadIdx.x;

    const int offset = out_id * dim * D + in_id;
    const T* __restrict__ src_row = src + offset;
    T* __restrict__ dst_row = dst + offset;

    // compute max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x) {
      max_val = max(max_val, static_cast<AccT>(src_row[dim_id * D]));
    }
    max_val = blockReduceMax(max_val, 0xffffffff);

    // compute sum value
    AccT sum_val(0);
    for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x){
      sum_val += Exp(static_cast<AccT>(src_row[dim_id * D]) - max_val);
    }
    sum_val = blockReduceSum(sum_val, 0xffffffff);

    // compute softmax result
    for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
      dst_row[dim_id * D] =
          static_cast<T>(Exp(static_cast<AccT>(src_row[dim_id * D]) - max_val)
            / sum_val);
  }
}

template<typename T>
float TimeOfBlockSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  const int threads = std::min(dim, 1024);
  const int grids = N * D;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeBlockSoftmaxForward<T, float>
      <<<grids, threads, 0, context.stream()>>>(
      out_data, in_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T, typename AccT>
__global__ void KeWarpSoftmaxForward(T *dst, const T *src, const int N,
                                      const int dim, const int D) {
  const int out_id = blockIdx.x;
  const int in_id = blockIdx.y * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  const int offset = out_id * dim * D + in_id;
  const T *out_src = src + offset;
  T *out_dst = dst + offset;

  // compute max value
  AccT max_val = -std::numeric_limits<AccT>::infinity();
  for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
    max_val = max(max_val, static_cast<AccT>(out_src[dim_id * D]));
  max_val = warpReduceMax(max_val, 0xffffffff);

  // compute sum value
  AccT sum_val(0);
  for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
    sum_val += Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val);
  sum_val = warpReduceSum(sum_val, 0xffffffff);

  // compute softmax result
  for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
    out_dst[dim_id * D] =
        static_cast<T>(Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val)
           / sum_val);
}

template<typename T>
float TimeOfWarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  const int dim_b = 32;
  const int inner_b = std::min(16, D);
  const int inner_g = (D + inner_b - 1) / inner_b;
  dim3 threads(dim_b, inner_b);
  dim3 grids(N, inner_g);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeWarpSoftmaxForward<T, float>
      <<<grids, threads, 0, context.stream()>>>(
      out_data, in_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
__forceinline__ __device__ T blockYReduceMax(T* __restrict__ s_data, T val) {
  s_data += threadIdx.x;
  s_data[threadIdx.y * blockDim.x] = val;
  __syncthreads();

  T res = -std::numeric_limits<T>::infinity();
  for(int i = 0; i < blockDim.y; i ++)
    res = max(s_data[i * blockDim.x], res);
  return res;
}
template<typename T>
__forceinline__ __device__ T blockYReduceSum(T* __restrict__ s_data, T val) {
  s_data += threadIdx.x;
  s_data[threadIdx.y * blockDim.x] = val;
  __syncthreads();

  T res(0);
  for(int i = 0; i < blockDim.y; i ++)
    res += s_data[i * blockDim.x];
  return res;
}

template<typename T, typename AccT>
__global__ void KeArrangeYSoftmaxForward(T* __restrict__ dst,
                          const T* __restrict__ src,
                          const int N, const int dim, const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  const int tidy = threadIdx.y;
  const int out_id = blockIdx.x;
  const int in_id = blockIdx.y * blockDim.x + threadIdx.x;

  const T *out_src = src + out_id * dim * D + in_id;
  T *out_dst = dst + out_id * dim * D + in_id;

  AccT max_val = -std::numeric_limits<AccT>::infinity();
  for(int dim_id = tidy; dim_id < dim; dim_id += blockDim.y)
    max_val = max(static_cast<AccT>(out_src[dim_id * D]), max_val);
  max_val = blockYReduceMax(s_data, max_val);

  // compute exponent value and sum value
  AccT sum_val(0);
  for(int dim_id = tidy; dim_id < dim; dim_id += blockDim.y)
    sum_val += Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val);
  sum_val = blockYReduceSum(s_data, sum_val);

  // compute softmax value
  // TODO(jiangcheng): how to eliminate twice Exp
  for(int dim_id = tidy; dim_id < dim; dim_id += blockDim.y)
    out_dst[dim_id * D] =
        static_cast<T>(Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val) /
        (sum_val + 1e-6f));
}

template<typename T>
float TimeOfArrangeYSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr int coalesce_size = 256 / sizeof(T);
  int dim_y = std::min(MAX_BLOCK_DIM / coalesce_size, dim);
  int D_x = MAX_BLOCK_DIM / dim_y;
  int D_y = (D + D_x - 1) / D_x;

  dim3 threads(D_x, dim_y);
  dim3 grids(N, D_y);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeArrangeYSoftmaxForward<T, float>
      <<<grids, threads, dim_y * D_x * sizeof(float), context.stream()>>>(
      out_data, in_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfCudnnSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
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
    status = cudnnSoftmaxForward(
        cudnn_hd, CUDNN_SOFTMAX_ACCURATE, mode,
        &one, desc_, in_data, &zero, desc_, out_data);
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
__global__ void KeLoopDimSoftmaxForward(T *dst, const T *src, const int N,
                                      const int dim, const int D) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_id = tid / D;
  if(out_id >= N) return;
  const int in_id = tid - out_id * D;

  const T *out_src = src + out_id * dim * D + in_id;
  T *out_dst = dst + out_id * dim * D + in_id;
  // compute max value
  AccT max_val = -std::numeric_limits<AccT>::infinity();
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    max_val = max(static_cast<AccT>(out_src[dim_id * D]), max_val);

  // compute exponent value and sum value
  AccT sum_val(0);
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    sum_val += Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val);

  // compute softmax value
  // TODO(jiangcheng): how to eliminate twice Exp
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    out_dst[dim_id * D] =
        static_cast<T>(Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val) /
        (sum_val + 1e-6f));
}

template<typename T, typename AccT, int VecSize>
__global__ void KeLoopDimSoftmaxForward(T *dst, const T *src, const int N,
                                      const int dim, const int D) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_id = tid / D;
  if(out_id >= N) return;
  const int in_id = tid - out_id * D;

  const T *out_src = src + out_id * dim * D + in_id;
  T *out_dst = dst + out_id * dim * D + in_id;
  // compute max value
  AccT max_val = -std::numeric_limits<AccT>::infinity();
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    max_val = max(static_cast<AccT>(out_src[dim_id * D]), max_val);

  // compute exponent value and sum value
  AccT sum_val(0);
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    sum_val += Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val);

  // compute softmax value
  // TODO(jiangcheng): how to eliminate twice Exp
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    out_dst[dim_id * D] =
        static_cast<T>(Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val) /
        (sum_val + 1e-6f));
}

template<typename T>
float TimeOfLoopDimSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  int threads = std::min(N * D, 1024);
  int grids = (N * D + threads - 1) / threads;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeLoopDimSoftmaxForward<T, float><<<grids, threads, 0, context.stream()>>>(
      out_data, in_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT>
__global__ void KeSpandDimDSoftmaxForward(T *dst, const T *src, const int N,
                                      const int dim, const int D) {
  extern __shared__ __align__(sizeof(AccT)) unsigned char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  const int tid = threadIdx.x;
  const int BlockDim = blockDim.x;
  for(int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const T *out_src = src + out_id * dim * D;
    T *out_dst = dst + out_id * dim * D;

    // Compute each thread's max value
    AccT max_val = -std::numeric_limits<AccT>::infinity();
    for(int id = tid; id < dim * D; id += BlockDim)
      max_val = max(static_cast<AccT>(out_src[id]), max_val);
    // write to shared memory
    s_data[tid] = max_val;
    __syncthreads();
    // compute total max value
    for(int id = tid % D; id < BlockDim; id += D)
      max_val = max(s_data[id], max_val);

    // Compute each thread's sum value
    AccT sum_val(0);
    for(int id = tid; id < dim * D; id += BlockDim)
      sum_val += Exp(static_cast<AccT>(out_src[id]) - max_val);
    // write to shared memory
    s_data[tid] = sum_val;
    __syncthreads();
    // compute total sum value
    sum_val = 0;
    for(int id = tid % D; id < BlockDim; id += D)
      sum_val += s_data[id];

    // Compute finally softmax result
    // TODO(jiangcheng): how to eliminate twice Exp
    for(int id = tid; id < dim * D; id += BlockDim)
      out_dst[id] = static_cast<T>(Exp(static_cast<AccT>(out_src[id]) - max_val) /
                      (sum_val + 1e-6f));
  }
}

template<typename T>
float TimeOfSpandDimDSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  if(D > 1024) return 0.0f;

  const int grids = N;
  const int threads = D * (1024 / D);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeSpandDimDSoftmaxForward<T, float>
      <<<grids, threads, threads * sizeof(float), context.stream()>>>(
      out_data, in_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfOneflowD1WarpSoftmax(CUDAStream &context, const DDim &dims,
                    const int in_axis, const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim <= 1024);

#define DEFINE_ONE_ELIF(col)                        \
  else if (dim <= (col) * WARP_SIZE) {             \
    DispatchOneflowD1WarpSoftmax<T, float, col>(    \
      context, in_data, out_data, N, dim);      \
  }

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  if(dim % 2 == 0 && dim > WARP_SIZE) {
    DispatchOneflowD1WarpSoftmaxCols<T, float, 2>(
        context, in_data, out_data, N, dim);
  } else {
    DispatchOneflowD1WarpSoftmaxCols<T, float, 1>(
        context, in_data, out_data, N, dim);
  }
  float cost = clock->stop();

  return cost;
}
/************************************************************************/
template<typename T>
float TimeOfD1WarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim <= 1024);

  const int cols_per_thread = (dim + WARP_SIZE - 1) / WARP_SIZE;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
    if(dim % 4 == 0 && cols_per_thread % 4 == 0) {
      DispatchD1WarpSoftmax<T, 4>(
        context, in_data, out_data, N, dim, cols_per_thread);
    } else if(dim % 2 == 0 && cols_per_thread % 2 == 0) {
      DispatchD1WarpSoftmax<T, 2>(
        context, in_data, out_data, N, dim, cols_per_thread);
    } else {
      DispatchD1WarpSoftmax<T, 1>(
        context, in_data, out_data, N, dim, cols_per_thread);
    }
  float cost = clock->stop();

  return cost;
}

template<typename T>
float TimeOfNoVecD1WarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim <= 1024);

  int N_b = std::min(8, N);
  dim3 threads(WARP_SIZE, N_b);
  int grids = (N + N_b - 1) / N_b;
  const int cols_per_thread = (dim + WARP_SIZE - 1) / WARP_SIZE;

#define LAUNCH_D1WARP_COLS(COLS)                    \
  case COLS:                                        \
    NoVec_KeD1WarpSoftmaxForward<T, float, COLS>    \
      <<<grids, threads, 0, context.stream()>>>(    \
        out_data, in_data, N, dim);                 \
    break;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
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
        fprintf(stderr, "[NoVec_KeD1WarpSoftmaxForward] "
                        "BAD PARAM (%d, %d) with %d\n",
                N, dim, cols_per_thread);
        break;
    }
#undef LAUNCH_D1WARP_COLS
  float cost = clock->stop();

  return cost;
}
/************************************************************************/

template<typename T>
float TimeOfD1BlockSharedSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim > 1024);
  assert(dim <= 4096);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    if(dim % 4 == 0) {
      LaunchD1BlockSharedSoftmax<T, 4>(
        context, in_data, out_data, N, dim);
    } else if(dim % 2 == 0) {
      LaunchD1BlockSharedSoftmax<T, 2>(
        context, in_data, out_data, N, dim);
    } else {
      LaunchD1BlockSharedSoftmax<T, 1>(
        context, in_data, out_data, N, dim);
    }
  }
  float cost = clock->stop();

  return cost;
}

template<typename T>
float TimeOfNoVecD1BlockSharedSoftmax(CUDAStream &context, const DDim &dims,
                        const int in_axis, const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim > 1024);
  assert(dim <= 4096);

  const int threads = std::min(dim, 1024);
  const int grids = N;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    NoVec_KeD1BlockSharedSoftmaxForward<T, float>
      <<<grids, threads, dim * sizeof(float), context.stream()>>>(
      out_data, in_data, N, dim);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfD1BlockSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim > 4096);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    if(dim % 4 == 0) {
      LaunchD1BlockSoftmax<T, 4>(
        context, in_data, out_data, N, dim);
    } else if(dim % 2 == 0) {
      LaunchD1BlockSoftmax<T, 2>(
        context, in_data, out_data, N, dim);
    } else {
      LaunchD1BlockSoftmax<T, 1>(
        context, in_data, out_data, N, dim);
    }
  }
  float cost = clock->stop();

  return cost;
}

template<typename T>
float TimeOfNoVecD1BlockSoftmax(CUDAStream &context, const DDim &dims,
                        const int in_axis, const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  assert(D == 1);
  assert(dim > 4096);

  const int threads = std::min(dim, 1024);
  const int grids = N;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    NoVec_KeD1BlockSoftmaxForward<T, float>
      <<<grids, threads, 0, context.stream()>>>(
      out_data, in_data, N, dim);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/

template<typename T>
float TimeOfNewSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);


  constexpr bool AccT_use_float =
        std::is_same<T, float>::value ||
        std::is_same<T, half>::value;
  bool optimize = false;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  if(D <= 512 && D < dim) {
    if(i == 0) printf("[TimeOfNewSoftmax] Using KeSpandDimDSoftmaxForward\n");
    optimize = true;
    const int grids = N;
    const int threads = D * (512 / D);

    if(AccT_use_float) {
      KeSpandDimDSoftmaxForward<T, float>
        <<<grids, threads, threads * sizeof(float),
          context.stream()>>>(
        out_data, in_data, N, dim, D);
    } else {
      KeSpandDimDSoftmaxForward<T, double>
        <<<grids, threads, threads * sizeof(double),
          context.stream()>>>(
        out_data, in_data, N, dim, D);
    }
  } else if(dim < 1024) {
    if(i == 0) printf("[TimeOfNewSoftmax] Using KeLoopDimSoftmaxForward\n");
    optimize = true;
    int threads = std::min(N * D, 256);
    int grids = (N * D + threads - 1) / threads;

    if(AccT_use_float) {
      KeLoopDimSoftmaxForward<T, float><<<grids, threads, 0,
        context.stream()>>>(
        out_data, in_data, N, dim, D);
    } else {
      KeLoopDimSoftmaxForward<T, double><<<grids, threads, 0,
        context.stream()>>>(
        out_data, in_data, N, dim, D);
    }
  }
  float cost = clock->stop();

  return cost;
}
/************************************************************************/
template<typename T>
int TestSoftmax(CUDAStream &context, const DDim &dims, const int in_axis) {
  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);
  const size_t num = GetSize(dims);

  MallocHost<T> input_h(num, context);
  MallocDevice<T> input(num, context);

  MallocDevice<T> out_cudnn(num, context);

  size_t n_oldvec(0), n_oldwarp(0), n_inner(0), n_DimD(0);
  size_t n_OneflowD1Warp(0), n_D1(0), n_D1NoVec(0);

  if(D == 1) {
    if(dim <= 320 && sizeof(T) <= 4){
      if(dim == 128 && N % 4 == 0) {
        n_oldvec = num;
      } else if(dim < 320) {
        n_oldwarp = num;
      }
    }
    if(dim <= 1024) {
      n_OneflowD1Warp = num;
    }
    n_D1 = n_D1NoVec = num;
  } else {
    if(dim <= 1024) {
      n_inner = num;
    }
    if(D <= 1024) {
      n_DimD = num;
    }
  }

  MallocDevice<T> out_oldvec(n_oldvec, context);
  MallocDevice<T> out_oldwarp(n_oldwarp, context);
  MallocDevice<T> out_inner(n_inner, context);
  MallocDevice<T> out_DimD(n_DimD, context);
  MallocDevice<T> out_OneflowD1Warp(n_OneflowD1Warp, context);
  MallocDevice<T> out_D1(n_D1, context);
  MallocDevice<T> out_D1NoVec(n_D1NoVec, context);
  
  // MallocDevice<T> out_rankY(num, context);
  // MallocDevice<T> out_block(num, context);
  // MallocDevice<T> out_warp(num, context);
  // MallocDevice<T> out_new(num, context);

  input_h.Random(static_cast<T>(-10), static_cast<T>(10));
  input.CopyFrom(input_h);
  T* in_data = input.data();

  std::vector<std::string> methods;
  std::vector<float> costs;
  std::vector<MallocDevice<T>*> results;

  float cost_cudnn = TimeOfCudnnSoftmax(
                      context, dims, in_axis, in_data, out_cudnn.data());
  printf("Cudnn cost %f\n", cost_cudnn);
  methods.push_back("Cudnn");
  costs.push_back(cost_cudnn);
  results.push_back(&out_cudnn);

  if(D == 1) {
    if(dim <= 320 && sizeof(T) <= 4){
      if(dim == 128 && N % 4 == 0) {
        // Vec和warp版本都只支持(axis == -1)，否则计算结果不正确
        float cost_oldvec = TimeOfOldVecSoftmax(context, dims, in_axis, in_data, out_oldvec.data());
        printf("Oldvec cost %f\n", cost_oldvec);
        methods.push_back("Oldvec");
        costs.push_back(cost_oldvec);
        results.push_back(&out_oldvec);
      } else if(dim < 320) {
        float cost_oldwarp = TimeOfOldWarpSoftmax(context, dims, in_axis, in_data, out_oldwarp.data());
        printf("Oldwarp cost %f\n", cost_oldwarp);
        methods.push_back("Oldwarp");
        costs.push_back(cost_oldwarp);
        results.push_back(&out_oldwarp);
      }
    }
    if(dim <= 1024) {
      float cost_OneflowD1Warp = TimeOfOneflowD1WarpSoftmax(
                                  context, dims, in_axis, in_data, out_OneflowD1Warp.data());
      printf("OneflowD1Warp cost %f\n", cost_OneflowD1Warp);
      methods.push_back("OneflowD1Warp");
      costs.push_back(cost_OneflowD1Warp);
      results.push_back(&out_OneflowD1Warp);

      float cost_D1Warp = TimeOfD1WarpSoftmax(
                            context, dims, in_axis, in_data, out_D1.data());
      printf("D1Warp cost %f\n", cost_D1Warp);
      methods.push_back("D1Warp");
      costs.push_back(cost_D1Warp);
      results.push_back(&out_D1);

      float cost_NoVecD1Warp = TimeOfNoVecD1WarpSoftmax(
                            context, dims, in_axis, in_data, out_D1NoVec.data());
      printf("NoVecD1Warp cost %f\n", cost_NoVecD1Warp);
      methods.push_back("NoVecD1Warp");
      costs.push_back(cost_NoVecD1Warp);
      results.push_back(&out_D1NoVec);
    } else if(dim <= 4096) {
      float cost_D1BlockShared = TimeOfD1BlockSharedSoftmax(
                                  context, dims, in_axis, in_data, out_D1.data());
      printf("D1BlockShared cost %f\n", cost_D1BlockShared);
      methods.push_back("D1BlockShared");
      costs.push_back(cost_D1BlockShared);
      results.push_back(&out_D1);

      float cost_NoVecD1BlockShared = TimeOfNoVecD1BlockSharedSoftmax(
                                  context, dims, in_axis, in_data, out_D1NoVec.data());
      printf("NoVecD1BlockShared cost %f\n", cost_NoVecD1BlockShared);
      methods.push_back("NoVecD1BlockShared");
      costs.push_back(cost_NoVecD1BlockShared);
      results.push_back(&out_D1NoVec);
    } else {
      float cost_NoVecD1Block = TimeOfNoVecD1BlockSoftmax(
                                  context, dims, in_axis, in_data, out_D1NoVec.data());
      printf("NoVecD1Block cost %f\n", cost_NoVecD1Block);
      methods.push_back("NoVecD1Block");
      costs.push_back(cost_NoVecD1Block);
      results.push_back(&out_D1NoVec);

      float cost_D1Block = TimeOfD1BlockSoftmax(
                                  context, dims, in_axis, in_data, out_D1.data());
      printf("D1Block cost %f\n", cost_D1Block);
      methods.push_back("D1Block");
      costs.push_back(cost_D1Block);
      results.push_back(&out_D1);
    }
  } else {
    if(dim <= 1024) {
      float cost_inner = TimeOfLoopDimSoftmax(
                        context, dims, in_axis, in_data, out_inner.data());
      printf("LoopDim cost %f\n", cost_inner);
      methods.push_back("LoopDim");
      costs.push_back(cost_inner);
      results.push_back(&out_inner);
    }

    if(D <= 1024) {
      float cost_DimD = TimeOfSpandDimDSoftmax(
                          context, dims, in_axis, in_data, out_DimD.data());
      printf("SpandDimD cost %f\n", cost_DimD);
      methods.push_back("SpandDimD");
      costs.push_back(cost_DimD);
      results.push_back(&out_DimD);
    }
  }
/*
  float cost_rankY = TimeOfArrangeYSoftmax(context, dims, in_axis, in_data, out_rankY.data());
  printf("ArrangeY cost %f\n", cost_rankY);
  methods.push_back("ArrangeY");
  costs.push_back(cost_rankY);
  float cost_block = TimeOfBlockSoftmax(context, dims, in_axis, in_data, out_block.data());
  printf("Block cost %f\n", cost_block);
  methods.push_back("Block");
  costs.push_back(cost_block);
  float cost_warp = TimeOfWarpSoftmax(context, dims, in_axis, in_data, out_warp.data());
  printf("Warp cost %f\n", cost_warp);
  methods.push_back("Warp");
  costs.push_back(cost_warp);
  */
  // float cost_new = TimeOfNewSoftmax(context, dims, in_axis, in_data, out_new.data());
  // printf("New cost %f\n", cost_new);
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

  float max_err(0);
  int large_one = 0;
  std::vector<float> vec_err(1, 0);
  for(int i = 1; i < results.size(); i ++) {
    float err_val = type2type<T, float>(results[0]->MaxError(*results[1]));
    vec_err.push_back(err_val);
    if(err_val > max_err) {
      max_err = err_val;
      large_one = i;
    }
  }
  if(max_err > 1e-5f) {
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
  typedef float T;
  do {
    DDim dims = {512, 896, 48};
    int in_axis = -1;
    printf("Please Input Dim [x, y, z]:\n");
    std::cin >> dims[0] >> dims[1] >> dims[2];
    printf("Please Input axis\n");
    std::cin >> in_axis;
    // dims[0] = rand() % 1000 + 1;
    // dims[1] = rand() % 8192 + 1;
    // dims[2] = rand() % 2048;
    print(dims);
    printf(", axis = %d\n", in_axis);
    if(TestSoftmax<T>(context, dims, in_axis) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
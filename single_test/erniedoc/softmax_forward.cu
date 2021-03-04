// class header file

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
template <typename T>
__inline__ __device__ T blockReduceMax(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMax(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (lane < block_span) ? shared[lane] : static_cast<T>(0.0f);
  val = warpReduceSum<T>(val, mask);

  return val;
}

template<typename T, typename AccT>
__global__ void KeBlockSoftmaxForward(T *dst, const T *src, const int N,
                                      const int dim, const int D) {
  const int out_id = blockIdx.x / D;
  const int in_id = blockIdx.x - out_id * D;
  const int tid = threadIdx.x;
  if(out_id >= N) return;

  const int offset = out_id * dim * D + in_id;
  const T *out_src = src + offset;
  T *out_dst = dst + offset;

  // compute max value
  AccT max_val = -std::numeric_limits<AccT>::infinity();
  for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
    max_val = max(max_val, static_cast<AccT>(out_src[dim_id * D]));
  max_val = blockReduceMax(max_val, 0xffffffff);

  // compute sum value
  AccT sum_val(0);
  for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
    sum_val += Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val);
  sum_val = blockReduceSum(sum_val, 0xffffffff);

  // compute softmax result
  for(int dim_id = tid; dim_id < dim; dim_id += blockDim.x)
    out_dst[dim_id * D] =
        static_cast<T>(Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val)
           / sum_val);
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

  const int threads = std::min(dim, 256);
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
  extern __shared__ char s_mem[];
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
  std::vector<int> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    strides[i] = dims[i + 1] * strides[i + 1];
  }

  std::vector<int> tensor_dims = {N, dim, D, 1};
  // cudnnTensorFormat_t layout = CUDNN_TENSOR_NCHW;
  cudnnDataType_t type = GetCudnnDataType<T>();

  cudnnTensorDescriptor_t desc_;
  cudnnCreateTensorDescriptor(&desc_);
  cudnnSetTensorNdDescriptor(desc_, type, rank, dims.data(), strides.data());

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

  int threads = std::min(N * D, 256);
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
  extern __shared__ char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  const int tid = threadIdx.x;
  const int BlockDim = blockDim.x;
  const int out_id = blockIdx.x;
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
  const size_t num = GetSize(dims);

  MallocHost<T> input_h(num, context);
  MallocDevice<T> input(num, context);

  MallocDevice<T> out_cudnn(num, context);
  /*
  MallocDevice<T> out_vec(num, context);
  MallocDevice<T> out_warp(num, context);
  */
  MallocDevice<T> out_inner(num, context);
  MallocDevice<T> out_DimD(num, context);
  // MallocDevice<T> out_rankY(num, context);
  // MallocDevice<T> out_block(num, context);
  // MallocDevice<T> out_warp(num, context);
  MallocDevice<T> out_new(num, context);

  input_h.Random(static_cast<T>(-10), static_cast<T>(10));
  input.CopyFrom(input_h);

  T* in_data = input.data();

  std::vector<std::string> methods;
  std::vector<float> costs;

  float cost_cudnn = TimeOfCudnnSoftmax(context, dims, in_axis, in_data, out_cudnn.data());
  printf("cudnn cost %f\n", cost_cudnn);
  methods.push_back("Cudnn");
  costs.push_back(cost_cudnn);
/*
  // Vec和warp版本都只支持(axis == -1)，否则计算结果不正确
  float cost_vec = TimeOfOldVecSoftmax(context, dims, in_axis, in_data, out_vec.data());
  printf("Old vec cost %f\n", cost_vec);

  float cost_warp = TimeOfOldWarpSoftmax(context, dims, in_axis, in_data, out_warp.data());
  printf("Old warp cost %f\n", cost_warp);
*/
  float cost_inner = TimeOfLoopDimSoftmax(context, dims, in_axis, in_data, out_inner.data());
  printf("LoopDim cost %f\n", cost_inner);
  methods.push_back("LoopDim");
  costs.push_back(cost_inner);

  float cost_DimD = TimeOfSpandDimDSoftmax(context, dims, in_axis, in_data, out_DimD.data());
  printf("SpandDimD cost %f\n", cost_DimD);
  methods.push_back("SpandDimD");
  costs.push_back(cost_DimD);
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

  float cost_new = TimeOfNewSoftmax(context, dims, in_axis, in_data, out_new.data());
  printf("New cost %f\n", cost_new);

  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "CUDA ERROR: %s\n", err);
    return CUDA_FAILED;
  }
  /*
  T vec_err = out_cudnn.MaxError(out_vec);
  std::cout<< "Vec Error: " << vec_err << std::endl;
  T warp_err = out_cudnn.MaxError(out_warp);
  std::cout<< "Warp Error: " << warp_err << std::endl;
  */
  T inner_err = out_cudnn.MaxError(out_inner);
  std::cout<< "LoopDim Error: " << type2type<T, float>(inner_err) << std::endl;
  T DimD_err = out_cudnn.MaxError(out_DimD);
  std::cout<< "SpandDimD Error: " << type2type<T, float>(DimD_err) << std::endl;
/*
  T rankY_err = out_cudnn.MaxError(out_rankY);
  std::cout<< "ArrangeY Error: " << type2type<T, float>(rankY_err) << std::endl;
  T block_err = out_cudnn.MaxError(out_block);
  std::cout<< "Block Error: " << type2type<T, float>(block_err) << std::endl;
  T warp_err = out_cudnn.MaxError(out_warp);
  std::cout<< "Warp Error: " << type2type<T, float>(warp_err) << std::endl;
  T new_err = out_cudnn.MaxError(out_new);
  std::cout<< "New Error: " << type2type<T, float>(new_err) << std::endl;
*/

  int min_idex = 0;
  for(int i = 1; i < costs.size(); i ++) {
    if(costs[i] != 0 && costs[i] < costs[min_idex]) min_idex = i;
  }
  printf("%s %s faster, which cost %f ms\n",
          ToString(dims).c_str(), methods[min_idex].c_str(),
          costs[min_idex]);
  return SUCCESS;
}

/************************************************************************/

int main() {
  srand(time(0));
  CUDAStream context;
  typedef float T;
  do {
    // DDim dims = {512, 896, 48};
    DDim dims = {2, 10, 10};
    // dims[0] = rand() % 1000;
    // dims[1] = rand() % 1000;
    // dims[2] = rand() % 1000;
    int in_axis = 1;
    print(dims, "\n");
    if(TestSoftmax<T>(context, dims, in_axis) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
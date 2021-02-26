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


constexpr int LOOPNUM=100;
typedef std::vector<int> DDim;

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
  float val = val4.x + val4.y + val4.z + val4.w;
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
float TimeOfVecSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                       const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr int warps_per_block = 4;
  clock->start();
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


template<typename T>
float TimeOfWarpSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
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
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfCudnnSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                       const T* in_data, T* out_data) {
  auto clock = TimeOfKernel::get(context);

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

  auto handle = context.cudnn_handle();
  auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                : CUDNN_SOFTMAX_MODE_CHANNEL;
  const T one = static_cast<T>(1), zero = static_cast<T>(0);

  clock->start();
  auto status = cudnnSoftmaxForward(
      handle, CUDNN_SOFTMAX_ACCURATE, mode,
      &one, desc_, in_data, &zero, desc_, out_data);
  float cost = clock->stop();

  auto err = context.GetError(status);
  if(err != "") {
    fprintf(stderr, "%s Cudnn ERROR: %s\n",
            ToString(dims), err);
    return 0.0f;
  }

  return cost;
}

/************************************************************************/
template<typename T>
int TestSoftmax(CUDAStream &context, const DDim &dims, const int in_axis) {
  const size_t num = GetSize(dims);

  MallocHost<T> input_h(num, context);
  MallocDevice<T> input(num, context);
  MallocDevice<T> out_vec(num, context);
  MallocDevice<T> out_warp(num, context);
  MallocDevice<T> out_cudnn(num, context);

  input_h.Random(static_cast<T>(-10), static_cast<T>(10));
  input.CopyFrom(input_h);

  T* in_data = input.data();

  float cost_vec = TimeOfVecSoftmax(context, dims, in_axis, in_data, out_vec.data());
  float cost_warp = TimeOfWarpSoftmax(context, dims, in_axis, in_data, out_vec.data());
  float cost_cudnn = TimeOfCudnnSoftmax(context, dims, in_axis, in_data, out_vec.data());

  printf("Vec cost %f vs warp cost %f vs cudnn cost %f\n", cost_vec, cost_warp, cost_cudnn);

  auto err = context.sync();
  if(err != "") {
    fprintf(stderr, "ERROR: %s\n", err);
    return CUDA_FAILED;
  }

  return SUCCESS;
}

/************************************************************************/

int main() {
  CUDAStream context;

  typedef float T;
  DDim dims = {512, 896, 4, 12};
  int in_axis = 1;

  TestSoftmax<T>(context, dims, in_axis);

  return 0;
}
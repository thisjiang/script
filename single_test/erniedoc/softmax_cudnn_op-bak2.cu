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

#define LAUNCH_SOFTMAX_WARP_FORWARD(Log2Elements)                  \
  case Log2Elements:                                               \
    WarpSoftmaxForward<T, float, Log2Elements><<<                  \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        out_data, x->data<T>(), N, dim, dim);                      \
    break;

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
  float max_val = math::warpReduceMax<float>(
      max(max(val4.x, val4.y), max(val4.z, val4.w)), 0xffffffff);
  float4 tmp4 = make_float4(__expf(val4.x - max_val), __expf(val4.y - max_val),
                            __expf(val4.z - max_val), __expf(val4.w - max_val));
  float* tmp4p = reinterpret_cast<float*>(&tmp4);
  float invsum = 1.f / (math::warpReduceSum<float>(
                            tmp4.x + tmp4.y + tmp4.z + tmp4.w, 0xffffffff) +
                        1e-6f);
  for (int i = 0; i < VPT; ++i) {
    bufp[i] = static_cast<T>(tmp4p[i] * invsum);
  }
  reinterpret_cast<VECT*>(&dst[offset + idx])[0] = buf;
}

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

template <typename T>
__global__ void MultiplyCUDAKernel(T* C, const T* A, const T* B, int N) {
  CUDA_KERNEL_LOOP(i, N) {
    C[i] = static_cast<T>(static_cast<float>(A[i]) * static_cast<float>(B[i]));
  }
}

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
  float sum_gy = math::warpReduceSum<float>(local_sum_gy, 0xffffffff);

  vec_t<T, VPT> local_dst;
  for (int i = 0; i < VPT; ++i) {
    local_dst.v[i] =
        static_cast<T>(static_cast<float>(local_src.v[i]) *
                       (static_cast<float>(local_grad.v[i]) - sum_gy));
  }
  reinterpret_cast<decltype(local_dst.s)*>(&dst[offset])[0] = local_dst.s;
}

template<typename T>
__forceinline__ __device__ T Exp(const T val) {
  return exp(val);
}
template<>
__forceinline__ __device__ float Exp<float>(const float val) {
  return __expf(val);
}
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
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    out_dst[dim_id * D] =
        static_cast<T>(Exp(static_cast<AccT>(out_src[dim_id * D]) - max_val) /
        (sum_val + 1e-6f));
}

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
    for(int id = tid; id < dim * D; id += BlockDim)
      out_dst[id] = static_cast<T>(Exp(static_cast<AccT>(out_src[id]) - max_val) /
                      (sum_val + 1e-6f));
  }
}

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT>
__global__ void KeSpandDimDSoftmaxBackward(T *dx, const T *out, const T *dout,
                        const int N, const int dim, const int D) {
  extern __shared__ char s_mem[];
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
    constexpr bool AccT_use_float =
        std::is_same<T, float>::value ||
        std::is_same<T, platform::float16>::value;
    if(D <= 512 && D < dim) {
      optimize = true;
      const int grids = N;
      const int threads = D * (512 / D);

      if(AccT_use_float) {
        KeSpandDimDSoftmaxForward<T, float>
          <<<grids, threads, threads * sizeof(float),
            ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), N, dim, D);
      } else {
        KeSpandDimDSoftmaxForward<T, double>
          <<<grids, threads, threads * sizeof(double),
            ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), N, dim, D);
      }
    } else if(dim < 1024) {
      optimize = true;
      int threads = std::min(N * D, 256);
      int grids = (N * D + threads - 1) / threads;

      if(AccT_use_float) {
        KeLoopDimSoftmaxForward<T, float><<<grids, threads, 0,
          ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), N, dim, D);
      } else {
        KeLoopDimSoftmaxForward<T, double><<<grids, threads, 0,
          ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), N, dim, D);
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

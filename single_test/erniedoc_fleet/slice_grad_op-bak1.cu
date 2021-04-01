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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

using float16 = plat::float16;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template<typename T, size_t D>
__global__ void KeSlicePaddingZero(
            const T *input, const int64_t *in_stride_array,
            const int64_t in_size, const int64_t *padding_start,
            T *output, const int64_t *out_stride_array) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // using register to speed up
  int64_t reg_in_stride_array[D], reg_out_stride_array[D], reg_padding_start[D];
#pragma unroll
  for(int i = 0; i < D; i ++) {
    reg_in_stride_array[i] = in_stride_array[i];
    reg_out_stride_array[i] = out_stride_array[i];
    reg_padding_start[i] = padding_start[i];
  }

  for(int64_t index = tid; index < in_size; index += blockDim.x * gridDim.x) {
    // compute each dimension index of input
    int dims[D]{0};
    int64_t k = index;
#pragma unroll
    for(int i = 0; i < D; i ++) {
      dims[i] = k / reg_in_stride_array[i];
      k -= dims[i] * reg_in_stride_array[i];
    }

    // compute the total index of output
    int64_t out_id = 0;
#pragma unroll
    for(int i = 0; i < D; i ++) {
      out_id += (dims[i] + reg_padding_start[i]) * reg_out_stride_array[i];
    }
    output[out_id] = input[index];
  }
}

template<typename T, size_t D>
void LaunchSlicePaddingKernel(const paddle::gpuStream_t &stream,
                const T *input, const int64_t *in_stride_array,
                const int64_t in_size, const int64_t *padding_start,
                T *output, const int64_t *out_stride_array,
                const int64_t out_size) {
  // Padding zero for output
  cudaMemsetAsync(output, 0, out_size * sizeof(T), stream);

  int threads = 256;
  int block_num = threads * 10;
  int grids = (in_size + block_num - 1) / block_num;
  KeSlicePaddingZero<T, D><<<grids, threads, 0, stream>>>(
      input, in_stride_array, in_size, padding_start, output, out_stride_array);
}

template<>
template<size_t D>
void SliceGradKernel<CUDADeviceContext, float16>::SliceComputeFunction(
          const framework::ExecutionContext& context,
          framework::Tensor *d_input, const framework::DDim& in_dims,
          const framework::Tensor *d_out, const framework::DDim& out_dims,
          const Eigen::array<std::pair<int64_t, int64_t>, D> &paddings) const {
  auto &dev_ctx = context.template device_context<CUDADeviceContext>();

  int64_t in_size = 1, out_size = 1;
#pragma unroll
  for(int i = 0; i < D; i ++) {
    in_size *= in_dims[i];
    out_size *= out_dims[i];
  }

  auto h_tmp_mem =
    memory::Alloc(platform::CPUPlace(), 3 * D * sizeof(int64_t));
  int64_t *h_in_strides = reinterpret_cast<int64_t *>(h_tmp_mem->ptr());
  int64_t *h_out_strides = h_in_strides + D;
  int64_t *h_pad_start = h_out_strides + D;

  auto d_tmp_mem = memory::Alloc(dev_ctx, 3 * D * sizeof(int64_t));
  int64_t *d_in_strides = reinterpret_cast<int64_t *>(d_tmp_mem->ptr());
  int64_t *d_out_strides = d_in_strides + D;
  int64_t *d_pad_start = d_out_strides + D;

  h_in_strides[D - 1] = h_out_strides[D - 1] = 1;
#pragma unroll
  for(int i = D - 2; i >= 0; i --) {
    h_in_strides[i] = h_in_strides[i + 1] * in_dims[i + 1];
    h_out_strides[i] = h_out_strides[i + 1] * out_dims[i + 1];
  }
#pragma unroll
  for(int i = 0; i < D; i ++) {
      h_pad_start[i] = paddings[i].first;
  }

  memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                d_in_strides, platform::CPUPlace(), h_in_strides,
                3 * D * sizeof(int64_t), dev_ctx.stream());

  LaunchSlicePaddingKernel<float16, D>(
                dev_ctx.stream(), d_out->data<float16>(),
                d_out_strides, out_size, d_pad_start,
                d_input->mutable_data<float16>(context.GetPlace()),
                d_in_strides, in_size);
}
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::complex128>);

REGISTER_OP_CUDA_KERNEL(
    slice_grad,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext,
                         plat::complex128>);

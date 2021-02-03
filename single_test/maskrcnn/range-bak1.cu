/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/range_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
__global__ void RangeKernel(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
}

template <typename T>
class CUDARangeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* start_t = context.Input<framework::Tensor>("Start");
    auto* end_t = context.Input<framework::Tensor>("End");
    auto* step_t = context.Input<framework::Tensor>("Step");
    auto* out = context.Output<framework::Tensor>("Out");
    auto &dev_ctx = context.template device_context<CUDADeviceContext>();
    auto stream = dev_ctx.stream();

    T start, end, step;
    memory::Copy(platform::CPUPlace(), &start,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 start_t->data<T>(), sizeof(T),
                 stream);
    memory::Copy(platform::CPUPlace(), &end,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 end_t->data<T>(), sizeof(T),
                 stream);
    memory::Copy(platform::CPUPlace(), &step,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 step_t->data<T>(), sizeof(T),
                 stream);
    dev_ctx.Wait();

    int64_t size = 0;
    GetSize(start, end, step, &size);
    out->Resize(framework::make_ddim({size}));
    T* out_data = out->mutable_data<T>(context.GetPlace());

    if(size >= 10000) { // use gpu
      int block = 1024;
      int grid = (size + block - 1) / block;
      RangeKernel<T><<<grid, block, 0, stream>>>(start, step, size, out_data);
    } else { // use cpu
      auto h_tmp_mem =  memory::Alloc(platform::CPUPlace(), size * sizeof(T));
      auto *tmp_data = reinterpret_cast<T *>(h_tmp_mem->ptr());
      RangeFunction(tmp_data, start, step, size); // cpu compute kernel
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 out_data, platform::CPUPlace(), tmp_data,
                 size * sizeof(T), stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(range, ops::CUDARangeKernel<int>,
                        ops::CUDARangeKernel<int64_t>,
                        ops::CUDARangeKernel<float>,
                        ops::CUDARangeKernel<double>);

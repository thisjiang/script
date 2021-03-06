/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/scan.h>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_index_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
struct CheckTrue {
  __host__ __device__ bool operator()(const T &val) {
    return static_cast<bool>(val);
  }
};
template <typename T>
__global__ void KeGetTrueNum(const T *cond_data, const int64_t numel,
                        int64_t *true_num_array, const int64_t vec_size) {
  const int tid_of_block = threadIdx.x;
  const int BLOCKDIM = blockDim.x;
  const int64_t tid = blockIdx.x * BLOCKDIM + tid_of_block;
  
  if(tid_of_block < BLOCKDIM) {
    for(int64_t idx = tid * vec_size; idx < numel; 
        idx += gridDim.x * BLOCKDIM * vec_size) {
      int ture_num = 0;
      const int64_t cond_over = min(idx + vec_size, numel);
      for(int64_t i = idx; i < cond_over; i ++) {
        if(CheckTrue<T>()(cond_data[i])) ture_num ++;
      }
      true_num_array[idx / vec_size] = ture_num;
    }
  }
}
template <typename T>
__global__ void KeSetTrueIndex(const T *cond_data,const int64_t numel,
                        const int64_t *true_num_array, const int64_t vec_size,
                        int64_t *true_index) {
  const int tid_of_block = threadIdx.x;
  const int BLOCKDIM = blockDim.x;
  const int64_t tid = blockIdx.x * BLOCKDIM + tid_of_block;

  if(tid_of_block < BLOCKDIM) {
    for(int64_t idx = tid * vec_size; idx < numel; 
        idx += gridDim.x * BLOCKDIM * vec_size) {
      int64_t index = (idx == 0 ) ? 0 : true_num_array[idx / vec_size - 1];
      const int64_t cond_over = min(idx + vec_size, numel);
      for(int64_t i = idx; i < cond_over; i ++) {
        if(CheckTrue<T>()(cond_data[i])) {
          true_index[index] = i;
          index ++;
        }
      }
    }
  }
}

template <typename T>
class CUDAWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* out = context.Output<framework::Tensor>("Out");
    auto& dev_ctx = context.template device_context<CUDADeviceContext>();

    // TODO(zhoukunsheng): Should optimize to ensure GPU is faster than CPU.
    const T* cond_data = condition->data<T>();
    int64_t numel = condition->numel();
    auto dims = condition->dims();
    int rank = dims.size();

    int grids = 8;
    int64_t num_of_block = (numel + grids - 1) / grids;
    int threads = std::min(static_cast<int64_t>(1024), num_of_block);
    int vec_size = (num_of_block + threads - 1) / threads;
    int num_of_vec = (numel + vec_size - 1) / vec_size;

    int64_t tmp_mem_size = rank + num_of_vec + numel;
    auto d_tmp_mem = memory::Alloc(dev_ctx, tmp_mem_size * sizeof(int64_t));
    auto h_tmp_mem = memory::Alloc(platform::CPUPlace(), 
                                  rank * sizeof(int64_t));

    int64_t *ptr_stride = reinterpret_cast<int64_t *>(d_tmp_mem->ptr());
    int64_t *ptr_true_num_array = ptr_stride + rank;
    int64_t *ptr_true_index = ptr_true_num_array + num_of_vec;

    int64_t *h_stride = reinterpret_cast<int64_t *>(h_tmp_mem->ptr());

    KeGetTrueNum<T><<<grids, threads, 0, dev_ctx.stream()>>>
              (cond_data, numel, ptr_true_num_array, vec_size);
    thrust::inclusive_scan(thrust::device, ptr_true_num_array, 
                          ptr_true_num_array + num_of_vec, 
                          ptr_true_num_array);
    KeSetTrueIndex<T><<<grids, threads, 0, dev_ctx.stream()>>>
      (cond_data, numel, ptr_true_num_array, vec_size, ptr_true_index);

    h_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      h_stride[i] = h_stride[i + 1] * dims[i + 1];
    }
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 ptr_stride, platform::CPUPlace(),
                 h_stride, rank * sizeof(int64_t),
                 dev_ctx.stream());

    int64_t true_num = 0;
    memory::Copy(platform::CPUPlace(), &true_num,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 ptr_true_num_array + num_of_vec - 1, sizeof(int64_t),
                 dev_ctx.stream());
    dev_ctx.Wait();

    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_ptr = out->mutable_data<int64_t>(context.GetPlace());

    if (true_num == 0) {
      return;
    }
    WhereIndexFunctor<int64_t> functor(ptr_true_index, true_num, ptr_stride,
                                       rank, out_ptr);
    platform::ForRange<CUDADeviceContext> for_range(dev_ctx, true_num);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(where_index, ops::CUDAWhereIndexKernel<int64_t>,
                        ops::CUDAWhereIndexKernel<int>,
                        ops::CUDAWhereIndexKernel<bool>,
                        ops::CUDAWhereIndexKernel<float>,
                        ops::CUDAWhereIndexKernel<double>);

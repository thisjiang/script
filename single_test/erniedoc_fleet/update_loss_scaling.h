// C system file
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// C++ system file
#include <iostream>
#include <vector>

// Library file
#include "../common.h"


constexpr int LOOPNUM = 100;

/*********************************************************************/
template <typename T>
__global__ void FillIf(T* data, const int64_t num, const T value,
                       const bool* has_inf) {
  if (*has_inf) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num; i += blockDim.x * gridDim.x) {
      data[i] = value;
    }
  }
}

template <typename T>
float TimeOfOldKernel(
            CUDAStream& dev_ctx, const bool* found_inf_data,
            const size_t xs_size, const int64_t* nums, T** outs) {
  auto* tt = TimeOfKernel::get(dev_ctx);
  tt->start();
  for(int id = 0; id < LOOPNUM; id ++) {
    for (size_t i = 0; i < xs_size; ++i) {
      T* out_data = outs[i];
      int64_t num = nums[i];
      int block = 1024;
      int grid = (block - 1 + num) / block;
      FillIf<<<grid, block, 0, dev_ctx.stream()>>>(
          out_data, num, static_cast<T>(0), found_inf_data);
    }
  }
  float cost = tt->stop();
  return cost;
}

/*********************************************************************/


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
template <typename T>
float TimeOfMemsetKernel(
            CUDAStream& dev_ctx, const bool* found_inf_data,
            const size_t xs_size, const int64_t* nums, T** outs) {
  auto* tt = TimeOfKernel::get(dev_ctx);
  tt->start();
  for(int id = 0; id < LOOPNUM; id ++) {
    for (size_t i = 0; i < xs_size; ++i) {
      T* out_data = outs[i];
      int64_t num = nums[i];
      cudaMemsetAsync(out_data, 0, num * sizeof(T), dev_ctx.stream());
    }
  }
  float cost = tt->stop();
  return cost;
}

/*********************************************************************/
template <typename T>
__global__ void FusedFillIf(T** outs, const int64_t xs_size,
                      const int64_t* starts, const T value,
                      const bool* has_inf) {
  if(!(*has_inf)) return;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // copy starts array from global memory to shared memory
  extern __shared__ int64_t starts_s[];
  for(int i = threadIdx.x; i <= xs_size; i += blockDim.x) {
    starts_s[i] = starts[i];
  }
  __syncthreads();

  const int64_t total_num = starts_s[xs_size];
  int out_index = 0;

  for (int64_t id = tid; id < total_num; id += blockDim.x * gridDim.x) {
    // get the out's index of thread
    int next_out_index = out_index;
    while(id < starts_s[next_out_index]) next_out_index ++;
    // avoid some tensor's numel is zero
    while(id >= starts_s[next_out_index]) next_out_index ++;
    out_index = next_out_index - 1;

    // get data pointer and index
    T* out_data = outs[out_index];
    int64_t idx = id - starts_s[out_index];

    // set value
    out_data[idx] = value;
  }
}

template <typename T>
float TimeOfFusedKernel(
            CUDAStream& dev_ctx, const bool* found_inf_data,
            const size_t xs_size, const int64_t* nums, T** outs) {
  auto* tt = TimeOfKernel::get(dev_ctx);

  MallocHost<int64_t> starts_h(xs_size + 1, dev_ctx);
  MallocDevice<int64_t> starts_d(xs_size + 1, dev_ctx);
  MallocDevice<T*> outs_d(xs_size, dev_ctx);

  int64_t *starts_h_data = starts_h.data();
  starts_h_data[0] = 0;
  for(int i = 0; i < xs_size; i ++) {
    starts_h_data[i + 1] = starts_h_data[i] + nums[i];
  }
  starts_d.CopyFrom(starts_h);
  outs_d.CopyFromHost(outs, xs_size * sizeof(T*));

  int64_t total_num = starts_h_data[xs_size];
  int64_t block = std::min(static_cast<int64_t>(1024), total_num);
  int64_t grid = (total_num + block - 1) / block;

  tt->start();
  for(int id = 0; id < LOOPNUM; id ++) {
    FusedFillIf<T><<<grid, block, (xs_size + 1) * sizeof(int64_t), dev_ctx.stream()>>>(
                outs_d.data(), xs_size, starts_d.data(),
                0, found_inf_data);
  }
  float cost = tt->stop();
  return cost;
}

/*********************************************************************/


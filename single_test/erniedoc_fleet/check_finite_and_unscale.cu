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

constexpr int LOOPNUM = 100;

template <typename MT>
__global__ void KeInitial(const MT scale_val, MT* scale, bool* found_inf) {
  *scale = scale_val;
  *found_inf = false;
}

template<typename MT>
float InitalKernel(CUDAStream &dev_ctx, MT* scale, bool* found_inf) {
  auto clock = TimeOfKernel::get(dev_ctx);

  MT scale_val;
  Random(&scale_val, 1);
  clock->start();
  KeInitial<<<1, 1, 0, dev_ctx.stream()>>>(scale_val, scale, found_inf);
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template <typename T, typename MT>
__global__ void OldCheckFiniteAndUnscale(const T* in, const MT* scale, int num,
                                      bool* found_inf, T* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < num) {
    MT val = static_cast<MT>(in[idx]) * (*scale);
    T narrow_val = static_cast<T>(val);
    out[idx] = narrow_val;
    if (!isfinite(narrow_val)) {
      *found_inf = true;
    }
  }
}

template<typename T, typename MT>
float TimeOfOldKernel(CUDAStream &dev_ctx, const T* in, const MT* scale, int num,
                          bool* found_inf, T* out) {
  auto clock = TimeOfKernel::get(dev_ctx);

  int block = 1024;
  int grid = (num + block - 1) / block;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    OldCheckFiniteAndUnscale<T, MT><<<grid, block, 0, dev_ctx.stream()>>>(
          in, scale, num, found_inf, out);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/

template <typename T, typename MT>
__global__ void CheckFiniteAndUnscaleLoop(const T* in, const MT* scale, int num,
                                      bool* found_inf, T* out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int idx = tid; idx < num; idx += gridDim.x * blockDim.x) {
    MT val = static_cast<MT>(in[idx]) * (*scale);
    T narrow_val = static_cast<T>(val);
    out[idx] = narrow_val;
    if (!isfinite(narrow_val)) {
      *found_inf = true;
    }
  }
}

template<typename T, typename MT>
float TimeOfLoopKernel(CUDAStream &dev_ctx, const T* in, const MT* scale, int num,
                          bool* found_inf, T* out) {
  auto clock = TimeOfKernel::get(dev_ctx);

  int block = std::min(256, num);
  int grid = (num + block - 1) / block;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    CheckFiniteAndUnscaleLoop<T, MT><<<grid, block, 0, dev_ctx.stream()>>>(
          in, scale, num, found_inf, out);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T, typename MT>
int TestKernel(CUDAStream &context, int num) {
  MallocHost<T> h_old_in(num, context), h_new_in(num, context);
  MallocDevice<T> d_old_in(num, context), d_new_in(num, context);
  MallocDevice<MT> d_scale(1, context);
  MallocDevice<bool> d_found_inf(1, context);

  MallocDevice<T> old_out(num, context), new_out(num, context);

  InitalKernel(context, d_scale.data(), d_found_inf.data());

  h_old_in.Random();
  d_old_in.CopyFrom(h_old_in);
  h_new_in.Random();
  d_new_in.CopyFrom(h_new_in);

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<char*> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  name = "Old Kernel";
  cost = TimeOfOldKernel(context, d_old_in.data(), d_scale.data(),
                              num, d_found_inf.data(), old_out.data());
  AfterRun();

  name = "Loop Kernel";
  cost = TimeOfLoopKernel(context, d_new_in.data(), d_scale.data(),
                              num, d_found_inf.data(), new_out.data());
  AfterRun();

  printf("*******************\n");
  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "CUDA ERROR: %s\n", err);
    return CUDA_FAILED;
  }

  if(!old_out.CheckSame(new_out)) {
    fprintf(stderr, "[%d] Result Check Failed\n", num);
    return CHECK_FAILED;
  }

  return SUCCESS;
}

int main() {
  srand(time(0));
  CUDAStream context;
  typedef float T;
  typedef float MT;

  do {
    int num = 21504000;
    // printf("Please Input num\n");
    // std::cin >> num;
    if(TestKernel<T, MT>(context, num) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
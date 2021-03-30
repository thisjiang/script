// C system file
#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// C++ system file

// Library file
#include "../common.h"

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
void LaunchOldKernel(const gpuStream_t &stream, const T* in, const MT* scale,
                    int64_t num, bool* found_inf, T* out) {
  int block = 1024;
  int grid = (num + block - 1) / block;
  OldCheckFiniteAndUnscale<T, MT><<<grid, block, 0, stream>>>(
          in, scale, num, found_inf, out);
}

template<typename T, typename MT>
float TimeOfOldKernel(CUDAStream &dev_ctx, const int size, int64_t* nums, const T* const * in,
                      const MT* scale, bool* found_inf, T** out) {
  auto clock = TimeOfKernel::get(dev_ctx);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    for(int k = 0; k < size; k ++) {
      LaunchOldKernel(dev_ctx.stream(), in[k], scale, nums[k], found_inf, out[k]);
    }
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template <typename T, typename MT>
__global__ void CheckFiniteAndUnscaleFused(const T* const * xs, const MT* scale,
                    int64_t size, int64_t* starts, bool* found_inf, T** outs) {
  const int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  // copy starts array from global memory to shared memory
  extern __shared__ int64_t s_starts[];
  for(int i = threadIdx.x; i <= size; i += blockDim.x) {
    s_starts[i] = starts[i];
  }
  __syncthreads();

  const int64_t num = s_starts[size];
  int pre_xs_index = 0;
  bool t_found_inf = false;
  const MT t_scale = *scale;
  for (int64_t idx = tid; idx < num; idx += gridDim.x * blockDim.x) {
    // get the xs's index of thread
    int xs_index = pre_xs_index;
    while(idx < s_starts[xs_index]) xs_index ++;
    // avoid some tensor's numel is zero
    while(idx >= s_starts[xs_index]) xs_index ++;
    pre_xs_index = xs_index - 1;

    // get in data and out data
    const T *in = xs[pre_xs_index];
    T *out = outs[pre_xs_index];
    int64_t in_idx = idx - s_starts[pre_xs_index];

    // Unscale
    MT val = static_cast<MT>(in[in_idx]) * t_scale;
    T narrow_val = static_cast<T>(val);
    out[in_idx] = narrow_val;

    //CheckFinite
    if (!isfinite(narrow_val)) {
      t_found_inf = true;
    }
  }
  if(t_found_inf){
    *found_inf = true;
  }
}

template<typename T, typename MT>
float TimeOfFusedKernel(CUDAStream &dev_ctx, const int size, int64_t* nums, const T* const * in,
                      const MT* scale, bool* found_inf, T** out) {
  auto clock = TimeOfKernel::get(dev_ctx);

  MallocHost<int64_t> h_starts(size + 1, dev_ctx);
  MallocDevice<int64_t> d_starts(size + 1, dev_ctx);
  MallocDevice<T*> d_in(size, dev_ctx), d_out(size, dev_ctx);

  int64_t *start_data = h_starts.data();
  start_data[0] = 0;
  for(int i = 1; i <= size; i ++) {
    start_data[i] = start_data[i - 1] + nums[i - 1];
  }
  int64_t total_num = start_data[size];

  d_starts.CopyFrom(h_starts);
  d_in.CopyFrom(in, size * sizeof(T*), get_host_place());
  d_out.CopyFrom(out, size * sizeof(T*), get_host_place());

  int threads = 1024;
  int grids = (total_num + threads - 1) / threads;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    CheckFiniteAndUnscaleFused<T, MT>
      <<<grids, threads, (size + 1) * sizeof(int64_t), dev_ctx.stream()>>>(
      d_in.data(), scale, size, d_starts.data(), found_inf, d_out.data()
    );
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
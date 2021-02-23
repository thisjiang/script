#include "../common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>

constexpr int LOOPNUM = 10;

constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
constexpr int ELEMWISE_WARP_SIZE = 32;

template <typename T>
struct MaxGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * (x > y);
  }
};

template <typename T>
struct MaxGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * (x <= y);
  }
};

namespace paddle {
namespace platform {
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))

template <typename T>
__forceinline__ __device__ T CudaShuffleDownSync(unsigned mask, T val,
                                                 int delta,
                                                 int width = warpSize) {
  return __shfl_down_sync(mask, val, static_cast<unsigned>(delta), width);
}

template <typename T>
__device__ T reduceSum(T val, int tid, int len) {
  // NOTE(zcd): The warp size should be taken from the
  // parameters of the GPU but not specified as 32 simply.
  // To make the reduceSum more efficiently,
  // I use Warp-Level Parallelism and assume the Warp size
  // is 32 which may be different for different GPU,
  // but most card's warp size is 32.
  constexpr int warpSize = 32;
  __shared__ T shm[warpSize];
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < len);

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += CudaShuffleDownSync(mask, val, offset);

  if (tid < warpSize) shm[tid] = 0;
  __syncthreads();

  if (tid % warpSize == 0) {
    shm[tid / warpSize] = val;
  }
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}
}
}

template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP>
static void OldElemwiseGradBroadcast2CUDA(cudaStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
}

template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2BlockCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  __shared__ T s_data;
  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    s_data = y[j];
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], s_data, out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], s_data, out[x_offset], dout[x_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    s_data = x[j];
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(s_data, y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(s_data, y[y_offset], out[y_offset], dout[y_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T>
__forceinline__ __device__ T ElemWarpReduceSum(T val, unsigned mask) {
  for (int offset = ELEMWISE_WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += paddle::platform::CudaShuffleDownSync(mask, val, offset);
  return val;
}

template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2WarpCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  const int tid = threadIdx.x % ELEMWISE_WARP_SIZE;
  const int warp_id = threadIdx.x / ELEMWISE_WARP_SIZE;
  const int wid = blockIdx.x * ELEMWISE_WARP_SIZE + warp_id;

  __shared__ T s_data[ELEMWISE_WARP_SIZE];

  for(int j = wid; j < n; j += gridDim.x * ELEMWISE_WARP_SIZE) {
    T val(0);
    int ttid = tid;

    if (is_xsize_larger) {
      s_data[warp_id] = y[j];
      // do not need __syncwarp()
      while (true) {
        int i = ttid / post;
        int k = ttid % post;
        if (i >= pre) break;

        int x_offset = i * n * post + j * post + k;

        if (dx != nullptr) {
          dx[x_offset] = dx_op(x[x_offset], s_data[warp_id], out[x_offset], dout[x_offset]);
        }

        if (dy != nullptr) {
          val += dy_op(x[x_offset], s_data[warp_id], out[x_offset], dout[x_offset]);
        }

        ttid += ELEMWISE_WARP_SIZE;
      }

      if (dy) {
        val = ElemWarpReduceSum(val, 0xffffffff);
        if (tid == 0) dy[j] = val;
      }
    } else {  // x.dims < y.dims, broadcast for x.
      s_data[warp_id] = x[j];
      // do not need __syncwarp()
      while (true) {
        int i = ttid / post;
        int k = ttid % post;
        if (i >= pre) break;

        int y_offset = i * n * post + j * post + k;

        if (dy != nullptr) {
          dy[y_offset] = dy_op(s_data[warp_id], y[y_offset], out[y_offset], dout[y_offset]);
        }

        if (dx != nullptr) {
          val += dx_op(s_data[warp_id], y[y_offset], out[y_offset], dout[y_offset]);
        }

        ttid += ELEMWISE_WARP_SIZE;
      }

      if (dx) {
        val = ElemWarpReduceSum(val, 0xffffffff);
        if (tid == 0) dx[j] = val;
      }
    }
  }
}

// each matrix is divided into "tile_num" tiles by dimension "n"
// each block deal with "tile_num * post" tiles
// loop "pre" outside assure read and write coalescing
template <typename T, typename DX_OP, typename DY_OP, int BLOCKDIM>
static __global__ void ElemwiseGradBroadcast2ThreadCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  const int tile_num = BLOCKDIM / post;
  const int tid = threadIdx.x;
  // return when thread exceed tile number
  if (tid >= tile_num * post) return;

  const int tile_id = tid / post;
  const int n_id = blockIdx.x * tile_num + tile_id;
  const int ttid = tid % post;

  __shared__ T s_data[BLOCKDIM];
  s_data[tid] = 0;

  if (is_xsize_larger) {
    for (int i = 0; i < pre; i++) {
      // x_offset is consecutive for all thread in block
      int x_offset = i * n * post + n_id * post + ttid;

      if (dx != nullptr) {
        // read and write dx,x,out,dout is coalescing
        dx[x_offset] =
            dx_op(x[x_offset], y[n_id], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        // using shared memory to store "ttid"'s "dy[nid]" value
        s_data[tid] +=
            dy_op(x[x_offset], y[n_id], out[x_offset], dout[x_offset]);
      }
    }
    __syncthreads();

    if (dy != nullptr && ttid == 0) {
      // first tile thread calculate total "dy[nid]" value
      T val(0);
      const int tile_over = (tile_id + 1) * post;
      for (int l = tile_over - post; l < tile_over; l++) val += s_data[l];
      dy[n_id] = val;
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int i = 0; i < pre; i++) {
      // y_offset is consecutive for all thread in block
      int y_offset = i * n * post + n_id * post + ttid;

      if (dy != nullptr) {
        dy[y_offset] =
            dy_op(x[n_id], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        s_data[tid] +=
            dx_op(x[n_id], y[y_offset], out[y_offset], dout[y_offset]);
      }
    }
    __syncthreads();

    if (dx != nullptr && ttid == 0) {
      T val(0);
      const int tile_over = (tile_id + 1) * post;
      for (int l = tile_over - post; l < tile_over; l++) val += s_data[l];
      dx[n_id] = val;
    }
  }
}

// when n is small, it's very useful
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcastAtomic2CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T* __restrict__ dx, T* __restrict__ dy) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nid = (tid / post) % n;

  if(tid > pre * n * post) return;

  if (is_xsize_larger) {
    int x_offset = tid;

    if (dx != nullptr) {
      dx[x_offset] = dx_op(x[x_offset], y[nid], out[x_offset], dout[x_offset]);
    }

    if (dy != nullptr) {
      T val = dy_op(x[x_offset], y[nid], out[x_offset], dout[x_offset]);
      atomicAdd(dy + nid, val);
    }
  } else {  // x.dims < y.dims, broadcast for x.
    int y_offset = tid;

    if (dy != nullptr) {
      dy[y_offset] = dy_op(x[nid], y[y_offset], out[y_offset], dout[y_offset]);
    }

    if (dx != nullptr) {
      T val = dx_op(x[nid], y[y_offset], out[y_offset], dout[y_offset]);
      atomicAdd(dx + nid, val);
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP>
static void NewElemwiseGradBroadcast2CUDA(cudaStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  int num = pre * post;
  if(post >= 32 || num >= ELEMWISE_MAX_BLOCK_DIM) {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, num);
    int grid_size = n;
    ElemwiseGradBroadcast2BlockCUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else {
    // each thread handle one operation
    constexpr int block_size = 256;
    int tile_num = block_size / post;
    int grid_size = (n + tile_num - 1) / tile_num;
    ElemwiseGradBroadcast2ThreadCUDAKernel<T, DX_OP, DY_OP, block_size>
        <<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
  }
}

template <typename T, typename DX_OP, typename DY_OP>
int ElemwiseGradBroadcast(CUDAStream &context, 
                          AllocHost &h_x, AllocDevice &d_x,
                          AllocHost &h_y, AllocDevice &d_y,
                          AllocHost &h_out, AllocDevice &d_out,
                          AllocHost &h_dout, AllocDevice &d_dout,
                          int pre, int n, int post, bool is_xsize_larger,
                          DX_OP dx_op, DY_OP dy_op,
                          AllocDevice &dx_old, AllocDevice &dy_old,
                          AllocDevice &dx_new, AllocDevice &dy_new){
  auto clock = TimeOfKernel::get(context);

  // check data size
  int out_num = pre * n * post;
  size_t x_num = 0, y_num = 0;
  if(is_xsize_larger) {
    x_num = out_num;
    y_num = n;
  } else {
    x_num = n;
    y_num = out_num;
  }

  h_x.resize(x_num * sizeof(T), true);
  d_x.resize(x_num * sizeof(T), true);
  h_y.resize(y_num * sizeof(T), true);
  d_y.resize(y_num * sizeof(T), true);

  T* x_ptr = h_x.data<T>();
  T* y_ptr = h_y.data<T>();
  T* out_ptr = h_out.data<T>();
  T *dout_ptr = h_dout.data<T>();

  // generate rand number
  Random<T>(x_ptr, x_num, 1, 0);
  Random<T>(y_ptr, y_num, 1, 0);
  Random<T>(out_ptr, out_num, 1, 0);
  Random<T>(dout_ptr, out_num, 1, 0);

  // copy data
  d_x.CopyFrom(h_x);
  d_y.CopyFrom(h_y);
  d_out.CopyFrom(h_out);
  d_dout.CopyFrom(h_dout);

  // Set output size
  dx_old.resize(x_num * sizeof(T));
  dy_old.resize(y_num * sizeof(T));
  dx_new.resize(x_num * sizeof(T));
  dy_new.resize(y_num * sizeof(T));

  // Initial kernel
  OldElemwiseGradBroadcast2CUDA(context.stream(), 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_old.data<T>(), dy_old.data<T>());

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  OldElemwiseGradBroadcast2CUDA(context.stream(), 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_old.data<T>(), dy_old.data<T>());
  float old_time = clock->stop();

  // Initial kernel
  dx_new.SetZero();
  dy_new.SetZero();
  NewElemwiseGradBroadcast2CUDA(context.stream(), 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_new.data<T>(), dy_new.data<T>());

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  NewElemwiseGradBroadcast2CUDA(context.stream(), 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_new.data<T>(), dy_new.data<T>());
  float new_time = clock->stop();
  printf("Old time %f vs New time %f\n", old_time, new_time);

  auto err_new = context.sync();
  if(err_new != "") {
    fprintf(stderr, "New ERROR: %s\n", err_new);
    return CUDA_FAILED;
  }

  // check result
  T dx_err = dx_old.MaxError<T>(dx_new);
  T dy_err = dy_old.MaxError<T>(dy_new);

  if(dx_err > 1e-4f || dy_err > 1e-4f) {
    fprintf(stderr, "[%d, %d, %d, %d] Error: dx ", 
          pre, n, post, static_cast<int>(is_xsize_larger));
    fprint(dx_err);
    fprintf(stderr, " dy ");
    fprint(dy_err);
    fprintf(stderr, "\n");

    if(pre * post * n > 0) return 1;

    fprintf(stderr, "dx_old\n");
    dx_old.Print<T>(1, x_num);
    fprintf(stderr, "dx_new\n");
    dx_new.Print<T>(1, x_num);
    fprintf(stderr, "dy_old\n");
    dy_old.Print<T>(1, y_num);
    fprintf(stderr, "dy_new\n");
    dy_new.Print<T>(1, y_num);

    return CHECK_FAILED;
  } else {
    printf("[%d, %d, %d, %d] Success!\n", 
          pre, n, post, static_cast<int>(is_xsize_larger));
  }

  return SUCCESS;
}

struct Param {
  int pre, post, n;
};

int main() {
    CUDAStream context;

    typedef float T;

    srand(time(NULL));

    int pre = 2, post = 3, n = 100;
    bool is_xsize_larger = true;

    do {
      pre = rand() % 100 + 1;
      post = rand() % 100 + 1;
      n = rand() % 10000 + 1000;

      int max_num = pre * post * n * sizeof(T);

      AllocHost h_x(max_num, context), h_y(max_num, context),
                h_out(max_num, context), h_dout(max_num, context);
      AllocDevice d_x(max_num, context), d_y(max_num, context),
                  d_out(max_num, context), d_dout(max_num, context);
      AllocDevice dx_old(max_num, context), dy_old(max_num, context),
                  dx_new(max_num, context), dy_new(max_num, context);

      is_xsize_larger = true;
      int res = ElemwiseGradBroadcast<T, MaxGradDx<T>, MaxGradDy<T>>
                            (context, h_x, d_x, h_y, d_y,
                            h_out, d_out, h_dout, d_dout,
                            pre, n, post, is_xsize_larger,
                            MaxGradDx<T>(), MaxGradDy<T>(),
                            dx_old, dy_old, dx_new, dy_new);
      if(res == CUDA_FAILED) {
        fprintf(stderr, "Compte Failed with CUDA error\n");
        return 1;
      }
      is_xsize_larger = false;
      res = ElemwiseGradBroadcast<T, MaxGradDx<T>, MaxGradDy<T>>
                            (context, h_x, d_x, h_y, d_y,
                            h_out, d_out, h_dout, d_dout,
                            pre, n, post, is_xsize_larger,
                            MaxGradDx<T>(), MaxGradDy<T>(),
                            dx_old, dy_old, dx_new, dy_new);
      if(res == CUDA_FAILED) {
        fprintf(stderr, "Compte Failed with CUDA error\n");
        return 1;
      }
    } while(true);
    
    return 0;
}
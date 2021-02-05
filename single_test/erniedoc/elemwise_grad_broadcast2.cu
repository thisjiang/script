#include "../common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>

constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
constexpr int ELEMWISE_MAX_WARP_NUM = 
              (ELEMWISE_MAX_BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE;
constexpr int HALF_WARP = 16;

#define HOSTDEVICE __host__ __device__

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

namespace math {
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
  const int warpSize = 32;
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
      val = math::reduceSum(val, tid, h);
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
      val = math::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T>
__forceinline__ __device__ T ElemWarpReduceSum(T val, unsigned mask) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += math::CudaShuffleDownSync(mask, val, offset);
  return val;
}

template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2WarpCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  const int tid = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int j = blockIdx.x * WARP_SIZE + warp_id;

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

      ttid += WARP_SIZE;
    }

    if (dy) {
      val = ElemWarpReduceSum(val, 0xffffffff);
      if (tid == 0) {
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

      ttid += WARP_SIZE;
    }

    if (dx) {
      val = ElemWarpReduceSum(val, 0xffffffff);
      if (tid == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2ThreadCUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  T val(0);

  if (is_xsize_larger) {
    for (int ttid = 0; ttid < pre * post; ttid ++) {
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
    }

    if (dy) dy[j] = val;
  } else {  // x.dims < y.dims, broadcast for x.
    for (int ttid = 0; ttid < pre * post; ttid ++) {
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
    }

    if (dx) dx[j] = val;
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
  fprintf(stderr, "%d %d\n", block_size, gird_size);
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
}

template <typename T, typename DX_OP, typename DY_OP>
static void NewElemwiseGradBroadcast2CUDA(cudaStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  int num = pre * post;
  if(num >= 256) {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, num);
    int gird_size = n;
    ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else if(num >= 32) {
    // each warp handle one operation, each block handle 32 operation
    int block_size = std::min(n * WARP_SIZE, ELEMWISE_MAX_BLOCK_DIM);
    int warp_size = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int gird_size = (n + warp_size - 1) / warp_size;
    ElemwiseGradBroadcast2WarpCUDAKernel<<<gird_size, block_size, 0, stream>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else {
    // each thread handle one operation
    int block_size = std::min(n, ELEMWISE_MAX_BLOCK_DIM);
    int gird_size = (n + block_size - 1) / block_size;
    ElemwiseGradBroadcast2ThreadCUDAKernel<<<gird_size, block_size, 0, stream>>>(
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
  T* x_ptr = h_x.data<T>();
  T* y_ptr = h_y.data<T>();
  T* out_ptr = h_out.data<T>();
  T *dout_ptr = h_dout.data<T>();

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

  h_x.resize(x_num, true);
  d_x.resize(x_num, true);
  h_y.resize(y_num, true);
  d_y.resize(y_num, true);

  // generate rand number
  Random<T>(x_ptr, x_num, 100);
  Random<T>(y_ptr, y_num, 100);
  Random<T>(out_ptr, out_num, 100);
  Random<T>(dout_ptr, out_num, 100);

  // copy data
  d_x.CopyFrom(h_x);
  d_y.CopyFrom(h_y);
  d_out.CopyFrom(h_out);
  d_dout.CopyFrom(h_dout);

  OldElemwiseGradBroadcast2CUDA(context.stream(), 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_old.data<T>(), dy_old.data<T>());
  NewElemwiseGradBroadcast2CUDA(context.stream(), 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_new.data<T>(), dy_new.data<T>());

  auto err = context.sync();
  if(err != "") {
    fprintf(stderr, "ERROR: %s\n", err);
    return CUDA_FAILED;
  }

  // check result
  T dx_err = dx_old.MaxError<T>(dx_new);
  T dy_err = dy_old.MaxError<T>(dy_new);

  if(dx_err > 1e-6f || dy_err > 1e-6f) {
    fprintf(stderr, "[%d, %d, %d, %d] Error: dx ", 
          pre, post, n, static_cast<int>(is_xsize_larger));
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
          pre, post, n, static_cast<int>(is_xsize_larger));
  }

  return SUCCESS;
}

struct Param {
  int pre, post, n;
};

int main() {
    CUDAStream context;

    // Param param[] = {{2, 3, 10},
    //                  {4, 8, 100},
    //                  {8, 8, 1024},
    //                  {64, 4, 2048},
    //                  {128, 128, 2048}};
    Param param[] = {{64, 4, 2048}};
    int pre = 2, post = 3, n = 100;
    bool is_xsize_larger = true;

    for(int i = 0; i < sizeof(param) / sizeof(Param); i ++) {
      pre = param[i].pre;
      post = param[i].post;
      n = param[i].n;

      int max_num = pre * post * n * sizeof(int64_t);

      AllocHost h_x(max_num, context), h_y(max_num, context),
                h_out(max_num, context), h_dout(max_num, context);
      AllocDevice d_x(max_num, context), d_y(max_num, context),
                  d_out(max_num, context), d_dout(max_num, context);
      AllocDevice dx_old(max_num, context), dy_old(max_num, context),
                  dx_new(max_num, context), dy_new(max_num, context);

      is_xsize_larger = true;
      int res = ElemwiseGradBroadcast<float, MaxGradDx<float>, MaxGradDy<float>>
                            (context, h_x, d_x, h_y, d_y,
                            h_out, d_out, h_dout, d_dout,
                            pre, n, post, is_xsize_larger,
                            MaxGradDx<float>(), MaxGradDy<float>(),
                            dx_old, dy_old, dx_new, dy_new);
      if(res == CUDA_FAILED) {
        fprintf(stderr, "Compte Failed with CUDA error\n");
        return 1;
      }
      is_xsize_larger = false;
      res = ElemwiseGradBroadcast<float, MaxGradDx<float>, MaxGradDy<float>>
                            (context, h_x, d_x, h_y, d_y,
                            h_out, d_out, h_dout, d_dout,
                            pre, n, post, is_xsize_larger,
                            MaxGradDx<float>(), MaxGradDy<float>(),
                            dx_old, dy_old, dx_new, dy_new);
      if(res == CUDA_FAILED) {
        fprintf(stderr, "Compte Failed with CUDA error\n");
        return 1;
      }
    }
    
    return 0;
}
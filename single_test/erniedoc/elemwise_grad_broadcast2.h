#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>

#include "../common.h"

constexpr int LOOPNUM = 100;

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

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
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
static float OldElemwiseGradBroadcast2CUDA(CUDAStream &context, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  auto tt = TimeOfKernel::get(context);

  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
    ElemwiseGradBroadcast2CUDAKernel<<<
        gird_size, block_size, 0, context.stream()>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
  float cost = tt->stop();
  return cost;
}

/**********************************************************************************/
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

/**********************************************************************/
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2CUDAKernelShared(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    T local_y = y[j];
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] =
            dx_op(x[x_offset], local_y, out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], local_y, out[x_offset], dout[x_offset]);
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
    T local_x = x[j];
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] =
            dy_op(local_x, y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(local_x, y[y_offset], out[y_offset], dout[y_offset]);
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

// each matrix is divided into "tile_num" tiles by dimension "n"
// each tile compute "pre * post" elemwise_grad result
// post number continuous threads deal with a tile
// each thread loop "pre" outside assure read and write coalescing
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2ThreadCUDAKernel(
    const T* __restrict__ x, const T* __restrict__ y,
    const T* __restrict__ out, const T* __restrict__ dout, int pre, int n,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op,
    T* __restrict__ dx, T* __restrict__ dy) {
  // each block deal with "tile_num" tiles, deal with t_num element
  const int tile_num = blockDim.x / post;
  const int t_num = tile_num * post;
  const int tid = threadIdx.x;
  // return when thread exceed tile number
  if(tid >= t_num) return;

  const int tile_id = tid / post;
  const int n_id = blockIdx.x * tile_num + tile_id;
  const int ttid = tid % post;
  if(n_id >= n) return;

  using AccT = typename GetAccType<T>::type;
  __shared__ __align__(sizeof(AccT)) unsigned char
                  s_mem[ELEMWISE_MAX_BLOCK_DIM * sizeof(AccT)];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);
  s_data[tid] = (AccT)0;

  if (is_xsize_larger) {
    for (int i = 0; i < pre; i ++) {
      // x_offset is consecutive for all thread in block
      int x_offset = i * n * post + n_id * post + ttid;

      if (dx != nullptr) {
        // read and write dx,x,out,dout is coalescing
        dx[x_offset] = dx_op(x[x_offset], y[n_id], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        // using shared memory to store "ttid"'s "dy[nid]" value
        s_data[tid] += static_cast<AccT>(
              dy_op(x[x_offset], y[n_id], out[x_offset], dout[x_offset]));
      }
    }
    __syncthreads();

    if (dy != nullptr && ttid == 0) {
      // first tile thread calculate total "dy[nid]" value
      AccT val(0);
      const int tile_beg = tile_id * post;
      const int tile_end = min(tile_beg + post, t_num);
      for(int l = tile_beg; l < tile_end; l ++) {
        val += s_data[l];
      }
      dy[n_id] = static_cast<T>(val);
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int i = 0; i < pre; i ++) {
      // y_offset is consecutive for all thread in block
      int y_offset = i * n * post + n_id * post + ttid;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(x[n_id], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        s_data[tid] += static_cast<AccT>(
              dx_op(x[n_id], y[y_offset], out[y_offset], dout[y_offset]));
      }
    }
    __syncthreads();

    if (dx != nullptr && ttid == 0) {
      AccT val(0);
      const int tile_beg = tile_id * post;
      const int tile_end = min(tile_beg + post, t_num);
      for(int l = tile_beg; l < tile_end; l ++) {
        val += s_data[l];
      }
      dx[n_id] = static_cast<T>(val);
    }
  }
}

// each matrix is divided into "tile_num" tiles by dimension "n"
// each tile compute "pre * post" elemwise_grad result
// post number continuous threads deal with a tile
// each thread loop "pre" outside assure read and write coalescing
template <typename T, typename DX_OP, typename DY_OP, int vec_size>
static __global__ void ElemwiseGradBroadcast2ThreadVecCUDAKernel(
    const T* __restrict__ x, const T* __restrict__ y,
    const T* __restrict__ out, const T* __restrict__ dout, int pre, int mid,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op,
    T* __restrict__ dx, T* __restrict__ dy) {
  assert((mid * post) % vec_size == 0);

  const int elem_per_block = (blockDim.x * vec_size / post) * post;
  const int elem_thread_start = threadIdx.x * vec_size;
  if(elem_thread_start >= elem_per_block) return;

  const int total_elem_num = mid * post;
  const int elem_global_start = blockIdx.x * elem_per_block + elem_thread_start;
  if(elem_global_start >= total_elem_num) return;

  // shared memory
  using AccT = typename MPTypeTrait<T>::Type;
  __shared__ __align__(sizeof(AccT)) unsigned char
                  s_mem[ELEMWISE_MAX_BLOCK_DIM * sizeof(AccT) * vec_size];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

#pragma unroll
  for(int k = 0; k < vec_size; k ++)
    s_data[elem_thread_start + k] = (AccT)0;

  // vectorize data read and write
  using VecT = typename GetVecType<T, vec_size>::Type;
  VecT vec_out, vec_dout;
  T *buf_out = reinterpret_cast<T*>(&vec_out);
  T *buf_dout = reinterpret_cast<T*>(&vec_dout);

  if (is_xsize_larger) {
    VecT vec_x, vec_dx;
    T *buf_x = reinterpret_cast<T*>(&vec_x);
    T *buf_dx = reinterpret_cast<T*>(&vec_dx);

    for (int i = 0; i < pre; i ++) {
      // x_offset is consecutive for all thread in block
      int x_offset = i * total_elem_num + elem_global_start;

      // vectorize read x to ensure coalescing
      vec_x = reinterpret_cast<const VecT*>(&x[x_offset])[0];
      vec_out = reinterpret_cast<const VecT*>(&out[x_offset])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout[x_offset])[0];
      if (dx != nullptr) {
#pragma unroll
        for(int k = 0; k < vec_size; k ++) {
          const int mid_id = (elem_global_start + k) / post;
          buf_dx[k] = dx_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]);
        }
        // vectorize write dx
        reinterpret_cast<VecT*>(&dx[x_offset])[0] = vec_dx;
      }

      if (dy != nullptr) {
#pragma unroll
        for(int k = 0; k < vec_size; k ++) {
          const int mid_id = (elem_global_start + k) / post;
          s_data[elem_thread_start + k] += static_cast<AccT>(
              dy_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]));
        }
      }
    }
    __syncthreads();

    if (dy != nullptr) {
      // calculate total "dy[nid]" value
#pragma unroll
      for(int k = 0; k < vec_size; k ++) {
        const int mid_id = (elem_global_start + k) / post;
        const int mid_id_in_block = (elem_thread_start + k) / post;
        const int tile_beg = mid_id_in_block * post;
        AccT val(0);
        for(int l = 0; l < post; l ++) {
          val += s_data[tile_beg + l];
        }
        dy[mid_id] = static_cast<T>(val);
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    VecT vec_y, vec_dy;
    T *buf_y = reinterpret_cast<T*>(&vec_y);
    T *buf_dy = reinterpret_cast<T*>(&vec_dy);

    for (int i = 0; i < pre; i ++) {
      // y_offset is consecutive for all thread in block
      int y_offset = i * total_elem_num + elem_global_start;

      // vectorize read x
      vec_y = reinterpret_cast<const VecT*>(&y[y_offset])[0];
      vec_out = reinterpret_cast<const VecT*>(&out[y_offset])[0];
      vec_dout = reinterpret_cast<const VecT*>(&dout[y_offset])[0];
      if (dy != nullptr) {
#pragma unroll
        for(int k = 0; k < vec_size; k ++) {
          const int mid_id = (elem_global_start + k) / post;
          buf_dy[k] = dy_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]);
        }
        // vectorize write dx
        reinterpret_cast<VecT*>(&dy[y_offset])[0] = vec_dy;
      }

      if (dx != nullptr) {
#pragma unroll
        for(int k = 0; k < vec_size; k ++) {
          const int mid_id = (elem_global_start + k) / post;
          s_data[elem_thread_start + k] += static_cast<AccT>(
              dx_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]));
        }
      }
    }
    __syncthreads();

    if (dx != nullptr) {
      // calculate total "dx[nid]" value
#pragma unroll
      for(int k = 0; k < vec_size; k ++) {
        const int mid_id = (elem_global_start + k) / post;
        const int mid_id_in_block = (elem_thread_start + k) / post;
        const int tile_beg = mid_id_in_block * post;
        AccT val(0);
        for(int l = 0; l < post; l ++) {
          val += s_data[tile_beg + l];
        }
        dx[mid_id] = static_cast<T>(val);
      }
    }
  }
}

#define LAUNCH_ELEMWISE_GRAD_KERNEL(vec_size)                     \
  int post_per_block = std::min(                                  \
                  ELEMWISE_MAX_BLOCK_DIM * vec_size / post, n);   \
  int elem_per_block = post_per_block * post;                     \
  while(elem_per_block % vec_size != 0) elem_per_block -= post;   \
  post_per_block = elem_per_block / post;                         \
  int thread_per_block = elem_per_block / vec_size;               \
  int block_per_grid = (n + post_per_block - 1) / post_per_block; \
  ElemwiseGradBroadcast2ThreadVecCUDAKernel<                      \
      T, DX_OP, DY_OP, vec_size><<<                               \
      block_per_grid, thread_per_block, 0, stream>>>(             \
      x, y, out, dout, pre, n, post, is_xsize_larger,             \
      dx_op, dy_op, dx, dy);

template <typename T, typename DX_OP, typename DY_OP>
static typename std::enable_if<
          !std::is_same<T, paddle::platform::float16>::value &&
          !std::is_same<T, float>::value &&
          !std::is_same<T, double>::value, void>::type
          LaunchElemwiseGradBroadcast2ThreadVecCUDAKernel(
                    gpuStream_t stream, const T *x,
                    const T *y, const T *out, const T *dout,
                    int pre, int n, int post,
                    bool is_xsize_larger, DX_OP dx_op,
                    DY_OP dy_op, T *dx, T *dy) {
  LAUNCH_ELEMWISE_GRAD_KERNEL(1)
}

template <typename T, typename DX_OP, typename DY_OP>
static typename std::enable_if<
          std::is_same<T, paddle::platform::float16>::value ||
          std::is_same<T, float>::value || 
          std::is_same<T, double>::value, void>::type
          LaunchElemwiseGradBroadcast2ThreadVecCUDAKernel(
                    gpuStream_t stream, const T *x,
                    const T *y, const T *out, const T *dout,
                    int pre, int n, int post,
                    bool is_xsize_larger, DX_OP dx_op,
                    DY_OP dy_op, T *dx, T *dy) {
  const int last_num = n * post;

  if(last_num % 4 == 0) {
    LAUNCH_ELEMWISE_GRAD_KERNEL(4)
  } else if(last_num % 2 == 0) {
    LAUNCH_ELEMWISE_GRAD_KERNEL(2)
  } else {
    LAUNCH_ELEMWISE_GRAD_KERNEL(1)
  }
}

#undef LAUNCH_ELEMWISE_GRAD_KERNEL

template <typename T, typename DX_OP, typename DY_OP>
static float NewElemwiseGradBroadcast2CUDA(CUDAStream &context, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post,
                                       bool is_xsize_larger, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  auto tt = TimeOfKernel::get(context);
  float cost = 0.0f;

  int num = pre * post;
  if(num <= 64 && n >= ELEMWISE_MAX_BLOCK_DIM * 3) {
    tt->start();
    for(int i = 0; i < LOOPNUM; i ++)
      LaunchElemwiseGradBroadcast2ThreadVecCUDAKernel<T, DX_OP, DY_OP>(
                          context.stream(), x, y, out, dout, pre, n, post, 
                          is_xsize_larger, dx_op, dy_op, dx, dy);
    cost = tt->stop();
  } else {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, num);
    int grid_size = n;
    tt->start();
    for(int i = 0; i < LOOPNUM; i ++)
      ElemwiseGradBroadcast2CUDAKernelShared<T, DX_OP, DY_OP>
          <<<grid_size, block_size, 0, context.stream()>>>(
          x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
    cost = tt->stop();
  }
  return cost;
}

/**********************************************************************/

// each matrix is divided into "post_num" tiles by dimension "n"
// each tile compute "pre * post" elemwise_grad result
// post number continuous threads deal with a tile
// each thread loop "pre" outside assure read and write coalescing
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2ThreadVectorizeCUDAKernel(
    const T* __restrict__ x, const T* __restrict__ y,
    const T* __restrict__ out, const T* __restrict__ dout, int pre, int mid,
    int post, bool is_xsize_larger, DX_OP dx_op, DY_OP dy_op,
    T* __restrict__ dx, T* __restrict__ dy) {
  constexpr int vec_size = 4;
  // each block deal with "post_num" tiles, deal with t_num element
  const int elem_per_block = ((blockDim.x * vec_size) / post) * post;
  const int elem_thread_start = threadIdx.x * vec_size;
  if(elem_thread_start >= elem_per_block) return;

  const int total_elem_num = mid * post;
  const int elem_global_start = blockIdx.x * elem_per_block + elem_thread_start;
  // return when the start element id exceed the total element number.
  if(elem_global_start >= total_elem_num) return;
  // the last thread is not full if total_elem_num % vec_size != 0
  int remaining_num = total_elem_num - elem_global_start;
  if(remaining_num >= vec_size) remaining_num = 0;

  using AccT = typename details::MPTypeTrait<T>::Type;
  __shared__ __align__(sizeof(AccT)) unsigned char
          s_mem[ELEMWISE_MAX_BLOCK_DIM * sizeof(AccT) * vec_size];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);
#pragma unroll
  for(int k = 0; k < vec_size; k ++)
    s_data[elem_thread_start + k] = static_cast<AccT>(0);

  // vectorize data read and write
  using VecT = typename details::GetVecType<T, vec_size>::Type;
  VecT vec_out, vec_dout;
  T *buf_out = reinterpret_cast<T*>(&vec_out);
  T *buf_dout = reinterpret_cast<T*>(&vec_dout);

  if (is_xsize_larger) {
    VecT vec_x;
    T *buf_x = reinterpret_cast<T*>(&vec_x);

    for (int i = 0; i < pre; i ++) {
      // x_offset is consecutive for all thread in block
      int x_offset = i * total_elem_num + elem_global_start;

      if(remaining_num > 0) { // the last thread not full
        for(int k = 0; k < remaining_num; k ++) {
          buf_x[k] = x[x_offset + k];
          buf_out[k] = out[x_offset + k];
          buf_dout[k] = dout[x_offset + k];
        }
      } else { // full thread, vectorize read data
        vec_x = reinterpret_cast<const VecT*>(&x[x_offset])[0];
        vec_out = reinterpret_cast<const VecT*>(&out[x_offset])[0];
        vec_dout = reinterpret_cast<const VecT*>(&dout[x_offset])[0];
      }

      if (dx != nullptr) {
        if(remaining_num > 0) { // not full
          for(int k = 0; k < remaining_num; k ++) {
            int mid_id = (elem_global_start + k) / post;
            // TODO(jiangcheng05): the last thread dx write not coalesced
            dx[x_offset + k] =
              dx_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]);
          }
        } else { // full thread, vectorize write data
          VecT vec_dx;
          T *buf_dx = reinterpret_cast<T*>(&vec_dx);
#pragma unroll
          for(int k = 0; k < vec_size; k ++) {
            int mid_id = (elem_global_start + k) / post;
            buf_dx[k] =
              dx_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]);
          }
          reinterpret_cast<VecT*>(&dx[x_offset])[0] = vec_dx;
        }
      }

      if (dy != nullptr) {
        if(remaining_num > 0) {
          for(int k = 0; k < remaining_num; k ++) {
            int mid_id = (elem_global_start + k) / post;
            s_data[elem_thread_start + k] +=
              dy_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]);
          }
        } else {
#pragma unroll
          for(int k = 0; k < vec_size; k ++) {
            int mid_id = (elem_global_start + k) / post;
            s_data[elem_thread_start + k] += static_cast<AccT>(
              dy_op(buf_x[k], y[mid_id], buf_out[k], buf_dout[k]));
          }
        }
      }
    }
    __syncthreads();

    if (dy != nullptr) {
      // Do not need distinguish full or not，
      // because all shared data are initialized to 0
#pragma unroll
      for(int k = 0; k < vec_size; k ++) {
        int mid_id = (elem_global_start + k) / post;
        int mid_id_in_block = (elem_thread_start + k) / post;
        int post_beg = mid_id_in_block * post;
        int post_end = post_beg + post;
        AccT val(0);
        for(int l = post_beg; l < post_end; l ++) {
          val += s_data[l];
        }
        if(mid_id < mid)
          dy[mid_id] = static_cast<T>(val);
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    VecT vec_y;
    T *buf_y = reinterpret_cast<T*>(&vec_y);

    for (int i = 0; i < pre; i ++) {
      // y_offset is consecutive for all thread in block
      int y_offset = i * total_elem_num + elem_global_start;

      if(remaining_num > 0) { // the last thread not full
        for(int k = 0; k < remaining_num; k ++) {
          buf_y[k] = y[y_offset + k];
          buf_out[k] = out[y_offset + k];
          buf_dout[k] = dout[y_offset + k];
        }
      } else { // full thread, vectorize read data
        vec_y = reinterpret_cast<const VecT*>(&y[y_offset])[0];
        vec_out = reinterpret_cast<const VecT*>(&out[y_offset])[0];
        vec_dout = reinterpret_cast<const VecT*>(&dout[y_offset])[0];
      }

      if (dx != nullptr) {
        if(remaining_num > 0) {
          for(int k = 0; k < remaining_num; k ++) {
            int mid_id = (elem_global_start + k) / post;
            dy[y_offset + k] =
              dy_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]);
          }
        } else {
          VecT vec_dy;
          T *buf_dy = reinterpret_cast<T*>(&vec_dy);
#pragma unroll
          for(int k = 0; k < vec_size; k ++) {
            int mid_id = (elem_global_start + k) / post;
            buf_dy[k] =
              dy_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]);
          }
          // full thread, vectorize write data
          reinterpret_cast<VecT*>(&dx[y_offset])[0] = vec_dy;
        }
      }

      if (dy != nullptr) {
        if(remaining_num > 0) {
          for(int k = 0; k < remaining_num; k ++) {
            int mid_id = (elem_global_start + k) / post;
            s_data[elem_thread_start + k] +=
              dx_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]);
          }
        } else {
#pragma unroll
          for(int k = 0; k < vec_size; k ++) {
            int mid_id = (elem_global_start + k) / post;
            s_data[elem_thread_start + k] += static_cast<AccT>(
              dx_op(x[mid_id], buf_y[k], buf_out[k], buf_dout[k]));
          }
        }
      }
    }
    __syncthreads();

    if (dy != nullptr) {
      // Do not need distinguish full or not，
      // because all shared data are initialized to 0
#pragma unroll
      for(int k = 0; k < vec_size; k ++) {
        int mid_id = (elem_global_start + k) / post;
        int mid_id_in_block = (elem_thread_start + k) / post;
        int post_beg = mid_id_in_block * post;
        int post_end = post_beg + post;
        AccT val(0);
        for(int l = post_beg; l < post_end; l ++) {
          val += s_data[l];
        }
        if(mid_id < mid)
          dx[mid_id] = static_cast<T>(val);
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP>
static float VectorizeElemwiseGradBroadcast2CUDA(
        CUDAStream &context, const T *x,
        const T *y, const T *out, const T *dout,
        int pre, int n, int post,
        bool is_xsize_larger, DX_OP dx_op,
        DY_OP dy_op, T *dx, T *dy) {
  auto tt = TimeOfKernel::get(context);
  float cost = 0.0f;

  int num = pre * post;
  if(num <= 64) {
    constexpr int vec_size = 4;
    int post_per_block = std::min(
                  ELEMWISE_MAX_BLOCK_DIM * vec_size / post, n);
    int elem_per_block = post_per_block * post;
    while(elem_per_block % vec_size != 0) elem_per_block -= post;
    post_per_block = elem_per_block / post;
    int thread_per_block = elem_per_block / vec_size;
    int block_per_grid = (n + post_per_block - 1) / post_per_block;

    tt->start();
    for(int i = 0; i < LOOPNUM; i ++)
      ElemwiseGradBroadcast2ThreadVectorizeCUDAKernel<T, DX_OP, DY_OP><<<
        block_per_grid, thread_per_block, 0, context.stream()>>>(
        x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
    cost = tt->stop();
  } else {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, num);
    int grid_size = n;

    tt->start();
    for(int i = 0; i < LOOPNUM; i ++)
      ElemwiseGradBroadcast2CUDAKernelShared<T, DX_OP, DY_OP>
          <<<grid_size, block_size, 0, context.stream()>>>(
          x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
    cost = tt->stop();
  }
  return cost;
}
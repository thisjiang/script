#include "../common.h"

#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cub/cub.cuh"

#include "stdio.h"
#include "time.h"

//#define CUDA_VERSION 10100

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

namespace platform {
template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl(val, src_line, width);
#else
  return __shfl_sync(mask, val, src_line, width);
#endif
}
}

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

inline static int GetDesiredBlockDim(int dim) {
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int64_t id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int64_t id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (v < value);
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (v > value);
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (v < in.v) || ((v == in.v) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (v > in.v) || ((v == in.v) && (id < in.id));
  }

  T v;
  int64_t id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size, const bool& largest) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (largest) {
      if (topk[k] < p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    } else {
      if (topk[k] > p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, int beam_size,
                                        const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T> tmp(src[idx], idx);
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T> tmp(src[idx], idx);
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, const Pair<T>& max,
                                        int beam_size, const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T> tmp(src[idx], idx);
        if (tmp < max) {
          AddTo<T>(topk, tmp, beam_size, largest);
        }
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T> tmp(src[idx], idx);
        if (tmp > max) {
          AddTo<T>(topk, tmp, beam_size, largest);
        }
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int* beam,
                                              int beam_size, const T* src,
                                              bool* firstStep, bool* is_empty,
                                              Pair<T>* max, int dim,
                                              const int tid, bool largest) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length, largest);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-static_cast<T>(INFINITY), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(topk + MaxLength - *beam, src, tid, dim, *max,
                              length, largest);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -static_cast<T>(1)) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T>* sh_topk, int* maxid,
                                            Pair<T> topk[], T** topVal,
                                            int64_t** topIds, int* beam, int* k,
                                            const int tid, const int warp,
                                            const bool& largest) {
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (largest) {
        if (sh_topk[tid] < sh_topk[tid + BlockSize / 2]) {
          maxid[tid] = tid + BlockSize / 2;
        } else {
          maxid[tid] = tid;
        }
      } else {
        if (sh_topk[tid] > sh_topk[tid + BlockSize / 2]) {
          maxid[tid] = tid + BlockSize / 2;
        } else {
          maxid[tid] = tid;
        }
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (largest) {
          if (sh_topk[maxid[tid]] < sh_topk[maxid[tid + stride]]) {
            maxid[tid] = maxid[tid + stride];
          }
        } else {
          if (sh_topk[maxid[tid]] > sh_topk[maxid[tid + stride]]) {
            maxid[tid] = maxid[tid + stride];
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = sh_topk[maxid[0]].v;
      **topIds = sh_topk[maxid[0]].id;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxid[0]) (*beam)++;
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == maxid[0]) {
      if (*beam < MaxLength) {
        sh_topk[tid] = topk[*beam];
      }
    }
    // NOTE(zcd): temporary solution
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (platform::CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) ==
          MaxLength)
        break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top MaxLength value;
 * 2. merge to sh_topk, block reduce and get max value;
 * 3. go to the second setp, until one thread's topk value is null;
 * 4. go to the first setp, until get the topk value.
 */

template <typename T, int MaxLength, int BlockSize>
__global__ void KeMatrixTopK(T* output, int output_stride, int64_t* indices,
                             const T* src, int lds, int dim, int k,
                             int grid_dim, int num, bool largest = true) {
  __shared__ Pair<T> sh_topk[BlockSize];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;

  const int bid = blockIdx.x;
  for (int i = bid; i < num; i += grid_dim) {
    int top_num = k;
    __shared__ int maxid[BlockSize / 2];
    T* out = output + i * output_stride;
    int64_t* inds = indices + i * k;
    Pair<T> topk[MaxLength]; //every thread get top MaxLength value;
    int beam = MaxLength;
    Pair<T> max;
    bool is_empty = false;
    bool firststep = true;

    for (int j = 0; j < MaxLength; j++) {
      if (largest) {
        topk[j].set(-static_cast<T>(INFINITY), -1);
      } else {
        topk[j].set(static_cast<T>(INFINITY), -1);
      }
    }
    while (top_num) {
      ThreadGetTopK<T, MaxLength, BlockSize>(topk, &beam, k, src + i * lds,
                                             &firststep, &is_empty, &max, dim,
                                             tid, largest);

      sh_topk[tid] = topk[0];
      BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &out, &inds,
                                           &beam, &top_num, tid, warp, largest);
    }
  }
}

float TestKeMatrixTopK(int num, int dim, int k,
                      CUDAStream &context, TimeOfKernel* sat) {
  MallocHost<float> src_h(num * dim, context);
  MallocDevice<float> src_d(num * dim, context);

  for(int i = 0; i < num * dim; i ++) {
      src_h.ptr()[i] = (rand() % 10000000)  * 0.0000001;
  }
  src_d.CopyFrom(src_h);

  MallocDevice<int64_t> indices(num * k, context);
  MallocDevice<float> out_d(num * k, context);

  const int kMaxHeight = 2048;
  int gridx = num < kMaxHeight ? num : kMaxHeight;
  float cost;

  switch (GetDesiredBlockDim(dim)) {
    FIXED_BLOCK_DIM(
        do {
          sat->start();
          KeMatrixTopK<float, 5,
                      kBlockDim><<<gridx, kBlockDim, 0, context.stream()>>>(
            out_d.ptr(), k, indices.ptr(), src_d.ptr(), num,
            num, static_cast<int>(k), gridx, dim);
          cost = sat->stop();
        } while(0);
        );
    default:
      fprintf(stderr, "not support size\n");
      return 0.0f;
  }

  #if 0
    MallocHost<float> out_h(num * k, context);
    out_h.CopyFrom(out_d);
    MallocHost<int64_t> ids_h(num * k, context);
    ids_h.CopyFrom(indices);
    context.sync();

    for(int i = 0; i < num; i ++) {
        for(int j = 0; j < k; j ++) {
            printf("%d %f ", ids_h.ptr()[i * k + j], out_h.ptr()[i * k + j]);
        }
        printf("\n");
    }
#endif

  return cost;
}

template <typename T>
__global__ void InitIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (int64_t j = row_id; j < num_rows; j += gridDim.x) {
    for (int64_t i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

struct SegmentOffsetIter {
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  __device__ __forceinline__ int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

float TestCubSort(int num, int dim, int k,
                  CUDAStream &context, TimeOfKernel* sat) {
  MallocHost<float> src_h(num * dim, context);
  MallocDevice<float> src_d(num * dim, context);

  for(int i = 0; i < num * dim; i ++) {
      src_h.ptr()[i] = (rand() % 10000000)  * 0.0000001;
  }
  src_d.CopyFrom(src_h);

  MallocDevice<int64_t> indices(num * dim, context);
  MallocDevice<int64_t> ids_dout(num * dim, context);

  MallocDevice<float> out_d(num * dim, context);

  // create iter for counting input
  cub::CountingInputIterator<int64_t> counting_iter(0);
  // segment_offset is used for move to next row    
  cub::TransformInputIterator<int64_t, SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num));

  sat->start();
  InitIndex<int64_t><<<num, 1024, 0, context.stream()>>>(
      indices.ptr(), dim, num);

  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr, temp_storage_bytes, src_d.ptr(), out_d.ptr(),
      indices.ptr(), ids_dout.ptr(), num * dim,
      dim, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(float) * 8,
      context.stream());
  MallocDevice<uint8_t> d_temp_storage(temp_storage_bytes, context);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage.ptr(), temp_storage_bytes, src_d.ptr(), out_d.ptr(),
      indices.ptr(), ids_dout.ptr(), num * dim,
      dim, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(float) * 8,
      context.stream());
  float cost = sat->stop();

#if 0
    MallocHost<float> out_h(num * k, context);
    out_h.CopyFrom(out_d);
    MallocHost<int64_t> ids_h(num * k, context]);
    ids_h.CopyFrom(ids_dout);
    context.sync();

    for(int i = 0; i < num; i ++) {
        for(int j = 0; j < k; j ++) {
            printf("%d %f ", ids_h.ptr()[i * k + j], out_h.ptr()[i * k + j]);
        }
        printf("\n");
    }
#endif

  return cost;
}

int main() {
    int num = 20;
    int dim = 242991;
    int k = 1;

    srand(time(NULL));
    CUDAStream context;
    TimeOfKernel* sat = TimeOfKernel::get(context);

    for(num = 20; num <= 242991; num <<= 2) {
      for(dim = 16; dim <= 242991; dim <<= 2) {
        float cost_topk = TestKeMatrixTopK(num, dim, k, context, sat);
        float cost_cub = TestCubSort(num, dim, k, context, sat);

        printf("(%d, %d, %d): KeMatrixTopK vs CubSort: %fms vs %fms\n", num, dim, k, cost_topk, cost_cub);

        if(cost_cub < cost_topk) {
          auto err = context.sync();
          if(err) printf("ERROR: %s\n", err);
          else printf("at (%d, %d), topk slower\n", num, dim);
          return 1;
        }
      }
    }

    return 0;
}
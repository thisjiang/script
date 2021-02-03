#include "../common.h"

#include "stdio.h"
#include "time.h"

#include <functional>
#include <vector>
#include <thrust/scan.h>
#include <cub/cub.cuh>

//#include "paddle/fluid/memory/malloc.h"

template <typename T>
struct CheckTrue {
  __host__ __device__ bool operator()(const T &val) {
    return static_cast<bool>(val);
  }
};
/*
template <typename T, int BLOCKDIM>
__global__ void KeGetTrueIndex(const T *cond_data, int64_t numel,
                               int64_t *true_index, int64_t *true_num) {
  const int tid = threadIdx.x;
  __shared__ int64_t s_num[BLOCKDIM + 2];
  s_num[BLOCKDIM + 1] = 0;

  if(tid < BLOCKDIM) {
    for(int64_t i = tid; i < numel; i += BLOCKDIM) {
      if(CheckTrue<T>()(cond_data[i])) s_num[tid] = i;
      else s_num[tid] = -1;
      __syncthreads();

      //TODO(jiangcheng):how about using warp match function?
      if(tid == 0) {
        int num = 0;
        int over = min(BLOCKDIM, static_cast<int>(numel - i));
        for(int ii = 0; ii < over; ii ++) {
          int64_t t_num = s_num[ii];
          if(t_num != -1) {
            s_num[num] = t_num;
            num ++;
          }
        }
        s_num[BLOCKDIM] = num;
        s_num[BLOCKDIM + 1] += num;
        *true_num = s_num[BLOCKDIM + 1];
      }
      __syncthreads();

      int tid_true_num = s_num[BLOCKDIM];
      int beg_index = s_num[BLOCKDIM + 1] - tid_true_num;
      if(tid < tid_true_num) {
        true_index[beg_index + tid] = s_num[tid];
      }
    }
  }
}
*/
/*
template <typename T, int BLOCKDIM>
__global__ void KeGetTrueIndex(int64_t *out_ptr, const T *cond_data, 
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank, int64_t *true_num) {
  const int tid = threadIdx.x;
  __shared__ int64_t s_num[BLOCKDIM + 2];
  s_num[tid + 1] = s_num[BLOCKDIM + 1] = 0;

  extern __shared__ int64_t s_stride[];
  if(tid < rank) s_stride[tid] = ptr_stride[tid];

  for(int64_t idx = tid; tid < BLOCKDIM && idx < numel; idx += BLOCKDIM) {
    bool is_true = CheckTrue<T>()(cond_data[idx]);
    s_num[tid + 1] = is_true ? 1 : 0;
    __syncthreads();

    if (tid == 0) {
      pre_sum = s_num[0] = s_num[BLOCKDIM + 1];
      for (int i = 1; i <= BLOCKDIM; i++)
        pre_sum = s_num[i] = pre_sum + s_num[i];

      s_num[BLOCKDIM + 1] = pre_sum;
      if(idx + BLOCKDIM >= numel) 
        *true_num = pre_sum;
    }
    __syncthreads();

    if (is_true) {
      int64_t place = s_num[tid];
      int index = idx;
      for (int j = 0; j < rank; j++) {
        int64_t stride_val = s_stride[j];
        int64_t out_val = index / stride_val;

        out_ptr[place * rank + j] = out_val;
        index -= out_val * stride_val;
      }
    }
    s_num[tid + 1] = 0;
  }
}
*/
/*
template <typename T, int BLOCKDIM>
__global__ void KeGetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank, int64_t *true_num) {
  const int tid = threadIdx.x;
  __shared__ int64_t s_num[BLOCKDIM + 1];
  s_num[tid + 1] = 0;
  const int64_t cond_block = (numel + BLOCKDIM - 1) / BLOCKDIM;
  const int64_t t_beg = tid * cond_block;
  const int64_t t_over = min(numel, t_beg + cond_block);

  if (tid < BLOCKDIM && t_beg < numel) {
    // first: each thread counting true condition number
    int64_t t_num = 0;
    for (int64_t i = t_beg; i < t_over; i++) {
      if (CheckTrue<T>()(cond_data[i])) t_num++;
    }
    s_num[tid + 1] = t_num;
    __syncthreads();

    // second: thread 0 counting all true condition number
    if (tid == 0) {
      int64_t pre_sum = s_num[0] = 0;
      for (int i = 1; i <= BLOCKDIM; i++) {
        pre_sum = s_num[i] = s_num[i] + pre_sum;
      }
      *true_num = pre_sum;
    }
    __syncthreads();

    // third: each thread set true index
    int64_t idx = s_num[tid];
    for (int64_t i = t_beg; i < t_over; i++) {
      if (CheckTrue<T>()(cond_data[i])) {
        int64_t index = i;
        for (int j = 0; j < rank; j++) {
          int64_t stride_val = ptr_stride[j];
          int64_t out_val = index / stride_val;
          
          out_ptr[idx * rank + j] = out_val;
          index -= out_val * stride_val;
        }
        idx++;
      }
    }
  }
}
*/
/*
template<typename T>
__device__ T ScanPrefixSum(T *data, const int num) {
  const int tid = threadIdx.x;
  for (int stride = 1; stride <= num; stride <<= 1) {
    __syncthreads();
    T tmp = 0;
    if(stride <= tid) tmp = data[tid - stride];
    __syncthreads();
    data[tid] += tmp;
  }
  __syncthreads();
  return data[num - 1];
}
template <typename T, int BLOCKDIM>
__global__ void KeGetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank, int64_t *true_num) {
  const int tid = threadIdx.x;
  __shared__ int64_t s_num[BLOCKDIM + 1];
  s_num[tid] = s_num[BLOCKDIM] = 0;
  const int64_t cond_block = (numel + BLOCKDIM - 1) / BLOCKDIM;
  const int64_t t_beg = tid * cond_block;
  const int64_t t_over = min(numel, t_beg + cond_block);
  const int act_num = (numel + cond_block - 1) / cond_block;

  if (tid < BLOCKDIM && t_beg < numel) {
    // first: each thread counting true condition number
    int64_t t_num = 0;
    for (int64_t i = t_beg; i < t_over; i++) {
      if (CheckTrue<T>()(cond_data[i])) t_num++;
    }
    s_num[tid + 1] = t_num;
    __syncthreads();

    // second: thread 0 counting all true condition number
    *true_num = ScanPrefixSum(s_num, act_num) + s_num[act_num];

    // third: each thread set true index
    int64_t idx = s_num[tid];
    for (int64_t i = t_beg; i < t_over; i++) {
      if (CheckTrue<T>()(cond_data[i])) {
        int64_t index = i;
        for (int j = 0; j < rank; j++) {
          int64_t stride_val = ptr_stride[j];
          int64_t out_val = index / stride_val;
          
          out_ptr[idx * rank + j] = out_val;
          index -= out_val * stride_val;
        }
        idx++;
      }
    }
  }
}
*/
/*
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
      true_num_array[idx / vec_size + 1] = ture_num;
    }
  }
}

template <typename T>
__global__ void KeGetIndexPlace(int64_t *true_num_array, 
                              const int64_t array_size) {
  if(blockIdx.x == 0 && threadIdx.x == 0) {
    true_num_array[0] = 0;
    for(int64_t i = 1; i <= array_size; i ++) {
      true_num_array[i] += true_num_array[i - 1];
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
      int64_t index = true_num_array[idx / vec_size];
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
*/
/*
template <typename T>
__global__ void KeGetTrueNum(const T *cond_data, const int64_t numel,
                             int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    true_num_array[idx] = CheckTrue<T>()(cond_data[idx]) ? 1 : 0;
  }
}
template <typename T, int BLOCKDIM>
__global__ void KeSetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank,
                               const int64_t *true_num_array,
                               const int64_t *block_reduce_sum) {
  const int tid = threadIdx.x;
  constexpr int tile_size = BLOCKDIM * 2;
  const int end_bid = (numel + tile_size - 1) / tile_size;

  for(int bid = blockIdx.x; bid < end_bid; bid += gridDim.x) {
    const int64_t tile_beg = bid * tile_size;
    const int64_t index = 2 * tid + tile_beg;

    const int64_t prefix_sum = block_reduce_sum[bid];
    for(int64_t idx = index; idx < index + 2 && idx < numel; idx ++) {
      if (CheckTrue<T>()(cond_data[idx])) {
        int64_t rank_index = idx;
        const int64_t true_index = true_num_array[idx] + prefix_sum;
        for (int j = 0; j < rank; j++) {
          const int64_t out_index = rank_index / ptr_stride[j];
          out_ptr[true_index * rank + j] = out_index;
          rank_index -= out_index * ptr_stride[j];
        }
      }
    }
  }
}

template<typename T, int BLOCKDIM>
__device__ T BlockPrefixSum(T *data, const int64_t index, 
                            const int64_t i_end) {
  const int tid = threadIdx.x;
  constexpr int tile_size = BLOCKDIM * 2;
  __shared__ T s_data[tile_size];
  s_data[2 * tid] = 0;
  s_data[2 * tid + 1] = 0;

  if(index < i_end) s_data[2 * tid] = data[index];
  if(index + 1 < i_end) s_data[2 * tid + 1] = data[index + 1];

  int offset = 1;
  T reduce_sum = 0;
  for(int i = tile_size>>1; i > 0; i >>= 1) {
    __syncthreads();
    if(tid < i) {
      int x = offset * (2 * tid + 1) - 1;
      int y = offset * (2 * tid + 2) - 1;

      s_data[y] += s_data[x];
    }
    offset <<= 1;
  }

  if(tid == 0) {
    reduce_sum = s_data[tile_size - 1];
    s_data[tile_size - 1] = 0;
  }

  for(int i = 1; i < tile_size; i <<= 1) {
    offset >>= 1;
    __syncthreads();

    if(tid < i) {
      int x = offset * (2 * tid + 1) - 1;
      int y = offset * (2 * tid + 2) - 1;

      T tmp = s_data[x];
      s_data[x] = s_data[y];
      s_data[y] += tmp;
    }
  }
  __syncthreads();
  if(index < i_end) data[index] = s_data[2 * tid];
  if(index + 1 < i_end) data[index + 1] = s_data[2 * tid + 1];

  return reduce_sum;
}
template<typename T, int BLOCKDIM>
__global__ void KeScanPrefixSum(T *data, const int64_t numel, T *block_reduce_sum) {
  const int tid = threadIdx.x;
  constexpr int tile_size = BLOCKDIM * 2;
  const int end_bid = (numel + tile_size - 1) / tile_size;

  for(int bid = blockIdx.x; bid < end_bid; bid += gridDim.x) {
    const int64_t tile_beg = bid * tile_size;
    const int64_t tile_end = min(tile_beg + tile_size, numel);
    const int64_t index = 2 * tid + tile_beg;

    T b_sum = BlockPrefixSum<T, BLOCKDIM>(data, index, tile_end);
    if(tid == 0) block_reduce_sum[bid] = b_sum;
  }
}
template<typename T, int BLOCKDIM>
__global__ void KeScanPrefixSum_OneBlock(T *data, const int64_t numel, T *reduce_sum) {
  if(blockIdx.x == 0) {
    const int tid = threadIdx.x;
    constexpr int tile_size = BLOCKDIM * 2;
    const int end_bid = (numel + tile_size - 1) / tile_size;
    __shared__ T base_sum;
    base_sum = 0;
    __syncthreads();

    for(int bid = 0; bid < end_bid; bid ++) {
      const int64_t tile_beg = bid * tile_size;
      const int64_t tile_end = min(tile_beg + tile_size, numel);
      const int64_t index = 2 * tid + tile_beg;

      T b_sum = BlockPrefixSum<T, BLOCKDIM>(data, index, tile_end);

      if(index < tile_end) data[index] += base_sum;
      if(index + 1 < tile_end) data[index + 1] += base_sum;
      __syncthreads();
      if(tid == 0) base_sum += b_sum;
      __syncthreads();
    }
    reduce_sum[0] = base_sum;
  }
}
template<typename T, int BLOCKDIM>
inline void KeWhereIndex(int64_t *out_ptr, int64_t *d_true_num,
                         const T *cond_data, const int64_t numel,
                         const int64_t *h_stride, const int64_t rank,
                         CUDAStream &ss) {
  constexpr int tile_size = BLOCKDIM * 2;
  const int64_t need_blocks = (numel + tile_size - 1) / tile_size;
  const int64_t grids = std::min(
      static_cast<int64_t>(ss.GetCUDAMaxGridDimSize().x), need_blocks);
  MallocDevice<int64_t> d_stride(rank, ss);
  d_stride.CopyFrom(h_stride, rank * sizeof(int64_t), Place::HOST);

  MallocDevice<int64_t> d_true_num_array(numel, ss);
  MallocDevice<int64_t> d_block_reduce_sum(need_blocks, ss);

  int64_t *ptr_stride = d_stride.data();
  int64_t *ptr_true_num_array = d_true_num_array.data();
  int64_t *ptr_block_reduce_sum = d_block_reduce_sum.data();

  KeGetTrueNum<T><<<grids, BLOCKDIM, 0, ss.stream()>>>
            (cond_data, numel, ptr_true_num_array);
  KeScanPrefixSum<int64_t, BLOCKDIM><<<grids, BLOCKDIM, 0, ss.stream()>>>
            (ptr_true_num_array, numel, ptr_block_reduce_sum);
  KeScanPrefixSum_OneBlock<int64_t, BLOCKDIM>
            <<<1, BLOCKDIM, 0, ss.stream()>>>
            (ptr_block_reduce_sum, need_blocks, d_true_num);
  KeSetTrueIndex<T, BLOCKDIM><<<grids, BLOCKDIM, 0, ss.stream()>>>
    (out_ptr, cond_data, numel, ptr_stride, 
     rank, ptr_true_num_array, ptr_block_reduce_sum);
}
*/
template <typename T>
__global__ void KeGetTrueNum(const T *cond_data, const int64_t numel,
                             int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    true_num_array[idx] = CheckTrue<T>()(cond_data[idx]) ? 1 : 0;
  }
}
template <typename T>
__global__ void KeSetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank,
                               const int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    if (CheckTrue<T>()(cond_data[idx])) {
        int64_t rank_index = idx;
        const int64_t true_index = true_num_array[idx] - 1;
        for (int j = 0; j < rank; j++) {
          const int64_t out_index = rank_index / ptr_stride[j];
          out_ptr[true_index * rank + j] = out_index;
          rank_index -= out_index * ptr_stride[j];
        }
      }
  }
}

template<typename T, int BLOCKDIM>
inline void KeWhereIndex(int64_t *out_ptr, int64_t *d_true_num,
                         const T *cond_data, const int64_t numel,
                         const int64_t *h_stride, const int64_t rank,
                         CUDAStream &ss) {
  const int threads = std::min(numel, static_cast<int64_t>(1024));
  const int64_t need_blocks = (numel + threads - 1) / threads;
  const int64_t grids = std::min(
      static_cast<int64_t>(ss.GetCUDAMaxGridDimSize().x), need_blocks);
  MallocDevice<int64_t> d_stride(rank, ss);
  d_stride.CopyFrom(h_stride, rank * sizeof(int64_t), Place::HOST);

  MallocDevice<int64_t> d_true_num_array(numel, ss);

  int64_t *ptr_stride = d_stride.data();
  int64_t *ptr_true_num_array = d_true_num_array.data();

  size_t tmp_size = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp_size,
                                ptr_true_num_array, ptr_true_num_array,
                                numel, ss.stream());
  MallocDevice<int64_t> d_tmp_mem(tmp_size, ss);
  int64_t *ptr_mem = d_tmp_mem.data();

  KeGetTrueNum<T><<<grids, threads, 0, ss.stream()>>>
            (cond_data, numel, ptr_true_num_array);
  cub::DeviceScan::InclusiveSum(ptr_mem, tmp_size,
                                ptr_true_num_array, ptr_true_num_array,
                                numel, ss.stream());
  cudaMemcpyAsync(d_true_num, ptr_true_num_array + numel - 1, 
                  sizeof(int64_t), cudaMemcpyDeviceToDevice, ss.stream());
  KeSetTrueIndex<T><<<grids, threads, 0, ss.stream()>>>
            (out_ptr, cond_data, numel, ptr_stride, rank, ptr_true_num_array);
}


/*
template <typename T>
class CUDAWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* out = context.Output<framework::Tensor>("Out");
    auto& dev_ctx = context.template device_context<CUDADeviceContext>();

    // TODO(zhoukunsheng): Should optimize to ensure GPU is faster than CPU.
    framework::Tensor cond_cpu;
    framework::TensorCopy(*condition, platform::CPUPlace(), &cond_cpu);

    const T* cond_data = cond_cpu.data<T>();
    int64_t numel = cond_cpu.numel();
    auto dims = cond_cpu.dims();
    int rank = dims.size();

    int64_t tmp_mem_size = numel + rank;
    auto d_tmp_mem = memory::Alloc(dev_ctx, tmp_mem_size * sizeof(int64_t));
    auto h_tmp_mem = memory::Alloc(platform::CPUPlace(), 
                                    tmp_mem_size * sizeof(int64_t));

    int64_t *ptr_true_index = reinterpret_cast<int64_t *>(d_tmp_mem->ptr());
    int64_t *ptr_stride = ptr_true_index + numel;

    int64_t *h_true_index = reinterpret_cast<int64_t *>(h_tmp_mem->ptr());
    int64_t *h_stride = h_true_index + numel;

    size_t true_num = 0;
    for (int64_t i = 0; i < numel; i++) {
      if (static_cast<bool>(cond_data[i])) {
        h_true_index[true_num] = i;
        true_num ++;
      }
    }

    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_ptr = out->mutable_data<int64_t>(context.GetPlace());

    if (true_num == 0) {
      return;
    }

    h_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      h_stride[i] = h_stride[i + 1] * dims[i + 1];
    }
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 ptr_true_index, platform::CPUPlace(),
                 h_true_index, tmp_mem_size * sizeof(int64_t),
                 dev_ctx.stream());

    WhereIndexFunctor<int64_t> functor(ptr_true_index, true_num, ptr_stride,
                                       rank, out_ptr);
    platform::ForRange<CUDADeviceContext> for_range(dev_ctx, true_num);
    for_range(functor);
  }
};
*/
/*
template<typename T>
int test_where_index(const std::vector<int64_t> &dims, CUDAStream &ss) {
  const int64_t rank = dims.size();
  int64_t numel = 1;
  for(auto val : dims) numel *= val;

  MallocHost<int64_t> stride_h(rank, ss);
  MallocDevice<int64_t> stride_d(rank, ss);
  int64_t *stride = stride_h.data();
  stride[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * dims[i + 1];
  }
  stride_d.CopyFrom(stride_h);

  MallocHost<T> cond_h(numel, ss);
  MallocDevice<T> cond_d(numel, ss);

  for(int i = 0; i < numel; i ++){
      cond_h.ptr()[i] = static_cast<T>(rand() % 2);
  }
  cond_d.CopyFrom(cond_h);
  T *cond_data = cond_d.data();

  MallocDevice<int64_t> out_d(numel * rank, ss);
  MallocDevice<int64_t> true_num_d(1, ss);

  int64_t *out_ptr = out_d.data();
  int64_t *ptr_true_num = true_num_d.data();

#define SetKenelParam(num)                                      \
  KeGetTrueIndex<T, num>                                        \
      <<<1, num, 0, ss.stream()>>>(        \
      out_ptr, cond_data, numel, stride, rank, ptr_true_num);

    if (numel > 1024) {
      SetKenelParam(1024)
    } else if (numel > 512) {
      SetKenelParam(512)
    } else if (numel > 256) {
      SetKenelParam(256)
    } else if (numel > 128) {
      SetKenelParam(128)
    } else if (numel > 64) {
      SetKenelParam(64)
    } else if (numel > 32) {
      SetKenelParam(32)
    } else {
      SetKenelParam(1)
    }
#undef SetKenelParam

  MallocHost<int64_t> out_h(numel * rank, ss);
  MallocHost<int64_t> true_num_h(1, ss);
  out_h.CopyFrom(out_d);
  true_num_h.CopyFrom(true_num_d);
  const char *err = ss.sync();
  if(err != "") {
      printf("Error: %s\n", err);
      return 1;
  }

  int64_t true_num = *true_num_h.data();  
  bool success = true;
  for(int i = 0; success && i < true_num; i ++) {
    int idx = 0;
    for(int j = 0; success && j < rank; j ++) {
      idx += stride[j] * out_h.data()[i * rank + j];
    }
    if(!CheckTrue<T>()(cond_h.data()[idx])) {
      success = false;
      fprintf(stderr, "ERROR: %d at %d\n", idx, i);
    }
  }
  if(!success) {
    fprintf(stderr, "Compute ERROR\n");
    if(numel <= 200) {
      fprintf(stderr, "Origin data:\n");
      cond_h.print(1, numel);
      fprintf(stderr, "total index:\n");
      if(rank == 1) out_h.print(1, true_num);
      else out_h.print(true_num, rank);
    }
    return 1;
  } else {
    printf("Success! True %d of %d\n", true_num, numel);
  }
  return 0;
}
*/
/*
template<typename T>
int test_where_index(const std::vector<int64_t> &dims, CUDAStream &ss) {
  const int64_t rank = dims.size();
  int64_t numel = 1;
  for(auto val : dims) numel *= val;

  MallocHost<int64_t> stride_h(rank, ss);
  MallocDevice<int64_t> stride_d(rank, ss);
  int64_t *stride = stride_h.data();
  stride[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * dims[i + 1];
  }
  stride_d.CopyFrom(stride_h);

  MallocHost<T> cond_h(numel, ss);
  MallocDevice<T> cond_d(numel, ss);

  for(int i = 0; i < numel; i ++){
      cond_h.ptr()[i] = static_cast<T>(rand() % 2);
  }
  cond_d.CopyFrom(cond_h);
  T *cond_data = cond_d.data();

  int vec_size = 4;
  int num_of_vec = (numel + vec_size - 1) / vec_size;

  MallocDevice<int64_t> true_index_d(numel, ss);
  MallocDevice<int64_t> true_num_d(num_of_vec + 1, ss);

  int64_t *true_index = true_index_d.data();
  int64_t *true_num_array = true_num_d.data();

#define SetKenelParam(num)                                      \
    int grids = (num_of_vec + num - 1) / num;                   \
    KeGetTrueNum<T, num><<<grids, num, 0, ss.stream()>>>        \
                  (cond_data, numel, true_num_array, vec_size); \
    KeGetIndexPlace<T, num><<<1, 1, 0, ss.stream()>>>           \
                  (true_num_array, num_of_vec);                 \
    KeSetTrueIndex<T, num><<<grids, num, 0, ss.stream()>>>      \
        (cond_data, numel, true_num_array, vec_size, true_index);

    if (numel > 1024) {
      SetKenelParam(1024)
    } else if (numel > 256) {
      SetKenelParam(256)
    } else if (numel > 64) {
      SetKenelParam(64)
    } else {
      SetKenelParam(32)
    }
#undef SetKenelParam

  MallocHost<int64_t> true_index_h(numel, ss);
  MallocHost<int64_t> true_num_h(num_of_vec + 1, ss);

  true_index_h.CopyFrom(true_index_d);
  true_num_h.CopyFrom(true_num_d);
  const char *err = ss.sync();
  if(err != "") {
      printf("Error: %s\n", err);
      return 1;
  }

  int64_t true_num = true_num_h.data()[num_of_vec];  
  bool success = true;
  for(int i = 0; i < true_num; i ++) {
    int64_t idx = true_index_h.data()[i];
    if((i == 0 || idx != 0) &&!CheckTrue<T>()(cond_h.data()[idx])) {
      fprintf(stderr, "ERROR:cond_data[%d] is flase\n", idx);
      success = false;
      break;
    }
  }
  for(int i = true_num; success && i < numel; i ++) {
    int64_t idx = true_index_h.data()[i];
    if(idx != 0) {
      fprintf(stderr, "ERROR:true_index_h[%d] not zero\n", idx);
      success = false;
      break;
    }
  }
  if(!success) {
    fprintf(stderr, "Compute ERROR\n");
    fprintf(stderr, "Origin data:\n");
    cond_h.print(1, numel);
    fprintf(stderr, "True number:\n");
    true_num_h.print(true_num, rank);
    fprintf(stderr, "True index:\n");
    true_index_h.print(true_num, rank);
    return 1;
  } else {
    printf("Success! True %d of %d\n", true_num, numel);
  }
  return 0;
}
*/

template<typename T>
int test_where_index(const std::vector<int64_t> &dims, CUDAStream &ss) {
  const int64_t rank = dims.size();
  int64_t numel = 1;
  for(auto val : dims) numel *= val;

  MallocHost<int64_t> stride_h(rank, ss);
  int64_t *stride = stride_h.data();
  stride[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * dims[i + 1];
  }

  MallocHost<T> cond_h(numel, ss);
  MallocDevice<T> cond_d(numel, ss);

  for(int i = 0; i < numel; i ++){
      cond_h.ptr()[i] = static_cast<T>(rand() % 2);
  }
  cond_d.CopyFrom(cond_h);

  MallocDevice<int64_t> out_d(numel * rank, ss);
  MallocDevice<int64_t> true_num_d(1, ss);

  T *cond_data = cond_d.data();
  int64_t *out_ptr = out_d.data();
  int64_t *ptr_true_num = true_num_d.data();
  
#define SelectBlockSize(BlockSize)                        \
  KeWhereIndex<T, BlockSize>(out_ptr, ptr_true_num,       \
                cond_data, numel, stride, rank, ss);

  if(numel / 1024 > 1) {
    SelectBlockSize(1024)
  } else if(numel / 256 > 1) {
    SelectBlockSize(256)
  } else {
    SelectBlockSize(32)
  }
#undef SelectBlockSize

  MallocHost<int64_t> out_h(numel * rank, ss);
  MallocHost<int64_t> true_num_h(1, ss);

  out_h.CopyFrom(out_d);
  true_num_h.CopyFrom(true_num_d);
  const char *err = ss.sync();
  if(err != "") {
      printf("Error: %s\n", err);
      return 1;
  }

  int64_t true_num = true_num_h.data()[0];
  bool success = true;
  for(int i = 0; success && i < true_num; i ++) {
    int idx = 0;
    for(int j = 0; success && j < rank; j ++) {
      idx += stride[j] * out_h.data()[i * rank + j];
    }
    if(!CheckTrue<T>()(cond_h.data()[idx])) {
      success = false;
      fprintf(stderr, "ERROR: %d at %d\n", idx, i);
    }
  }
  if(!success) {
    fprintf(stderr, 
            "Compute ERROR with numel %d true_num %d\n",
            numel, true_num);
    if(numel <= 200) {
      fprintf(stderr, "Origin data:\n");
      int row = 1, col = numel;
      if(dims.size() > 1) {
        row = dims[0];
        col = 1;
        for(int dim_i = 1; dim_i < dims.size(); dim_i ++)
          col *= dims[dim_i];
      }
      cond_h.print(row, col);

      fprintf(stderr, "output:\n");
      if(rank == 1) out_h.print(1, true_num);
      else out_h.print(true_num, rank);
    }
    return 1;
  } else {
    printf("Success! True %d of %d\n", true_num, numel);
  }
  return 0;
}

int main() {
    srand(time(NULL));
    CUDAStream ss;

    std::vector<int64_t> dims = {100, 100, 100};
    do {
      if(test_where_index<int>(dims, ss)) break;
      if(test_where_index<bool>(dims, ss)) break;
      if(test_where_index<int64_t>(dims, ss)) break;
      if(test_where_index<float>(dims, ss)) break;
    } while(true);
    
    return 0;
}

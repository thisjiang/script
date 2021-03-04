// class header file

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


constexpr int LOOPNUM = 10;
typedef std::vector<int> DDim;

#define CUDA_KERNEL_LOOP(i, num)                             \
  int64_t __index__ = blockIdx.x * blockDim.x + threadIdx.x; \
  for (int i = __index__; __index__ < (num);                 \
       __index__ += blockDim.x * gridDim.x, i = __index__)

template <typename T>
__global__ void MultiplyCUDAKernel(T* C, const T* A, const T* B, int N) {
  CUDA_KERNEL_LOOP(i, N) {
    C[i] = static_cast<T>(static_cast<float>(A[i]) * static_cast<float>(B[i]));
  }
}

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T, int VLEN>
union vec_t {
  static_assert(sizeof(T) == -1, "vec_t is only available by specialization.");
};

template <>
union vec_t<float, 4> {
  float4 s;
  float v[4];
};

template <>
union vec_t<float16, 4> {
  int2 s;
  float16 v[4];
};

template <typename T, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxBackward(T* dst, const T* grad, const T* src,
                                   const int batch_size,
                                   const int softmax_ele) {
  const int offset =
      blockIdx.x * softmax_ele * WARP_PER_BLOCK + threadIdx.x * VPT;

  float local_sum_gy = 0.f;
  vec_t<T, VPT> local_grad;
  vec_t<T, VPT> local_src;

  local_grad.s =
      reinterpret_cast<const decltype(local_grad.s)*>(&grad[offset])[0];
  local_src.s = reinterpret_cast<const decltype(local_src.s)*>(&src[offset])[0];

  for (int i = 0; i < VPT; ++i) {
    local_sum_gy += static_cast<float>(local_grad.v[i]) *
                    static_cast<float>(local_src.v[i]);
  }
  float sum_gy = warpReduceSum<float>(local_sum_gy, 0xffffffff);

  vec_t<T, VPT> local_dst;
  for (int i = 0; i < VPT; ++i) {
    local_dst.v[i] =
        static_cast<T>(static_cast<float>(local_src.v[i]) *
                       (static_cast<float>(local_grad.v[i]) - sum_gy));
  }
  reinterpret_cast<decltype(local_dst.s)*>(&dst[offset])[0] = local_dst.s;
}

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_sum(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T sum_val = __shfl_xor_sync(0xFFFFFFFF, sum[i], offset, WARP_SIZE);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_max(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T max_val = __shfl_xor_sync(0xFFFFFFFF, sum[i], offset, WARP_SIZE);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, typename AccT, int Log2Elements>
__global__ void softmax_warp_backward(T* gradInput, const T* grad,
                                      const T* output, int batch_size,
                                      int stride, int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x % warp_size_softmax;

  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // load data from global memory
  AccT grad_reg[WARP_BATCH][WARP_ITERATIONS];
  AccT output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        grad_reg[i][it] =
            static_cast<AccT>(grad[i * element_count + it * warp_size_softmax]);
        output_reg[i][it] = static_cast<AccT>(
            output[i * element_count + it * warp_size_softmax]);
      } else {
        grad_reg[i][it] = AccT(0);
        output_reg[i][it] = AccT(0);
      }
    }
  }

  AccT sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        // compute gradients
        gradInput[i * element_count + it * warp_size_softmax] =
            (grad_reg[i][it] - output_reg[i][it] * sum[i]);
      }
    }
  }
}

#define LAUNCH_SOFTMAX_WARP_BACKWARD(Log2Elements)                 \
  case Log2Elements:                                               \
    softmax_warp_backward<T, float, Log2Elements><<<               \
        blocks, threads, 0, context.stream()>>>( \
        dx_data, mul_grad.data<T>(), out->data<T>(), N, dim, dim); \
    break;

/************************************************************************/
template<typename T>
float TimeOfCudnnSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  CUDNNHandle handle(context.stream());

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  // the format is not used now, will add later
  std::vector<int> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) {
    strides[i] = dims[i + 1] * strides[i + 1];
  }

  std::vector<int> tensor_dims = {N, dim, D, 1};
  // cudnnTensorFormat_t layout = CUDNN_TENSOR_NCHW;
  cudnnDataType_t type = GetCudnnDataType<T>();

  cudnnTensorDescriptor_t desc_;
  cudnnCreateTensorDescriptor(&desc_);
  cudnnSetTensorNdDescriptor(desc_, type, rank, dims.data(), strides.data());

  auto cudnn_hd = handle.cudnn_handle();
  auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                : CUDNN_SOFTMAX_MODE_CHANNEL;
  const T one = static_cast<T>(1), zero = static_cast<T>(0);
  cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    status = cudnnSoftmaxBackward(
        cudnn_hd, CUDNN_SOFTMAX_ACCURATE, mode,
        &one, desc_, out_data, desc_, dout_data,
        &zero, desc_, dx_data);
  }
  float cost = clock->stop();

  cudnnDestroyTensorDescriptor(desc_);
  auto err = handle.CheckError(status);
  if(err != EMPTY_STRING) {
    fprintf(stderr, "%s Cudnn ERROR: %s\n",
            ToString(dims).c_str(), err);
    return 0.0f;
  }

  return cost;
}

/************************************************************************/
// When D is larg and dim is small:
// Each block arranged by N，each thread arranged by D
// each thread compute dim number's softmax
template<typename T, typename AccT>
__global__ void KeLoopDimSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim, const int D) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_id = tid / D;
  if(out_id >= N) return;
  const int in_id = tid - out_id * D;

  const T *src_out = out + out_id * dim * D + in_id;
  const T *src_dout = dout + out_id * dim * D + in_id;
  T *dst_dx = dx + out_id * dim * D + in_id;

  // compute multiply then sum value
  AccT sum_val(0);
  for(int dim_id = 0; dim_id < dim; dim_id ++) {
    sum_val += static_cast<AccT>(src_out[dim_id * D]) *
               static_cast<AccT>(src_dout[dim_id * D]);
  }

  // compute softmax gradient value
  for(int dim_id = 0; dim_id < dim; dim_id ++)
    dst_dx[dim_id * D] =
        static_cast<T>(static_cast<AccT>(src_out[dim_id * D]) *
        (static_cast<AccT>(src_dout[dim_id * D]) - sum_val));
}

template<typename T>
float TimeOfLoopDimSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);
  CUDNNHandle handle(context.stream());

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  int threads = std::min(N * D, 256);
  int grids = (N * D + threads - 1) / threads;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeLoopDimSoftmaxBackward<T, float><<<grids, threads, 0, context.stream()>>>(
      dx_data, out_data, dout_data, N, dim, D);
  }

  float cost = clock->stop();

  return cost;
}

/************************************************************************/

// When D is small and (dim * D) is larger
// Each block arranged by N，each thread arranged by dim * D
// each block compute (dim * D) number's softmax
template<typename T, typename AccT>
__global__ void KeSpandDimDSoftmaxBackward(T* __restrict__ dx,
                const T* __restrict__ out, const T* __restrict__ dout,
                const int N, const int dim, const int D) {
  extern __shared__ char s_mem[];
  AccT* s_data = reinterpret_cast<AccT*>(s_mem);

  const int tid = threadIdx.x;
  const int BlockDim = blockDim.x;
  for(int out_id = blockIdx.x; out_id < N; out_id += gridDim.x) {
    const T *src_out = out + out_id * dim * D;
    const T *src_dout = dout + out_id * dim * D;
    T *dst_dx = dx + out_id * dim * D;

    // Compute each thread's sum value
    AccT sum_val(0);
    for(int id = tid; id < dim * D; id += BlockDim)
      sum_val += static_cast<AccT>(src_out[id]) *
                 static_cast<AccT>(src_dout[id]);
    // write to shared memory
    s_data[tid] = sum_val;
    __syncthreads();
    // compute total sum value
    sum_val = 0;
    for(int id = tid % D; id < BlockDim; id += D)
      sum_val += s_data[id];

    // Compute finally softmax result
    for(int id = tid; id < dim * D; id += BlockDim)
      dst_dx[id] =
        static_cast<T>(static_cast<AccT>(src_out[id]) *
        (static_cast<AccT>(src_dout[id]) - sum_val));
  }
}

template<typename T>
float TimeOfSpandDimDSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  if(D > 1024) return 0.0f;

  const int grids = N;
  const int threads = D * (1024 / D);

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    KeSpandDimDSoftmaxBackward<T, float>
      <<<grids, threads, threads * sizeof(float), context.stream()>>>(
      dx_data, out_data, dout_data, N, dim, D);
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/
template<typename T>
float TimeOfNewSoftmax(CUDAStream &context, const DDim &dims, const int in_axis,
                        const T* out_data, T* dout_data, T *dx_data) {
  auto clock = TimeOfKernel::get(context);

  const int rank = dims.size();
  const int axis = CanonicalAxis(in_axis, rank);
  const int dim = dims[axis];
  const int N = SizeToAxis(axis, dims);
  const int D = SizeOutAxis(axis, dims);

  constexpr bool AccT_use_float =
        std::is_same<T, float>::value ||
        std::is_same<T, half>::value;
  bool optimize = false;

  clock->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
  if(D <= 1024 && D < dim) {
    if(i == 0) printf("[TimeOfNewSoftmax] Using KeSpandDimDSoftmaxBackward\n");
    optimize = true;
    const int grids = N;
    const int threads = D * (1024 / D);

    if(AccT_use_float) {
      KeSpandDimDSoftmaxBackward<T, float>
        <<<grids, threads, threads * sizeof(float),
          context.stream()>>>(
        dx_data, out_data, dout_data, N, dim, D);
    } else {
      KeSpandDimDSoftmaxBackward<T, double>
        <<<grids, threads, threads * sizeof(double),
          context.stream()>>>(
        dx_data, out_data, dout_data, N, dim, D);
    }
  } else if(dim < 1024) {
    if(i == 0) printf("[TimeOfNewSoftmax] Using KeLoopDimSoftmaxBackward\n");
    optimize = true;
    int threads = std::min(N * D, 256);
    int grids = (N * D + threads - 1) / threads;

    if(AccT_use_float) {
      KeLoopDimSoftmaxBackward<T, float><<<grids, threads, 0,
        context.stream()>>>(
        dx_data, out_data, dout_data, N, dim, D);
    } else {
      KeLoopDimSoftmaxBackward<T, double><<<grids, threads, 0,
        context.stream()>>>(
        dx_data, out_data, dout_data, N, dim, D);
    }
  }
  float cost = clock->stop();

  return cost;
}

/************************************************************************/


/************************************************************************/
template<typename T>
int TestSoftmax(CUDAStream &context, const DDim &dims, const int in_axis) {
  const size_t num = GetSize(dims);

  MallocHost<T> out_h(num, context);
  MallocDevice<T> out(num, context);

  MallocHost<T> dout_h(num, context);
  MallocDevice<T> dout(num, context);

  MallocDevice<T> dx_cudnn(num, context);
  MallocDevice<T> dx_inner(num, context);
  MallocDevice<T> dx_DimD(num, context);
  // MallocDevice<T> dx_new(num, context);

  out_h.Random(static_cast<T>(0), static_cast<T>(1));
  out.CopyFrom(out_h);
  dout_h.Random(static_cast<T>(-10), static_cast<T>(10));
  dout.CopyFrom(dout_h);

  T* out_data = out.data();
  T* dout_data = dout.data();

  std::vector<std::string> methods;
  std::vector<float> costs;

  float cost_cudnn = TimeOfCudnnSoftmax(context, dims, in_axis, out_data, dout_data, dx_cudnn.data());
  printf("cudnn cost %f\n", cost_cudnn);
  methods.push_back("Cudnn");
  costs.push_back(cost_cudnn);
  float cost_inner = TimeOfLoopDimSoftmax(context, dims, in_axis, out_data, dout_data, dx_inner.data());
  printf("LoopDim cost %f\n", cost_inner);
  methods.push_back("LoopDim");
  costs.push_back(cost_inner);
  float cost_DimD = TimeOfSpandDimDSoftmax(context, dims, in_axis, out_data, dout_data, dx_DimD.data());
  printf("SpandDimD cost %f\n", cost_DimD);
  methods.push_back("SpandDimD");
  costs.push_back(cost_DimD);
  // float cost_new = TimeOfNewSoftmax(context, dims, in_axis, out_data, dout_data, dx_new.data());
  // printf("New cost %f\n", cost_new);

  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "CUDA ERROR: %s\n", err);
    return CUDA_FAILED;
  }

  T inner_err = dx_cudnn.MaxError(dx_inner);
  std::cout<< "LoopDim Error: " << type2type<T, float>(inner_err) << std::endl;
  T DimD_err = dx_cudnn.MaxError(dx_DimD);
  std::cout<< "SpandDimD Error: " << type2type<T, float>(DimD_err) << std::endl;
  // T new_err = dx_cudnn.MaxError(dx_new);
  // std::cout<< "New Error: " << type2type<T, float>(new_err) << std::endl;

  int min_idex = 0;
  for(int i = 1; i < costs.size(); i ++) {
    if(costs[i] != 0 && costs[i] < costs[min_idex]) min_idex = i;
  }
  printf("%s %s faster, which cost %f ms\n",
          ToString(dims).c_str(), methods[min_idex].c_str(),
          costs[min_idex]);
  
  if(num < 500) {
    printf("Cudnn Result\n");
    dx_cudnn.Print(dims[0], dims[1], dims[2]);
    printf("LoopDim Result\n");
    dx_inner.Print(dims[0], dims[1], dims[2]);
    printf("SpandDim Result\n");
    dx_DimD.Print(dims[0], dims[1], dims[2]);
  }
  return SUCCESS;
}

/************************************************************************/

int main() {
  srand(time(0));
  CUDAStream context;
  typedef float T;
  do {
    DDim dims = {512, 896, 48};
    // dims[0] = rand() % 1024 + 1;
    // dims[1] = rand() % 1024 + 1;
    // dims[2] = rand() % 1024 + 1;
    int in_axis = 1;
    printf("%s\n", ToString(dims).c_str());
    if(TestSoftmax<T>(context, dims, in_axis) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
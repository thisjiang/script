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

template<typename T, size_t rank>
using EigenType = Eigen::TensorMap<Eigen::Tensor<T, rank, 
                                  Eigen::RowMajor, Eigen::DenseIndex>>;
template<size_t rank>
using EigenIndexArray = Eigen::array<Eigen::Index, rank>;

template<size_t rank>
using EigenPairArray = Eigen::array<std::pair<int64_t, int64_t>, rank>;

template<typename T, size_t rank>
void LaunchEigenKernel(Eigen::GpuDevice &dev_ctx,
                 EigenType<T, rank> &input_eigen,
                 const Eigen::array<std::pair<int, int>, rank> &pad_eigen,
                 EigenType<T, rank> &output_eigen) {
  output_eigen.device(dev_ctx) = input_eigen.pad(pad_eigen, T(0));
}

template<typename T, size_t rank>
float TimeEigenKernel(CUDAStream &context,
                 const std::array<int, rank> &input_dims,
                 const std::array<int, rank> &pad_start,
                 const std::array<int, rank> &pad_end,
                 const std::array<int, rank> &output_dims,
                 T *input_data,
                 T *output_data) {
  auto tt = TimeOfKernel::get(context);
  Eigen::GpuStreamDevice stream(&context.stream());
  Eigen::GpuDevice dev_ctx(&stream);

  Eigen::array<Eigen::Index, rank> indims_eigen, outdims_eigen;
  for(int i = 0; i < rank; i ++) indims_eigen[i] = input_dims[i];
  for(int i = 0; i < rank; i ++) outdims_eigen[i] = output_dims[i];
  
  Eigen::array<std::pair<int, int>, rank> pad_eigen;
  for(int i = 0; i < rank; i ++)
    pad_eigen[i] = std::make_pair(pad_start[i], pad_end[i]);

  EigenType<T, rank> input_eigen(input_data, indims_eigen);
  EigenType<T, rank> output_eigen(output_data, outdims_eigen);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchEigenKernel<T, rank>(dev_ctx, input_eigen, pad_eigen, output_eigen);
  }
  float cost = tt->stop();
  return cost;
}

/************************************************************************/
template<typename T, size_t rank>
__global__ void KePaddingCpy(const T *input, const int64_t *in_stride_array,
                              const int64_t in_size, const int64_t *padding_start,
                              T *output, const int64_t *out_stride_array) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // using register to speed up
  int64_t reg_in_stride_array[rank], reg_out_stride_array[rank];
  int64_t reg_padding_start[rank];
#pragma unroll
  for(int i = 0; i < rank; i ++) {
    reg_in_stride_array[i] = in_stride_array[i];
    reg_out_stride_array[i] = out_stride_array[i];
    reg_padding_start[i] = padding_start[i];
  }

  for(int index = tid; index < in_size; index += blockDim.x * gridDim.x) {
    // compute each dimension index of input
    int dims[rank]{0};
    int64_t k = index;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      dims[i] = k / reg_in_stride_array[i];
      k -= dims[i] * reg_in_stride_array[i];
    }

    // compute the total index of output
    int64_t out_id = 0;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      out_id += (dims[i] + reg_padding_start[i]) * reg_out_stride_array[i];
    }
    output[out_id] = input[index];
  }
}

template<typename T, size_t rank>
void LaunchPaddingZeroKernel(CUDAStream &context,
                const T *input, const int64_t *in_stride_array,
                const int64_t in_size, const int64_t *padding_start,
                T *output, const int64_t *out_stride_array, const int64_t out_size) {
  // Padding zero for output
  cudaMemsetAsync(output, 0, out_size * sizeof(T), context.stream());

  int threads = 256;
  int block_num = threads * 10;
  int grids = (in_size + block_num - 1) / block_num;
  KePaddingCpy<T, rank><<<grids, threads, 0, context.stream()>>>(
      input, in_stride_array, in_size, padding_start, output, out_stride_array);
}


template<typename T, size_t rank>
float TestPaddingKernel(CUDAStream &context,
                 const std::array<int, rank> &input_dims,
                 const std::array<int, rank> &pad_start,
                 const std::array<int, rank> &pad_end,
                 const std::array<int, rank> &output_dims,
                 const T *input_data,
                 T *output_data) {
  auto tt = TimeOfKernel::get(context);

  int64_t in_size = GetSize(input_dims);
  int64_t out_size = GetSize(output_dims);

  std::array<int64_t, rank> in_stride_array, out_stride_array, padding_start;
  in_stride_array[rank - 1] = out_stride_array[rank - 1] = 1;
  for(int i = rank - 2; i >= 0; i --) {
    in_stride_array[i] = in_stride_array[i + 1] * input_dims[i + 1];
    out_stride_array[i] = out_stride_array[i + 1] * output_dims[i + 1];
  }
  for(int i = 0; i < rank; i ++) {
    padding_start[i] = pad_start[i];
  }

  MallocDevice<int64_t> din_stride_array(rank, context);
  MallocDevice<int64_t> dout_stride_array(rank, context);
  MallocDevice<int64_t> padding(rank, context);

  din_stride_array.CopyFromHost(in_stride_array.data(), rank * sizeof(int64_t));
  dout_stride_array.CopyFromHost(out_stride_array.data(), rank * sizeof(int64_t));
  padding.CopyFromHost(padding_start.data(), rank * sizeof(int64_t));

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchPaddingZeroKernel<T, rank>(context, input_data, din_stride_array.data(),
                                     in_size, padding.data(), output_data,
                                     dout_stride_array.data(), out_size);
  }
  float cost = tt->stop();

  return cost;
}
/************************************************************************/

template<typename T, size_t rank>
__global__ void KePaddingMemcpy(const T *input, const int *in_stride_array,
                              const int last_dimension,
                              const int prefix_size, const int *padding_start,
                              T *output, const int *out_stride_array) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // using register to speed up
  int reg_in_stride_array[rank], reg_out_stride_array[rank], reg_padding_start[rank];
#pragma unroll
  for(int i = 0; i < rank; i ++) {
    reg_in_stride_array[i] = in_stride_array[i];
    reg_out_stride_array[i] = out_stride_array[i];
    reg_padding_start[i] = padding_start[i];
  }

  for(int index = tid; index < prefix_size; index += blockDim.x * gridDim.x) {
    int in_id = index * last_dimension;
    int dims[rank]{0};
    int k = index;
#pragma unroll
    for(int i = 0; i < rank - 1; i ++) {
      dims[i] = k / reg_in_stride_array[i];
      k -= dims[i] * reg_in_stride_array[i];
    }

    int out_id = 0;
#pragma unroll
    for(int i = 0; i < rank - 1; i ++) {
      out_id += (dims[i] + reg_padding_start[i]) * reg_out_stride_array[i];
    }

    memcpy(reinterpret_cast<void*>(output + out_id),
           reinterpret_cast<const void*>(input + in_id),
           last_dimension * sizeof(T));
  }
}

template<typename T, size_t rank>
void LaunchPaddingMemcpyKernel(CUDAStream &context, const T *input, 
                const int *in_stride_array, const int last_dimension,
                const int prefix_size, const int *padding_start,
                T *output, const int *out_stride_array, const int out_size) {
  cudaMemsetAsync(output, 0, out_size * sizeof(T), context.stream());

  cudaMemcpyAsync(output, input, prefix_size * last_dimension * sizeof(T),
                  cudaMemcpyDeviceToDevice, context.stream());
/*
  int block_size = 256;
  int grid_size = (prefix_size + block_size - 1) / block_size;
  KePaddingMemcpy<T, rank><<<grid_size, block_size, 0, context.stream()>>>(
      input, in_stride_array, last_dimension,
      prefix_size, padding_start, output, out_stride_array);
*/
}


template<typename T, size_t rank>
float TestPaddingMemcpyKernel(CUDAStream &context,
                 const std::array<int, rank> &input_dims,
                 const std::array<int, rank> &pad_start,
                 const std::array<int, rank> &pad_end,
                 const std::array<int, rank> &output_dims,
                 const T *input_data,
                 T *output_data) {
  auto tt = TimeOfKernel::get(context);

  int in_size = GetSize(input_dims);
  int out_size = GetSize(output_dims);

  int last_dimension = input_dims[rank - 1];
  int prefix_size = in_size / last_dimension;

  std::array<int, rank> in_stride_array, out_stride_array;
  in_stride_array[rank - 1] = in_stride_array[rank - 2] = 1;
  for(int i = rank - 3; i >= 0; i --) {
    in_stride_array[i] = in_stride_array[i + 1] * input_dims[i + 1];
  }
  out_stride_array[rank - 1] = 1;
  for(int i = rank - 2; i >= 0; i --) {
    out_stride_array[i] = out_stride_array[i + 1] * output_dims[i + 1];
  }

  MallocDevice<int> din_stride_array(rank, context);
  MallocDevice<int> dout_stride_array(rank, context);
  MallocDevice<int> padding(rank, context);

  din_stride_array.CopyFromHost(in_stride_array.data(), rank * sizeof(int));
  dout_stride_array.CopyFromHost(out_stride_array.data(), rank * sizeof(int));
  padding.CopyFromHost(pad_start.data(), rank * sizeof(int));

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchPaddingMemcpyKernel<T, rank>(context, input_data,
                                     din_stride_array.data(), last_dimension,
                                     prefix_size, padding.data(), output_data,
                                     dout_stride_array.data(), out_size);
  }
  float cost = tt->stop();

  return cost;
}

/************************************************************************/
template<typename T, size_t rank>
__global__ void KePaddingOrCopy(const T *input_data,
                              const EigenIndexArray<rank> in_stride_array,
                              const int64_t in_size,
                              const EigenPairArray<rank> paddings,
                              T *output_data,
                              const EigenIndexArray<rank> out_stride_array,
                              const int64_t out_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for(int64_t index = tid; index < out_size; index += blockDim.x * gridDim.x) {
    int dims[rank]{0};
    int64_t k = index;
    bool set_zero = false;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      dims[i] = k / out_stride_array[i];
      if(dims[i] < paddings[i].first || dims[i] >= paddings[i].second){
        set_zero = true;
        break;
      }
      k -= dims[i] * out_stride_array[i];
    }
    if(set_zero) {
      output_data[index] = (T)0;
    } else {
      int64_t in_id = 0;
#pragma unroll
      for(int i = 0; i < rank; i ++) {
        in_id += (dims[i] - paddings[i].first) * in_stride_array[i];
      }
      output_data[index] = input_data[in_id];
    }
  }
}

template<typename T, size_t rank>
void LaunchPaddingOrCopyKernel(const gpuStream_t &stream,
                              const T *input_data,
                              const EigenIndexArray<rank> &in_stride_array,
                              const int64_t in_size,
                              const EigenPairArray<rank> &paddings,
                              T *output_data,
                              const EigenIndexArray<rank> &out_stride_array,
                              const int64_t out_size) {
  int threads = 256;
  int block_num = threads * 10;
  int grids = (out_size + block_num - 1) / block_num;
  KePaddingOrCopy<T, rank><<<grids, threads, 0, stream>>>(
      input_data, in_stride_array, in_size,
      paddings, output_data, out_stride_array, out_size);
}

template<typename T, size_t rank>
float TestPaddingOrCopyKernel(CUDAStream &context,
                              const std::array<int, rank> &input_dims,
                              const std::array<int, rank> &pad_start,
                              const std::array<int, rank> &pad_end,
                              const std::array<int, rank> &output_dims,
                              const T *input_data,
                              T *output_data) {
  auto tt = TimeOfKernel::get(context);

  int64_t in_size = GetSize(input_dims);
  int64_t out_size = GetSize(output_dims);

  EigenIndexArray<rank> in_stride_array, out_stride_array;
  in_stride_array[rank - 1] = out_stride_array[rank - 1] = 1;
  for(int i = rank - 2; i >= 0; i --) {
    in_stride_array[i] = in_stride_array[i + 1] * input_dims[i + 1];
    out_stride_array[i] = out_stride_array[i + 1] * output_dims[i + 1];
  }

  EigenPairArray<rank> paddings;
  for(int i = 0; i < rank; i ++)
    paddings[i] = std::make_pair(pad_start[i], input_dims[i] + pad_start[i]);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchPaddingOrCopyKernel<T, rank>(context.stream(), input_data, in_stride_array,
                                      in_size, paddings, output_data,
                                      out_stride_array, out_size);
  }
  float cost = tt->stop();

  return cost;
}

/************************************************************************/
template<typename T, size_t rank>
__global__ void KePaddingEigen(const T *input_data,
                              const EigenIndexArray<rank> in_stride_array,
                              const int64_t in_size,
                              const EigenPairArray<rank> paddings,
                              T *output_data,
                              const EigenIndexArray<rank> out_stride_array,
                              const int64_t out_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for(int index = tid; index < in_size; index += blockDim.x * gridDim.x) {
    // compute each dimension index of input
    int dims[rank]{0};
    int64_t k = index;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      dims[i] = k / in_stride_array[i];
      k -= dims[i] * in_stride_array[i];
    }

    // compute the total index of output
    int64_t out_id = 0;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      out_id += (dims[i] + paddings[i].first) * out_stride_array[i];
    }
    output_data[out_id] = input_data[index];
  }
}

template<typename T, size_t rank>
void LaunchPaddingEigenKernel(const gpuStream_t &stream,
                              const T *input_data,
                              const EigenIndexArray<rank> &in_stride_array,
                              const int64_t in_size,
                              const EigenPairArray<rank> &paddings,
                              T *output_data,
                              const EigenIndexArray<rank> &out_stride_array,
                              const int64_t out_size) {
  // Padding zero for output
  cudaMemsetAsync(output_data, 0, out_size * sizeof(T), stream);

  int threads = 256;
  int block_num = threads * 10;
  int grids = (in_size + block_num - 1) / block_num;
  KePaddingEigen<T, rank><<<grids, threads, 0, stream>>>(
      input_data, in_stride_array, in_size,
      paddings, output_data, out_stride_array, out_size);
}


template<typename T, size_t rank>
float TestPaddingEigenKernel(CUDAStream &context,
                 const std::array<int, rank> &input_dims,
                 const std::array<int, rank> &pad_start,
                 const std::array<int, rank> &pad_end,
                 const std::array<int, rank> &output_dims,
                 const T *input_data,
                 T *output_data) {
  auto tt = TimeOfKernel::get(context);

  int64_t in_size = GetSize(input_dims);
  int64_t out_size = GetSize(output_dims);

  EigenIndexArray<rank> in_stride_array, out_stride_array;
  in_stride_array[rank - 1] = out_stride_array[rank - 1] = 1;
  for(int i = rank - 2; i >= 0; i --) {
    in_stride_array[i] = in_stride_array[i + 1] * input_dims[i + 1];
    out_stride_array[i] = out_stride_array[i + 1] * output_dims[i + 1];
  }

  EigenPairArray<rank> pad_eigen;
  for(int i = 0; i < rank; i ++)
    pad_eigen[i] = std::make_pair(pad_start[i], pad_end[i]);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchPaddingEigenKernel<T, rank>(context.stream(), input_data, in_stride_array,
                                     in_size, pad_eigen, output_data,
                                     out_stride_array, out_size);
  }
  float cost = tt->stop();

  return cost;
}
/************************************************************************/
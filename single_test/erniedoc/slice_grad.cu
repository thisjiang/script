#include "../common.h"

#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <utility>
#include <array>

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

constexpr int LOOPNUM = 100;

template<typename T, size_t rank>
using EigenType = Eigen::TensorMap<Eigen::Tensor<T, rank, 
                                  Eigen::RowMajor, Eigen::DenseIndex>>;

template<typename T, size_t rank>
void RunEigenKernel(Eigen::GpuDevice &dev_ctx,
                 const EigenType<T, rank> &input_eigen,
                 const Eigen::array<std::pair<int, int>, rank> &pad_eigen,
                 EigenType<T, rank> &output_eigen) {
  output_eigen.device(dev_ctx) = input_eigen.pad(pad_eigen, T(0));
}

template<typename T, size_t rank>
float TestEigen(CUDAStream &context,
                 const std::array<int, rank> &input_dims,
                 const std::array<int, rank> &pad_start,
                 const std::array<int, rank> &pad_end,
                 const std::array<int, rank> &output_dims,
                 const MallocDevice<T> &input,
                 MallocDevice<T> &output) {
  auto tt = TimeOfKernel::get(context);
  Eigen::GpuStreamDevice stream(&context.stream());
  Eigen::GpuDevice dev_ctx(&stream);

  Eigen::array<Eigen::Index, rank> indims_eigen, outdims_eigen;
  for(int i = 0; i < rank; i ++) indims_eigen[i] = input_dims[i];
  for(int i = 0; i < rank; i ++) outdims_eigen[i] = output_dims[i];
  
  Eigen::array<std::pair<int, int>, rank> pad_eigen;
  for(int i = 0; i < rank; i ++)
    pad_eigen[i] = std::make_pair(pad_start[i], pad_end[i]);

  EigenType<T, rank> input_eigen(input.data(), indims_eigen);
  EigenType<T, rank> output_eigen(output.data(), outdims_eigen);

  RunEigenKernel<T, rank>(dev_ctx, input_eigen, pad_eigen, output_eigen);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
    RunEigenKernel<T, rank>(dev_ctx, input_eigen, pad_eigen, output_eigen);
  float cost = tt->stop();
  return cost;
}

template<typename T, size_t rank>
__global__ void KePaddingCpy(const T *input, const int *in_stride,
                              const int in_size, const int *padding_start,
                              T *output, const int *out_stride, const int out_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int index = tid; index < in_size; index += blockDim.x * gridDim.x) {
    int dims[rank]{0};
    int k = index;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      dims[i] = k / in_stride[i];
      k -= dims[i] * in_stride[i];
    }

    int out_id = 0;
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      out_id += (dims[i] + padding_start[i]) * out_stride[i];
    }

    output[out_id] = input[index];
  }
}

template<typename T, size_t rank>
void RunPaddingZero(CUDAStream &context,
                const T *input, const int *in_stride,
                const int in_size, const int *padding_start,
                T *output, const int *out_stride, const int out_size) {
  cudaMemsetAsync(output, 0, out_size * sizeof(T), context.stream());
  
  int block_size = 256;
  int grid_size = (in_size + block_size - 1) / block_size;
  KePaddingCpy<T, rank><<<grid_size, block_size, 0, context.stream()>>>(
      input, in_stride, in_size, padding_start, output, out_stride, out_size);
}

template<typename T, size_t rank>
float TestPaddingZero(CUDAStream &context,
                 const std::array<int, rank> &input_dims,
                 const std::array<int, rank> &pad_start,
                 const std::array<int, rank> &pad_end,
                 const std::array<int, rank> &output_dims,
                 const MallocDevice<T> &input,
                 MallocDevice<T> &output) {
  auto tt = TimeOfKernel::get(context);

  int in_size = GetSize(input_dims);
  int out_size = GetSize(output_dims);

  std::array<int, rank> in_stride, out_stride;
  in_stride[rank - 1] = out_stride[rank - 1] = 1;
  for(int i = rank - 2; i >= 0; i --) {
    in_stride[i] = in_stride[i + 1] * input_dims[i + 1];
    out_stride[i] = out_stride[i + 1] * output_dims[i + 1];
  }

  MallocDevice<int> din_stride(rank, context);
  MallocDevice<int> dout_stride(rank, context);
  MallocDevice<int> padding(rank, context);

  din_stride.CopyFromHost(in_stride.data(), rank * sizeof(int));
  dout_stride.CopyFromHost(out_stride.data(), rank * sizeof(int));
  padding.CopyFromHost(pad_start.data(), rank * sizeof(int));

  RunPaddingZero<T, rank>(context, input.data(), din_stride.data(), in_size,
                padding.data(), output.data(), dout_stride.data(), out_size);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++)
    RunPaddingZero<T, rank>(context, input.data(), din_stride.data(), in_size,
                padding.data(), output.data(), dout_stride.data(), out_size);
  float cost = tt->stop();

  return cost;
}

template<typename T, size_t rank>
int TestKernel(CUDAStream &context,
               const std::array<int, rank> &input_dims,
               const std::array<int, rank> &pad_start,
               const std::array<int, rank> &pad_end) {
  std::array<int, rank> output_dims;
  for(int i = 0; i < rank; i ++)
    output_dims[i] = input_dims[i] + pad_start[i] + pad_end[i];

  int in_size = GetSize(input_dims);
  int out_size = GetSize(output_dims);
  MallocHost<T> input_cpu(in_size, context);
  MallocDevice<T> input(in_size, context);
  MallocDevice<T> out_eigen(out_size, context);
  MallocDevice<T> out_padzero(out_size, context);

  input_cpu.Random(0, 100);
  input.CopyFrom(input_cpu);
  float eigen_cost = TestEigen<T, rank>(context, input_dims, pad_start, 
                            pad_end, output_dims, input, out_eigen);
  float self_cost = TestPaddingZero<T, rank>(context, input_dims, pad_start, 
                            pad_end, output_dims, input, out_padzero);
  printf("Eigen cost %f ms vs PaddingZero cost %f\n", eigen_cost, self_cost);

  auto err = context.sync();
  if(err != "") {
    fprintf(stderr, "ERROR: %s\n", err);
    return CUDA_FAILED;
  }

  bool is_same = out_padzero.CheckSame(out_eigen);
  if(!is_same) {
    fprintf(stderr, "Check Failed!\n");
    return CHECK_FAILED;
  } else {
    printf("Check Success!\n");
  }

  return SUCCESS;
}


int main() {
  CUDAStream context;

  constexpr int rank = 4;
  std::array<int, rank> input_dims = {1407, 512, 4, 12};
  std::array<int, rank> pad_start = {1, 0, 0, 0};
  std::array<int, rank> pad_end = {0, 0, 0, 0};

  TestKernel<float, rank>(context, input_dims, pad_start, pad_end);
  TestKernel<half, rank>(context, input_dims, pad_start, pad_end);
  return 0;
}
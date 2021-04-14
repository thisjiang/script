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
template<typename T, size_t rank>
using EigenMapType = Eigen::TensorMap<Eigen::Tensor<T, rank,
                                  Eigen::RowMajor, int>, Eigen::Aligned>;

template<size_t rank>
using EigenInt64Array = Eigen::array<int64_t, rank>;

template<size_t rank>
using EigenIntArray = Eigen::DSizes<int, rank>;

template<typename T, size_t rank>
void LaunchEigenKernel(Eigen::GpuDevice &dev_ctx,
                 EigenInt64Array<rank> &offsets,
                 EigenInt64Array<rank> &extents,
                 EigenType<T, rank> &input,
                 EigenType<T, rank> &output) {
  output.device(dev_ctx) = input.slice(offsets, extents);
}

template<typename T, size_t rank>
float TimeEigenKernel(CUDAStream &context,
                 T *input_data,
                 const std::array<int64_t, rank> &input_dims,
                 T *output_data,
                 const std::array<int64_t, rank> &output_dims,
                 const std::array<int64_t, rank> &offsets,
                 const std::array<int64_t, rank> &extents) {
  auto tt = TimeOfKernel::get(context);
  Eigen::GpuStreamDevice stream(&context.stream());
  Eigen::GpuDevice dev_ctx(&stream);

  EigenInt64Array<rank> offsets_eigen, extents_eigen;
  for(int i = 0; i < rank; i ++) offsets_eigen[i] = offsets[i];
  for(int i = 0; i < rank; i ++) extents_eigen[i] = extents[i];

  EigenInt64Array<rank> indims_eigen, outdims_eigen;
  for(int i = 0; i < rank; i ++) indims_eigen[i] = input_dims[i];
  for(int i = 0; i < rank; i ++) outdims_eigen[i] = output_dims[i];

  EigenType<T, rank> input_eigen(input_data, indims_eigen);
  EigenType<T, rank> output_eigen(output_data, outdims_eigen);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchEigenKernel<T, rank>(dev_ctx, offsets_eigen, extents_eigen,
                               input_eigen, output_eigen);
  }
  float cost = tt->stop();
  return cost;
}

/***********************************************************/

template<typename T, size_t rank>
void LaunchTFEigenKernel(Eigen::GpuDevice &dev_ctx,
                 EigenIntArray<rank> &offsets,
                 EigenIntArray<rank> &extents,
                 EigenMapType<T, rank> &input,
                 EigenMapType<T, rank> &output) {
  output.device(dev_ctx) = input.slice(offsets, extents);
}

template<typename T, size_t rank>
float TimeTFEigenKernel(CUDAStream &context,
                 T *input_data,
                 const std::array<int64_t, rank> &input_dims,
                 T *output_data,
                 const std::array<int64_t, rank> &output_dims,
                 const std::array<int64_t, rank> &offsets,
                 const std::array<int64_t, rank> &extents) {
  auto tt = TimeOfKernel::get(context);
  Eigen::GpuStreamDevice stream(&context.stream());
  Eigen::GpuDevice dev_ctx(&stream);

  EigenIntArray<rank> offsets_eigen, extents_eigen;
  for(int i = 0; i < rank; i ++) offsets_eigen[i] = offsets[i];
  for(int i = 0; i < rank; i ++) extents_eigen[i] = extents[i];

  EigenIntArray<rank> indims_eigen, outdims_eigen;
  for(int i = 0; i < rank; i ++) indims_eigen[i] = input_dims[i];
  for(int i = 0; i < rank; i ++) outdims_eigen[i] = output_dims[i];

  EigenMapType<T, rank> input_eigen(input_data, indims_eigen);
  EigenMapType<T, rank> output_eigen(output_data, outdims_eigen);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    LaunchTFEigenKernel<T, rank>(dev_ctx, offsets_eigen, extents_eigen,
                               input_eigen, output_eigen);
  }
  float cost = tt->stop();
  return cost;
}
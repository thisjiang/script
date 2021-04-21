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
  float cost = 0.0f;
  Eigen::GpuStreamDevice stream(&context.stream());
  Eigen::GpuDevice dev_ctx(&stream);

  if(GetSize(input_dims) >= std::numeric_limits<int>::max()) {
    cost = TimeEigenKernel(context, input_data, input_dims,
                            output_data, output_dims,
                            offsets, extents);
  } else {
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
    cost = tt->stop();
  }
  return cost;
}

/***********************************************************/

template<typename T, size_t rank>
float TimeReshapeEigenKernel(CUDAStream &context,
                 T *input_data,
                 const std::array<int64_t, rank> &input_dims,
                 T *output_data,
                 const std::array<int64_t, rank> &output_dims,
                 const std::array<int64_t, rank> &offsets,
                 const std::array<int64_t, rank> &extents) {
  float cost = 0.0f;
  auto tt = TimeOfKernel::get(context);

  if(rank <= 3) {
    cost = TimeTFEigenKernel(context, input_data, input_dims,
                            output_data, output_dims,
                            offsets, extents);
  } else {
    int slice_num = 0, slice_dim = -1;
    for(int i = 0; i < rank; i ++) {
      if(offsets[i] != 0 || extents[i] != input_dims[i]) {
        slice_num ++;
        slice_dim = i;
      }
    }
    printf("%d (%d, %d)\n", rank, slice_num, slice_dim);

    if(slice_num == 0) {
      size_t in_size = GetSize(input_dims);
      tt->start();
      #pragma unroll
      for(int i = 0; i < LOOPNUM; i ++) {
        cudaMemcpyAsync(output_data, input_data, in_size * sizeof(T), cudaMemcpyDeviceToDevice, context.stream());
      }
      cost = tt->stop();
    } else if(slice_num == 1) {
      if(slice_dim == 0) {
        // reshape to [slice, succeeding]
        std::array<int64_t, 2> offsets_toreshpae, extents_toreshape;
        offsets_toreshpae.fill(1), extents_toreshape.fill(1);
        offsets_toreshpae[0] = offsets[slice_dim];
        extents_toreshape[0] = extents[slice_dim];
        for(int i = slice_dim + 1; i < rank; i ++) {
          offsets_toreshpae[1] *= offsets[i];
          extents_toreshape[1] *= extents[i];
        }

        std::array<int64_t, 2> input_dims_toreshape, output_dims_toreshape;
        input_dims_toreshape.fill(1), output_dims_toreshape.fill(1);
        input_dims_toreshape[0] = input_dims[slice_dim];
        output_dims_toreshape[0] = output_dims[slice_dim];
        for(int i = slice_dim + 1; i < rank; i ++) {
          input_dims_toreshape[1] *= input_dims[i];
          output_dims_toreshape[1] *= output_dims[i];
        }
        cost = TimeTFEigenKernel(context, input_data, input_dims_toreshape,
                            output_data, output_dims_toreshape,
                            offsets_toreshpae, extents_toreshape);
      } else if(slice_dim == rank - 1) {
        // reshape to [preceding, slice]
        std::array<int64_t, 2> offsets_toreshpae, extents_toreshape;
        offsets_toreshpae.fill(1), extents_toreshape.fill(1);
        for(int i = 0; i < slice_dim; i ++) {
          offsets_toreshpae[0] *= offsets[i];
          extents_toreshape[0] *= extents[i];
        }
        offsets_toreshpae[1] = offsets[slice_dim];
        extents_toreshape[1] = extents[slice_dim];

        std::array<int64_t, 2> input_dims_toreshape, output_dims_toreshape;
        input_dims_toreshape.fill(1), output_dims_toreshape.fill(1);
        for(int i = 0; i < slice_dim; i ++) {
          input_dims_toreshape[0] *= input_dims[i];
          output_dims_toreshape[0] *= output_dims[i];
        }
        input_dims_toreshape[1] = input_dims[slice_dim];
        output_dims_toreshape[1] = output_dims[slice_dim];
        
        cost = TimeTFEigenKernel(context, input_data, input_dims_toreshape,
                            output_data, output_dims_toreshape,
                            offsets_toreshpae, extents_toreshape);
      } else {
        // reshape to [preceding, slice, succeeding]
        std::array<int64_t, 3> offsets_toreshpae, extents_toreshape;
        offsets_toreshpae.fill(1), extents_toreshape.fill(1);
        for(int i = 0; i < slice_dim; i ++) {
          offsets_toreshpae[0] *= offsets[i];
          extents_toreshape[0] *= extents[i];
        }
        offsets_toreshpae[1] = offsets[slice_dim];
        extents_toreshape[1] = extents[slice_dim];
        for(int i = slice_dim + 1; i < rank; i ++) {
          offsets_toreshpae[2] *= offsets[i];
          extents_toreshape[2] *= extents[i];
        }

        std::array<int64_t, 3> input_dims_toreshape, output_dims_toreshape;
        input_dims_toreshape.fill(1), output_dims_toreshape.fill(1);
        for(int i = 0; i < slice_dim; i ++) {
          input_dims_toreshape[0] *= input_dims[i];
          output_dims_toreshape[0] *= output_dims[i];
        }
        input_dims_toreshape[1] = input_dims[slice_dim];
        output_dims_toreshape[1] = output_dims[slice_dim];
        for(int i = slice_dim + 1; i < rank; i ++) {
          input_dims_toreshape[2] *= input_dims[i];
          output_dims_toreshape[2] *= output_dims[i];
        }

        cost = TimeTFEigenKernel(context, input_data, input_dims_toreshape,
                            output_data, output_dims_toreshape,
                            offsets_toreshpae, extents_toreshape);
      }
    } else {
      cost = TimeTFEigenKernel(context, input_data, input_dims,
                            output_data, output_dims,
                            offsets, extents);
    }
  }

  return cost;
}
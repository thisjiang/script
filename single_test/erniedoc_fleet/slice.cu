// header file
#include "slice.h"

// C system file
#include <stdio.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// C++ system file
#include <iostream>
#include <vector>

// Library file
#include "../common.h"


template<typename T, size_t rank>
int TestSliceKernel(CUDAStream &context,
                 const std::array<int64_t, rank> &input_dims,
                 const std::array<int64_t, rank> &output_dims,
                 const std::array<int64_t, rank> &offsets,
                 const std::array<int64_t, rank> &extents) {
  int in_size = 1, out_size = 1;
  for(int i = 0; i < rank; i ++) {
    in_size *= input_dims[i];
    out_size *= output_dims[i];
  }

  MallocHost<T> h_input(in_size, context);
  MallocDevice<T> d_input(in_size, context);
  MallocDevice<T> old_outs(out_size, context);
  MallocDevice<T> new_outs(out_size, context);

  h_input.Random(0, 10);
  d_input.CopyFrom(h_input);

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<char*> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  name = "Eigen Kernel";
  cost = TimeEigenKernel<T, rank>(context,
                                  d_input.data(), input_dims,
                                  old_outs.data(), output_dims,
                                  offsets, extents);
  AfterRun();

  name = "TF Eigen Kernel";
  cost = TimeTFEigenKernel<T, rank>(context,
                                  d_input.data(), input_dims,
                                  new_outs.data(), output_dims,
                                  offsets, extents);
  AfterRun();

  name = "Reshaped Eigen Kernel";
  cost = TimeReshapeEigenKernel<T, rank>(context,
                                  d_input.data(), input_dims,
                                  new_outs.data(), output_dims,
                                  offsets, extents);
  AfterRun();

  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "%s%s CUDA ERROR: %s\n",
            ToString<int64_t, rank>(input_dims).c_str(),
            ToString<int64_t, rank>(output_dims).c_str(),
            err);
    printf("*******************\n");
    return CUDA_FAILED;
  }

  if(!old_outs.CheckSame(new_outs)) {
    fprintf(stderr, "%s%s Result Check Failed\n",
            ToString<int64_t, rank>(input_dims).c_str(),
            ToString<int64_t, rank>(output_dims).c_str());
    printf("*******************\n");
    return CHECK_FAILED;
  } else {
    printf("%s%s Success\n",
            ToString<int64_t, rank>(input_dims).c_str(),
            ToString<int64_t, rank>(output_dims).c_str());
    printf("*******************\n");
  }

  return SUCCESS;
}

int main() {
  CUDAStream context;
  srand(time(0));

  do {
    constexpr int rank = 4;
#if 1
    std::array<int64_t, rank> input_dims = {4, 12, 512, 1151};
    std::array<int64_t, rank> output_dims = {4, 12, 512, 640};
    std::array<int64_t, rank> offsets = {0, 0, 0, 0};
    std::array<int64_t, rank> extents = {4, 12, 512, 640};
#else
    std::array<int64_t, rank> input_dims = {4, 12, 1152, 512};
    std::array<int64_t, rank> output_dims = {4, 12, 1151, 512};
    std::array<int64_t, rank> offsets = {0, 0, 1, 0, };
    std::array<int64_t, rank> extents = {4, 12, 1151, 512};
#endif

#if 1
    for(int i = 0; i < rank; i ++) {
      input_dims[i] = rand() % 100 + 1;
      offsets[i] = 0;
      extents[i] = input_dims[i];
      output_dims[i] = input_dims[i];
    }
    int dim = rand() % rank;
    offsets[dim] = rand() % (input_dims[dim] - 1);
    extents[dim] = rand() % (input_dims[dim] - offsets[dim] - 1) + 1;
    output_dims[dim] = offsets[dim] + extents[dim];
#endif
    auto a = Eigen::array<std::pair<int64_t, int64_t>, 2>({{0, 0}, {0, 0}});
    if(!TestSliceKernel<float, rank>(context, input_dims, output_dims,
                                       offsets, extents) == SUCCESS) {
      fprintf(stderr, "Float Error\n");
      // break;
    }
    if(!TestSliceKernel<half, rank>(context, input_dims, output_dims,
                                       offsets, extents) == SUCCESS) {
      fprintf(stderr, "Half Error\n");
      // break;
    }
    printf("\n****************************************\n");
    sleep(1);
  } while(true);
  return 0;
}
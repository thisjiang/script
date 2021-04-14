// header file
#include "slice_pad.h"

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
int TestPaddingKernel(CUDAStream &context,
               const std::array<int, rank> &input_dims,
               const std::array<int, rank> &pad_start,
               const std::array<int, rank> &pad_end,
               const std::array<int, rank> &output_dims) {
  int in_size = GetSize(input_dims), out_size = GetSize(output_dims);
  int pad_size = 0;
  for(int i = 0; i < rank; i ++) {
    pad_size += pad_start[i] + pad_end[i];
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
  cost = TimeEigenKernel<T, rank>(context, input_dims, pad_start, 
                                  pad_end, output_dims, d_input.data(),
                                  old_outs.data());
  AfterRun();


  // name = "Memcpy Kernel";
  // cost = TestPaddingMemcpyKernel<T, rank>(context, input_dims, pad_start, 
  //                                   pad_end, output_dims, d_input.data(),
  //                                   new_outs.data());
  // AfterRun();

  // name = "Padding Kernel";
  // cost = TestPaddingKernel<T, rank>(context, input_dims, pad_start, 
  //                                   pad_end, output_dims, d_input.data(),
  //                                   new_outs.data());
  // AfterRun();

  // name = "PaddingEigen Kernel";
  // cost = TestPaddingEigenKernel<T, rank>(context, input_dims, pad_start, 
  //                                   pad_end, output_dims, d_input.data(),
  //                                   new_outs.data());
  // AfterRun();

  name = "PaddingEigen Kernel";
  cost = TimeReshapeEigenKernel<T, rank>(context, input_dims, pad_start, 
                                    pad_end, output_dims, d_input.data(),
                                    new_outs.data());
  AfterRun();

  // name = "PaddingOrCopy Kernel";
  // cost = TestPaddingOrCopyKernel<T, rank>(context, input_dims, pad_start, 
  //                                   pad_end, output_dims, d_input.data(),
  //                                   new_outs.data());
  // AfterRun();

  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "%s%s[%d] CUDA ERROR: %s\n",
            ToString<int, rank>(input_dims).c_str(),
            ToString<int, rank>(output_dims).c_str(),
            pad_size,
            err);
    printf("*******************\n");
    return CUDA_FAILED;
  }

  if(out_size <= 1000) {
    printf("Input:\n");
    h_input.Print(input_dims);
    printf("Old Output:\n");
    old_outs.Print(output_dims);
    printf("New Output:\n");
    new_outs.Print(output_dims);
  }

  if(!old_outs.CheckSame(new_outs)) {
    fprintf(stderr, "%s%s[%d] Result Check Failed\n",
            ToString<int, rank>(input_dims).c_str(),
            ToString<int, rank>(output_dims).c_str(),
            pad_size);
    printf("*******************\n");
    return CHECK_FAILED;
  } else {
    printf("%s%s[%d] Success\n",
            ToString<int, rank>(input_dims).c_str(),
            ToString<int, rank>(output_dims).c_str(),
            pad_size);
    printf("*******************\n");
  }

  return SUCCESS;
}

int main() {
  CUDAStream context;
  srand(time(0));

  do {
    constexpr int rank = 4;
#if 0
    std::array<int, rank> input_dims = {4, 12, 512, 640};
    std::array<int, rank> output_dims = {4, 12, 512, 1151};
    std::array<int, rank> pad_start = {0, 0, 0, 0};
    std::array<int, rank> pad_end = {0, 0, 0, 511};
#else
    std::array<int, rank> input_dims = {4, 12, 1151, 512};
    std::array<int, rank> output_dims = {4, 12, 1152, 512};
    std::array<int, rank> pad_start = {0, 0, 1, 0};
    std::array<int, rank> pad_end = {0, 0, 0, 0};
#endif

#if 0
#pragma unroll
    for(int i = 0; i < rank; i ++) {
      input_dims[i] = rand() % 100 + 1;
      pad_start[i] = 0;
      pad_end[i] = 0;
      output_dims[i] = input_dims[i];
    }
    int dim = rand() % rank;
    pad_start[dim] = rand() % 20;
    pad_end[dim] = rand() % 20;
    output_dims[dim] = input_dims[dim] + pad_start[dim] + pad_end[dim];
#endif

#if 0
    for(int i = 0; i < rank; i ++) {
      input_dims[i] = 2;
      pad_start[i] = pad_end[i] = 0;
      output_dims[i] = 2;
    }

    int dim = 2;
    pad_start[dim] = 1;
    pad_end[dim] = 2;
    output_dims[dim] = input_dims[dim] + pad_start[dim] + pad_end[dim];
#endif

    if(!TestPaddingKernel<float, rank>(context, input_dims, pad_start,
                                  pad_end, output_dims) == SUCCESS) {
      fprintf(stderr, "Float Error\n");
      // break;
    }
    if(!TestPaddingKernel<half, rank>(context, input_dims, pad_start,
                                  pad_end, output_dims) == SUCCESS) {
      fprintf(stderr, "Half Error\n");
      // break;
    }
    printf("\n\n");
    sleep(1);
  } while(false);
  return 0;
}
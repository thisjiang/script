// header file
#include "update_loss_scaling.h"

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

constexpr int MAXRAND = 100000;

/************************************************************************/
template<typename T2>
struct AllEqualToZero{
  HOSTDEVICE bool operator()(const T2 &num) {
    return static_cast<T2>(0) == num;
  }
};

template<typename T>
class AllocParam {
public:
  AllocParam(const size_t xs_size, const bool found_inf, CUDAStream &context)
  : _context(context), _xs_size(xs_size), _found_inf_data(found_inf),
    found_inf_mem(1, context)
  {
    // copy found_inf from host to device
    found_inf_mem.CopyFromHost(&found_inf, sizeof(bool));
    found_inf_data = found_inf_mem.data();

    // set nums array
    nums = new int64_t[xs_size];
    std::vector<int64_t> offsets(xs_size + 1, 0);
    for(size_t i = 0; i < xs_size; i ++) {
      nums[i] = rand() % MAXRAND + 1;
      offsets[i + 1] = offsets[i] + nums[i];
    }
    total_num = offsets[xs_size];

    // alloc outs memory
    outs_mem = new MallocDevice<T>(total_num, context);
    T* outs_data = outs_mem->data();

    // set outs array pointer
    outs = new T*[xs_size];
    for(size_t i = 0; i < xs_size; i ++) {
      outs[i] = outs_data + offsets[i];
    }
  }

  ~AllocParam() {
    delete [] nums;
    delete [] outs;
    delete outs_mem;

    nums = nullptr;
    outs = nullptr;
    found_inf_data = nullptr;
    outs_mem = nullptr;
  }

  bool CheckResult() const {
    return _found_inf_data ?
            CheckAllTrueDevice<T, AllEqualToZero>(outs_mem->data(), total_num, _context.stream()) :
            true;
  }

public:
  T** outs;
  int64_t* nums;
  bool* found_inf_data;
  int64_t total_num;

private:
  const CUDAStream &_context;
  const int64_t _xs_size;
  const bool _found_inf_data;

  MallocDevice<T> *outs_mem;
  MallocDevice<bool> found_inf_mem;
};
/************************************************************************/

template<typename T>
int TestKernel(CUDAStream &context, const size_t xs_size, const bool found_inf) {
  AllocParam<T> params(xs_size, found_inf, context);

  auto err2 = context.sync();
  if(err2 != EMPTY_STRING) {
    fprintf(stderr, "[%zd, %ld, %d] Alloc CUDA ERROR: %s\n",
            xs_size, params.total_num, static_cast<int>(found_inf), err2);
    return CUDA_FAILED;
  }

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<std::string> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  name = "Old Kernel";
  cost = TimeOfOldKernel(context, params.found_inf_data, xs_size, params.nums, params.outs);
  AfterRun();

  printf("*******************\n");
  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "[%zd, %ld, %d] CUDA ERROR: %s\n",
            xs_size, params.total_num, static_cast<int>(found_inf), err);
    return CUDA_FAILED;
  }

  if(!params.CheckResult()) {
    fprintf(stderr, "[%zd, %ld, %d] Result Check Failed\n",
            xs_size, params.total_num, static_cast<int>(found_inf));
    return CHECK_FAILED;
  } else {
    printf("[%zd, %ld, %d] Success\n",
            xs_size, params.total_num, static_cast<int>(found_inf));
  }

  auto err3 = context.sync();
  if(err3 != EMPTY_STRING) {
    fprintf(stderr, "[%zd, %ld, %d] Check CUDA ERROR: %s\n",
            xs_size, params.total_num, static_cast<int>(found_inf), err3);
    return CUDA_FAILED;
  }

  return SUCCESS;
}

int main() {
  srand(time(0));
  CUDAStream context;

  do {
    size_t size = rand() % 100 + 1;
    // printf("Please Input num\n");
    // std::cin >> num;
    printf("Float:\n");
    if(TestKernel<float>(context, size, true) != SUCCESS) break;
    if(TestKernel<float>(context, size, false) != SUCCESS) break;
    printf("Half:\n");
    if(TestKernel<half>(context, size, true) != SUCCESS) break;
    if(TestKernel<half>(context, size, false) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
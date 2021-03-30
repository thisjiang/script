// header file
#include "check_finite_and_unscale.h"

// C system file
#include "stdio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// C++ system file
#include <iostream>
#include <vector>
// Library file
#include "../common.h"

constexpr int MAXRAND = 100000;

/************************************************************************/
template<typename T>
class AllocParam {
public:
  AllocParam(const int size, CUDAStream &context)
  : _context(context), _size(size) {
    // generate input size
    nums = new int64_t[size];
    std::vector<int64_t> offset(size + 1, 0);
    for(int i = 0; i < size; i ++) {
      nums[i] = rand() % MAXRAND + 1;
      offset[i + 1] = offset[i] + nums[i];
    }
    // alloc memory
    int64_t total_num = offset[size];
    h_mem = new MallocHost<T>(total_num, context);
    d_mem = new MallocDevice<T>(3 * total_num, context);
    // generate input and copy to device
    h_mem->Random();
    d_mem->CopyFrom(*h_mem); // Copy size = total_num and offset = 0
    // alloc memory
    h_xs = new T*[size];
    d_xs = new T*[size];
    old_outs = new T*[size];
    new_outs = new T*[size];
    // set array address
    T* h_data = h_mem->data();
    T* in_data = d_mem->data();
    T* old_out_data = in_data + total_num;
    T* new_out_data = old_out_data + total_num;
    for(int i = 0; i < size; i ++) {
      h_xs[i] = h_data + offset[i];
      d_xs[i] = in_data + offset[i];
      old_outs[i] = old_out_data + offset[i];
      new_outs[i] = new_out_data + offset[i];
    }
  }

  ~AllocParam() {
    delete nums;
    nums = nullptr;
    delete h_mem, d_mem;
    h_mem = nullptr;
    d_mem = nullptr;
    delete h_xs, d_xs, old_outs, new_outs;
    h_xs = d_xs = old_outs = new_outs = nullptr;
  }

public:
  int64_t *nums;
  T** h_xs, **d_xs, **old_outs, **new_outs;

private:
  CUDAStream &_context;
  const int _size;
  MallocHost<T>* h_mem;
  MallocDevice<T>* d_mem;
};

template<typename T>
bool CheckResult(CUDAStream &context, const int size, int64_t* nums, T** old_outs, T** new_outs) {
  return CheckSameDevice(old_outs[0], new_outs[0], GetSum(nums, size), context.stream());
}
/************************************************************************/

template<typename T, typename MT>
int TestKernel(CUDAStream &context, const int size) {
  MallocDevice<MT> d_scale(1, context);
  MallocDevice<bool> d_found_inf(1, context);
  InitalKernel(context, d_scale.data(), d_found_inf.data());

  AllocParam<T> params(size, context);

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<char*> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  name = "Old Kernel";
  cost = TimeOfOldKernel(context, size, params.nums, params.d_xs, d_scale.data(),
                         d_found_inf.data(), params.old_outs);
  AfterRun();

  name = "Fused Kernel";
  cost = TimeOfFusedKernel(context, size, params.nums, params.d_xs, d_scale.data(),
                         d_found_inf.data(), params.new_outs);
  AfterRun();

  printf("*******************\n");
  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "[%d][%s] CUDA ERROR: %s\n",
            GetSum(params.nums, size), ToString(params.nums, size).c_str(), err);
    return CUDA_FAILED;
  }

  if(!CheckResult(context, size, params.nums, params.old_outs, params.new_outs)) {
    fprintf(stderr, "[%d][%s] Result Check Failed\n",
            GetSum(params.nums, size), ToString(params.nums, size).c_str());
    return CHECK_FAILED;
  } else {
    printf("[%d][%s] Success\n",
          GetSum(params.nums, size), ToString(params.nums, size).c_str());
  }

  return SUCCESS;
}

int main() {
  srand(time(0));
  CUDAStream context;
  typedef float T;
  typedef float MT;

  do {
    int size = rand() % 20 + 1;
    // printf("Please Input num\n");
    // std::cin >> num;
    if(TestKernel<T, MT>(context, size) != SUCCESS) break;
    printf("\n");
  } while(true);

  return 0;
}
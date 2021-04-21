// header file
#include "adam.h"

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

template<typename T, typename MT>
class AllocParam {
public:
  AllocParam(int ndim, CUDAStream &context)
  : ndim(ndim),
    h_moment1(ndim, context), d_moment1(ndim, context),
    h_moment2(ndim, context), d_moment2(ndim, context),
    h_param(ndim, context), d_param(ndim, context),
    h_grad(ndim, context), d_grad(ndim, context),
    h_master_param(ndim, context), d_master_param(ndim, context),
    h_lr(1, context), d_lr(1, context),
    moment1_out_old(ndim, context), moment1_out_new(ndim, context),
    moment2_out_old(ndim, context), moment2_out_new(ndim, context),
    param_out_old(ndim, context), param_out_new(ndim, context),
    master_param_out_old(ndim, context), master_param_out_new(ndim, context)
  {
    h_moment1.Random(0, 1);
    d_moment1.CopyFrom(h_moment1);
    h_moment2.Random(0, 1);
    d_moment2.CopyFrom(h_moment2);
    h_master_param.Random(0, 1);
    d_master_param.CopyFrom(h_master_param);
    h_param.Random(0, 1);
    d_param.CopyFrom(h_param);
    h_grad.Random(0, 1);
    d_grad.CopyFrom(h_grad);
    h_lr.Random(0, 1);
    d_lr.CopyFrom(h_lr);

    beta1 = rand() % 1 + 1e-6f;
    beta2 = rand() % 1 + 1e-6f;
    beta1_pow = rand() % 1 + 1e-6f;
    beta2_pow = rand() % 1 + 1e-6f;
    epsilon = rand() % 1 + 1e-6f;
  }

  bool CheckResult() {
    bool check_failed = false;
  #if 1
    auto moment1_out_err = moment1_out_old.MaxError(moment1_out_new);
    auto moment2_out_err = moment2_out_old.MaxError(moment2_out_new);
    auto param_out_err = param_out_old.MaxError(param_out_new);
    auto master_param_out_err = master_param_out_old.MaxError(master_param_out_new);

    printf("Error:(moment1_out %f), (moment2_out %f) (param_out %f) (master_param_out %f)\n",
            moment1_out_err, moment2_out_err, param_out_err, master_param_out_err);
    if(moment1_out_err >= 1e-6f) {
      fprintf(stderr, "[%d] moment1_out result check error: %f\n",
              ndim, moment1_out_err);
      check_failed = true;
    }
    if(moment2_out_err >= 1e-6f) {
      fprintf(stderr, "[%d] moment2_out result check error: %f\n",
              ndim, moment2_out_err);
      check_failed = true;
    }
    if(param_out_err >= 1e-6f) {
      fprintf(stderr, "[%d] param_out result check error: %f\n",
              ndim, param_out_err);
      check_failed = true;
    }
    if(master_param_out_err >= 1e-6f) {
      fprintf(stderr, "[%d] master_param_out result check error: %f\n",
              ndim, master_param_out_err);
      check_failed = true;
    }
  #else
    auto moment1_out_err = moment1_out_old.CheckSame(moment1_out_new);
    auto moment2_out_err = moment2_out_old.CheckSame(moment2_out_new);
    auto param_out_err = param_out_old.CheckSame(param_out_new);
    auto master_param_out_err = master_param_out_old.CheckSame(master_param_out_new);

    if(moment1_out_err) {
      fprintf(stderr, "[%d] moment1_out result check same failed\n",
              ndim);
      check_failed = true;
    }
    if(moment2_out_err) {
      fprintf(stderr, "[%d] moment2_out result check same failed\n",
              ndim);
      check_failed = true;
    }
    if(param_out_err) {
      fprintf(stderr, "[%d] param_out result check same failed\n",
              ndim);
      check_failed = true;
    }
    if(master_param_out_err) {
      fprintf(stderr, "[%d] master_param_out result check same failed\n",
              ndim);
      check_failed = true;
    }
  #endif
    return check_failed;
  }

public:
  MallocHost<MT> h_moment1, h_moment2, h_master_param, h_lr;
  MallocDevice<MT> d_moment1, d_moment2, d_master_param, d_lr;
  MallocHost<T> h_param, h_grad;
  MallocDevice<T> d_param, d_grad;

  MallocDevice<MT> moment1_out_old, moment1_out_new;
  MallocDevice<MT> moment2_out_old, moment2_out_new;
  MallocDevice<MT> master_param_out_old, master_param_out_new;
  MallocDevice<T> param_out_old, param_out_new;

  MT beta1, beta2, beta1_pow, beta2_pow, epsilon;

private:
  int ndim;
};

template<typename T>
int TestAdamKernel(CUDAStream &context, int ndim) {
  using MT = typename GetAccType<T>::type;

  AllocParam<T, MT> params(ndim, context);

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<char*> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  name = "Adam Baseline Kernel";
  cost = TimeAdamKernelREG<T, MT>(context, 
                                params.beta1, params.beta2, params.epsilon, params.beta1_pow,
                                params.beta2_pow, params.d_moment1.data(), params.moment1_out_old.data(),
                                params.d_moment2.data(), params.moment2_out_old.data(), params.d_lr.data(),
                                params.d_grad.data(), params.d_param.data(), params.param_out_old.data(),
                                params.d_master_param.data(), params.master_param_out_old.data(),
                                ndim);
  AfterRun();

  name = "Adam Vec Kernel";
  cost = TimeVecAdamKernelREG<T, MT>(context, 
                                params.beta1, params.beta2, params.epsilon, params.beta1_pow,
                                params.beta2_pow, params.d_moment1.data(), params.moment1_out_new.data(),
                                params.d_moment2.data(), params.moment2_out_new.data(), params.d_lr.data(),
                                params.d_grad.data(), params.d_param.data(), params.param_out_new.data(),
                                params.d_master_param.data(), params.master_param_out_new.data(),
                                ndim);
  AfterRun();

  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "[%d] CUDA ERROR: %s\n",
            ndim, err);
    printf("*******************\n");
    return CUDA_FAILED;
  }

  bool check_failed = params.CheckResult();
  if(check_failed) {
    printf("*******************\n");
    return CHECK_FAILED;
  }

  printf("*******************\n");
  printf("[%d] SUCCESS\n", ndim);
  return SUCCESS;
}

int main() {
  CUDAStream context;
  srand(time(0));

  do {
    int ndim = rand() % 1000000 + 1000;
    if(!TestAdamKernel<float>(context, ndim) == SUCCESS) {
      fprintf(stderr, "Float Error\n");
      // break;
    }
    if(!TestAdamKernel<half>(context, ndim) == SUCCESS) {
      fprintf(stderr, "Half Error\n");
      // break;
    }
    printf("\n\n");
    printf("*******************************************\n");
    sleep(1);
  } while(false);
}
// header file
#include "reduce_sum_fp16.h"

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
template<typename Tx, typename Ty>
class AllocParam {
public:
  AllocParam(const int reduce_num, CUDAStream &context)
      : _context(context), _reduce_num(reduce_num),
        x_h(reduce_num, context), x(reduce_num, context),
        y_old(1, context), y_new(1, context) {
    // generate input size
    x_h.Random(0, 1);
    x.CopyFrom(x_h);

    result = Ty(0);
    if(std::is_same<Tx, half>::value) {
      float res_f = 0.0f;
      for(int i = 0; i < reduce_num; i ++)
        res_f += type2type<Tx, float>(x_h.data()[i]);
      result = type2type<float, Ty>(res_f);
    } else {
      for(int i = 0; i < reduce_num; i ++)
        result += x_h.data()[i];
    }
  }

  bool CheckResult() {
    return y_old.CheckSame(y_new);
  }

  MallocHost<Tx> x_h;
  MallocDevice<Tx> x;
  MallocDevice<Ty> y_old, y_new;
  Ty result;

private:
  CUDAStream &_context;
  const int _reduce_num;
};

/************************************************************************/
template <typename T>
struct IdentityFunctor {
  HOSTDEVICE explicit IdentityFunctor() {}

  HOSTDEVICE T operator()(const T& x) const { return x; }
};

/************************************************************************/

template<typename Tx, typename Ty, typename TransformOp, typename ReduceOp>
int TestKernel(CUDAStream &context,
               const int reduce_num,
               const TransformOp& transformer,
               const ReduceOp& reducer,
               const Ty& init) {
  AllocParam<Tx, Ty> params(reduce_num, context);

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<char*> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  name = "Cub Kernel";
  cost = TimeOfCubKernel(
            params.x.data(), params.y_old.data(),
            context, reducer, transformer,
            init, reduce_num);
  AfterRun();

  name = "Reduce Kernel";
  cost = TimeOfReduceKernel(
            params.x.data(), params.y_new.data(),
            context, reducer, transformer,
            init, reduce_num);
  AfterRun();

  name = "Grid Reduce Kernel";
  cost = TimeOfReduceKernelGrid(
            params.x.data(), params.y_new.data(),
            context, reducer, transformer,
            init, reduce_num);
  AfterRun();

  printf("*******************\n");
  auto err = context.sync();
  if(err != EMPTY_STRING) {
    fprintf(stderr, "[%d] CUDA ERROR: %s\n",
            reduce_num, err);
    return CUDA_FAILED;
  }

  if(reduce_num <= 100) {
    fprintf(stderr, "input X:\n");
    params.x_h.Print(1, reduce_num);

    fprintf(stderr, "Cpu result:\n");
    print(params.result, "\n");

    fprintf(stderr, "Old result:\n");
    params.y_old.Print(1, 1);

    fprintf(stderr, "New result:\n");
    params.y_new.Print(1, 1);
  }

  if(!params.CheckResult()) {
    fprintf(stderr, "[%d] Result Check Failed\n",
            reduce_num);
    return CHECK_FAILED;
  } else {
    printf("[%d] Success\n", reduce_num);
  }

  return SUCCESS;
}

int main() {
  srand(time(0));
  CUDAStream context;
  typedef half Tx;
  typedef half Ty;

  do {
    int reduce_num = 100;
    // printf("Please Input reduce_num\n");
    // std::cin >> reduce_num;
    if(TestKernel
          <Tx, Ty, IdentityFunctor<Tx>, cub::Sum>(
          context, reduce_num, IdentityFunctor<Tx>(),
          cub::Sum(), static_cast<Ty>(0)) != SUCCESS) break;
    printf("\n");
  } while(false);

  return 0;
}
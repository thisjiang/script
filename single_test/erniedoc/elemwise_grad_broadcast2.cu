#include "elemwise_grad_broadcast2.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>

#include "../common.h"


template <typename T, typename DX_OP, typename DY_OP>
int ElemwiseGradBroadcast(CUDAStream &context, 
                          AllocHost &h_x, AllocDevice &d_x,
                          AllocHost &h_y, AllocDevice &d_y,
                          AllocHost &h_out, AllocDevice &d_out,
                          AllocHost &h_dout, AllocDevice &d_dout,
                          int pre, int n, int post, bool is_xsize_larger,
                          DX_OP dx_op, DY_OP dy_op,
                          AllocDevice &dx_old, AllocDevice &dy_old,
                          AllocDevice &dx_new, AllocDevice &dy_new){
  auto clock = TimeOfKernel::get(context);

  // check data size
  int out_num = pre * n * post;
  size_t x_num = 0, y_num = 0;
  if(is_xsize_larger) {
    x_num = out_num;
    y_num = n;
  } else {
    x_num = n;
    y_num = out_num;
  }

  h_x.resize(x_num * sizeof(T), true);
  d_x.resize(x_num * sizeof(T), true);
  h_y.resize(y_num * sizeof(T), true);
  d_y.resize(y_num * sizeof(T), true);

  // generate rand number
  h_x.Random<T>(0, 1);
  h_y.Random<T>(0, 1);
  h_out.Random<T>(0, 1);
  h_dout.Random<T>(0, 1);

  // copy data
  d_x.CopyFrom(h_x);
  d_y.CopyFrom(h_y);
  d_out.CopyFrom(h_out);
  d_dout.CopyFrom(h_dout);

  dx_old.SetZero();
  dy_old.SetZero();
  dx_new.SetZero();
  dy_new.SetZero();

  // Set output size
  dx_old.resize(x_num * sizeof(T));
  dy_old.resize(y_num * sizeof(T));
  dx_new.resize(x_num * sizeof(T));
  dy_new.resize(y_num * sizeof(T));

  char* name;
  float cost;
  std::vector<float> costs;
  std::vector<char*> names;

#define AfterRun()  \
  printf("%s cost %f\n", name, cost); \
  costs.push_back(cost);  \
  names.push_back(name);

  // Initial kernel
  name = "Origin Kernel";
  cost = OldElemwiseGradBroadcast2CUDA(context, 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_old.data<T>(), dy_old.data<T>());
  AfterRun();

  name = "Vec by template";
  cost = NewElemwiseGradBroadcast2CUDA(context, 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_new.data<T>(), dy_new.data<T>());
  AfterRun();

  name = "Vec fix 4";
  cost = VectorizeElemwiseGradBroadcast2CUDA(context, 
                                d_x.data<T>(), d_y.data<T>(),
                                d_out.data<T>(), d_dout.data<T>(),
                                pre, n, post, is_xsize_larger,
                                dx_op, dy_op, 
                                dx_new.data<T>(), dy_new.data<T>());
  AfterRun();

  auto err_new = context.sync();
  if(err_new != "") {
    fprintf(stderr, "New ERROR: %s\n", err_new);
    return CUDA_FAILED;
  }

  // check result
  using AccT = typename GetAccType<T>::type;
  AccT dx_err = dx_old.MaxError<T>(dx_new);
  AccT dy_err = dy_old.MaxError<T>(dy_new);

  if(dx_err > 0 || dy_err > 0) {
    fprintf(stderr, "[%d, %d, %d, %d] Error: dx ", 
          pre, n, post, static_cast<int>(is_xsize_larger));
    fprint(dx_err);
    fprintf(stderr, " dy ");
    fprint(dy_err);
    fprintf(stderr, "\n");

    if(pre * post * n > 500) return CHECK_FAILED;

    fprintf(stderr, "dx_old\n");
    dx_old.Print<T>(1, x_num);
    fprintf(stderr, "dx_new\n");
    dx_new.Print<T>(1, x_num);
    fprintf(stderr, "dy_old\n");
    dy_old.Print<T>(1, y_num);
    fprintf(stderr, "dy_new\n");
    dy_new.Print<T>(1, y_num);

    return CHECK_FAILED;
  } else {
    printf("[%d, %d, %d, %d] Success!\n", 
          pre, n, post, static_cast<int>(is_xsize_larger));
  }
  return SUCCESS;
}

struct Param {
  int pre, post, n;
};

int main() {
    CUDAStream context;

    typedef float T;

    srand(time(NULL));

    int pre = 2, post = 3, n = 10000;
    bool is_xsize_larger = true;

    do {
      pre = rand() % 8 + 1;
      post = rand() % 8 + 1;
      n = rand() % 10000 + 10000;

      // printf("Please Input dims [pre, n, post]\n");
      // std::cin >> pre >> n >> post;

      printf("shape = [%d, %d, %d]\n", pre, n, post);
      int max_num = pre * post * n * sizeof(T);

      AllocHost h_x(max_num, context), h_y(max_num, context),
                h_out(max_num, context), h_dout(max_num, context);
      AllocDevice d_x(max_num, context), d_y(max_num, context),
                  d_out(max_num, context), d_dout(max_num, context);
      AllocDevice dx_old(max_num, context), dy_old(max_num, context),
                  dx_new(max_num, context), dy_new(max_num, context);

      is_xsize_larger = true;
      int res = ElemwiseGradBroadcast<T, MulGradDX<T>, MulGradDY<T>>
                            (context, h_x, d_x, h_y, d_y,
                            h_out, d_out, h_dout, d_dout,
                            pre, n, post, is_xsize_larger,
                            MulGradDX<T>(), MulGradDY<T>(),
                            dx_old, dy_old, dx_new, dy_new);
      if(res == CUDA_FAILED) {
        fprintf(stderr, "Compute Failed with CUDA error\n");
        return 1;
      }
      printf("\n");
      is_xsize_larger = false;
      res = ElemwiseGradBroadcast<T, MulGradDX<T>, MulGradDY<T>>
                            (context, h_x, d_x, h_y, d_y,
                            h_out, d_out, h_dout, d_dout,
                            pre, n, post, is_xsize_larger,
                            MulGradDX<T>(), MulGradDY<T>(),
                            dx_old, dy_old, dx_new, dy_new);
      if(res == CUDA_FAILED) {
        fprintf(stderr, "Compute Failed with CUDA error\n");
        return 1;
      }
      printf("**************************************************\n");
    } while(true);

    return 0;
}
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

template <typename T, typename MT>
__global__ void AdamKernelREG(MT beta1, MT beta2, MT epsilon, MT beta1_pow_,
                              MT beta2_pow_, const MT* moment1, MT* moment1_out,
                              const MT* moment2, MT* moment2_out, const MT* lr_,
                              const T* grad, const T* param, T* param_out,
                              const MT* master_param, MT* master_param_out,
                              int ndim) {
  MT lr = *lr_;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
        (static_cast<MT>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = moment1[id];
    MT mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
    p -= lr * (mom1 /
               (sqrt(mom2) + epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template<typename T, typename MT>
float TimeAdamKernelREG(CUDAStream &dev_ctx,
                        MT beta1, MT beta2, MT epsilon, MT beta1_pow_,
                        MT beta2_pow_, const MT* moment1, MT* moment1_out,
                        const MT* moment2, MT* moment2_out, const MT* lr_,
                        const T* grad, const T* param, T* param_out,
                        const MT* master_param, MT* master_param_out,
                        int ndim) {
  auto tt = TimeOfKernel::get(dev_ctx);

  int threads = 512;
  int blocks = (ndim + threads - 1) / threads;

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    AdamKernelREG<T, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
      beta1, beta2, epsilon, beta1_pow_,
      beta2_pow_, moment1, moment1_out,
      moment2, moment2_out, lr_,
      grad, param, param_out,
      master_param, master_param_out,
      ndim);
  }
  float cost = tt->stop();
  return cost;
}

/*************************************************************************/
template <typename T, typename MT, int vec_size>
__global__ void VecAdamKernelREG(MT beta1, MT beta2, MT epsilon, MT beta1_pow_,
                              MT beta2_pow_, const MT* moment1, MT* moment1_out,
                              const MT* moment2, MT* moment2_out, const MT* lr_,
                              const T* grad, const T* param, T* param_out,
                              const MT* master_param, MT* master_param_out,
                              int ndim) {
  MT lr = *lr_;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  lr *= sqrt(static_cast<MT>(1.0f) - beta2_pow) /
        (static_cast<MT>(1.0f) - beta1_pow);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  int vec_num = ndim / vec_size;
  int remainder = ndim - vec_num * vec_size;

  using VecT = typename GetVecType<T, vec_size>::type;
  using VecMT = typename GetVecType<MT, vec_size>::type;

  VecMT moment1_vec, moment2_vec;
  VecT grad_vec;

  MT *moment1_vec_data = reinterpret_cast<MT*>(&moment1_vec);
  MT *moment2_vec_data = reinterpret_cast<MT*>(&moment2_vec);
  T *grad_vec_data = reinterpret_cast<T*>(&grad_vec);

  VecMT p_vec;
  MT *p_vec_data = reinterpret_cast<MT*>(&p_vec);

  for(int vec_id = tid; vec_id < vec_num; vec_id += stride) {
    int id = vec_id * vec_size;
#pragma unroll
    for(int i = 0; i < vec_size; i ++) {
      p_vec_data[i] = master_param ? master_param[id + i] : static_cast<MT>(param[id + i]);
    }

    grad_vec = *reinterpret_cast<const VecT*>(&grad[id]);
    moment1_vec = *reinterpret_cast<const VecMT*>(&moment1[id]);
    moment2_vec = *reinterpret_cast<const VecMT*>(&moment2[id]);
#pragma unroll
    for(int i = 0; i < vec_size; i ++) {
      MT g = static_cast<MT>(grad_vec_data[i]);
      moment1_vec_data[i] = beta1 * moment1_vec_data[i] +
                            (static_cast<MT>(1.0f) - beta1) * g;
      moment2_vec_data[i] = beta2 * moment2_vec_data[i] +
                            (static_cast<MT>(1.0f) - beta2) * g * g;
      p_vec_data[i] -= lr * (moment1_vec_data[i] /
                  (sqrt(moment2_vec_data[i]) +
                  epsilon * sqrt(static_cast<MT>(1.0f) - beta2_pow)));
    }
    *reinterpret_cast<VecMT*>(&moment1_out[id]) = moment1_vec;
    *reinterpret_cast<VecMT*>(&moment2_out[id]) = moment2_vec;
#pragma unroll
    for(int i = 0; i < vec_size; i ++) {
      param_out[id + i] = static_cast<T>(p_vec_data[i]);
    }
    if (master_param_out) {
      *reinterpret_cast<VecMT*>(&master_param_out[id]) = p_vec;
    }
  }

  if(tid == 0 && remainder != 0) {
    for(int id = ndim -  remainder; id < ndim; id ++) {
      MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
      MT g = static_cast<MT>(grad[id]);
      MT mom1 = moment1[id];
      MT mom2 = moment2[id];
      mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
      mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
      p -= lr * (mom1 /
                (sqrt(mom2) + epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

      moment1_out[id] = mom1;
      moment2_out[id] = mom2;
      param_out[id] = static_cast<T>(p);
      if (master_param_out) {
        master_param_out[id] = p;
      }
    }
  }
}

template<typename T, typename MT>
float TimeVecAdamKernelREG(CUDAStream &dev_ctx,
                        MT beta1, MT beta2, MT epsilon, MT beta1_pow_,
                        MT beta2_pow_, const MT* moment1, MT* moment1_out,
                        const MT* moment2, MT* moment2_out, const MT* lr_,
                        const T* grad, const T* param, T* param_out,
                        const MT* master_param, MT* master_param_out,
                        int ndim) {
  auto tt = TimeOfKernel::get(dev_ctx);

  constexpr int vec_size = 4;

  int threads = 512;
  int blocks = (ndim + threads * vec_size - 1) / (threads * vec_size);

  tt->start();
#pragma unroll
  for(int i = 0; i < LOOPNUM; i ++) {
    VecAdamKernelREG<T, MT, vec_size><<<blocks, threads, 0, dev_ctx.stream()>>>(
      beta1, beta2, epsilon, beta1_pow_,
      beta2_pow_, moment1, moment1_out,
      moment2, moment2_out, lr_,
      grad, param, param_out,
      master_param, master_param_out,
      ndim);
  }
  float cost = tt->stop();
  return cost;
}
2021-12-01 11:31:01
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
A new filed (is_distributed) detected!
[2021/12/01 03:31:04] root INFO: 
===========================================================
==        PaddleClas is powered by PaddlePaddle !        ==
===========================================================
==                                                       ==
==   For more info please go to the following website.   ==
==                                                       ==
==       https://github.com/PaddlePaddle/PaddleClas      ==
===========================================================

[2021/12/01 03:31:04] root INFO: Arch : 
[2021/12/01 03:31:04] root INFO:     class_num : 1000
[2021/12/01 03:31:04] root INFO:     name : ResNet50
[2021/12/01 03:31:04] root INFO: DataLoader : 
[2021/12/01 03:31:04] root INFO:     Eval : 
[2021/12/01 03:31:04] root INFO:         dataset : 
[2021/12/01 03:31:04] root INFO:             cls_label_path : ./dataset/ILSVRC2012/val_list.txt
[2021/12/01 03:31:04] root INFO:             image_root : ./dataset/ILSVRC2012/
[2021/12/01 03:31:04] root INFO:             name : ImageNetDataset
[2021/12/01 03:31:04] root INFO:             transform_ops : 
[2021/12/01 03:31:04] root INFO:                 DecodeImage : 
[2021/12/01 03:31:04] root INFO:                     channel_first : False
[2021/12/01 03:31:04] root INFO:                     to_rgb : True
[2021/12/01 03:31:04] root INFO:                 ResizeImage : 
[2021/12/01 03:31:04] root INFO:                     resize_short : 256
[2021/12/01 03:31:04] root INFO:                 CropImage : 
[2021/12/01 03:31:04] root INFO:                     size : 224
[2021/12/01 03:31:04] root INFO:                 NormalizeImage : 
[2021/12/01 03:31:04] root INFO:                     mean : [0.485, 0.456, 0.406]
[2021/12/01 03:31:04] root INFO:                     order : 
[2021/12/01 03:31:04] root INFO:                     scale : 1.0/255.0
[2021/12/01 03:31:04] root INFO:                     std : [0.229, 0.224, 0.225]
[2021/12/01 03:31:04] root INFO:         loader : 
[2021/12/01 03:31:04] root INFO:             num_workers : 4
[2021/12/01 03:31:04] root INFO:             use_shared_memory : True
[2021/12/01 03:31:04] root INFO:         sampler : 
[2021/12/01 03:31:04] root INFO:             batch_size : 64
[2021/12/01 03:31:04] root INFO:             drop_last : False
[2021/12/01 03:31:04] root INFO:             name : DistributedBatchSampler
[2021/12/01 03:31:04] root INFO:             shuffle : False
[2021/12/01 03:31:04] root INFO:     Train : 
[2021/12/01 03:31:04] root INFO:         dataset : 
[2021/12/01 03:31:04] root INFO:             cls_label_path : ./dataset/ILSVRC2012/train_list.txt
[2021/12/01 03:31:04] root INFO:             image_root : ./dataset/ILSVRC2012/
[2021/12/01 03:31:04] root INFO:             name : ImageNetDataset
[2021/12/01 03:31:04] root INFO:             transform_ops : 
[2021/12/01 03:31:04] root INFO:                 DecodeImage : 
[2021/12/01 03:31:04] root INFO:                     channel_first : False
[2021/12/01 03:31:04] root INFO:                     to_rgb : True
[2021/12/01 03:31:04] root INFO:                 RandCropImage : 
[2021/12/01 03:31:04] root INFO:                     size : 224
[2021/12/01 03:31:04] root INFO:                 RandFlipImage : 
[2021/12/01 03:31:04] root INFO:                     flip_code : 1
[2021/12/01 03:31:04] root INFO:                 NormalizeImage : 
[2021/12/01 03:31:04] root INFO:                     mean : [0.485, 0.456, 0.406]
[2021/12/01 03:31:04] root INFO:                     order : 
[2021/12/01 03:31:04] root INFO:                     scale : 1.0/255.0
[2021/12/01 03:31:04] root INFO:                     std : [0.229, 0.224, 0.225]
[2021/12/01 03:31:04] root INFO:         loader : 
[2021/12/01 03:31:04] root INFO:             num_workers : 4
[2021/12/01 03:31:04] root INFO:             use_shared_memory : True
[2021/12/01 03:31:04] root INFO:         sampler : 
[2021/12/01 03:31:04] root INFO:             batch_size : 32
[2021/12/01 03:31:04] root INFO:             drop_last : False
[2021/12/01 03:31:04] root INFO:             name : DistributedBatchSampler
[2021/12/01 03:31:04] root INFO:             shuffle : True
[2021/12/01 03:31:04] root INFO: Global : 
[2021/12/01 03:31:04] root INFO:     checkpoints : None
[2021/12/01 03:31:04] root INFO:     device : gpu
[2021/12/01 03:31:04] root INFO:     epochs : 120
[2021/12/01 03:31:04] root INFO:     eval_during_train : False
[2021/12/01 03:31:04] root INFO:     eval_interval : 1
[2021/12/01 03:31:04] root INFO:     image_shape : [3, 224, 224]
[2021/12/01 03:31:04] root INFO:     is_distributed : False
[2021/12/01 03:31:04] root INFO:     output_dir : ./output/
[2021/12/01 03:31:04] root INFO:     pretrained_model : None
[2021/12/01 03:31:04] root INFO:     print_batch_step : 10
[2021/12/01 03:31:04] root INFO:     save_inference_dir : ./inference
[2021/12/01 03:31:04] root INFO:     save_interval : 1
[2021/12/01 03:31:04] root INFO:     to_static : False
[2021/12/01 03:31:04] root INFO:     use_dali : False
[2021/12/01 03:31:04] root INFO:     use_visualdl : False
[2021/12/01 03:31:04] root INFO: Infer : 
[2021/12/01 03:31:04] root INFO:     PostProcess : 
[2021/12/01 03:31:04] root INFO:         class_id_map_file : ppcls/utils/imagenet1k_label_list.txt
[2021/12/01 03:31:04] root INFO:         name : Topk
[2021/12/01 03:31:04] root INFO:         topk : 5
[2021/12/01 03:31:04] root INFO:     batch_size : 10
[2021/12/01 03:31:04] root INFO:     infer_imgs : docs/images/whl/demo.jpg
[2021/12/01 03:31:04] root INFO:     transforms : 
[2021/12/01 03:31:04] root INFO:         DecodeImage : 
[2021/12/01 03:31:04] root INFO:             channel_first : False
[2021/12/01 03:31:04] root INFO:             to_rgb : True
[2021/12/01 03:31:04] root INFO:         ResizeImage : 
[2021/12/01 03:31:04] root INFO:             resize_short : 256
[2021/12/01 03:31:04] root INFO:         CropImage : 
[2021/12/01 03:31:04] root INFO:             size : 224
[2021/12/01 03:31:04] root INFO:         NormalizeImage : 
[2021/12/01 03:31:04] root INFO:             mean : [0.485, 0.456, 0.406]
[2021/12/01 03:31:04] root INFO:             order : 
[2021/12/01 03:31:04] root INFO:             scale : 1.0/255.0
[2021/12/01 03:31:04] root INFO:             std : [0.229, 0.224, 0.225]
[2021/12/01 03:31:04] root INFO:         ToCHWImage : None
[2021/12/01 03:31:04] root INFO: Loss : 
[2021/12/01 03:31:04] root INFO:     Eval : 
[2021/12/01 03:31:04] root INFO:         CELoss : 
[2021/12/01 03:31:04] root INFO:             weight : 1.0
[2021/12/01 03:31:04] root INFO:     Train : 
[2021/12/01 03:31:04] root INFO:         CELoss : 
[2021/12/01 03:31:04] root INFO:             weight : 1.0
[2021/12/01 03:31:04] root INFO: Metric : 
[2021/12/01 03:31:04] root INFO:     Eval : 
[2021/12/01 03:31:04] root INFO:         TopkAcc : 
[2021/12/01 03:31:04] root INFO:             topk : [1, 5]
[2021/12/01 03:31:04] root INFO:     Train : 
[2021/12/01 03:31:04] root INFO:         TopkAcc : 
[2021/12/01 03:31:04] root INFO:             topk : [1, 5]
[2021/12/01 03:31:04] root INFO: Optimizer : 
[2021/12/01 03:31:04] root INFO:     lr : 
[2021/12/01 03:31:04] root INFO:         decay_epochs : [30, 60, 90]
[2021/12/01 03:31:04] root INFO:         learning_rate : 0.0125
[2021/12/01 03:31:04] root INFO:         name : Piecewise
[2021/12/01 03:31:04] root INFO:         values : [0.1, 0.01, 0.001, 0.0001]
[2021/12/01 03:31:04] root INFO:     momentum : 0.9
[2021/12/01 03:31:04] root INFO:     name : Momentum
[2021/12/01 03:31:04] root INFO:     regularizer : 
[2021/12/01 03:31:04] root INFO:         coeff : 0.0001
[2021/12/01 03:31:04] root INFO:         name : L2
W1201 03:31:14.362438 25336 device_context.cc:451] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.4, Runtime API Version: 11.2
W1201 03:31:14.362490 25336 device_context.cc:469] device: 0, cuDNN Version: 8.1.
I1201 03:31:20.678622 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 3 230 230 64 3 7 7 32 64 112 112
I1201 03:31:23.584873 25408 compiler.cc:73] [CUDA] host module:
Module module_host {

function fn_conv2d_0 (args__ptr, num_args)
{
  fn_conv2d_0_kernel(args__ptr, num_args)
}
function fn_reduce_sum_3 (args__ptr, num_args)
{
  fn_reduce_sum_3_kernel(args__ptr, num_args)
}
function fn_reduce_sum_4 (args__ptr, num_args)
{
  fn_reduce_sum_4_kernel(args__ptr, num_args)
}
function fn_identity_9_elementwise_mul_10_reduce_sum_11_fused (args__ptr, num_args)
{
  fn_identity_9_elementwise_mul_10_reduce_sum_11_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_1_broadcast_to_2_divide_5_fused (args__ptr, num_args)
{
  fn_const_scalar_1_broadcast_to_2_divide_5_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_12 (args__ptr, num_args)
{
  fn_reduce_sum_12_kernel(args__ptr, num_args)
}
function fn_const_scalar_28_const_scalar_30_broadcast_to_29_broadcast_to_31_elementwise_mul_33_elementwise_mul_32_elementwise_add_34_fused (args__ptr, num_args)
{
  fn_const_scalar_28_const_scalar_30_broadcast_to_29_broadcast_to_31_elementwise_mul_33_elementwise_mul_32_elementwise_add_34_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_7_broadcast_to_8_identity_14_elementwise_mul_15_divide_13_substract_16_fused (args__ptr, num_args)
{
  fn_const_scalar_7_broadcast_to_8_identity_14_elementwise_mul_15_divide_13_substract_16_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_35_const_scalar_37_broadcast_to_36_broadcast_to_38_elementwise_mul_40_elementwise_mul_39_elementwise_add_41_fused (args__ptr, num_args)
{
  fn_const_scalar_35_const_scalar_37_broadcast_to_36_broadcast_to_38_elementwise_mul_40_elementwise_mul_39_elementwise_add_41_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_17_const_scalar_42_broadcast_to_22_broadcast_to_23_broadcast_to_18_broadcast_to_43_broadcast_to_6_substract_24_broadcast_to_19_elementwise_add_20_rsqrt_21_elementwise_mul_25_elementwise_mul_26_elementwise_add_27_max_44_fused (args__ptr, num_args)
{
  fn_const_scalar_17_const_scalar_42_broadcast_to_22_broadcast_to_23_broadcast_to_18_broadcast_to_43_broadcast_to_6_substract_24_broadcast_to_19_elementwise_add_20_rsqrt_21_elementwise_mul_25_elementwise_mul_26_elementwise_add_27_max_44_fused_kernel(args__ptr, num_args)
}


}
I1201 03:31:23.584986 25408 compiler.cc:76] [CUDA] device module:
Module module_gpu_device {

function fn_conv2d_0_kernel (_data, _conv2d_0__w_0, _Conv2d_nchw_out)
{
  for (i, 0, 32)
  {
    if ((blockIdx.z < 4)) {
      if ((blockIdx.y < 112)) {
        if ((threadIdx.z < 8)) {
          if ((threadIdx.x < 112)) {
            {
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_write_cache__reduce_init[0, j_inner, 0, 0] = 0
                for (rc_outer, 0, 3)
                {
                  for (ry, 0, 7)
                  {
                    for (rx, 0, 7)
                    {
                      Conv2d_nchw_out_write_cache[0, j_inner, 0, 0] = (Conv2d_nchw_out_write_cache[0, j_inner, 0, 0] + (select(((((((blockIdx.y * 2) + (ry * 1)) >= 3) and (((blockIdx.y * 2) + (ry * 1)) < (224 + 3))) and (((threadIdx.x * 2) + (rx * 1)) >= 3)) and (((threadIdx.x * 2) + (rx * 1)) < (224 + 3))), data[i, rc_outer, (-3 + ((2 * blockIdx.y) + ry)), (-3 + ((2 * threadIdx.x) + rx))], 0) * conv2d_0__w_0[((16 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), rc_outer, ry, rx]))
                    }
                  }
                }
              }
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out[i, ((16 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), blockIdx.y, threadIdx.x] = Conv2d_nchw_out_write_cache[0, j_inner, 0, 0]
              }
            }
          }
        }
      }
    }
  }
}
function fn_reduce_sum_3_kernel (_var_3, _reduce_sum_out)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 112)) {
      {
        reduce_sum_out__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk, 0, 32)
        {
          for (kk_0, 0, 112)
          {
            reduce_sum_out[blockIdx.x, threadIdx.x] = (reduce_sum_out[blockIdx.x, threadIdx.x] + var_3[kk, blockIdx.x, kk_0, threadIdx.x])
          }
        }
      }
    }
  }
}
function fn_reduce_sum_4_kernel (_var_26, _reduce_sum_out_0)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_0__reduce_init[threadIdx.x] = 0
      for (kk_1, 0, 112)
      {
        reduce_sum_out_0[threadIdx.x] = (reduce_sum_out_0[threadIdx.x] + var_26[threadIdx.x, kk_1])
      }
    }
  }
}
function fn_identity_9_elementwise_mul_10_reduce_sum_11_fused_kernel (_var_3, _reduce_sum_out_1)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 112)) {
      {
        reduce_sum_out_1__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_2, 0, 32)
        {
          for (kk_3, 0, 112)
          {
            reduce_sum_out_1[blockIdx.x, threadIdx.x] = (reduce_sum_out_1[blockIdx.x, threadIdx.x] + (var_3[kk_2, blockIdx.x, kk_3, threadIdx.x] * var_3[kk_2, blockIdx.x, kk_3, threadIdx.x]))
          }
        }
      }
    }
  }
}
function fn_const_scalar_1_broadcast_to_2_divide_5_fused_kernel (_var_28, _divide_Out)
{
  if ((threadIdx.x < 64)) {
    divide_Out[threadIdx.x] = (2.49123e-06 * var_28[threadIdx.x])
  }
}
function fn_reduce_sum_12_kernel (_var_36, _reduce_sum_out_2)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_2__reduce_init[threadIdx.x] = 0
      for (kk_4, 0, 112)
      {
        reduce_sum_out_2[threadIdx.x] = (reduce_sum_out_2[threadIdx.x] + var_36[threadIdx.x, kk_4])
      }
    }
  }
}
function fn_const_scalar_28_const_scalar_30_broadcast_to_29_broadcast_to_31_elementwise_mul_33_elementwise_mul_32_elementwise_add_34_fused_kernel (_batch_norm_0__w_1, _var_14, _elementwise_add_Out)
{
  if ((threadIdx.x < 64)) {
    elementwise_add_Out[threadIdx.x] = ((0.9 * batch_norm_0__w_1[threadIdx.x]) + (0.1 * var_14[threadIdx.x]))
  }
}
function fn_const_scalar_7_broadcast_to_8_identity_14_elementwise_mul_15_divide_13_substract_16_fused_kernel (_var_14, _var_38, _substract_Out)
{
  if ((threadIdx.x < 64)) {
    substract_Out[threadIdx.x] = ((2.49123e-06 * var_38[threadIdx.x]) - (var_14[threadIdx.x] * var_14[threadIdx.x]))
  }
}
function fn_const_scalar_35_const_scalar_37_broadcast_to_36_broadcast_to_38_elementwise_mul_40_elementwise_mul_39_elementwise_add_41_fused_kernel (_batch_norm_0__w_2, _var_15, _elementwise_add_Out_0)
{
  if ((threadIdx.x < 64)) {
    elementwise_add_Out_0[threadIdx.x] = ((0.9 * batch_norm_0__w_2[threadIdx.x]) + (0.1 * var_15[threadIdx.x]))
  }
}
function fn_const_scalar_17_const_scalar_42_broadcast_to_22_broadcast_to_23_broadcast_to_18_broadcast_to_43_broadcast_to_6_substract_24_broadcast_to_19_elementwise_add_20_rsqrt_21_elementwise_mul_25_elementwise_mul_26_elementwise_add_27_max_44_fused_kernel (_batch_norm_0__w_0, _batch_norm_0__b_0, _var_14, _var_3, _var_15, _max_Out)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, 98)
      {
        max_Out[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] = cinn_max(((batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (rsqrt((var_15[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)) * var_3[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))])) + ((-1 * (batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (rsqrt((var_15[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)) * var_14[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)]))) + batch_norm_0__b_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)])), 0)
      }
    }
  }
}


}
I1201 03:31:23.931758 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(896) fn_conv2d_0_kernel(const float* __restrict__ data, const float* __restrict__ conv2d_0__w_0, float* __restrict__ Conv2d_nchw_out)
{
  float _Conv2d_nchw_out_write_cache [ 2 ];
  float* Conv2d_nchw_out_write_cache = _Conv2d_nchw_out_write_cache;
  float* Conv2d_nchw_out_write_cache__reduce_init = _Conv2d_nchw_out_write_cache;
  for (int32_t i = 0; i < 32; i += 1) {
    if (((int)blockIdx.z < 4)) {
      if (((int)blockIdx.y < 112)) {
        if (((int)threadIdx.z < 8)) {
          if (((int)threadIdx.x < 112)) {
          {
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_write_cache__reduce_init[j_inner] = 0;
              for (int32_t rc_outer = 0; rc_outer < 3; rc_outer += 1) {
                for (int32_t ry = 0; ry < 7; ry += 1) {
                  for (int32_t rx = 0; rx < 7; rx += 1) {
                    Conv2d_nchw_out_write_cache[j_inner] = (Conv2d_nchw_out_write_cache[j_inner] + ((((((((((int)blockIdx.y * 2) + (ry * 1)) >= 3) && ((((int)blockIdx.y * 2) + (ry * 1)) < (224 + 3))) && ((((int)threadIdx.x * 2) + (rx * 1)) >= 3)) && ((((int)threadIdx.x * 2) + (rx * 1)) < (224 + 3)))) ? data[(-675 + ((448 * (int)blockIdx.y) + ((150528 * i) + ((50176 * rc_outer) + ((224 * ry) + ((2 * (int)threadIdx.x) + rx))))))] : 0) * conv2d_0__w_0[((2352 * (int)blockIdx.z) + ((147 * j_inner) + ((49 * rc_outer) + ((7 * ry) + ((294 * (int)threadIdx.z) + rx)))))]));
                  };
                };
              };
            };
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out[((112 * (int)blockIdx.y) + ((200704 * (int)blockIdx.z) + ((802816 * i) + ((12544 * j_inner) + ((25088 * (int)threadIdx.z) + (int)threadIdx.x)))))] = Conv2d_nchw_out_write_cache[j_inner];
            };
          }
          };
        };
      };
    };
  };
}__global__
void __launch_bounds__(112) fn_reduce_sum_3_kernel(const float* __restrict__ var_3, float* __restrict__ reduce_sum_out)
{
  float* reduce_sum_out__reduce_init = reduce_sum_out;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 112)) {
    {
      reduce_sum_out__reduce_init[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk = 0; kk < 32; kk += 1) {
        for (int32_t kk_0 = 0; kk_0 < 112; kk_0 += 1) {
          reduce_sum_out[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out[((112 * (int)blockIdx.x) + (int)threadIdx.x)] + var_3[((12544 * (int)blockIdx.x) + ((802816 * kk) + ((112 * kk_0) + (int)threadIdx.x)))]);
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_4_kernel(const float* __restrict__ var_26, float* __restrict__ reduce_sum_out_0)
{
  float* reduce_sum_out_0__reduce_init = reduce_sum_out_0;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_0__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_1 = 0; kk_1 < 112; kk_1 += 1) {
      reduce_sum_out_0[(int)threadIdx.x] = (reduce_sum_out_0[(int)threadIdx.x] + var_26[((112 * (int)threadIdx.x) + kk_1)]);
    };
  }
  };
}__global__
void __launch_bounds__(112) fn_identity_9_elementwise_mul_10_reduce_sum_11_fused_kernel(const float* __restrict__ var_3, float* __restrict__ reduce_sum_out_1)
{
  float* reduce_sum_out_1__reduce_init = reduce_sum_out_1;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 112)) {
    {
      reduce_sum_out_1__reduce_init[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_2 = 0; kk_2 < 32; kk_2 += 1) {
        for (int32_t kk_3 = 0; kk_3 < 112; kk_3 += 1) {
          reduce_sum_out_1[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_1[((112 * (int)blockIdx.x) + (int)threadIdx.x)] + (var_3[((12544 * (int)blockIdx.x) + ((802816 * kk_2) + ((112 * kk_3) + (int)threadIdx.x)))] * var_3[((12544 * (int)blockIdx.x) + ((802816 * kk_2) + ((112 * kk_3) + (int)threadIdx.x)))]));
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_1_broadcast_to_2_divide_5_fused_kernel(const float* __restrict__ var_28, float* __restrict__ divide_Out)
{
  if (((int)threadIdx.x < 64)) {
    divide_Out[(int)threadIdx.x] = (2.49123e-06 * var_28[(int)threadIdx.x]);
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_12_kernel(const float* __restrict__ var_36, float* __restrict__ reduce_sum_out_2)
{
  float* reduce_sum_out_2__reduce_init = reduce_sum_out_2;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_2__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_4 = 0; kk_4 < 112; kk_4 += 1) {
      reduce_sum_out_2[(int)threadIdx.x] = (reduce_sum_out_2[(int)threadIdx.x] + var_36[((112 * (int)threadIdx.x) + kk_4)]);
    };
  }
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_28_const_scalar_30_broadcast_to_29_broadcast_to_31_elementwise_mul_33_elementwise_mul_32_elementwise_add_34_fused_kernel(const float* __restrict__ batch_norm_0__w_1, const float* __restrict__ var_14, float* __restrict__ elementwise_add_Out)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_add_Out[(int)threadIdx.x] = ((0.9 * batch_norm_0__w_1[(int)threadIdx.x]) + (0.1 * var_14[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_7_broadcast_to_8_identity_14_elementwise_mul_15_divide_13_substract_16_fused_kernel(const float* __restrict__ var_14, const float* __restrict__ var_38, float* __restrict__ substract_Out)
{
  if (((int)threadIdx.x < 64)) {
    substract_Out[(int)threadIdx.x] = ((2.49123e-06 * var_38[(int)threadIdx.x]) - (var_14[(int)threadIdx.x] * var_14[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_35_const_scalar_37_broadcast_to_36_broadcast_to_38_elementwise_mul_40_elementwise_mul_39_elementwise_add_41_fused_kernel(const float* __restrict__ batch_norm_0__w_2, const float* __restrict__ var_15, float* __restrict__ elementwise_add_Out_0)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_add_Out_0[(int)threadIdx.x] = ((0.9 * batch_norm_0__w_2[(int)threadIdx.x]) + (0.1 * var_15[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(1024) fn_const_scalar_17_const_scalar_42_broadcast_to_22_broadcast_to_23_broadcast_to_18_broadcast_to_43_broadcast_to_6_substract_24_broadcast_to_19_elementwise_add_20_rsqrt_21_elementwise_mul_25_elementwise_mul_26_elementwise_add_27_max_44_fused_kernel(const float* __restrict__ batch_norm_0__w_0, const float* __restrict__ batch_norm_0__b_0, const float* __restrict__ var_14, const float* __restrict__ var_3, const float* __restrict__ var_15, float* __restrict__ max_Out)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < 98; i_j_fused_k_fused_a_fused_outer += 1) {
        max_Out[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] = cinn_nvgpu_max_fp32(((batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (rsqrt((var_15[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)) * var_3[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))])) + ((-1 * (batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (rsqrt((var_15[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)) * var_14[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)]))) + batch_norm_0__b_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)])), 0);
      };
    };
  };
}

}
I1201 03:31:23.934296 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
I1201 03:31:26.272509 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 56 56 256 64 1 1 32 256 56 56
I1201 03:31:27.196255 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 56 56 64 64 1 1 32 64 56 56
I1201 03:31:29.390262 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 64 58 58 64 64 3 3 32 64 56 56
I1201 03:31:29.393805 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 58 58 64 64 3 3 32 64 56 56
I1201 03:31:31.408644 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 56 56 256 64 1 1 32 256 56 56
I1201 03:31:35.256006 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 56 56 64 256 1 1 32 64 56 56
I1201 03:31:37.229835 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 64 58 58 64 64 3 3 32 64 56 56
I1201 03:31:37.233323 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 58 58 64 64 3 3 32 64 56 56
I1201 03:31:39.239567 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 56 56 256 64 1 1 32 256 56 56
I1201 03:31:42.064908 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 56 56 64 256 1 1 32 64 56 56
I1201 03:31:44.022848 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 64 58 58 64 64 3 3 32 64 56 56
I1201 03:31:44.026345 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 58 58 64 64 3 3 32 64 56 56
I1201 03:31:46.026661 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 64 56 56 256 64 1 1 32 256 56 56
I1201 03:31:48.851321 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 56 56 128 256 1 1 32 128 56 56
I1201 03:31:49.766079 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 56 56 512 256 1 1 32 512 28 28
I1201 03:31:52.027139 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 58 58 128 128 3 3 32 128 28 28
I1201 03:31:54.034723 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 28 28 512 128 1 1 32 512 28 28
I1201 03:31:57.446014 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 28 28 128 512 1 1 32 128 28 28
I1201 03:31:59.499769 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 128 30 30 128 128 3 3 32 128 28 28
I1201 03:31:59.503269 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 30 30 128 128 3 3 32 128 28 28
I1201 03:32:01.514506 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 28 28 512 128 1 1 32 512 28 28
I1201 03:32:04.081689 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 28 28 128 512 1 1 32 128 28 28
I1201 03:32:06.030527 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 128 30 30 128 128 3 3 32 128 28 28
I1201 03:32:06.033984 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 30 30 128 128 3 3 32 128 28 28
I1201 03:32:08.044147 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 28 28 512 128 1 1 32 512 28 28
I1201 03:32:10.600977 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 28 28 128 512 1 1 32 128 28 28
I1201 03:32:12.564077 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 128 30 30 128 128 3 3 32 128 28 28
I1201 03:32:12.567616 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 30 30 128 128 3 3 32 128 28 28
I1201 03:32:14.581682 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 128 28 28 512 128 1 1 32 512 28 28
I1201 03:32:17.145094 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 28 28 1024 512 1 1 32 1024 14 14
I1201 03:32:18.049397 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 28 28 256 512 1 1 32 256 28 28
I1201 03:32:20.303815 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 30 30 256 256 3 3 32 256 14 14
I1201 03:32:22.288280 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 14 14 1024 256 1 1 32 1024 14 14
I1201 03:32:26.033980 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 256 1024 1 1 32 256 14 14
I1201 03:32:27.992290 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:27.995831 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:29.978171 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 14 14 1024 256 1 1 32 1024 14 14
I1201 03:32:32.782605 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 256 1024 1 1 32 256 14 14
I1201 03:32:34.719066 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:34.722553 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:36.692699 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 14 14 1024 256 1 1 32 1024 14 14
I1201 03:32:39.497040 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 256 1024 1 1 32 256 14 14
I1201 03:32:41.451270 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:41.454746 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:43.449793 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 14 14 1024 256 1 1 32 1024 14 14
I1201 03:32:46.269007 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 256 1024 1 1 32 256 14 14
I1201 03:32:48.208557 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:48.211977 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:50.199674 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 14 14 1024 256 1 1 32 1024 14 14
I1201 03:32:53.002326 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 256 1024 1 1 32 256 14 14
I1201 03:32:54.933966 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:54.937443 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 16 16 256 256 3 3 32 256 14 14
I1201 03:32:56.918126 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 256 14 14 1024 256 1 1 32 1024 14 14
I1201 03:32:59.731729 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 512 1024 1 1 32 512 14 14
I1201 03:33:00.636890 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 1024 14 14 2048 1024 1 1 32 2048 7 7
I1201 03:33:02.895272 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 16 16 512 512 3 3 32 512 7 7
I1201 03:33:04.914631 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 7 7 2048 512 1 1 32 2048 7 7
I1201 03:33:08.719436 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 2048 7 7 512 2048 1 1 32 512 7 7
I1201 03:33:10.676563 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 512 9 9 512 512 3 3 32 512 7 7
I1201 03:33:10.679972 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 9 9 512 512 3 3 32 512 7 7
I1201 03:33:12.682508 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 7 7 2048 512 1 1 32 2048 7 7
I1201 03:33:15.558076 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 2048 7 7 512 2048 1 1 32 512 7 7
I1201 03:33:17.532346 25408 nn.cc:260] Didn't find saved winograd_conv2d schedule param! key is: CudaWinogradConvSchedule 32 512 9 9 512 512 3 3 32 512 7 7
I1201 03:33:17.535804 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 9 9 512 512 3 3 32 512 7 7
I1201 03:33:19.542748 25408 schedule.cc:1736] Didn't find saved param, key is: CudaDirectConvSchedule 32 512 7 7 2048 512 1 1 32 2048 7 7
I1201 03:33:36.155557 25408 compiler.cc:73] [CUDA] host module:
Module module_0_host {

function fn_conv2d_0_1 (args__ptr, num_args)
{
  fn_conv2d_0_1_kernel(args__ptr, num_args)
}
function fn_conv2d_1 (args__ptr, num_args)
{
  fn_conv2d_1_kernel(args__ptr, num_args)
}
function fn_reduce_sum_4_1 (args__ptr, num_args)
{
  fn_reduce_sum_4_1_kernel(args__ptr, num_args)
}
function fn_reduce_sum_45 (args__ptr, num_args)
{
  fn_reduce_sum_45_kernel(args__ptr, num_args)
}
function fn_reduce_sum_5 (args__ptr, num_args)
{
  fn_reduce_sum_5_kernel(args__ptr, num_args)
}
function fn_reduce_sum_46 (args__ptr, num_args)
{
  fn_reduce_sum_46_kernel(args__ptr, num_args)
}
function fn_const_scalar_2_broadcast_to_3_divide_6_fused (args__ptr, num_args)
{
  fn_const_scalar_2_broadcast_to_3_divide_6_fused_kernel(args__ptr, num_args)
}
function fn_identity_10_elementwise_mul_11_reduce_sum_12_fused (args__ptr, num_args)
{
  fn_identity_10_elementwise_mul_11_reduce_sum_12_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_43_broadcast_to_44_divide_47_fused (args__ptr, num_args)
{
  fn_const_scalar_43_broadcast_to_44_divide_47_fused_kernel(args__ptr, num_args)
}
function fn_identity_51_elementwise_mul_52_reduce_sum_53_fused (args__ptr, num_args)
{
  fn_identity_51_elementwise_mul_52_reduce_sum_53_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_13 (args__ptr, num_args)
{
  fn_reduce_sum_13_kernel(args__ptr, num_args)
}
function fn_reduce_sum_54 (args__ptr, num_args)
{
  fn_reduce_sum_54_kernel(args__ptr, num_args)
}
function fn_const_scalar_29_const_scalar_31_broadcast_to_30_broadcast_to_32_elementwise_mul_34_elementwise_mul_33_elementwise_add_35_fused (args__ptr, num_args)
{
  fn_const_scalar_29_const_scalar_31_broadcast_to_30_broadcast_to_32_elementwise_mul_34_elementwise_mul_33_elementwise_add_35_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_70_const_scalar_72_broadcast_to_71_broadcast_to_73_elementwise_mul_75_elementwise_mul_74_elementwise_add_76_fused (args__ptr, num_args)
{
  fn_const_scalar_70_const_scalar_72_broadcast_to_71_broadcast_to_73_elementwise_mul_75_elementwise_mul_74_elementwise_add_76_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_8_broadcast_to_9_identity_15_elementwise_mul_16_divide_14_substract_17_fused (args__ptr, num_args)
{
  fn_const_scalar_8_broadcast_to_9_identity_15_elementwise_mul_16_divide_14_substract_17_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_49_broadcast_to_50_identity_56_elementwise_mul_57_divide_55_substract_58_fused (args__ptr, num_args)
{
  fn_const_scalar_49_broadcast_to_50_identity_56_elementwise_mul_57_divide_55_substract_58_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_36_const_scalar_38_broadcast_to_37_broadcast_to_39_elementwise_mul_41_elementwise_mul_40_elementwise_add_42_fused (args__ptr, num_args)
{
  fn_const_scalar_36_const_scalar_38_broadcast_to_37_broadcast_to_39_elementwise_mul_41_elementwise_mul_40_elementwise_add_42_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_77_const_scalar_79_broadcast_to_78_broadcast_to_80_elementwise_mul_82_elementwise_mul_81_elementwise_add_83_fused (args__ptr, num_args)
{
  fn_const_scalar_77_const_scalar_79_broadcast_to_78_broadcast_to_80_elementwise_mul_82_elementwise_mul_81_elementwise_add_83_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_59_const_scalar_84_broadcast_to_64_broadcast_to_65_broadcast_to_60_broadcast_to_85_broadcast_to_48_substract_66_broadcast_to_61_elementwise_add_62_rsqrt_63_elementwise_mul_67_elementwise_mul_68_elementwise_add_69_max_86_fused (args__ptr, num_args)
{
  fn_const_scalar_59_const_scalar_84_broadcast_to_64_broadcast_to_65_broadcast_to_60_broadcast_to_85_broadcast_to_48_substract_66_broadcast_to_61_elementwise_add_62_rsqrt_63_elementwise_mul_67_elementwise_mul_68_elementwise_add_69_max_86_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_87 (args__ptr, num_args)
{
  fn_conv2d_87_kernel(args__ptr, num_args)
}
function fn_reduce_sum_90 (args__ptr, num_args)
{
  fn_reduce_sum_90_kernel(args__ptr, num_args)
}
function fn_reduce_sum_91 (args__ptr, num_args)
{
  fn_reduce_sum_91_kernel(args__ptr, num_args)
}
function fn_const_scalar_88_broadcast_to_89_divide_92_fused (args__ptr, num_args)
{
  fn_const_scalar_88_broadcast_to_89_divide_92_fused_kernel(args__ptr, num_args)
}
function fn_identity_96_elementwise_mul_97_reduce_sum_98_fused (args__ptr, num_args)
{
  fn_identity_96_elementwise_mul_97_reduce_sum_98_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_99 (args__ptr, num_args)
{
  fn_reduce_sum_99_kernel(args__ptr, num_args)
}
function fn_const_scalar_115_const_scalar_117_broadcast_to_116_broadcast_to_118_elementwise_mul_120_elementwise_mul_119_elementwise_add_121_fused (args__ptr, num_args)
{
  fn_const_scalar_115_const_scalar_117_broadcast_to_116_broadcast_to_118_elementwise_mul_120_elementwise_mul_119_elementwise_add_121_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_94_broadcast_to_95_identity_101_elementwise_mul_102_divide_100_substract_103_fused (args__ptr, num_args)
{
  fn_const_scalar_94_broadcast_to_95_identity_101_elementwise_mul_102_divide_100_substract_103_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_122_const_scalar_124_broadcast_to_123_broadcast_to_125_elementwise_mul_127_elementwise_mul_126_elementwise_add_128_fused (args__ptr, num_args)
{
  fn_const_scalar_122_const_scalar_124_broadcast_to_123_broadcast_to_125_elementwise_mul_127_elementwise_mul_126_elementwise_add_128_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_104_const_scalar_129_broadcast_to_109_broadcast_to_110_broadcast_to_105_broadcast_to_130_broadcast_to_93_substract_111_broadcast_to_106_elementwise_add_107_rsqrt_108_elementwise_mul_112_elementwise_mul_113_elementwise_add_114_max_131_fused (args__ptr, num_args)
{
  fn_const_scalar_104_const_scalar_129_broadcast_to_109_broadcast_to_110_broadcast_to_105_broadcast_to_130_broadcast_to_93_substract_111_broadcast_to_106_elementwise_add_107_rsqrt_108_elementwise_mul_112_elementwise_mul_113_elementwise_add_114_max_131_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_132 (args__ptr, num_args)
{
  fn_conv2d_132_kernel(args__ptr, num_args)
}
function fn_reduce_sum_135 (args__ptr, num_args)
{
  fn_reduce_sum_135_kernel(args__ptr, num_args)
}
function fn_reduce_sum_136 (args__ptr, num_args)
{
  fn_reduce_sum_136_kernel(args__ptr, num_args)
}
function fn_const_scalar_133_broadcast_to_134_divide_137_fused (args__ptr, num_args)
{
  fn_const_scalar_133_broadcast_to_134_divide_137_fused_kernel(args__ptr, num_args)
}
function fn_identity_141_elementwise_mul_142_reduce_sum_143_fused (args__ptr, num_args)
{
  fn_identity_141_elementwise_mul_142_reduce_sum_143_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_144 (args__ptr, num_args)
{
  fn_reduce_sum_144_kernel(args__ptr, num_args)
}
function fn_const_scalar_160_const_scalar_162_broadcast_to_161_broadcast_to_163_elementwise_mul_165_elementwise_mul_164_elementwise_add_166_fused (args__ptr, num_args)
{
  fn_const_scalar_160_const_scalar_162_broadcast_to_161_broadcast_to_163_elementwise_mul_165_elementwise_mul_164_elementwise_add_166_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_139_broadcast_to_140_identity_146_elementwise_mul_147_divide_145_substract_148_fused (args__ptr, num_args)
{
  fn_const_scalar_139_broadcast_to_140_identity_146_elementwise_mul_147_divide_145_substract_148_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_167_const_scalar_169_broadcast_to_168_broadcast_to_170_elementwise_mul_172_elementwise_mul_171_elementwise_add_173_fused (args__ptr, num_args)
{
  fn_const_scalar_167_const_scalar_169_broadcast_to_168_broadcast_to_170_elementwise_mul_172_elementwise_mul_171_elementwise_add_173_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_18_const_scalar_149_const_scalar_175_broadcast_to_23_broadcast_to_24_broadcast_to_154_broadcast_to_155_broadcast_to_19_broadcast_to_150_broadcast_to_176_broadcast_to_7_substract_25_broadcast_to_20_elementwise_add_21_rsqrt_22_elementwise_mul_26_elementwise_mul_27_elementwise_add_28_broadcast_to_138_substract_156_broadcast_to_151_elementwise_add_152_rsqrt_153_elementwise_mul_157_elementwise_mul_158_elementwise_add_159_elementwise_add_174_max_177_fused (args__ptr, num_args)
{
  fn_const_scalar_18_const_scalar_149_const_scalar_175_broadcast_to_23_broadcast_to_24_broadcast_to_154_broadcast_to_155_broadcast_to_19_broadcast_to_150_broadcast_to_176_broadcast_to_7_substract_25_broadcast_to_20_elementwise_add_21_rsqrt_22_elementwise_mul_26_elementwise_mul_27_elementwise_add_28_broadcast_to_138_substract_156_broadcast_to_151_elementwise_add_152_rsqrt_153_elementwise_mul_157_elementwise_mul_158_elementwise_add_159_elementwise_add_174_max_177_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_178 (args__ptr, num_args)
{
  fn_conv2d_178_kernel(args__ptr, num_args)
}
function fn_reduce_sum_181 (args__ptr, num_args)
{
  fn_reduce_sum_181_kernel(args__ptr, num_args)
}
function fn_reduce_sum_182 (args__ptr, num_args)
{
  fn_reduce_sum_182_kernel(args__ptr, num_args)
}
function fn_const_scalar_179_broadcast_to_180_divide_183_fused (args__ptr, num_args)
{
  fn_const_scalar_179_broadcast_to_180_divide_183_fused_kernel(args__ptr, num_args)
}
function fn_identity_187_elementwise_mul_188_reduce_sum_189_fused (args__ptr, num_args)
{
  fn_identity_187_elementwise_mul_188_reduce_sum_189_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_190 (args__ptr, num_args)
{
  fn_reduce_sum_190_kernel(args__ptr, num_args)
}
function fn_const_scalar_206_const_scalar_208_broadcast_to_207_broadcast_to_209_elementwise_mul_211_elementwise_mul_210_elementwise_add_212_fused (args__ptr, num_args)
{
  fn_const_scalar_206_const_scalar_208_broadcast_to_207_broadcast_to_209_elementwise_mul_211_elementwise_mul_210_elementwise_add_212_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_185_broadcast_to_186_identity_192_elementwise_mul_193_divide_191_substract_194_fused (args__ptr, num_args)
{
  fn_const_scalar_185_broadcast_to_186_identity_192_elementwise_mul_193_divide_191_substract_194_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_213_const_scalar_215_broadcast_to_214_broadcast_to_216_elementwise_mul_218_elementwise_mul_217_elementwise_add_219_fused (args__ptr, num_args)
{
  fn_const_scalar_213_const_scalar_215_broadcast_to_214_broadcast_to_216_elementwise_mul_218_elementwise_mul_217_elementwise_add_219_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_195_const_scalar_220_broadcast_to_200_broadcast_to_201_broadcast_to_196_broadcast_to_221_broadcast_to_184_substract_202_broadcast_to_197_elementwise_add_198_rsqrt_199_elementwise_mul_203_elementwise_mul_204_elementwise_add_205_max_222_fused (args__ptr, num_args)
{
  fn_const_scalar_195_const_scalar_220_broadcast_to_200_broadcast_to_201_broadcast_to_196_broadcast_to_221_broadcast_to_184_substract_202_broadcast_to_197_elementwise_add_198_rsqrt_199_elementwise_mul_203_elementwise_mul_204_elementwise_add_205_max_222_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_223 (args__ptr, num_args)
{
  fn_conv2d_223_kernel(args__ptr, num_args)
}
function fn_reduce_sum_226 (args__ptr, num_args)
{
  fn_reduce_sum_226_kernel(args__ptr, num_args)
}
function fn_reduce_sum_227 (args__ptr, num_args)
{
  fn_reduce_sum_227_kernel(args__ptr, num_args)
}
function fn_const_scalar_224_broadcast_to_225_divide_228_fused (args__ptr, num_args)
{
  fn_const_scalar_224_broadcast_to_225_divide_228_fused_kernel(args__ptr, num_args)
}
function fn_identity_232_elementwise_mul_233_reduce_sum_234_fused (args__ptr, num_args)
{
  fn_identity_232_elementwise_mul_233_reduce_sum_234_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_235 (args__ptr, num_args)
{
  fn_reduce_sum_235_kernel(args__ptr, num_args)
}
function fn_const_scalar_251_const_scalar_253_broadcast_to_252_broadcast_to_254_elementwise_mul_256_elementwise_mul_255_elementwise_add_257_fused (args__ptr, num_args)
{
  fn_const_scalar_251_const_scalar_253_broadcast_to_252_broadcast_to_254_elementwise_mul_256_elementwise_mul_255_elementwise_add_257_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_230_broadcast_to_231_identity_237_elementwise_mul_238_divide_236_substract_239_fused (args__ptr, num_args)
{
  fn_const_scalar_230_broadcast_to_231_identity_237_elementwise_mul_238_divide_236_substract_239_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_258_const_scalar_260_broadcast_to_259_broadcast_to_261_elementwise_mul_263_elementwise_mul_262_elementwise_add_264_fused (args__ptr, num_args)
{
  fn_const_scalar_258_const_scalar_260_broadcast_to_259_broadcast_to_261_elementwise_mul_263_elementwise_mul_262_elementwise_add_264_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_240_const_scalar_265_broadcast_to_245_broadcast_to_246_broadcast_to_241_broadcast_to_266_broadcast_to_229_substract_247_broadcast_to_242_elementwise_add_243_rsqrt_244_elementwise_mul_248_elementwise_mul_249_elementwise_add_250_max_267_fused (args__ptr, num_args)
{
  fn_const_scalar_240_const_scalar_265_broadcast_to_245_broadcast_to_246_broadcast_to_241_broadcast_to_266_broadcast_to_229_substract_247_broadcast_to_242_elementwise_add_243_rsqrt_244_elementwise_mul_248_elementwise_mul_249_elementwise_add_250_max_267_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_268 (args__ptr, num_args)
{
  fn_conv2d_268_kernel(args__ptr, num_args)
}
function fn_reduce_sum_271 (args__ptr, num_args)
{
  fn_reduce_sum_271_kernel(args__ptr, num_args)
}
function fn_reduce_sum_272 (args__ptr, num_args)
{
  fn_reduce_sum_272_kernel(args__ptr, num_args)
}
function fn_const_scalar_269_broadcast_to_270_divide_273_fused (args__ptr, num_args)
{
  fn_const_scalar_269_broadcast_to_270_divide_273_fused_kernel(args__ptr, num_args)
}
function fn_identity_277_elementwise_mul_278_reduce_sum_279_fused (args__ptr, num_args)
{
  fn_identity_277_elementwise_mul_278_reduce_sum_279_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_280 (args__ptr, num_args)
{
  fn_reduce_sum_280_kernel(args__ptr, num_args)
}
function fn_const_scalar_296_const_scalar_298_broadcast_to_297_broadcast_to_299_elementwise_mul_301_elementwise_mul_300_elementwise_add_302_fused (args__ptr, num_args)
{
  fn_const_scalar_296_const_scalar_298_broadcast_to_297_broadcast_to_299_elementwise_mul_301_elementwise_mul_300_elementwise_add_302_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_275_broadcast_to_276_identity_282_elementwise_mul_283_divide_281_substract_284_fused (args__ptr, num_args)
{
  fn_const_scalar_275_broadcast_to_276_identity_282_elementwise_mul_283_divide_281_substract_284_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_303_const_scalar_305_broadcast_to_304_broadcast_to_306_elementwise_mul_308_elementwise_mul_307_elementwise_add_309_fused (args__ptr, num_args)
{
  fn_const_scalar_303_const_scalar_305_broadcast_to_304_broadcast_to_306_elementwise_mul_308_elementwise_mul_307_elementwise_add_309_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_285_const_scalar_311_broadcast_to_290_broadcast_to_291_broadcast_to_286_broadcast_to_312_broadcast_to_274_substract_292_broadcast_to_287_elementwise_add_288_rsqrt_289_elementwise_mul_293_elementwise_mul_294_elementwise_add_295_elementwise_add_310_max_313_fused (args__ptr, num_args)
{
  fn_const_scalar_285_const_scalar_311_broadcast_to_290_broadcast_to_291_broadcast_to_286_broadcast_to_312_broadcast_to_274_substract_292_broadcast_to_287_elementwise_add_288_rsqrt_289_elementwise_mul_293_elementwise_mul_294_elementwise_add_295_elementwise_add_310_max_313_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_314 (args__ptr, num_args)
{
  fn_conv2d_314_kernel(args__ptr, num_args)
}
function fn_reduce_sum_317 (args__ptr, num_args)
{
  fn_reduce_sum_317_kernel(args__ptr, num_args)
}
function fn_reduce_sum_318 (args__ptr, num_args)
{
  fn_reduce_sum_318_kernel(args__ptr, num_args)
}
function fn_const_scalar_315_broadcast_to_316_divide_319_fused (args__ptr, num_args)
{
  fn_const_scalar_315_broadcast_to_316_divide_319_fused_kernel(args__ptr, num_args)
}
function fn_identity_323_elementwise_mul_324_reduce_sum_325_fused (args__ptr, num_args)
{
  fn_identity_323_elementwise_mul_324_reduce_sum_325_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_326 (args__ptr, num_args)
{
  fn_reduce_sum_326_kernel(args__ptr, num_args)
}
function fn_const_scalar_342_const_scalar_344_broadcast_to_343_broadcast_to_345_elementwise_mul_347_elementwise_mul_346_elementwise_add_348_fused (args__ptr, num_args)
{
  fn_const_scalar_342_const_scalar_344_broadcast_to_343_broadcast_to_345_elementwise_mul_347_elementwise_mul_346_elementwise_add_348_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_321_broadcast_to_322_identity_328_elementwise_mul_329_divide_327_substract_330_fused (args__ptr, num_args)
{
  fn_const_scalar_321_broadcast_to_322_identity_328_elementwise_mul_329_divide_327_substract_330_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_349_const_scalar_351_broadcast_to_350_broadcast_to_352_elementwise_mul_354_elementwise_mul_353_elementwise_add_355_fused (args__ptr, num_args)
{
  fn_const_scalar_349_const_scalar_351_broadcast_to_350_broadcast_to_352_elementwise_mul_354_elementwise_mul_353_elementwise_add_355_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_331_const_scalar_356_broadcast_to_336_broadcast_to_337_broadcast_to_332_broadcast_to_357_broadcast_to_320_substract_338_broadcast_to_333_elementwise_add_334_rsqrt_335_elementwise_mul_339_elementwise_mul_340_elementwise_add_341_max_358_fused (args__ptr, num_args)
{
  fn_const_scalar_331_const_scalar_356_broadcast_to_336_broadcast_to_337_broadcast_to_332_broadcast_to_357_broadcast_to_320_substract_338_broadcast_to_333_elementwise_add_334_rsqrt_335_elementwise_mul_339_elementwise_mul_340_elementwise_add_341_max_358_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_359 (args__ptr, num_args)
{
  fn_conv2d_359_kernel(args__ptr, num_args)
}
function fn_reduce_sum_362 (args__ptr, num_args)
{
  fn_reduce_sum_362_kernel(args__ptr, num_args)
}
function fn_reduce_sum_363 (args__ptr, num_args)
{
  fn_reduce_sum_363_kernel(args__ptr, num_args)
}
function fn_const_scalar_360_broadcast_to_361_divide_364_fused (args__ptr, num_args)
{
  fn_const_scalar_360_broadcast_to_361_divide_364_fused_kernel(args__ptr, num_args)
}
function fn_identity_368_elementwise_mul_369_reduce_sum_370_fused (args__ptr, num_args)
{
  fn_identity_368_elementwise_mul_369_reduce_sum_370_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_371 (args__ptr, num_args)
{
  fn_reduce_sum_371_kernel(args__ptr, num_args)
}
function fn_const_scalar_387_const_scalar_389_broadcast_to_388_broadcast_to_390_elementwise_mul_392_elementwise_mul_391_elementwise_add_393_fused (args__ptr, num_args)
{
  fn_const_scalar_387_const_scalar_389_broadcast_to_388_broadcast_to_390_elementwise_mul_392_elementwise_mul_391_elementwise_add_393_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_366_broadcast_to_367_identity_373_elementwise_mul_374_divide_372_substract_375_fused (args__ptr, num_args)
{
  fn_const_scalar_366_broadcast_to_367_identity_373_elementwise_mul_374_divide_372_substract_375_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_394_const_scalar_396_broadcast_to_395_broadcast_to_397_elementwise_mul_399_elementwise_mul_398_elementwise_add_400_fused (args__ptr, num_args)
{
  fn_const_scalar_394_const_scalar_396_broadcast_to_395_broadcast_to_397_elementwise_mul_399_elementwise_mul_398_elementwise_add_400_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_376_const_scalar_401_broadcast_to_381_broadcast_to_382_broadcast_to_377_broadcast_to_402_broadcast_to_365_substract_383_broadcast_to_378_elementwise_add_379_rsqrt_380_elementwise_mul_384_elementwise_mul_385_elementwise_add_386_max_403_fused (args__ptr, num_args)
{
  fn_const_scalar_376_const_scalar_401_broadcast_to_381_broadcast_to_382_broadcast_to_377_broadcast_to_402_broadcast_to_365_substract_383_broadcast_to_378_elementwise_add_379_rsqrt_380_elementwise_mul_384_elementwise_mul_385_elementwise_add_386_max_403_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_404 (args__ptr, num_args)
{
  fn_conv2d_404_kernel(args__ptr, num_args)
}
function fn_reduce_sum_407 (args__ptr, num_args)
{
  fn_reduce_sum_407_kernel(args__ptr, num_args)
}
function fn_reduce_sum_408 (args__ptr, num_args)
{
  fn_reduce_sum_408_kernel(args__ptr, num_args)
}
function fn_const_scalar_405_broadcast_to_406_divide_409_fused (args__ptr, num_args)
{
  fn_const_scalar_405_broadcast_to_406_divide_409_fused_kernel(args__ptr, num_args)
}
function fn_identity_413_elementwise_mul_414_reduce_sum_415_fused (args__ptr, num_args)
{
  fn_identity_413_elementwise_mul_414_reduce_sum_415_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_416 (args__ptr, num_args)
{
  fn_reduce_sum_416_kernel(args__ptr, num_args)
}
function fn_const_scalar_432_const_scalar_434_broadcast_to_433_broadcast_to_435_elementwise_mul_437_elementwise_mul_436_elementwise_add_438_fused (args__ptr, num_args)
{
  fn_const_scalar_432_const_scalar_434_broadcast_to_433_broadcast_to_435_elementwise_mul_437_elementwise_mul_436_elementwise_add_438_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_411_broadcast_to_412_identity_418_elementwise_mul_419_divide_417_substract_420_fused (args__ptr, num_args)
{
  fn_const_scalar_411_broadcast_to_412_identity_418_elementwise_mul_419_divide_417_substract_420_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_439_const_scalar_441_broadcast_to_440_broadcast_to_442_elementwise_mul_444_elementwise_mul_443_elementwise_add_445_fused (args__ptr, num_args)
{
  fn_const_scalar_439_const_scalar_441_broadcast_to_440_broadcast_to_442_elementwise_mul_444_elementwise_mul_443_elementwise_add_445_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_421_const_scalar_447_broadcast_to_426_broadcast_to_427_broadcast_to_422_broadcast_to_448_broadcast_to_410_substract_428_broadcast_to_423_elementwise_add_424_rsqrt_425_elementwise_mul_429_elementwise_mul_430_elementwise_add_431_elementwise_add_446_max_449_fused (args__ptr, num_args)
{
  fn_const_scalar_421_const_scalar_447_broadcast_to_426_broadcast_to_427_broadcast_to_422_broadcast_to_448_broadcast_to_410_substract_428_broadcast_to_423_elementwise_add_424_rsqrt_425_elementwise_mul_429_elementwise_mul_430_elementwise_add_431_elementwise_add_446_max_449_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_451 (args__ptr, num_args)
{
  fn_conv2d_451_kernel(args__ptr, num_args)
}
function fn_conv2d_450 (args__ptr, num_args)
{
  fn_conv2d_450_kernel(args__ptr, num_args)
}
function fn_reduce_sum_495 (args__ptr, num_args)
{
  fn_reduce_sum_495_kernel(args__ptr, num_args)
}
function fn_reduce_sum_454 (args__ptr, num_args)
{
  fn_reduce_sum_454_kernel(args__ptr, num_args)
}
function fn_reduce_sum_496 (args__ptr, num_args)
{
  fn_reduce_sum_496_kernel(args__ptr, num_args)
}
function fn_reduce_sum_455 (args__ptr, num_args)
{
  fn_reduce_sum_455_kernel(args__ptr, num_args)
}
function fn_const_scalar_493_broadcast_to_494_divide_497_fused (args__ptr, num_args)
{
  fn_const_scalar_493_broadcast_to_494_divide_497_fused_kernel(args__ptr, num_args)
}
function fn_identity_501_elementwise_mul_502_reduce_sum_503_fused (args__ptr, num_args)
{
  fn_identity_501_elementwise_mul_502_reduce_sum_503_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_452_broadcast_to_453_divide_456_fused (args__ptr, num_args)
{
  fn_const_scalar_452_broadcast_to_453_divide_456_fused_kernel(args__ptr, num_args)
}
function fn_identity_460_elementwise_mul_461_reduce_sum_462_fused (args__ptr, num_args)
{
  fn_identity_460_elementwise_mul_461_reduce_sum_462_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_504 (args__ptr, num_args)
{
  fn_reduce_sum_504_kernel(args__ptr, num_args)
}
function fn_reduce_sum_463 (args__ptr, num_args)
{
  fn_reduce_sum_463_kernel(args__ptr, num_args)
}
function fn_const_scalar_520_const_scalar_522_broadcast_to_521_broadcast_to_523_elementwise_mul_525_elementwise_mul_524_elementwise_add_526_fused (args__ptr, num_args)
{
  fn_const_scalar_520_const_scalar_522_broadcast_to_521_broadcast_to_523_elementwise_mul_525_elementwise_mul_524_elementwise_add_526_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_479_const_scalar_481_broadcast_to_480_broadcast_to_482_elementwise_mul_484_elementwise_mul_483_elementwise_add_485_fused (args__ptr, num_args)
{
  fn_const_scalar_479_const_scalar_481_broadcast_to_480_broadcast_to_482_elementwise_mul_484_elementwise_mul_483_elementwise_add_485_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_499_broadcast_to_500_identity_506_elementwise_mul_507_divide_505_substract_508_fused (args__ptr, num_args)
{
  fn_const_scalar_499_broadcast_to_500_identity_506_elementwise_mul_507_divide_505_substract_508_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_458_broadcast_to_459_identity_465_elementwise_mul_466_divide_464_substract_467_fused (args__ptr, num_args)
{
  fn_const_scalar_458_broadcast_to_459_identity_465_elementwise_mul_466_divide_464_substract_467_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_527_const_scalar_529_broadcast_to_528_broadcast_to_530_elementwise_mul_532_elementwise_mul_531_elementwise_add_533_fused (args__ptr, num_args)
{
  fn_const_scalar_527_const_scalar_529_broadcast_to_528_broadcast_to_530_elementwise_mul_532_elementwise_mul_531_elementwise_add_533_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_486_const_scalar_488_broadcast_to_487_broadcast_to_489_elementwise_mul_491_elementwise_mul_490_elementwise_add_492_fused (args__ptr, num_args)
{
  fn_const_scalar_486_const_scalar_488_broadcast_to_487_broadcast_to_489_elementwise_mul_491_elementwise_mul_490_elementwise_add_492_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_509_const_scalar_534_broadcast_to_514_broadcast_to_515_broadcast_to_510_broadcast_to_535_broadcast_to_498_substract_516_broadcast_to_511_elementwise_add_512_rsqrt_513_elementwise_mul_517_elementwise_mul_518_elementwise_add_519_max_536_fused (args__ptr, num_args)
{
  fn_const_scalar_509_const_scalar_534_broadcast_to_514_broadcast_to_515_broadcast_to_510_broadcast_to_535_broadcast_to_498_substract_516_broadcast_to_511_elementwise_add_512_rsqrt_513_elementwise_mul_517_elementwise_mul_518_elementwise_add_519_max_536_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_537 (args__ptr, num_args)
{
  fn_conv2d_537_kernel(args__ptr, num_args)
}
function fn_reduce_sum_540 (args__ptr, num_args)
{
  fn_reduce_sum_540_kernel(args__ptr, num_args)
}
function fn_reduce_sum_541 (args__ptr, num_args)
{
  fn_reduce_sum_541_kernel(args__ptr, num_args)
}
function fn_const_scalar_538_broadcast_to_539_divide_542_fused (args__ptr, num_args)
{
  fn_const_scalar_538_broadcast_to_539_divide_542_fused_kernel(args__ptr, num_args)
}
function fn_identity_546_elementwise_mul_547_reduce_sum_548_fused (args__ptr, num_args)
{
  fn_identity_546_elementwise_mul_547_reduce_sum_548_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_549 (args__ptr, num_args)
{
  fn_reduce_sum_549_kernel(args__ptr, num_args)
}
function fn_const_scalar_565_const_scalar_567_broadcast_to_566_broadcast_to_568_elementwise_mul_570_elementwise_mul_569_elementwise_add_571_fused (args__ptr, num_args)
{
  fn_const_scalar_565_const_scalar_567_broadcast_to_566_broadcast_to_568_elementwise_mul_570_elementwise_mul_569_elementwise_add_571_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_544_broadcast_to_545_identity_551_elementwise_mul_552_divide_550_substract_553_fused (args__ptr, num_args)
{
  fn_const_scalar_544_broadcast_to_545_identity_551_elementwise_mul_552_divide_550_substract_553_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_572_const_scalar_574_broadcast_to_573_broadcast_to_575_elementwise_mul_577_elementwise_mul_576_elementwise_add_578_fused (args__ptr, num_args)
{
  fn_const_scalar_572_const_scalar_574_broadcast_to_573_broadcast_to_575_elementwise_mul_577_elementwise_mul_576_elementwise_add_578_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_554_const_scalar_579_broadcast_to_559_broadcast_to_560_broadcast_to_555_broadcast_to_580_broadcast_to_543_substract_561_broadcast_to_556_elementwise_add_557_rsqrt_558_elementwise_mul_562_elementwise_mul_563_elementwise_add_564_max_581_fused (args__ptr, num_args)
{
  fn_const_scalar_554_const_scalar_579_broadcast_to_559_broadcast_to_560_broadcast_to_555_broadcast_to_580_broadcast_to_543_substract_561_broadcast_to_556_elementwise_add_557_rsqrt_558_elementwise_mul_562_elementwise_mul_563_elementwise_add_564_max_581_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_582 (args__ptr, num_args)
{
  fn_conv2d_582_kernel(args__ptr, num_args)
}
function fn_reduce_sum_585 (args__ptr, num_args)
{
  fn_reduce_sum_585_kernel(args__ptr, num_args)
}
function fn_reduce_sum_586 (args__ptr, num_args)
{
  fn_reduce_sum_586_kernel(args__ptr, num_args)
}
function fn_const_scalar_583_broadcast_to_584_divide_587_fused (args__ptr, num_args)
{
  fn_const_scalar_583_broadcast_to_584_divide_587_fused_kernel(args__ptr, num_args)
}
function fn_identity_591_elementwise_mul_592_reduce_sum_593_fused (args__ptr, num_args)
{
  fn_identity_591_elementwise_mul_592_reduce_sum_593_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_594 (args__ptr, num_args)
{
  fn_reduce_sum_594_kernel(args__ptr, num_args)
}
function fn_const_scalar_610_const_scalar_612_broadcast_to_611_broadcast_to_613_elementwise_mul_615_elementwise_mul_614_elementwise_add_616_fused (args__ptr, num_args)
{
  fn_const_scalar_610_const_scalar_612_broadcast_to_611_broadcast_to_613_elementwise_mul_615_elementwise_mul_614_elementwise_add_616_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_589_broadcast_to_590_identity_596_elementwise_mul_597_divide_595_substract_598_fused (args__ptr, num
I1201 03:33:36.166527 25408 compiler.cc:76] [CUDA] device module:
Module module_0_gpu_device {

function fn_conv2d_0_1_kernel (_pool2d_0__tmp_0, _conv2d_4__w_0, _Conv2d_nchw_out_0)
{
  for (i, 0, 32)
  {
    if ((blockIdx.z < 8)) {
      if ((blockIdx.y < 56)) {
        if ((threadIdx.z < 16)) {
          if ((threadIdx.x < 56)) {
            {
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_0_write_cache__reduce_init[0, j_inner, 0, 0] = 0
                for (rc_0_outer, 0, 16)
                {
                  for (rc_0_inner, 0, 4)
                  {
                    Conv2d_nchw_out_0_write_cache[0, j_inner, 0, 0] = (Conv2d_nchw_out_0_write_cache[0, j_inner, 0, 0] + (pool2d_0__tmp_0[i, ((4 * rc_0_outer) + rc_0_inner), blockIdx.y, threadIdx.x] * conv2d_4__w_0[((32 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), ((4 * rc_0_outer) + rc_0_inner), 0, 0]))
                  }
                }
              }
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_0[i, ((32 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), blockIdx.y, threadIdx.x] = Conv2d_nchw_out_0_write_cache[0, j_inner, 0, 0]
              }
            }
          }
        }
      }
    }
  }
}
function fn_conv2d_1_kernel (_pool2d_0__tmp_0, _conv2d_1__w_0, _Conv2d_nchw_out_1)
{
  for (i, 0, 32)
  {
    if ((blockIdx.z < 4)) {
      if ((blockIdx.y < 56)) {
        if ((threadIdx.z < 8)) {
          if ((threadIdx.x < 56)) {
            {
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_1_write_cache__reduce_init[0, j_inner, 0, 0] = 0
                for (rc_1_outer, 0, 16)
                {
                  for (rc_1_inner, 0, 4)
                  {
                    Conv2d_nchw_out_1_write_cache[0, j_inner, 0, 0] = (Conv2d_nchw_out_1_write_cache[0, j_inner, 0, 0] + (pool2d_0__tmp_0[i, ((4 * rc_1_outer) + rc_1_inner), blockIdx.y, threadIdx.x] * conv2d_1__w_0[((16 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), ((4 * rc_1_outer) + rc_1_inner), 0, 0]))
                  }
                }
              }
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_1[i, ((16 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), blockIdx.y, threadIdx.x] = Conv2d_nchw_out_1_write_cache[0, j_inner, 0, 0]
              }
            }
          }
        }
      }
    }
  }
}
function fn_reduce_sum_4_1_kernel (_var_82, _reduce_sum_out_3)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_3__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_5, 0, 32)
        {
          for (kk_6, 0, 56)
          {
            reduce_sum_out_3[blockIdx.x, threadIdx.x] = (reduce_sum_out_3[blockIdx.x, threadIdx.x] + var_82[kk_5, blockIdx.x, kk_6, threadIdx.x])
          }
        }
      }
    }
  }
}
function fn_reduce_sum_45_kernel (_var_86, _reduce_sum_out_4)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_4__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_7, 0, 32)
        {
          for (kk_8, 0, 56)
          {
            reduce_sum_out_4[blockIdx.x, threadIdx.x] = (reduce_sum_out_4[blockIdx.x, threadIdx.x] + var_86[kk_7, blockIdx.x, kk_8, threadIdx.x])
          }
        }
      }
    }
  }
}
function fn_reduce_sum_5_kernel (_var_1251, _reduce_sum_out_5)
{
  if ((threadIdx.x < 256)) {
    {
      reduce_sum_out_5__reduce_init[threadIdx.x] = 0
      for (kk_9, 0, 56)
      {
        reduce_sum_out_5[threadIdx.x] = (reduce_sum_out_5[threadIdx.x] + var_1251[threadIdx.x, kk_9])
      }
    }
  }
}
function fn_reduce_sum_46_kernel (_var_1301, _reduce_sum_out_6)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_6__reduce_init[threadIdx.x] = 0
      for (kk_10, 0, 56)
      {
        reduce_sum_out_6[threadIdx.x] = (reduce_sum_out_6[threadIdx.x] + var_1301[threadIdx.x, kk_10])
      }
    }
  }
}
function fn_const_scalar_2_broadcast_to_3_divide_6_fused_kernel (_var_1253, _divide_Out_1)
{
  if ((threadIdx.x < 256)) {
    divide_Out_1[threadIdx.x] = (9.96492e-06 * var_1253[threadIdx.x])
  }
}
function fn_identity_10_elementwise_mul_11_reduce_sum_12_fused_kernel (_var_82, _reduce_sum_out_7)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_7__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_11, 0, 32)
        {
          for (kk_12, 0, 56)
          {
            reduce_sum_out_7[blockIdx.x, threadIdx.x] = (reduce_sum_out_7[blockIdx.x, threadIdx.x] + (var_82[kk_11, blockIdx.x, kk_12, threadIdx.x] * var_82[kk_11, blockIdx.x, kk_12, threadIdx.x]))
          }
        }
      }
    }
  }
}
function fn_const_scalar_43_broadcast_to_44_divide_47_fused_kernel (_var_1303, _divide_Out_2)
{
  if ((threadIdx.x < 64)) {
    divide_Out_2[threadIdx.x] = (9.96492e-06 * var_1303[threadIdx.x])
  }
}
function fn_identity_51_elementwise_mul_52_reduce_sum_53_fused_kernel (_var_86, _reduce_sum_out_8)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_8__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_13, 0, 32)
        {
          for (kk_14, 0, 56)
          {
            reduce_sum_out_8[blockIdx.x, threadIdx.x] = (reduce_sum_out_8[blockIdx.x, threadIdx.x] + (var_86[kk_13, blockIdx.x, kk_14, threadIdx.x] * var_86[kk_13, blockIdx.x, kk_14, threadIdx.x]))
          }
        }
      }
    }
  }
}
function fn_reduce_sum_13_kernel (_var_1261, _reduce_sum_out_9)
{
  if ((threadIdx.x < 256)) {
    {
      reduce_sum_out_9__reduce_init[threadIdx.x] = 0
      for (kk_15, 0, 56)
      {
        reduce_sum_out_9[threadIdx.x] = (reduce_sum_out_9[threadIdx.x] + var_1261[threadIdx.x, kk_15])
      }
    }
  }
}
function fn_reduce_sum_54_kernel (_var_1311, _reduce_sum_out_10)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_10__reduce_init[threadIdx.x] = 0
      for (kk_16, 0, 56)
      {
        reduce_sum_out_10[threadIdx.x] = (reduce_sum_out_10[threadIdx.x] + var_1311[threadIdx.x, kk_16])
      }
    }
  }
}
function fn_const_scalar_29_const_scalar_31_broadcast_to_30_broadcast_to_32_elementwise_mul_34_elementwise_mul_33_elementwise_add_35_fused_kernel (_batch_norm_4__w_1, _var_97, _elementwise_add_Out_3)
{
  if ((threadIdx.x < 256)) {
    elementwise_add_Out_3[threadIdx.x] = ((0.9 * batch_norm_4__w_1[threadIdx.x]) + (0.1 * var_97[threadIdx.x]))
  }
}
function fn_const_scalar_70_const_scalar_72_broadcast_to_71_broadcast_to_73_elementwise_mul_75_elementwise_mul_74_elementwise_add_76_fused_kernel (_batch_norm_1__w_1, _var_113, _elementwise_add_Out_4)
{
  if ((threadIdx.x < 64)) {
    elementwise_add_Out_4[threadIdx.x] = ((0.9 * batch_norm_1__w_1[threadIdx.x]) + (0.1 * var_113[threadIdx.x]))
  }
}
function fn_const_scalar_8_broadcast_to_9_identity_15_elementwise_mul_16_divide_14_substract_17_fused_kernel (_var_97, _var_1263, _substract_Out_1)
{
  if ((threadIdx.x < 256)) {
    substract_Out_1[threadIdx.x] = ((9.96492e-06 * var_1263[threadIdx.x]) - (var_97[threadIdx.x] * var_97[threadIdx.x]))
  }
}
function fn_const_scalar_49_broadcast_to_50_identity_56_elementwise_mul_57_divide_55_substract_58_fused_kernel (_var_113, _var_1313, _substract_Out_2)
{
  if ((threadIdx.x < 64)) {
    substract_Out_2[threadIdx.x] = ((9.96492e-06 * var_1313[threadIdx.x]) - (var_113[threadIdx.x] * var_113[threadIdx.x]))
  }
}
function fn_const_scalar_36_const_scalar_38_broadcast_to_37_broadcast_to_39_elementwise_mul_41_elementwise_mul_40_elementwise_add_42_fused_kernel (_batch_norm_4__w_2, _var_98, _elementwise_add_Out_5)
{
  if ((threadIdx.x < 256)) {
    elementwise_add_Out_5[threadIdx.x] = ((0.9 * batch_norm_4__w_2[threadIdx.x]) + (0.1 * var_98[threadIdx.x]))
  }
}
function fn_const_scalar_77_const_scalar_79_broadcast_to_78_broadcast_to_80_elementwise_mul_82_elementwise_mul_81_elementwise_add_83_fused_kernel (_batch_norm_1__w_2, _var_114, _elementwise_add_Out_6)
{
  if ((threadIdx.x < 64)) {
    elementwise_add_Out_6[threadIdx.x] = ((0.9 * batch_norm_1__w_2[threadIdx.x]) + (0.1 * var_114[threadIdx.x]))
  }
}
function fn_const_scalar_59_const_scalar_84_broadcast_to_64_broadcast_to_65_broadcast_to_60_broadcast_to_85_broadcast_to_48_substract_66_broadcast_to_61_elementwise_add_62_rsqrt_63_elementwise_mul_67_elementwise_mul_68_elementwise_add_69_max_86_fused_kernel (_batch_norm_1__w_0, _batch_norm_1__b_0, _var_113, _var_86, _var_114, _max_Out_0)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (25 + (((128 + blockIdx.x) / 256) * -1)))
      {
        max_Out_0[((-1 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + (blockIdx.x + ((2 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((64 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))] = cinn_max(((batch_norm_1__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (rsqrt((var_114[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)) * var_86[((-1 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + (blockIdx.x + ((2 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((64 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))])) + ((-1 * (batch_norm_1__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (rsqrt((var_114[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)) * var_113[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)]))) + batch_norm_1__b_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)])), 0)
      }
    }
  }
}
function fn_conv2d_87_kernel (_var_120, _conv2d_2__w_0, _Conv2d_nchw_out_2)
{
  for (i, 0, 32)
  {
    if ((blockIdx.z < 4)) {
      if ((blockIdx.y < 56)) {
        if ((threadIdx.z < 8)) {
          if ((threadIdx.x < 56)) {
            {
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_2_write_cache__reduce_init[0, j_inner, 0, 0] = 0
                for (rc_2_outer, 0, 16)
                {
                  for (rc_2_inner, 0, 4)
                  {
                    for (ry_2, 0, 3)
                    {
                      for (rx_2, 0, 3)
                      {
                        Conv2d_nchw_out_2_write_cache[0, j_inner, 0, 0] = (Conv2d_nchw_out_2_write_cache[0, j_inner, 0, 0] + (select(((((((blockIdx.y * 1) + (ry_2 * 1)) >= 1) and (((blockIdx.y * 1) + (ry_2 * 1)) < (56 + 1))) and (((threadIdx.x * 1) + (rx_2 * 1)) >= 1)) and (((threadIdx.x * 1) + (rx_2 * 1)) < (56 + 1))), var_120[i, ((4 * rc_2_outer) + rc_2_inner), (-1 + (blockIdx.y + ry_2)), (-1 + (rx_2 + threadIdx.x))], 0) * conv2d_2__w_0[((16 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), ((4 * rc_2_outer) + rc_2_inner), ry_2, rx_2]))
                      }
                    }
                  }
                }
              }
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_2[i, ((16 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), blockIdx.y, threadIdx.x] = Conv2d_nchw_out_2_write_cache[0, j_inner, 0, 0]
              }
            }
          }
        }
      }
    }
  }
}
function fn_reduce_sum_90_kernel (_var_124, _reduce_sum_out_11)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_11__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_17, 0, 32)
        {
          for (kk_18, 0, 56)
          {
            reduce_sum_out_11[blockIdx.x, threadIdx.x] = (reduce_sum_out_11[blockIdx.x, threadIdx.x] + var_124[kk_17, blockIdx.x, kk_18, threadIdx.x])
          }
        }
      }
    }
  }
}
function fn_reduce_sum_91_kernel (_var_1355, _reduce_sum_out_12)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_12__reduce_init[threadIdx.x] = 0
      for (kk_19, 0, 56)
      {
        reduce_sum_out_12[threadIdx.x] = (reduce_sum_out_12[threadIdx.x] + var_1355[threadIdx.x, kk_19])
      }
    }
  }
}
function fn_const_scalar_88_broadcast_to_89_divide_92_fused_kernel (_var_1357, _divide_Out_5)
{
  if ((threadIdx.x < 64)) {
    divide_Out_5[threadIdx.x] = (9.96492e-06 * var_1357[threadIdx.x])
  }
}
function fn_identity_96_elementwise_mul_97_reduce_sum_98_fused_kernel (_var_124, _reduce_sum_out_13)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_13__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_20, 0, 32)
        {
          for (kk_21, 0, 56)
          {
            reduce_sum_out_13[blockIdx.x, threadIdx.x] = (reduce_sum_out_13[blockIdx.x, threadIdx.x] + (var_124[kk_20, blockIdx.x, kk_21, threadIdx.x] * var_124[kk_20, blockIdx.x, kk_21, threadIdx.x]))
          }
        }
      }
    }
  }
}
function fn_reduce_sum_99_kernel (_var_1365, _reduce_sum_out_14)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_14__reduce_init[threadIdx.x] = 0
      for (kk_22, 0, 56)
      {
        reduce_sum_out_14[threadIdx.x] = (reduce_sum_out_14[threadIdx.x] + var_1365[threadIdx.x, kk_22])
      }
    }
  }
}
function fn_const_scalar_115_const_scalar_117_broadcast_to_116_broadcast_to_118_elementwise_mul_120_elementwise_mul_119_elementwise_add_121_fused_kernel (_batch_norm_2__w_1, _var_135, _elementwise_add_Out_9)
{
  if ((threadIdx.x < 64)) {
    elementwise_add_Out_9[threadIdx.x] = ((0.9 * batch_norm_2__w_1[threadIdx.x]) + (0.1 * var_135[threadIdx.x]))
  }
}
function fn_const_scalar_94_broadcast_to_95_identity_101_elementwise_mul_102_divide_100_substract_103_fused_kernel (_var_135, _var_1367, _substract_Out_4)
{
  if ((threadIdx.x < 64)) {
    substract_Out_4[threadIdx.x] = ((9.96492e-06 * var_1367[threadIdx.x]) - (var_135[threadIdx.x] * var_135[threadIdx.x]))
  }
}
function fn_const_scalar_122_const_scalar_124_broadcast_to_123_broadcast_to_125_elementwise_mul_127_elementwise_mul_126_elementwise_add_128_fused_kernel (_batch_norm_2__w_2, _var_136, _elementwise_add_Out_10)
{
  if ((threadIdx.x < 64)) {
    elementwise_add_Out_10[threadIdx.x] = ((0.9 * batch_norm_2__w_2[threadIdx.x]) + (0.1 * var_136[threadIdx.x]))
  }
}
function fn_const_scalar_104_const_scalar_129_broadcast_to_109_broadcast_to_110_broadcast_to_105_broadcast_to_130_broadcast_to_93_substract_111_broadcast_to_106_elementwise_add_107_rsqrt_108_elementwise_mul_112_elementwise_mul_113_elementwise_add_114_max_131_fused_kernel (_batch_norm_2__w_0, _batch_norm_2__b_0, _var_135, _var_124, _var_136, _max_Out_1)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (25 + (((128 + blockIdx.x) / 256) * -1)))
      {
        max_Out_1[((-1 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + (blockIdx.x + ((2 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((64 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))] = cinn_max(((batch_norm_2__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (rsqrt((var_136[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)) * var_124[((-1 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + (blockIdx.x + ((2 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((64 * ((200703 + ((199680 * blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * threadIdx.x)))) / 200704)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))])) + ((-1 * (batch_norm_2__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (rsqrt((var_136[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)) * var_135[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)]))) + batch_norm_2__b_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-63 * blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)])), 0)
      }
    }
  }
}
function fn_conv2d_132_kernel (_var_142, _conv2d_3__w_0, _Conv2d_nchw_out_3)
{
  for (i, 0, 32)
  {
    if ((blockIdx.z < 8)) {
      if ((blockIdx.y < 56)) {
        if ((threadIdx.z < 16)) {
          if ((threadIdx.x < 56)) {
            {
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_3_write_cache__reduce_init[0, j_inner, 0, 0] = 0
                for (rc_3_outer, 0, 16)
                {
                  for (rc_3_inner, 0, 4)
                  {
                    Conv2d_nchw_out_3_write_cache[0, j_inner, 0, 0] = (Conv2d_nchw_out_3_write_cache[0, j_inner, 0, 0] + (var_142[i, ((4 * rc_3_outer) + rc_3_inner), blockIdx.y, threadIdx.x] * conv2d_3__w_0[((32 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), ((4 * rc_3_outer) + rc_3_inner), 0, 0]))
                  }
                }
              }
              for (j_inner, 0, 2)
              {
                Conv2d_nchw_out_3[i, ((32 * blockIdx.z) + ((2 * threadIdx.z) + j_inner)), blockIdx.y, threadIdx.x] = Conv2d_nchw_out_3_write_cache[0, j_inner, 0, 0]
              }
            }
          }
        }
      }
    }
  }
}
function fn_reduce_sum_135_kernel (_var_146, _reduce_sum_out_15)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_15__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_23, 0, 32)
        {
          for (kk_24, 0, 56)
          {
            reduce_sum_out_15[blockIdx.x, threadIdx.x] = (reduce_sum_out_15[blockIdx.x, threadIdx.x] + var_146[kk_23, blockIdx.x, kk_24, threadIdx.x])
          }
        }
      }
    }
  }
}
function fn_reduce_sum_136_kernel (_var_1409, _reduce_sum_out_16)
{
  if ((threadIdx.x < 256)) {
    {
      reduce_sum_out_16__reduce_init[threadIdx.x] = 0
      for (kk_25, 0, 56)
      {
        reduce_sum_out_16[threadIdx.x] = (reduce_sum_out_16[threadIdx.x] + var_1409[threadIdx.x, kk_25])
      }
    }
  }
}
function fn_const_scalar_133_broadcast_to_134_divide_137_fused_kernel (_var_1411, _divide_Out_7)
{
  if ((threadIdx.x < 256)) {
    divide_Out_7[threadIdx.x] = (9.96492e-06 * var_1411[threadIdx.x])
  }
}
function fn_identity_141_elementwise_mul_142_reduce_sum_143_fused_kernel (_var_146, _reduce_sum_out_17)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 56)) {
      {
        reduce_sum_out_17__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_26, 0, 32)
        {
          for (kk_27, 0, 56)
          {
            reduce_sum_out_17[blockIdx.x, threadIdx.x] = (reduce_sum_out_17[blockIdx.x, threadIdx.x] + (var_146[kk_26, blockIdx.x, kk_27, threadIdx.x] * var_146[kk_26, blockIdx.x, kk_27, threadIdx.x]))
          }
        }
      }
    }
  }
}
function fn_reduce_sum_144_kernel (_var_1419, _reduce_sum_out_18)
{
  if ((threadIdx.x < 256)) {
    {
      reduce_sum_out_18__reduce_init[threadIdx.x] = 0
      for (kk_28, 0, 56)
      {
        reduce_sum_out_18[threadIdx.x] = (reduce_sum_out_18[threadIdx.x] + var_1419[threadIdx.x, kk_28])
      }
    }
  }
}
function fn_const_scalar_160_const_scalar_162_broadcast_to_161_broadcast_to_163_elementwise_mul_165_elementwise_mul_164_elementwise_add_166_fused_kernel (_batch_norm_3__w_1, _var_157, _elementwise_add_Out_13)
{
  if ((threadIdx.x < 256)) {
    elementwise_add_Out_13[threadIdx.x] = ((0.9 * batch_norm_3__w_1[threadIdx.x]) + (0.1 * var_157[threadIdx.x]))
  }
}
function fn_const_scalar_139_broadcast_to_140_identity_146_elementwise_mul_147_divide_145_substract_148_fused_kernel (_var_157, _var_1421, _substract_Out_6)
{
  if ((threadIdx.x < 256)) {
    substract_Out_6[threadIdx.x] = ((9.96492e-06 * var_1421[threadIdx.x]) - (var_157[threadIdx.x] * var_157[threadIdx.x]))
  }
}
function fn_const_scalar_167_const_scalar_169_broadcast_to_168_broadcast_to_170_elementwise_mul_172_elementwise_mul_171_elementwise_add_173_fused_kernel (_batch_norm_3__w_2, _var_158, _elementwise_add_Out_14)
{
  if ((threadIdx.x < 256)) {
    elementwise_add_Out_14[threadIdx.x] = ((0.9 * batch_norm_3__w_2[threadIdx.x]) + (0.1 * var_158[threadIdx.x]))
  }
}
function fn_const_scalar_18_const_scalar_149_const_scalar_175_broadcast_to_23_broadcast_to_24_broadcast_to_154_broadcast_to_155_broadcast_to_19_broadcast_to_150_broadcast_to_176_broadcast_to_7_substract_25_broadcast_to_20_elementwise_add_21_rsqrt_22_elementwise_mul_26_elementwise_mul_27_elementwise_add_28_broadcast_to_138_substract_156_broadcast_to_151_elementwise_add_152_rsqrt_153_elementwise_mul_157_elementwise_mul_158_elementwise_add_159_elementwise_add_174_max_177_fused_kernel (_batch_norm_4__w_0, _batch_norm_4__b_0, _batch_norm_3__w_0, _batch_norm_3__b_0, _var_97, _var_82, _var_98, _var_157, _var_146, _var_158, _max_Out_2, _elementwise_add_Out_18, _elementwise_add_Out_16)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, 98)
      {
        {
          elementwise_add_Out_18[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((256 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))] = ((batch_norm_3__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)] * (rsqrt((var_158[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)] + 1e-05)) * var_146[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((256 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))])) + ((-1 * (batch_norm_3__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)] * (rsqrt((var_158[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)] + 1e-05)) * var_157[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)]))) + batch_norm_3__b_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)]))
          elementwise_add_Out_16[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((256 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x))))), ((-1 * ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) / 56)) + ((56 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-37 * blockIdx.x) + ((-22 * i_j_fused_k_fused_a_fused_outer) + (-55 * threadIdx.x))))), (55 - ((55 + ((40 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * threadIdx.x)))) % 56))] = ((batch_norm_4__w_0[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_outer) + (-255 * threadIdx.x)))) % 256)] * (rsqrt((var_98[(((-1 * ((3135 + ((2112 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * threadIdx.x)))) / 3136)) + ((-255 * blockIdx.x) + ((-172 * i_j_fused_k_fused_a_fused_o
I1201 03:34:00.830487 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(896) fn_conv2d_0_1_kernel(const float* __restrict__ pool2d_0__tmp_0, const float* __restrict__ conv2d_4__w_0, float* __restrict__ Conv2d_nchw_out_0)
{
  float _Conv2d_nchw_out_0_write_cache [ 2 ];
  float* Conv2d_nchw_out_0_write_cache = _Conv2d_nchw_out_0_write_cache;
  float* Conv2d_nchw_out_0_write_cache__reduce_init = _Conv2d_nchw_out_0_write_cache;
  for (int32_t i = 0; i < 32; i += 1) {
    if (((int)blockIdx.z < 8)) {
      if (((int)blockIdx.y < 56)) {
        if (((int)threadIdx.z < 16)) {
          if (((int)threadIdx.x < 56)) {
          {
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_0_write_cache__reduce_init[j_inner] = 0;
              for (int32_t rc_0_outer = 0; rc_0_outer < 16; rc_0_outer += 1) {
                for (int32_t rc_0_inner = 0; rc_0_inner < 4; rc_0_inner += 1) {
                  Conv2d_nchw_out_0_write_cache[j_inner] = (Conv2d_nchw_out_0_write_cache[j_inner] + (pool2d_0__tmp_0[((56 * (int)blockIdx.y) + ((200704 * i) + ((3136 * rc_0_inner) + ((12544 * rc_0_outer) + (int)threadIdx.x))))] * conv2d_4__w_0[((2048 * (int)blockIdx.z) + ((64 * j_inner) + ((4 * rc_0_outer) + ((128 * (int)threadIdx.z) + rc_0_inner))))]));
                };
              };
            };
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_0[((56 * (int)blockIdx.y) + ((100352 * (int)blockIdx.z) + ((802816 * i) + ((3136 * j_inner) + ((6272 * (int)threadIdx.z) + (int)threadIdx.x)))))] = Conv2d_nchw_out_0_write_cache[j_inner];
            };
          }
          };
        };
      };
    };
  };
}__global__
void __launch_bounds__(448) fn_conv2d_1_kernel(const float* __restrict__ pool2d_0__tmp_0, const float* __restrict__ conv2d_1__w_0, float* __restrict__ Conv2d_nchw_out_1)
{
  float _Conv2d_nchw_out_1_write_cache [ 2 ];
  float* Conv2d_nchw_out_1_write_cache = _Conv2d_nchw_out_1_write_cache;
  float* Conv2d_nchw_out_1_write_cache__reduce_init = _Conv2d_nchw_out_1_write_cache;
  for (int32_t i = 0; i < 32; i += 1) {
    if (((int)blockIdx.z < 4)) {
      if (((int)blockIdx.y < 56)) {
        if (((int)threadIdx.z < 8)) {
          if (((int)threadIdx.x < 56)) {
          {
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_1_write_cache__reduce_init[j_inner] = 0;
              for (int32_t rc_1_outer = 0; rc_1_outer < 16; rc_1_outer += 1) {
                for (int32_t rc_1_inner = 0; rc_1_inner < 4; rc_1_inner += 1) {
                  Conv2d_nchw_out_1_write_cache[j_inner] = (Conv2d_nchw_out_1_write_cache[j_inner] + (pool2d_0__tmp_0[((56 * (int)blockIdx.y) + ((200704 * i) + ((3136 * rc_1_inner) + ((12544 * rc_1_outer) + (int)threadIdx.x))))] * conv2d_1__w_0[((1024 * (int)blockIdx.z) + ((64 * j_inner) + ((4 * rc_1_outer) + ((128 * (int)threadIdx.z) + rc_1_inner))))]));
                };
              };
            };
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_1[((56 * (int)blockIdx.y) + ((50176 * (int)blockIdx.z) + ((200704 * i) + ((3136 * j_inner) + ((6272 * (int)threadIdx.z) + (int)threadIdx.x)))))] = Conv2d_nchw_out_1_write_cache[j_inner];
            };
          }
          };
        };
      };
    };
  };
}__global__
void __launch_bounds__(56) fn_reduce_sum_4_1_kernel(const float* __restrict__ var_82, float* __restrict__ reduce_sum_out_3)
{
  float* reduce_sum_out_3__reduce_init = reduce_sum_out_3;
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_3__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_5 = 0; kk_5 < 32; kk_5 += 1) {
        for (int32_t kk_6 = 0; kk_6 < 56; kk_6 += 1) {
          reduce_sum_out_3[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_3[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + var_82[((3136 * (int)blockIdx.x) + ((802816 * kk_5) + ((56 * kk_6) + (int)threadIdx.x)))]);
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(56) fn_reduce_sum_45_kernel(const float* __restrict__ var_86, float* __restrict__ reduce_sum_out_4)
{
  float* reduce_sum_out_4__reduce_init = reduce_sum_out_4;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_4__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_7 = 0; kk_7 < 32; kk_7 += 1) {
        for (int32_t kk_8 = 0; kk_8 < 56; kk_8 += 1) {
          reduce_sum_out_4[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_4[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + var_86[((3136 * (int)blockIdx.x) + ((200704 * kk_7) + ((56 * kk_8) + (int)threadIdx.x)))]);
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(256) fn_reduce_sum_5_kernel(const float* __restrict__ var_1251, float* __restrict__ reduce_sum_out_5)
{
  float* reduce_sum_out_5__reduce_init = reduce_sum_out_5;
  if (((int)threadIdx.x < 256)) {
  {
    reduce_sum_out_5__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_9 = 0; kk_9 < 56; kk_9 += 1) {
      reduce_sum_out_5[(int)threadIdx.x] = (reduce_sum_out_5[(int)threadIdx.x] + var_1251[((56 * (int)threadIdx.x) + kk_9)]);
    };
  }
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_46_kernel(const float* __restrict__ var_1301, float* __restrict__ reduce_sum_out_6)
{
  float* reduce_sum_out_6__reduce_init = reduce_sum_out_6;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_6__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_10 = 0; kk_10 < 56; kk_10 += 1) {
      reduce_sum_out_6[(int)threadIdx.x] = (reduce_sum_out_6[(int)threadIdx.x] + var_1301[((56 * (int)threadIdx.x) + kk_10)]);
    };
  }
  };
}__global__
void __launch_bounds__(256) fn_const_scalar_2_broadcast_to_3_divide_6_fused_kernel(const float* __restrict__ var_1253, float* __restrict__ divide_Out_1)
{
  if (((int)threadIdx.x < 256)) {
    divide_Out_1[(int)threadIdx.x] = (9.96492e-06 * var_1253[(int)threadIdx.x]);
  };
}__global__
void __launch_bounds__(56) fn_identity_10_elementwise_mul_11_reduce_sum_12_fused_kernel(const float* __restrict__ var_82, float* __restrict__ reduce_sum_out_7)
{
  float* reduce_sum_out_7__reduce_init = reduce_sum_out_7;
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_7__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_11 = 0; kk_11 < 32; kk_11 += 1) {
        for (int32_t kk_12 = 0; kk_12 < 56; kk_12 += 1) {
          reduce_sum_out_7[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_7[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + (var_82[((3136 * (int)blockIdx.x) + ((802816 * kk_11) + ((56 * kk_12) + (int)threadIdx.x)))] * var_82[((3136 * (int)blockIdx.x) + ((802816 * kk_11) + ((56 * kk_12) + (int)threadIdx.x)))]));
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_43_broadcast_to_44_divide_47_fused_kernel(const float* __restrict__ var_1303, float* __restrict__ divide_Out_2)
{
  if (((int)threadIdx.x < 64)) {
    divide_Out_2[(int)threadIdx.x] = (9.96492e-06 * var_1303[(int)threadIdx.x]);
  };
}__global__
void __launch_bounds__(56) fn_identity_51_elementwise_mul_52_reduce_sum_53_fused_kernel(const float* __restrict__ var_86, float* __restrict__ reduce_sum_out_8)
{
  float* reduce_sum_out_8__reduce_init = reduce_sum_out_8;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_8__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_13 = 0; kk_13 < 32; kk_13 += 1) {
        for (int32_t kk_14 = 0; kk_14 < 56; kk_14 += 1) {
          reduce_sum_out_8[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_8[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + (var_86[((3136 * (int)blockIdx.x) + ((200704 * kk_13) + ((56 * kk_14) + (int)threadIdx.x)))] * var_86[((3136 * (int)blockIdx.x) + ((200704 * kk_13) + ((56 * kk_14) + (int)threadIdx.x)))]));
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(256) fn_reduce_sum_13_kernel(const float* __restrict__ var_1261, float* __restrict__ reduce_sum_out_9)
{
  float* reduce_sum_out_9__reduce_init = reduce_sum_out_9;
  if (((int)threadIdx.x < 256)) {
  {
    reduce_sum_out_9__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_15 = 0; kk_15 < 56; kk_15 += 1) {
      reduce_sum_out_9[(int)threadIdx.x] = (reduce_sum_out_9[(int)threadIdx.x] + var_1261[((56 * (int)threadIdx.x) + kk_15)]);
    };
  }
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_54_kernel(const float* __restrict__ var_1311, float* __restrict__ reduce_sum_out_10)
{
  float* reduce_sum_out_10__reduce_init = reduce_sum_out_10;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_10__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_16 = 0; kk_16 < 56; kk_16 += 1) {
      reduce_sum_out_10[(int)threadIdx.x] = (reduce_sum_out_10[(int)threadIdx.x] + var_1311[((56 * (int)threadIdx.x) + kk_16)]);
    };
  }
  };
}__global__
void __launch_bounds__(256) fn_const_scalar_29_const_scalar_31_broadcast_to_30_broadcast_to_32_elementwise_mul_34_elementwise_mul_33_elementwise_add_35_fused_kernel(const float* __restrict__ batch_norm_4__w_1, const float* __restrict__ var_97, float* __restrict__ elementwise_add_Out_3)
{
  if (((int)threadIdx.x < 256)) {
    elementwise_add_Out_3[(int)threadIdx.x] = ((0.9 * batch_norm_4__w_1[(int)threadIdx.x]) + (0.1 * var_97[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_70_const_scalar_72_broadcast_to_71_broadcast_to_73_elementwise_mul_75_elementwise_mul_74_elementwise_add_76_fused_kernel(const float* __restrict__ batch_norm_1__w_1, const float* __restrict__ var_113, float* __restrict__ elementwise_add_Out_4)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_add_Out_4[(int)threadIdx.x] = ((0.9 * batch_norm_1__w_1[(int)threadIdx.x]) + (0.1 * var_113[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(256) fn_const_scalar_8_broadcast_to_9_identity_15_elementwise_mul_16_divide_14_substract_17_fused_kernel(const float* __restrict__ var_97, const float* __restrict__ var_1263, float* __restrict__ substract_Out_1)
{
  if (((int)threadIdx.x < 256)) {
    substract_Out_1[(int)threadIdx.x] = ((9.96492e-06 * var_1263[(int)threadIdx.x]) - (var_97[(int)threadIdx.x] * var_97[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_49_broadcast_to_50_identity_56_elementwise_mul_57_divide_55_substract_58_fused_kernel(const float* __restrict__ var_113, const float* __restrict__ var_1313, float* __restrict__ substract_Out_2)
{
  if (((int)threadIdx.x < 64)) {
    substract_Out_2[(int)threadIdx.x] = ((9.96492e-06 * var_1313[(int)threadIdx.x]) - (var_113[(int)threadIdx.x] * var_113[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(256) fn_const_scalar_36_const_scalar_38_broadcast_to_37_broadcast_to_39_elementwise_mul_41_elementwise_mul_40_elementwise_add_42_fused_kernel(const float* __restrict__ batch_norm_4__w_2, const float* __restrict__ var_98, float* __restrict__ elementwise_add_Out_5)
{
  if (((int)threadIdx.x < 256)) {
    elementwise_add_Out_5[(int)threadIdx.x] = ((0.9 * batch_norm_4__w_2[(int)threadIdx.x]) + (0.1 * var_98[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_77_const_scalar_79_broadcast_to_78_broadcast_to_80_elementwise_mul_82_elementwise_mul_81_elementwise_add_83_fused_kernel(const float* __restrict__ batch_norm_1__w_2, const float* __restrict__ var_114, float* __restrict__ elementwise_add_Out_6)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_add_Out_6[(int)threadIdx.x] = ((0.9 * batch_norm_1__w_2[(int)threadIdx.x]) + (0.1 * var_114[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(1024) fn_const_scalar_59_const_scalar_84_broadcast_to_64_broadcast_to_65_broadcast_to_60_broadcast_to_85_broadcast_to_48_substract_66_broadcast_to_61_elementwise_add_62_rsqrt_63_elementwise_mul_67_elementwise_mul_68_elementwise_add_69_max_86_fused_kernel(const float* __restrict__ batch_norm_1__w_0, const float* __restrict__ batch_norm_1__b_0, const float* __restrict__ var_113, const float* __restrict__ var_86, const float* __restrict__ var_114, float* __restrict__ max_Out_0)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (25 + (((128 + (int)blockIdx.x) / 256) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        max_Out_0[(55 + ((-200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-56 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) / 56)) + ((3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-1 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) % 56)) + ((1064 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (56 * (int)threadIdx.x))))))))))] = cinn_nvgpu_max_fp32(((batch_norm_1__w_0[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (rsqrt((var_114[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)) * var_86[(55 + ((-200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-56 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) / 56)) + ((3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-1 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) % 56)) + ((1064 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (56 * (int)threadIdx.x))))))))))])) + ((-1 * (batch_norm_1__w_0[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (rsqrt((var_114[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)) * var_113[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)]))) + batch_norm_1__b_0[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)])), 0);
      };
    };
  };
}__global__
void __launch_bounds__(448) fn_conv2d_87_kernel(const float* __restrict__ var_120, const float* __restrict__ conv2d_2__w_0, float* __restrict__ Conv2d_nchw_out_2)
{
  float _Conv2d_nchw_out_2_write_cache [ 2 ];
  float* Conv2d_nchw_out_2_write_cache = _Conv2d_nchw_out_2_write_cache;
  float* Conv2d_nchw_out_2_write_cache__reduce_init = _Conv2d_nchw_out_2_write_cache;
  for (int32_t i = 0; i < 32; i += 1) {
    if (((int)blockIdx.z < 4)) {
      if (((int)blockIdx.y < 56)) {
        if (((int)threadIdx.z < 8)) {
          if (((int)threadIdx.x < 56)) {
          {
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_2_write_cache__reduce_init[j_inner] = 0;
              for (int32_t rc_2_outer = 0; rc_2_outer < 16; rc_2_outer += 1) {
                for (int32_t rc_2_inner = 0; rc_2_inner < 4; rc_2_inner += 1) {
                  for (int32_t ry_2 = 0; ry_2 < 3; ry_2 += 1) {
                    for (int32_t rx_2 = 0; rx_2 < 3; rx_2 += 1) {
                      Conv2d_nchw_out_2_write_cache[j_inner] = (Conv2d_nchw_out_2_write_cache[j_inner] + ((((((((((int)blockIdx.y * 1) + (ry_2 * 1)) >= 1) && ((((int)blockIdx.y * 1) + (ry_2 * 1)) < (56 + 1))) && ((((int)threadIdx.x * 1) + (rx_2 * 1)) >= 1)) && ((((int)threadIdx.x * 1) + (rx_2 * 1)) < (56 + 1)))) ? var_120[(-57 + ((56 * (int)blockIdx.y) + ((200704 * i) + ((3136 * rc_2_inner) + ((12544 * rc_2_outer) + ((56 * ry_2) + (rx_2 + (int)threadIdx.x)))))))] : 0) * conv2d_2__w_0[((9216 * (int)blockIdx.z) + ((576 * j_inner) + ((9 * rc_2_inner) + ((36 * rc_2_outer) + ((3 * ry_2) + ((1152 * (int)threadIdx.z) + rx_2))))))]));
                    };
                  };
                };
              };
            };
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_2[((56 * (int)blockIdx.y) + ((50176 * (int)blockIdx.z) + ((200704 * i) + ((3136 * j_inner) + ((6272 * (int)threadIdx.z) + (int)threadIdx.x)))))] = Conv2d_nchw_out_2_write_cache[j_inner];
            };
          }
          };
        };
      };
    };
  };
}__global__
void __launch_bounds__(56) fn_reduce_sum_90_kernel(const float* __restrict__ var_124, float* __restrict__ reduce_sum_out_11)
{
  float* reduce_sum_out_11__reduce_init = reduce_sum_out_11;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_11__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_17 = 0; kk_17 < 32; kk_17 += 1) {
        for (int32_t kk_18 = 0; kk_18 < 56; kk_18 += 1) {
          reduce_sum_out_11[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_11[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + var_124[((3136 * (int)blockIdx.x) + ((200704 * kk_17) + ((56 * kk_18) + (int)threadIdx.x)))]);
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_91_kernel(const float* __restrict__ var_1355, float* __restrict__ reduce_sum_out_12)
{
  float* reduce_sum_out_12__reduce_init = reduce_sum_out_12;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_12__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_19 = 0; kk_19 < 56; kk_19 += 1) {
      reduce_sum_out_12[(int)threadIdx.x] = (reduce_sum_out_12[(int)threadIdx.x] + var_1355[((56 * (int)threadIdx.x) + kk_19)]);
    };
  }
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_88_broadcast_to_89_divide_92_fused_kernel(const float* __restrict__ var_1357, float* __restrict__ divide_Out_5)
{
  if (((int)threadIdx.x < 64)) {
    divide_Out_5[(int)threadIdx.x] = (9.96492e-06 * var_1357[(int)threadIdx.x]);
  };
}__global__
void __launch_bounds__(56) fn_identity_96_elementwise_mul_97_reduce_sum_98_fused_kernel(const float* __restrict__ var_124, float* __restrict__ reduce_sum_out_13)
{
  float* reduce_sum_out_13__reduce_init = reduce_sum_out_13;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_13__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_20 = 0; kk_20 < 32; kk_20 += 1) {
        for (int32_t kk_21 = 0; kk_21 < 56; kk_21 += 1) {
          reduce_sum_out_13[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_13[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + (var_124[((3136 * (int)blockIdx.x) + ((200704 * kk_20) + ((56 * kk_21) + (int)threadIdx.x)))] * var_124[((3136 * (int)blockIdx.x) + ((200704 * kk_20) + ((56 * kk_21) + (int)threadIdx.x)))]));
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_99_kernel(const float* __restrict__ var_1365, float* __restrict__ reduce_sum_out_14)
{
  float* reduce_sum_out_14__reduce_init = reduce_sum_out_14;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_14__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_22 = 0; kk_22 < 56; kk_22 += 1) {
      reduce_sum_out_14[(int)threadIdx.x] = (reduce_sum_out_14[(int)threadIdx.x] + var_1365[((56 * (int)threadIdx.x) + kk_22)]);
    };
  }
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_115_const_scalar_117_broadcast_to_116_broadcast_to_118_elementwise_mul_120_elementwise_mul_119_elementwise_add_121_fused_kernel(const float* __restrict__ batch_norm_2__w_1, const float* __restrict__ var_135, float* __restrict__ elementwise_add_Out_9)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_add_Out_9[(int)threadIdx.x] = ((0.9 * batch_norm_2__w_1[(int)threadIdx.x]) + (0.1 * var_135[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_94_broadcast_to_95_identity_101_elementwise_mul_102_divide_100_substract_103_fused_kernel(const float* __restrict__ var_135, const float* __restrict__ var_1367, float* __restrict__ substract_Out_4)
{
  if (((int)threadIdx.x < 64)) {
    substract_Out_4[(int)threadIdx.x] = ((9.96492e-06 * var_1367[(int)threadIdx.x]) - (var_135[(int)threadIdx.x] * var_135[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_122_const_scalar_124_broadcast_to_123_broadcast_to_125_elementwise_mul_127_elementwise_mul_126_elementwise_add_128_fused_kernel(const float* __restrict__ batch_norm_2__w_2, const float* __restrict__ var_136, float* __restrict__ elementwise_add_Out_10)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_add_Out_10[(int)threadIdx.x] = ((0.9 * batch_norm_2__w_2[(int)threadIdx.x]) + (0.1 * var_136[(int)threadIdx.x]));
  };
}__global__
void __launch_bounds__(1024) fn_const_scalar_104_const_scalar_129_broadcast_to_109_broadcast_to_110_broadcast_to_105_broadcast_to_130_broadcast_to_93_substract_111_broadcast_to_106_elementwise_add_107_rsqrt_108_elementwise_mul_112_elementwise_mul_113_elementwise_add_114_max_131_fused_kernel(const float* __restrict__ batch_norm_2__w_0, const float* __restrict__ batch_norm_2__b_0, const float* __restrict__ var_135, const float* __restrict__ var_124, const float* __restrict__ var_136, float* __restrict__ max_Out_1)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (25 + (((128 + (int)blockIdx.x) / 256) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        max_Out_1[(55 + ((-200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-56 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) / 56)) + ((3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-1 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) % 56)) + ((1064 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (56 * (int)threadIdx.x))))))))))] = cinn_nvgpu_max_fp32(((batch_norm_2__w_0[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (rsqrt((var_136[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)) * var_124[(55 + ((-200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((200704 * ((200703 + ((199680 * (int)blockIdx.x) + ((139264 * i_j_fused_k_fused_a_fused_outer) + (200703 * (int)threadIdx.x)))) / 200704)) + ((-56 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) / 56)) + ((3136 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-1 * ((55 + ((40 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (55 * (int)threadIdx.x)))) % 56)) + ((1064 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (56 * (int)threadIdx.x))))))))))])) + ((-1 * (batch_norm_2__w_0[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (rsqrt((var_136[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)) * var_135[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)]))) + batch_norm_2__b_0[(((-1 * ((3135 + ((2112 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (3135 * (int)threadIdx.x)))) / 3136)) + ((-63 * (int)blockIdx.x) + ((-44 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)])), 0);
      };
    };
  };
}__global__
void __launch_bounds__(896) fn_conv2d_132_kernel(const float* __restrict__ var_142, const float* __restrict__ conv2d_3__w_0, float* __restrict__ Conv2d_nchw_out_3)
{
  float _Conv2d_nchw_out_3_write_cache [ 2 ];
  float* Conv2d_nchw_out_3_write_cache = _Conv2d_nchw_out_3_write_cache;
  float* Conv2d_nchw_out_3_write_cache__reduce_init = _Conv2d_nchw_out_3_write_cache;
  for (int32_t i = 0; i < 32; i += 1) {
    if (((int)blockIdx.z < 8)) {
      if (((int)blockIdx.y < 56)) {
        if (((int)threadIdx.z < 16)) {
          if (((int)threadIdx.x < 56)) {
          {
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_3_write_cache__reduce_init[j_inner] = 0;
              for (int32_t rc_3_outer = 0; rc_3_outer < 16; rc_3_outer += 1) {
                for (int32_t rc_3_inner = 0; rc_3_inner < 4; rc_3_inner += 1) {
                  Conv2d_nchw_out_3_write_cache[j_inner] = (Conv2d_nchw_out_3_write_cache[j_inner] + (var_142[((56 * (int)blockIdx.y) + ((200704 * i) + ((3136 * rc_3_inner) + ((12544 * rc_3_outer) + (int)threadIdx.x))))] * conv2d_3__w_0[((2048 * (int)blockIdx.z) + ((64 * j_inner) + ((4 * rc_3_outer) + ((128 * (int)threadIdx.z) + rc_3_inner))))]));
                };
              };
            };
            for (int32_t j_inner = 0; j_inner < 2; j_inner += 1) {
              Conv2d_nchw_out_3[((56 * (int)blockIdx.y) + ((100352 * (int)blockIdx.z) + ((802816 * i) + ((3136 * j_inner) + ((6272 * (int)threadIdx.z) + (int)threadIdx.x)))))] = Conv2d_nchw_out_3_write_cache[j_inner];
            };
          }
          };
        };
      };
    };
  };
}__global__
void __launch_bounds__(56) fn_reduce_sum_135_kernel(const float* __restrict__ var_146, float* __restrict__ reduce_sum_out_15)
{
  float* reduce_sum_out_15__reduce_init = reduce_sum_out_15;
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 56)) {
    {
      reduce_sum_out_15__reduce_init[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_23 = 0; kk_23 < 32; kk_23 += 1) {
        for (int32_t kk_24 = 0; kk_24 < 56; kk_24 += 1) {
          reduce_sum_out_15[((56 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_15[((56 * (int)blockIdx.x) + (int)threadIdx.x)] + var_146[((3136 * (int)blockIdx.x) + ((802816 * kk_23) + ((56 * kk_24) + (int)threadIdx.x)))]);
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(256) fn_reduce_sum_136_kernel(const float* __restrict__ var_1409, float* __restrict__ reduce_sum_out_16)
{
  float* reduce_sum_out_16__reduce_init = reduce_sum_out_16;
  if (((int)threadIdx.x < 256)) {
  {
    reduce_sum_out_16__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_25 = 0; kk_25 < 56; kk_25 += 1) {
      reduce_sum_out_16[(int)threadIdx.x] = (reduce_sum_out_16[(int)threadIdx.x] + var_1409[((56 * (int)threadIdx.x) + kk_25)]);
    };
  }
  };
}__global__
void __launch_bounds__(256) fn_const_scalar_133_broadcast_to_134_divide_137_fused_kernel(const float* __restrict__ var_1411, float* __restrict__ divide_Out_7)
{
  if (((int)threadIdx.x < 256)) {
    divide_Out_7[(int)threadIdx.x] = (9.96492e-06 * var_1411[(int)threadIdx.x]);
  };
}__global__
void __launch_bounds__(56) fn_identity_141_elementwise_mul_1
I1201 03:34:00.832091 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
I1201 03:34:17.656563 25408 compiler.cc:73] [CUDA] host module:
Module module_1_host {

function fn_broadcast_to_0_elementwise_add_1_fused (args__ptr, num_args)
{
  fn_broadcast_to_0_elementwise_add_1_fused_kernel(args__ptr, num_args)
}


}
I1201 03:34:17.656635 25408 compiler.cc:76] [CUDA] device module:
Module module_1_gpu_device {

function fn_broadcast_to_0_elementwise_add_1_fused_kernel (_linear_0__b_0, _linear_1__tmp_0, _elementwise_add_Out_227)
{
  if ((blockIdx.x < 32)) {
    if ((threadIdx.x < cinn_min(1024, (32000 + (-1024 * blockIdx.x))))) {
      elementwise_add_Out_227[((((24 * blockIdx.x) + threadIdx.x) / 1000) + blockIdx.x), (999 - ((31999 + ((-1024 * blockIdx.x) - threadIdx.x)) % 1000))] = (linear_1__tmp_0[((((24 * blockIdx.x) + threadIdx.x) / 1000) + blockIdx.x), (999 - ((31999 + ((-1024 * blockIdx.x) - threadIdx.x)) % 1000))] + linear_0__b_0[((999 - ((31999 + ((-1024 * blockIdx.x) - threadIdx.x)) % 1000)) % 1000)])
    }
  }
}


}
I1201 03:34:17.678232 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(1024) fn_broadcast_to_0_elementwise_add_1_fused_kernel(const float* __restrict__ linear_0__b_0, const float* __restrict__ linear_1__tmp_0, float* __restrict__ elementwise_add_Out_227)
{
  if (((int)blockIdx.x < 32)) {
    if (((int)threadIdx.x < cinn_nvgpu_min_fp32(1024, (32000 + (-1024 * (int)blockIdx.x))))) {
      elementwise_add_Out_227[(999 + ((1000 * (((24 * (int)blockIdx.x) + (int)threadIdx.x) / 1000)) + ((-1 * ((31999 + ((-1024 * (int)blockIdx.x) - (int)threadIdx.x)) % 1000)) + (1000 * (int)blockIdx.x))))] = (linear_1__tmp_0[(999 + ((1000 * (((24 * (int)blockIdx.x) + (int)threadIdx.x) / 1000)) + ((-1 * ((31999 + ((-1024 * (int)blockIdx.x) - (int)threadIdx.x)) % 1000)) + (1000 * (int)blockIdx.x))))] + linear_0__b_0[((999 - ((31999 + ((-1024 * (int)blockIdx.x) - (int)threadIdx.x)) % 1000)) % 1000)]);
    };
  };
}

}
I1201 03:34:17.678347 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
I1201 03:34:18.475033 25408 compiler.cc:73] [CUDA] host module:
Module module_2_host {

function fn_reduce_sum_1 (args__ptr, num_args)
{
  fn_reduce_sum_1_kernel(args__ptr, num_args)
}
function fn_identity_0 (args__ptr, num_args)
{
  fn_identity_0_kernel(args__ptr, num_args)
}
function fn_reshape_2 (args__ptr, num_args)
{
  fn_reshape_2_kernel(args__ptr, num_args)
}


}
I1201 03:34:18.475113 25408 compiler.cc:76] [CUDA] device module:
Module module_2_gpu_device {

function fn_reduce_sum_1_kernel (_linear_1__tmp_1____GRAD, _reduce_sum_out_211)
{
  if ((blockIdx.x < 2)) {
    if ((threadIdx.x < cinn_min(512, (1000 + (-512 * blockIdx.x))))) {
      {
        reduce_sum_out_211__reduce_init[0, ((512 * blockIdx.x) + threadIdx.x)] = 0
        for (kk_317, 0, 32)
        {
          reduce_sum_out_211[0, ((512 * blockIdx.x) + threadIdx.x)] = (reduce_sum_out_211[0, ((512 * blockIdx.x) + threadIdx.x)] + linear_1__tmp_1____GRAD[kk_317, ((512 * blockIdx.x) + threadIdx.x)])
        }
      }
    }
  }
}
function fn_identity_0_kernel (_linear_1__tmp_1____GRAD, _identity_Out_105)
{
  if ((blockIdx.x < 32)) {
    if ((threadIdx.x < cinn_min(1024, (32000 + (-1024 * blockIdx.x))))) {
      identity_Out_105[((((24 * blockIdx.x) + threadIdx.x) / 1000) + blockIdx.x), (999 - ((31999 + ((-1024 * blockIdx.x) - threadIdx.x)) % 1000))] = linear_1__tmp_1____GRAD[((((24 * blockIdx.x) + threadIdx.x) / 1000) + blockIdx.x), (999 - ((31999 + ((-1024 * blockIdx.x) - threadIdx.x)) % 1000))]
    }
  }
}
function fn_reshape_2_kernel (_var_4115, _Reshape_out)
{
  if ((threadIdx.x < 1000)) {
    Reshape_out[threadIdx.x] = var_4115_reshape[threadIdx.x]
  }
}


}
I1201 03:34:18.496805 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(512) fn_reduce_sum_1_kernel(const float* __restrict__ linear_1__tmp_1____GRAD, float* __restrict__ reduce_sum_out_211)
{
  float* reduce_sum_out_211__reduce_init = reduce_sum_out_211;
  if (((int)blockIdx.x < 2)) {
    if (((int)threadIdx.x < cinn_nvgpu_min_fp32(512, (1000 + (-512 * (int)blockIdx.x))))) {
    {
      reduce_sum_out_211__reduce_init[((512 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_317 = 0; kk_317 < 32; kk_317 += 1) {
        reduce_sum_out_211[((512 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_211[((512 * (int)blockIdx.x) + (int)threadIdx.x)] + linear_1__tmp_1____GRAD[((512 * (int)blockIdx.x) + ((1000 * kk_317) + (int)threadIdx.x))]);
      };
    }
    };
  };
}__global__
void __launch_bounds__(1024) fn_identity_0_kernel(const float* __restrict__ linear_1__tmp_1____GRAD, float* __restrict__ identity_Out_105)
{
  if (((int)blockIdx.x < 32)) {
    if (((int)threadIdx.x < cinn_nvgpu_min_fp32(1024, (32000 + (-1024 * (int)blockIdx.x))))) {
      identity_Out_105[(999 + ((1000 * (((24 * (int)blockIdx.x) + (int)threadIdx.x) / 1000)) + ((-1 * ((31999 + ((-1024 * (int)blockIdx.x) - (int)threadIdx.x)) % 1000)) + (1000 * (int)blockIdx.x))))] = linear_1__tmp_1____GRAD[(999 + ((1000 * (((24 * (int)blockIdx.x) + (int)threadIdx.x) / 1000)) + ((-1 * ((31999 + ((-1024 * (int)blockIdx.x) - (int)threadIdx.x)) % 1000)) + (1000 * (int)blockIdx.x))))];
    };
  };
}__global__
void __launch_bounds__(1000) fn_reshape_2_kernel(const float* __restrict__ var_4115, float* __restrict__ Reshape_out)
{
  const float* var_4115_reshape = var_4115;
  if (((int)threadIdx.x < 1000)) {
    Reshape_out[(int)threadIdx.x] = var_4115_reshape[(int)threadIdx.x];
  };
}

}
I1201 03:34:18.496902 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
I1201 03:34:19.086549 25408 compiler.cc:73] [CUDA] host module:
Module module_3_host {

function fn_identity_0_1 (args__ptr, num_args)
{
  fn_identity_0_1_kernel(args__ptr, num_args)
}


}
I1201 03:34:19.086607 25408 compiler.cc:76] [CUDA] device module:
Module module_3_gpu_device {

function fn_identity_0_1_kernel (_tmp_0, _identity_Out_106)
{
  identity_Out_106[0] = tmp_0[0]
}


}
I1201 03:34:19.086851 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(1) fn_identity_0_1_kernel(const float* __restrict__ tmp_0, float* __restrict__ identity_Out_106)
{
  identity_Out_106[0] = tmp_0[0];
}

}
I1201 03:34:19.086915 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
I1201 03:38:07.570636 25408 compiler.cc:73] [CUDA] host module:
Module module_4_host {

function fn_identity_6 (args__ptr, num_args)
{
  fn_identity_6_kernel(args__ptr, num_args)
}
function fn_identity_48 (args__ptr, num_args)
{
  fn_identity_48_kernel(args__ptr, num_args)
}
function fn_identity_90 (args__ptr, num_args)
{
  fn_identity_90_kernel(args__ptr, num_args)
}
function fn_identity_136 (args__ptr, num_args)
{
  fn_identity_136_kernel(args__ptr, num_args)
}
function fn_identity_178 (args__ptr, num_args)
{
  fn_identity_178_kernel(args__ptr, num_args)
}
function fn_identity_220 (args__ptr, num_args)
{
  fn_identity_220_kernel(args__ptr, num_args)
}
function fn_identity_266 (args__ptr, num_args)
{
  fn_identity_266_kernel(args__ptr, num_args)
}
function fn_identity_302 (args__ptr, num_args)
{
  fn_identity_302_kernel(args__ptr, num_args)
}
function fn_identity_346 (args__ptr, num_args)
{
  fn_identity_346_kernel(args__ptr, num_args)
}
function fn_identity_388 (args__ptr, num_args)
{
  fn_identity_388_kernel(args__ptr, num_args)
}
function fn_identity_434 (args__ptr, num_args)
{
  fn_identity_434_kernel(args__ptr, num_args)
}
function fn_identity_476 (args__ptr, num_args)
{
  fn_identity_476_kernel(args__ptr, num_args)
}
function fn_identity_518 (args__ptr, num_args)
{
  fn_identity_518_kernel(args__ptr, num_args)
}
function fn_identity_564 (args__ptr, num_args)
{
  fn_identity_564_kernel(args__ptr, num_args)
}
function fn_identity_606 (args__ptr, num_args)
{
  fn_identity_606_kernel(args__ptr, num_args)
}
function fn_identity_648 (args__ptr, num_args)
{
  fn_identity_648_kernel(args__ptr, num_args)
}
function fn_identity_694 (args__ptr, num_args)
{
  fn_identity_694_kernel(args__ptr, num_args)
}
function fn_identity_736 (args__ptr, num_args)
{
  fn_identity_736_kernel(args__ptr, num_args)
}
function fn_identity_778 (args__ptr, num_args)
{
  fn_identity_778_kernel(args__ptr, num_args)
}
function fn_identity_824 (args__ptr, num_args)
{
  fn_identity_824_kernel(args__ptr, num_args)
}
function fn_identity_866 (args__ptr, num_args)
{
  fn_identity_866_kernel(args__ptr, num_args)
}
function fn_identity_908 (args__ptr, num_args)
{
  fn_identity_908_kernel(args__ptr, num_args)
}
function fn_identity_954 (args__ptr, num_args)
{
  fn_identity_954_kernel(args__ptr, num_args)
}
function fn_identity_996 (args__ptr, num_args)
{
  fn_identity_996_kernel(args__ptr, num_args)
}
function fn_identity_1038 (args__ptr, num_args)
{
  fn_identity_1038_kernel(args__ptr, num_args)
}
function fn_identity_1084 (args__ptr, num_args)
{
  fn_identity_1084_kernel(args__ptr, num_args)
}
function fn_identity_1120 (args__ptr, num_args)
{
  fn_identity_1120_kernel(args__ptr, num_args)
}
function fn_identity_1164 (args__ptr, num_args)
{
  fn_identity_1164_kernel(args__ptr, num_args)
}
function fn_identity_1206 (args__ptr, num_args)
{
  fn_identity_1206_kernel(args__ptr, num_args)
}
function fn_identity_1252 (args__ptr, num_args)
{
  fn_identity_1252_kernel(args__ptr, num_args)
}
function fn_identity_1294 (args__ptr, num_args)
{
  fn_identity_1294_kernel(args__ptr, num_args)
}
function fn_identity_1336 (args__ptr, num_args)
{
  fn_identity_1336_kernel(args__ptr, num_args)
}
function fn_identity_1382 (args__ptr, num_args)
{
  fn_identity_1382_kernel(args__ptr, num_args)
}
function fn_identity_1424 (args__ptr, num_args)
{
  fn_identity_1424_kernel(args__ptr, num_args)
}
function fn_identity_1466 (args__ptr, num_args)
{
  fn_identity_1466_kernel(args__ptr, num_args)
}
function fn_identity_1512 (args__ptr, num_args)
{
  fn_identity_1512_kernel(args__ptr, num_args)
}
function fn_identity_1554 (args__ptr, num_args)
{
  fn_identity_1554_kernel(args__ptr, num_args)
}
function fn_identity_1596 (args__ptr, num_args)
{
  fn_identity_1596_kernel(args__ptr, num_args)
}
function fn_identity_1642 (args__ptr, num_args)
{
  fn_identity_1642_kernel(args__ptr, num_args)
}
function fn_identity_1678 (args__ptr, num_args)
{
  fn_identity_1678_kernel(args__ptr, num_args)
}
function fn_identity_1722 (args__ptr, num_args)
{
  fn_identity_1722_kernel(args__ptr, num_args)
}
function fn_identity_1764 (args__ptr, num_args)
{
  fn_identity_1764_kernel(args__ptr, num_args)
}
function fn_identity_1810 (args__ptr, num_args)
{
  fn_identity_1810_kernel(args__ptr, num_args)
}
function fn_identity_1852 (args__ptr, num_args)
{
  fn_identity_1852_kernel(args__ptr, num_args)
}
function fn_identity_1894 (args__ptr, num_args)
{
  fn_identity_1894_kernel(args__ptr, num_args)
}
function fn_identity_1940 (args__ptr, num_args)
{
  fn_identity_1940_kernel(args__ptr, num_args)
}
function fn_identity_1982 (args__ptr, num_args)
{
  fn_identity_1982_kernel(args__ptr, num_args)
}
function fn_identity_2024 (args__ptr, num_args)
{
  fn_identity_2024_kernel(args__ptr, num_args)
}
function fn_identity_2070 (args__ptr, num_args)
{
  fn_identity_2070_kernel(args__ptr, num_args)
}
function fn_identity_2106 (args__ptr, num_args)
{
  fn_identity_2106_kernel(args__ptr, num_args)
}
function fn_identity_2150 (args__ptr, num_args)
{
  fn_identity_2150_kernel(args__ptr, num_args)
}
function fn_identity_2192 (args__ptr, num_args)
{
  fn_identity_2192_kernel(args__ptr, num_args)
}
function fn_broadcast_to_9_substract_10_fused (args__ptr, num_args)
{
  fn_broadcast_to_9_substract_10_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_51_substract_52_fused (args__ptr, num_args)
{
  fn_broadcast_to_51_substract_52_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_93_substract_94_fused (args__ptr, num_args)
{
  fn_broadcast_to_93_substract_94_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_139_substract_140_fused (args__ptr, num_args)
{
  fn_broadcast_to_139_substract_140_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_181_substract_182_fused (args__ptr, num_args)
{
  fn_broadcast_to_181_substract_182_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_223_substract_224_fused (args__ptr, num_args)
{
  fn_broadcast_to_223_substract_224_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_269_substract_270_fused (args__ptr, num_args)
{
  fn_broadcast_to_269_substract_270_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_305_substract_306_fused (args__ptr, num_args)
{
  fn_broadcast_to_305_substract_306_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_349_substract_350_fused (args__ptr, num_args)
{
  fn_broadcast_to_349_substract_350_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_391_substract_392_fused (args__ptr, num_args)
{
  fn_broadcast_to_391_substract_392_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_437_substract_438_fused (args__ptr, num_args)
{
  fn_broadcast_to_437_substract_438_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_479_substract_480_fused (args__ptr, num_args)
{
  fn_broadcast_to_479_substract_480_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_521_substract_522_fused (args__ptr, num_args)
{
  fn_broadcast_to_521_substract_522_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_567_substract_568_fused (args__ptr, num_args)
{
  fn_broadcast_to_567_substract_568_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_609_substract_610_fused (args__ptr, num_args)
{
  fn_broadcast_to_609_substract_610_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_651_substract_652_fused (args__ptr, num_args)
{
  fn_broadcast_to_651_substract_652_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_697_substract_698_fused (args__ptr, num_args)
{
  fn_broadcast_to_697_substract_698_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_739_substract_740_fused (args__ptr, num_args)
{
  fn_broadcast_to_739_substract_740_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_781_substract_782_fused (args__ptr, num_args)
{
  fn_broadcast_to_781_substract_782_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_827_substract_828_fused (args__ptr, num_args)
{
  fn_broadcast_to_827_substract_828_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_869_substract_870_fused (args__ptr, num_args)
{
  fn_broadcast_to_869_substract_870_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_911_substract_912_fused (args__ptr, num_args)
{
  fn_broadcast_to_911_substract_912_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_957_substract_958_fused (args__ptr, num_args)
{
  fn_broadcast_to_957_substract_958_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_999_substract_1000_fused (args__ptr, num_args)
{
  fn_broadcast_to_999_substract_1000_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1041_substract_1042_fused (args__ptr, num_args)
{
  fn_broadcast_to_1041_substract_1042_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1087_substract_1088_fused (args__ptr, num_args)
{
  fn_broadcast_to_1087_substract_1088_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1123_substract_1124_fused (args__ptr, num_args)
{
  fn_broadcast_to_1123_substract_1124_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1167_substract_1168_fused (args__ptr, num_args)
{
  fn_broadcast_to_1167_substract_1168_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1209_substract_1210_fused (args__ptr, num_args)
{
  fn_broadcast_to_1209_substract_1210_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1255_substract_1256_fused (args__ptr, num_args)
{
  fn_broadcast_to_1255_substract_1256_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1297_substract_1298_fused (args__ptr, num_args)
{
  fn_broadcast_to_1297_substract_1298_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1339_substract_1340_fused (args__ptr, num_args)
{
  fn_broadcast_to_1339_substract_1340_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1385_substract_1386_fused (args__ptr, num_args)
{
  fn_broadcast_to_1385_substract_1386_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1427_substract_1428_fused (args__ptr, num_args)
{
  fn_broadcast_to_1427_substract_1428_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1469_substract_1470_fused (args__ptr, num_args)
{
  fn_broadcast_to_1469_substract_1470_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1515_substract_1516_fused (args__ptr, num_args)
{
  fn_broadcast_to_1515_substract_1516_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1557_substract_1558_fused (args__ptr, num_args)
{
  fn_broadcast_to_1557_substract_1558_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1599_substract_1600_fused (args__ptr, num_args)
{
  fn_broadcast_to_1599_substract_1600_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1645_substract_1646_fused (args__ptr, num_args)
{
  fn_broadcast_to_1645_substract_1646_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1681_substract_1682_fused (args__ptr, num_args)
{
  fn_broadcast_to_1681_substract_1682_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1725_substract_1726_fused (args__ptr, num_args)
{
  fn_broadcast_to_1725_substract_1726_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1767_substract_1768_fused (args__ptr, num_args)
{
  fn_broadcast_to_1767_substract_1768_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1813_substract_1814_fused (args__ptr, num_args)
{
  fn_broadcast_to_1813_substract_1814_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1855_substract_1856_fused (args__ptr, num_args)
{
  fn_broadcast_to_1855_substract_1856_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1897_substract_1898_fused (args__ptr, num_args)
{
  fn_broadcast_to_1897_substract_1898_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1943_substract_1944_fused (args__ptr, num_args)
{
  fn_broadcast_to_1943_substract_1944_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_1985_substract_1986_fused (args__ptr, num_args)
{
  fn_broadcast_to_1985_substract_1986_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_2027_substract_2028_fused (args__ptr, num_args)
{
  fn_broadcast_to_2027_substract_2028_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_2073_substract_2074_fused (args__ptr, num_args)
{
  fn_broadcast_to_2073_substract_2074_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_2109_substract_2110_fused (args__ptr, num_args)
{
  fn_broadcast_to_2109_substract_2110_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_2153_substract_2154_fused (args__ptr, num_args)
{
  fn_broadcast_to_2153_substract_2154_fused_kernel(args__ptr, num_args)
}
function fn_broadcast_to_2195_substract_2196_fused (args__ptr, num_args)
{
  fn_broadcast_to_2195_substract_2196_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_0_broadcast_to_1_greater_2_select_3_fused (args__ptr, num_args)
{
  fn_const_scalar_0_broadcast_to_1_greater_2_select_3_fused_kernel(args__ptr, num_args)
}
function fn_identity_4 (args__ptr, num_args)
{
  fn_identity_4_kernel(args__ptr, num_args)
}
function fn_reduce_sum_7 (args__ptr, num_args)
{
  fn_reduce_sum_7_kernel(args__ptr, num_args)
}
function fn_reduce_sum_8 (args__ptr, num_args)
{
  fn_reduce_sum_8_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_11_reduce_sum_12_fused (args__ptr, num_args)
{
  fn_elementwise_mul_11_reduce_sum_12_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_13_1 (args__ptr, num_args)
{
  fn_reduce_sum_13_1_kernel(args__ptr, num_args)
}
function fn_const_scalar_14_broadcast_to_15_elementwise_add_16_rsqrt_17_elementwise_mul_18_fused (args__ptr, num_args)
{
  fn_const_scalar_14_broadcast_to_15_elementwise_add_16_rsqrt_17_elementwise_mul_18_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_19_const_scalar_21_const_scalar_28_const_scalar_34_broadcast_to_20_broadcast_to_22_broadcast_to_29_broadcast_to_35_elementwise_add_23_elementwise_add_36_rsqrt_24_broadcast_to_37_elementwise_mul_25_elementwise_mul_30_divide_26_broadcast_to_27_broadcast_to_31_substract_39_broadcast_to_32_elementwise_mul_33_divide_38_substract_40_elementwise_mul_41_fused (args__ptr, num_args)
{
  fn_const_scalar_19_const_scalar_21_const_scalar_28_const_scalar_34_broadcast_to_20_broadcast_to_22_broadcast_to_29_broadcast_to_35_elementwise_add_23_elementwise_add_36_rsqrt_24_broadcast_to_37_elementwise_mul_25_elementwise_mul_30_divide_26_broadcast_to_27_broadcast_to_31_substract_39_broadcast_to_32_elementwise_mul_33_divide_38_substract_40_elementwise_mul_41_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_43 (args__ptr, num_args)
{
  fn_conv2d_43_kernel(args__ptr, num_args)
}
function fn_conv2d_42 (args__ptr, num_args)
{
  fn_conv2d_42_kernel(args__ptr, num_args)
}
function fn_const_scalar_44_broadcast_to_45_greater_46_select_47_fused (args__ptr, num_args)
{
  fn_const_scalar_44_broadcast_to_45_greater_46_select_47_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_49 (args__ptr, num_args)
{
  fn_reduce_sum_49_kernel(args__ptr, num_args)
}
function fn_reduce_sum_50 (args__ptr, num_args)
{
  fn_reduce_sum_50_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_53_reduce_sum_54_fused (args__ptr, num_args)
{
  fn_elementwise_mul_53_reduce_sum_54_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_55 (args__ptr, num_args)
{
  fn_reduce_sum_55_kernel(args__ptr, num_args)
}
function fn_const_scalar_56_broadcast_to_57_elementwise_add_58_rsqrt_59_elementwise_mul_60_fused (args__ptr, num_args)
{
  fn_const_scalar_56_broadcast_to_57_elementwise_add_58_rsqrt_59_elementwise_mul_60_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_61_const_scalar_63_const_scalar_70_const_scalar_76_broadcast_to_62_broadcast_to_64_broadcast_to_71_broadcast_to_77_elementwise_add_65_elementwise_add_78_rsqrt_66_broadcast_to_79_elementwise_mul_67_divide_68_broadcast_to_69_elementwise_mul_72_broadcast_to_73_substract_81_broadcast_to_74_elementwise_mul_75_divide_80_substract_82_elementwise_mul_83_fused (args__ptr, num_args)
{
  fn_const_scalar_61_const_scalar_63_const_scalar_70_const_scalar_76_broadcast_to_62_broadcast_to_64_broadcast_to_71_broadcast_to_77_elementwise_add_65_elementwise_add_78_rsqrt_66_broadcast_to_79_elementwise_mul_67_divide_68_broadcast_to_69_elementwise_mul_72_broadcast_to_73_substract_81_broadcast_to_74_elementwise_mul_75_divide_80_substract_82_elementwise_mul_83_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_85 (args__ptr, num_args)
{
  fn_conv2d_85_kernel(args__ptr, num_args)
}
function fn_conv2d_84 (args__ptr, num_args)
{
  fn_conv2d_84_kernel(args__ptr, num_args)
}
function fn_const_scalar_86_broadcast_to_87_greater_88_select_89_fused (args__ptr, num_args)
{
  fn_const_scalar_86_broadcast_to_87_greater_88_select_89_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_91_1 (args__ptr, num_args)
{
  fn_reduce_sum_91_1_kernel(args__ptr, num_args)
}
function fn_reduce_sum_92 (args__ptr, num_args)
{
  fn_reduce_sum_92_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_95_reduce_sum_96_fused (args__ptr, num_args)
{
  fn_elementwise_mul_95_reduce_sum_96_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_97 (args__ptr, num_args)
{
  fn_reduce_sum_97_kernel(args__ptr, num_args)
}
function fn_const_scalar_98_broadcast_to_99_elementwise_add_100_rsqrt_101_elementwise_mul_102_fused (args__ptr, num_args)
{
  fn_const_scalar_98_broadcast_to_99_elementwise_add_100_rsqrt_101_elementwise_mul_102_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_103_const_scalar_105_const_scalar_112_const_scalar_118_broadcast_to_104_broadcast_to_106_broadcast_to_113_broadcast_to_119_elementwise_add_107_elementwise_add_120_rsqrt_108_broadcast_to_121_elementwise_mul_109_divide_110_broadcast_to_111_elementwise_mul_114_broadcast_to_115_substract_123_broadcast_to_116_elementwise_mul_117_divide_122_substract_124_elementwise_mul_125_fused (args__ptr, num_args)
{
  fn_const_scalar_103_const_scalar_105_const_scalar_112_const_scalar_118_broadcast_to_104_broadcast_to_106_broadcast_to_113_broadcast_to_119_elementwise_add_107_elementwise_add_120_rsqrt_108_broadcast_to_121_elementwise_mul_109_divide_110_broadcast_to_111_elementwise_mul_114_broadcast_to_115_substract_123_broadcast_to_116_elementwise_mul_117_divide_122_substract_124_elementwise_mul_125_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_127 (args__ptr, num_args)
{
  fn_conv2d_127_kernel(args__ptr, num_args)
}
function fn_conv2d_126 (args__ptr, num_args)
{
  fn_conv2d_126_kernel(args__ptr, num_args)
}
function fn_const_scalar_130_broadcast_to_131_greater_132_identity_5_identity_128_elementwise_add_129_select_133_fused (args__ptr, num_args)
{
  fn_const_scalar_130_broadcast_to_131_greater_132_identity_5_identity_128_elementwise_add_129_select_133_fused_kernel(args__ptr, num_args)
}
function fn_identity_134 (args__ptr, num_args)
{
  fn_identity_134_kernel(args__ptr, num_args)
}
function fn_reduce_sum_137 (args__ptr, num_args)
{
  fn_reduce_sum_137_kernel(args__ptr, num_args)
}
function fn_reduce_sum_138 (args__ptr, num_args)
{
  fn_reduce_sum_138_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_141_reduce_sum_142_fused (args__ptr, num_args)
{
  fn_elementwise_mul_141_reduce_sum_142_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_143 (args__ptr, num_args)
{
  fn_reduce_sum_143_kernel(args__ptr, num_args)
}
function fn_const_scalar_144_broadcast_to_145_elementwise_add_146_rsqrt_147_elementwise_mul_148_fused (args__ptr, num_args)
{
  fn_const_scalar_144_broadcast_to_145_elementwise_add_146_rsqrt_147_elementwise_mul_148_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_149_const_scalar_151_const_scalar_158_const_scalar_164_broadcast_to_150_broadcast_to_152_broadcast_to_159_broadcast_to_165_elementwise_add_153_elementwise_add_166_rsqrt_154_broadcast_to_167_elementwise_mul_155_divide_156_broadcast_to_157_elementwise_mul_160_broadcast_to_161_substract_169_broadcast_to_162_elementwise_mul_163_divide_168_substract_170_elementwise_mul_171_fused (args__ptr, num_args)
{
  fn_const_scalar_149_const_scalar_151_const_scalar_158_const_scalar_164_broadcast_to_150_broadcast_to_152_broadcast_to_159_broadcast_to_165_elementwise_add_153_elementwise_add_166_rsqrt_154_broadcast_to_167_elementwise_mul_155_divide_156_broadcast_to_157_elementwise_mul_160_broadcast_to_161_substract_169_broadcast_to_162_elementwise_mul_163_divide_168_substract_170_elementwise_mul_171_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_173 (args__ptr, num_args)
{
  fn_conv2d_173_kernel(args__ptr, num_args)
}
function fn_conv2d_172 (args__ptr, num_args)
{
  fn_conv2d_172_kernel(args__ptr, num_args)
}
function fn_const_scalar_174_broadcast_to_175_greater_176_select_177_fused (args__ptr, num_args)
{
  fn_const_scalar_174_broadcast_to_175_greater_176_select_177_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_179 (args__ptr, num_args)
{
  fn_reduce_sum_179_kernel(args__ptr, num_args)
}
function fn_reduce_sum_180 (args__ptr, num_args)
{
  fn_reduce_sum_180_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_183_reduce_sum_184_fused (args__ptr, num_args)
{
  fn_elementwise_mul_183_reduce_sum_184_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_185 (args__ptr, num_args)
{
  fn_reduce_sum_185_kernel(args__ptr, num_args)
}
function fn_const_scalar_186_broadcast_to_187_elementwise_add_188_rsqrt_189_elementwise_mul_190_fused (args__ptr, num_args)
{
  fn_const_scalar_186_broadcast_to_187_elementwise_add_188_rsqrt_189_elementwise_mul_190_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_191_const_scalar_193_const_scalar_200_const_scalar_206_broadcast_to_192_broadcast_to_194_broadcast_to_201_broadcast_to_207_elementwise_add_195_elementwise_add_208_rsqrt_196_broadcast_to_209_elementwise_mul_197_divide_198_broadcast_to_199_elementwise_mul_202_broadcast_to_203_substract_211_broadcast_to_204_elementwise_mul_205_divide_210_substract_212_elementwise_mul_213_fused (args__ptr, num_args)
{
  fn_const_scalar_191_const_scalar_193_const_scalar_200_const_scalar_206_broadcast_to_192_broadcast_to_194_broadcast_to_201_broadcast_to_207_elementwise_add_195_elementwise_add_208_rsqrt_196_broadcast_to_209_elementwise_mul_197_divide_198_broadcast_to_199_elementwise_mul_202_broadcast_to_203_substract_211_broadcast_to_204_elementwise_mul_205_divide_210_substract_212_elementwise_mul_213_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_215 (args__ptr, num_args)
{
  fn_conv2d_215_kernel(args__ptr, num_args)
}
function fn_conv2d_214 (args__ptr, num_args)
{
  fn_conv2d_214_kernel(args__ptr, num_args)
}
function fn_const_scalar_216_broadcast_to_217_greater_218_select_219_fused (args__ptr, num_args)
{
  fn_const_scalar_216_broadcast_to_217_greater_218_select_219_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_221 (args__ptr, num_args)
{
  fn_reduce_sum_221_kernel(args__ptr, num_args)
}
function fn_reduce_sum_222 (args__ptr, num_args)
{
  fn_reduce_sum_222_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_225_reduce_sum_226_fused (args__ptr, num_args)
{
  fn_elementwise_mul_225_reduce_sum_226_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_227_1 (args__ptr, num_args)
{
  fn_reduce_sum_227_1_kernel(args__ptr, num_args)
}
function fn_const_scalar_228_broadcast_to_229_elementwise_add_230_rsqrt_231_elementwise_mul_232_fused (args__ptr, num_args)
{
  fn_const_scalar_228_broadcast_to_229_elementwise_add_230_rsqrt_231_elementwise_mul_232_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_233_const_scalar_235_const_scalar_242_const_scalar_248_broadcast_to_234_broadcast_to_236_broadcast_to_243_broadcast_to_249_elementwise_add_237_elementwise_add_250_rsqrt_238_broadcast_to_251_elementwise_mul_239_divide_240_broadcast_to_241_elementwise_mul_244_broadcast_to_245_substract_253_broadcast_to_246_elementwise_mul_247_divide_252_substract_254_elementwise_mul_255_fused (args__ptr, num_args)
{
  fn_const_scalar_233_const_scalar_235_const_scalar_242_const_scalar_248_broadcast_to_234_broadcast_to_236_broadcast_to_243_broadcast_to_249_elementwise_add_237_elementwise_add_250_rsqrt_238_broadcast_to_251_elementwise_mul_239_divide_240_broadcast_to_241_elementwise_mul_244_broadcast_to_245_substract_253_broadcast_to_246_elementwise_mul_247_divide_252_substract_254_elementwise_mul_255_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_257 (args__ptr, num_args)
{
  fn_conv2d_257_kernel(args__ptr, num_args)
}
function fn_conv2d_256 (args__ptr, num_args)
{
  fn_conv2d_256_kernel(args__ptr, num_args)
}
function fn_const_scalar_260_broadcast_to_261_greater_262_identity_135_identity_258_elementwise_add_259_select_263_fused (args__ptr, num_args)
{
  fn_const_scalar_260_broadcast_to_261_greater_262_identity_135_identity_258_elementwise_add_259_select_263_fused_kernel(args__ptr, num_args)
}
function fn_identity_265 (args__ptr, num_args)
{
  fn_identity_265_kernel(args__ptr, num_args)
}
function fn_identity_264 (args__ptr, num_args)
{
  fn_identity_264_kernel(args__ptr, num_args)
}
function fn_reduce_sum_267 (args__ptr, num_args)
{
  fn_reduce_sum_267_kernel(args__ptr, num_args)
}
function fn_reduce_sum_303 (args__ptr, num_args)
{
  fn_reduce_sum_303_kernel(args__ptr, num_args)
}
function fn_reduce_sum_268 (args__ptr, num_args)
{
  fn_reduce_sum_268_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_271_reduce_sum_272_fused (args__ptr, num_args)
{
  fn_elementwise_mul_271_reduce_sum_272_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_304 (args__ptr, num_args)
{
  fn_reduce_sum_304_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_307_reduce_sum_308_fused (args__ptr, num_args)
{
  fn_elementwise_mul_307_reduce_sum_308_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_273 (args__ptr, num_args)
{
  fn_reduce_sum_273_kernel(args__ptr, num_args)
}
function fn_reduce_sum_309 (args__ptr, num_args)
{
  fn_reduce_sum_309_kernel(args__ptr, num_args)
}
function fn_const_scalar_274_broadcast_to_275_elementwise_add_276_rsqrt_277_elementwise_mul_278_fused (args__ptr, num_args)
{
  fn_const_scalar_274_broadcast_to_275_elementwise_add_276_rsqrt_277_elementwise_mul_278_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_310_broadcast_to_311_elementwise_add_312_rsqrt_313_elementwise_mul_314_fused (args__ptr, num_args)
{
  fn_const_scalar_310_broadcast_to_311_elementwise_add_312_rsqrt_313_elementwise_mul_314_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_279_const_scalar_281_const_scalar_288_const_scalar_294_broadcast_to_280_broadcast_to_282_broadcast_to_289_broadcast_to_295_elementwise_add_283_elementwise_add_296_rsqrt_284_broadcast_to_297_elementwise_mul_285_divide_286_broadcast_to_287_elementwise_mul_290_broadcast_to_291_substract_299_broadcast_to_292_elementwise_mul_293_divide_298_substract_300_elementwise_mul_301_fused (args__ptr, num_args)
{
  fn_const_scalar_279_const_scalar_281_const_scalar_288_const_scalar_294_broadcast_to_280_broadcast_to_282_broadcast_to_289_broadcast_to_295_elementwise_add_283_elementwise_add_296_rsqrt_284_broadcast_to_297_elementwise_mul_285_divide_286_broadcast_to_287_elementwise_mul_290_broadcast_to_291_substract_299_broadcast_to_292_elementwise_mul_293_divide_298_substract_300_elementwise_mul_301_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_315_const_scalar_317_const_scalar_324_const_scalar_330_broadcast_to_316_broadcast_to_318_broadcast_to_325_broadcast_to_331_elementwise_add_319_elementwise_add_332_rsqrt_320_broadcast_to_333_elementwise_mul_321_divide_322_broadcast_to_323_elementwise_mul_326_broadcast_to_327_substract_335_broadcast_to_328_elementwise_mul_329_divide_334_substract_336_elementwise_mul_337_fused (args__ptr, num_args)
{
  fn_const_scalar_315_const_scalar_317_const_scalar_324_const_scalar_330_broadcast_to_316_broadcast_to_318_broadcast_to_325_broadcast_to_331_elementwise_add_319_elementwise_add_332_rsqrt_320_broadcast_to_333_elementwise_mul_321_divide_322_broadcast_to_323_elementwise_mul_326_broadcast_to_327_substract_335_broadcast_to_328_elementwise_mul_329_divide_334_substract_336_elementwise_mul_337_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_339 (args__ptr, num_args)
{
  fn_conv2d_339_kernel(args__ptr, num_args)
}
function fn_conv2d_338 (args__ptr, num_args)
{
  fn_conv2d_338_kernel(args__ptr, num_args)
}
function fn_conv2d_341 (args__ptr, num_args)
{
  fn_conv2d_341_kernel(args__ptr, num_args)
}
function fn_conv2d_340 (args__ptr, num_args)
{
  fn_conv2d_340_kernel(args__ptr, num_args)
}
function fn_const_scalar_342_broadcast_to_343_greater_344_select_345_fused (args__ptr, num_args)
{
  fn_const_scalar_342_broadcast_to_343_greater_344_select_345_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_347 (args__ptr, num_args)
{
  fn_reduce_sum_347_kernel(args__ptr, num_args)
}
function fn_reduce_sum_348 (args__ptr, num_args)
{
  fn_reduce_sum_348_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_351_reduce_sum_352_fused (args__ptr, num_args)
{
  fn_elementwise_mul_351_reduce_sum_352_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_353 (args__ptr, num_args)
{
  fn_reduce_sum_353_kernel(args__ptr, num_args)
}
function fn_const_scalar_354_broadcast_to_355_elementwise_add_356_rsqrt_357_elementwise_mul_358_fused (args__ptr, num_args)
{
  fn_const_scalar_354_broadcast_to_355_elementwise_add_356_rsqrt_357_elementwise_mul_358_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_359_const_scalar_361_const_scalar_368_const_scalar_374_broadcast_to_360_broadcast_to_362_broadcast_to_369_broadcast_to_375_elementwise_add_363_elementwise_add_376_rsqrt_364_broadcast_to_377_elementwise_mul_365_divide_366_broadcast_to_367_elementwise_mul_370_broadcast_to_371_substract_379_broadcast_to_372_elementwise_mul_373_divide_378_substract_380_elementwise_mul_381_fused (args__ptr, num_args)
{
  fn_const_scalar_359_const_scalar_361_const_scalar_368_const_scalar_374_broadcast_to_360_broadcast_to_362_broadcast_to_369_broadca
I1201 03:38:07.573915 25408 compiler.cc:76] [CUDA] device module:
Module module_4_gpu_device {

function fn_identity_6_kernel (_batch_norm_52__b_0, _identity_Out_107)
{
  if ((blockIdx.x < 2)) {
    if ((threadIdx.x < 1024)) {
      identity_Out_107[((1024 * blockIdx.x) + threadIdx.x)] = batch_norm_52__b_0[((1024 * blockIdx.x) + threadIdx.x)]
    }
  }
}
function fn_identity_48_kernel (_batch_norm_51__b_0, _identity_Out_108)
{
  if ((threadIdx.x < 512)) {
    identity_Out_108[threadIdx.x] = batch_norm_51__b_0[threadIdx.x]
  }
}
function fn_identity_90_kernel (_batch_norm_50__b_0, _identity_Out_109)
{
  if ((threadIdx.x < 512)) {
    identity_Out_109[threadIdx.x] = batch_norm_50__b_0[threadIdx.x]
  }
}
function fn_identity_136_kernel (_batch_norm_49__b_0, _identity_Out_110)
{
  if ((blockIdx.x < 2)) {
    if ((threadIdx.x < 1024)) {
      identity_Out_110[((1024 * blockIdx.x) + threadIdx.x)] = batch_norm_49__b_0[((1024 * blockIdx.x) + threadIdx.x)]
    }
  }
}
function fn_identity_178_kernel (_batch_norm_48__b_0, _identity_Out_111)
{
  if ((threadIdx.x < 512)) {
    identity_Out_111[threadIdx.x] = batch_norm_48__b_0[threadIdx.x]
  }
}
function fn_identity_220_kernel (_batch_norm_47__b_0, _identity_Out_112)
{
  if ((threadIdx.x < 512)) {
    identity_Out_112[threadIdx.x] = batch_norm_47__b_0[threadIdx.x]
  }
}
function fn_identity_266_kernel (_batch_norm_46__b_0, _identity_Out_113)
{
  if ((blockIdx.x < 2)) {
    if ((threadIdx.x < 1024)) {
      identity_Out_113[((1024 * blockIdx.x) + threadIdx.x)] = batch_norm_46__b_0[((1024 * blockIdx.x) + threadIdx.x)]
    }
  }
}
function fn_identity_302_kernel (_batch_norm_45__b_0, _identity_Out_114)
{
  if ((blockIdx.x < 2)) {
    if ((threadIdx.x < 1024)) {
      identity_Out_114[((1024 * blockIdx.x) + threadIdx.x)] = batch_norm_45__b_0[((1024 * blockIdx.x) + threadIdx.x)]
    }
  }
}
function fn_identity_346_kernel (_batch_norm_44__b_0, _identity_Out_115)
{
  if ((threadIdx.x < 512)) {
    identity_Out_115[threadIdx.x] = batch_norm_44__b_0[threadIdx.x]
  }
}
function fn_identity_388_kernel (_batch_norm_43__b_0, _identity_Out_116)
{
  if ((threadIdx.x < 512)) {
    identity_Out_116[threadIdx.x] = batch_norm_43__b_0[threadIdx.x]
  }
}
function fn_identity_434_kernel (_batch_norm_42__b_0, _identity_Out_117)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_117[threadIdx.x] = batch_norm_42__b_0[threadIdx.x]
  }
}
function fn_identity_476_kernel (_batch_norm_41__b_0, _identity_Out_118)
{
  if ((threadIdx.x < 256)) {
    identity_Out_118[threadIdx.x] = batch_norm_41__b_0[threadIdx.x]
  }
}
function fn_identity_518_kernel (_batch_norm_40__b_0, _identity_Out_119)
{
  if ((threadIdx.x < 256)) {
    identity_Out_119[threadIdx.x] = batch_norm_40__b_0[threadIdx.x]
  }
}
function fn_identity_564_kernel (_batch_norm_39__b_0, _identity_Out_120)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_120[threadIdx.x] = batch_norm_39__b_0[threadIdx.x]
  }
}
function fn_identity_606_kernel (_batch_norm_38__b_0, _identity_Out_121)
{
  if ((threadIdx.x < 256)) {
    identity_Out_121[threadIdx.x] = batch_norm_38__b_0[threadIdx.x]
  }
}
function fn_identity_648_kernel (_batch_norm_37__b_0, _identity_Out_122)
{
  if ((threadIdx.x < 256)) {
    identity_Out_122[threadIdx.x] = batch_norm_37__b_0[threadIdx.x]
  }
}
function fn_identity_694_kernel (_batch_norm_36__b_0, _identity_Out_123)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_123[threadIdx.x] = batch_norm_36__b_0[threadIdx.x]
  }
}
function fn_identity_736_kernel (_batch_norm_35__b_0, _identity_Out_124)
{
  if ((threadIdx.x < 256)) {
    identity_Out_124[threadIdx.x] = batch_norm_35__b_0[threadIdx.x]
  }
}
function fn_identity_778_kernel (_batch_norm_34__b_0, _identity_Out_125)
{
  if ((threadIdx.x < 256)) {
    identity_Out_125[threadIdx.x] = batch_norm_34__b_0[threadIdx.x]
  }
}
function fn_identity_824_kernel (_batch_norm_33__b_0, _identity_Out_126)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_126[threadIdx.x] = batch_norm_33__b_0[threadIdx.x]
  }
}
function fn_identity_866_kernel (_batch_norm_32__b_0, _identity_Out_127)
{
  if ((threadIdx.x < 256)) {
    identity_Out_127[threadIdx.x] = batch_norm_32__b_0[threadIdx.x]
  }
}
function fn_identity_908_kernel (_batch_norm_31__b_0, _identity_Out_128)
{
  if ((threadIdx.x < 256)) {
    identity_Out_128[threadIdx.x] = batch_norm_31__b_0[threadIdx.x]
  }
}
function fn_identity_954_kernel (_batch_norm_30__b_0, _identity_Out_129)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_129[threadIdx.x] = batch_norm_30__b_0[threadIdx.x]
  }
}
function fn_identity_996_kernel (_batch_norm_29__b_0, _identity_Out_130)
{
  if ((threadIdx.x < 256)) {
    identity_Out_130[threadIdx.x] = batch_norm_29__b_0[threadIdx.x]
  }
}
function fn_identity_1038_kernel (_batch_norm_28__b_0, _identity_Out_131)
{
  if ((threadIdx.x < 256)) {
    identity_Out_131[threadIdx.x] = batch_norm_28__b_0[threadIdx.x]
  }
}
function fn_identity_1084_kernel (_batch_norm_26__b_0, _identity_Out_132)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_132[threadIdx.x] = batch_norm_26__b_0[threadIdx.x]
  }
}
function fn_identity_1120_kernel (_batch_norm_27__b_0, _identity_Out_133)
{
  if ((threadIdx.x < 1024)) {
    identity_Out_133[threadIdx.x] = batch_norm_27__b_0[threadIdx.x]
  }
}
function fn_identity_1164_kernel (_batch_norm_25__b_0, _identity_Out_134)
{
  if ((threadIdx.x < 256)) {
    identity_Out_134[threadIdx.x] = batch_norm_25__b_0[threadIdx.x]
  }
}
function fn_identity_1206_kernel (_batch_norm_24__b_0, _identity_Out_135)
{
  if ((threadIdx.x < 256)) {
    identity_Out_135[threadIdx.x] = batch_norm_24__b_0[threadIdx.x]
  }
}
function fn_identity_1252_kernel (_batch_norm_23__b_0, _identity_Out_136)
{
  if ((threadIdx.x < 512)) {
    identity_Out_136[threadIdx.x] = batch_norm_23__b_0[threadIdx.x]
  }
}
function fn_identity_1294_kernel (_batch_norm_22__b_0, _identity_Out_137)
{
  if ((threadIdx.x < 128)) {
    identity_Out_137[threadIdx.x] = batch_norm_22__b_0[threadIdx.x]
  }
}
function fn_identity_1336_kernel (_batch_norm_21__b_0, _identity_Out_138)
{
  if ((threadIdx.x < 128)) {
    identity_Out_138[threadIdx.x] = batch_norm_21__b_0[threadIdx.x]
  }
}
function fn_identity_1382_kernel (_batch_norm_20__b_0, _identity_Out_139)
{
  if ((threadIdx.x < 512)) {
    identity_Out_139[threadIdx.x] = batch_norm_20__b_0[threadIdx.x]
  }
}
function fn_identity_1424_kernel (_batch_norm_19__b_0, _identity_Out_140)
{
  if ((threadIdx.x < 128)) {
    identity_Out_140[threadIdx.x] = batch_norm_19__b_0[threadIdx.x]
  }
}
function fn_identity_1466_kernel (_batch_norm_18__b_0, _identity_Out_141)
{
  if ((threadIdx.x < 128)) {
    identity_Out_141[threadIdx.x] = batch_norm_18__b_0[threadIdx.x]
  }
}
function fn_identity_1512_kernel (_batch_norm_17__b_0, _identity_Out_142)
{
  if ((threadIdx.x < 512)) {
    identity_Out_142[threadIdx.x] = batch_norm_17__b_0[threadIdx.x]
  }
}
function fn_identity_1554_kernel (_batch_norm_16__b_0, _identity_Out_143)
{
  if ((threadIdx.x < 128)) {
    identity_Out_143[threadIdx.x] = batch_norm_16__b_0[threadIdx.x]
  }
}
function fn_identity_1596_kernel (_batch_norm_15__b_0, _identity_Out_144)
{
  if ((threadIdx.x < 128)) {
    identity_Out_144[threadIdx.x] = batch_norm_15__b_0[threadIdx.x]
  }
}
function fn_identity_1642_kernel (_batch_norm_14__b_0, _identity_Out_145)
{
  if ((threadIdx.x < 512)) {
    identity_Out_145[threadIdx.x] = batch_norm_14__b_0[threadIdx.x]
  }
}
function fn_identity_1678_kernel (_batch_norm_13__b_0, _identity_Out_146)
{
  if ((threadIdx.x < 512)) {
    identity_Out_146[threadIdx.x] = batch_norm_13__b_0[threadIdx.x]
  }
}
function fn_identity_1722_kernel (_batch_norm_12__b_0, _identity_Out_147)
{
  if ((threadIdx.x < 128)) {
    identity_Out_147[threadIdx.x] = batch_norm_12__b_0[threadIdx.x]
  }
}
function fn_identity_1764_kernel (_batch_norm_11__b_0, _identity_Out_148)
{
  if ((threadIdx.x < 128)) {
    identity_Out_148[threadIdx.x] = batch_norm_11__b_0[threadIdx.x]
  }
}
function fn_identity_1810_kernel (_batch_norm_10__b_0, _identity_Out_149)
{
  if ((threadIdx.x < 256)) {
    identity_Out_149[threadIdx.x] = batch_norm_10__b_0[threadIdx.x]
  }
}
function fn_identity_1852_kernel (_batch_norm_9__b_0, _identity_Out_150)
{
  if ((threadIdx.x < 64)) {
    identity_Out_150[threadIdx.x] = batch_norm_9__b_0[threadIdx.x]
  }
}
function fn_identity_1894_kernel (_batch_norm_8__b_0, _identity_Out_151)
{
  if ((threadIdx.x < 64)) {
    identity_Out_151[threadIdx.x] = batch_norm_8__b_0[threadIdx.x]
  }
}
function fn_identity_1940_kernel (_batch_norm_7__b_0, _identity_Out_152)
{
  if ((threadIdx.x < 256)) {
    identity_Out_152[threadIdx.x] = batch_norm_7__b_0[threadIdx.x]
  }
}
function fn_identity_1982_kernel (_batch_norm_6__b_0, _identity_Out_153)
{
  if ((threadIdx.x < 64)) {
    identity_Out_153[threadIdx.x] = batch_norm_6__b_0[threadIdx.x]
  }
}
function fn_identity_2024_kernel (_batch_norm_5__b_0, _identity_Out_154)
{
  if ((threadIdx.x < 64)) {
    identity_Out_154[threadIdx.x] = batch_norm_5__b_0[threadIdx.x]
  }
}
function fn_identity_2070_kernel (_batch_norm_4__b_0, _identity_Out_155)
{
  if ((threadIdx.x < 256)) {
    identity_Out_155[threadIdx.x] = batch_norm_4__b_0[threadIdx.x]
  }
}
function fn_identity_2106_kernel (_batch_norm_3__b_0, _identity_Out_156)
{
  if ((threadIdx.x < 256)) {
    identity_Out_156[threadIdx.x] = batch_norm_3__b_0[threadIdx.x]
  }
}
function fn_identity_2150_kernel (_batch_norm_2__b_0, _identity_Out_157)
{
  if ((threadIdx.x < 64)) {
    identity_Out_157[threadIdx.x] = batch_norm_2__b_0[threadIdx.x]
  }
}
function fn_identity_2192_kernel (_batch_norm_1__b_0, _identity_Out_158)
{
  if ((threadIdx.x < 64)) {
    identity_Out_158[threadIdx.x] = batch_norm_1__b_0[threadIdx.x]
  }
}
function fn_broadcast_to_9_substract_10_fused_kernel (_batch_norm_52__tmp_0, _conv2d_105__tmp_0, _substract_Out_105)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (13 + (((196608 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_105[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_105__tmp_0[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_52__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x)))) % 2048)])
      }
    }
  }
}
function fn_broadcast_to_51_substract_52_fused_kernel (_batch_norm_51__tmp_0, _conv2d_104__tmp_0, _substract_Out_106)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (4 + (((245760 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_106[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_104__tmp_0[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_51__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x)))) % 512)])
      }
    }
  }
}
function fn_broadcast_to_93_substract_94_fused_kernel (_batch_norm_50__tmp_0, _conv2d_103__tmp_0, _substract_Out_107)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (4 + (((245760 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_107[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_103__tmp_0[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_50__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x)))) % 512)])
      }
    }
  }
}
function fn_broadcast_to_139_substract_140_fused_kernel (_batch_norm_49__tmp_0, _conv2d_102__tmp_0, _substract_Out_108)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (13 + (((196608 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_108[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_102__tmp_0[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_49__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x)))) % 2048)])
      }
    }
  }
}
function fn_broadcast_to_181_substract_182_fused_kernel (_batch_norm_48__tmp_0, _conv2d_101__tmp_0, _substract_Out_109)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (4 + (((245760 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_109[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_101__tmp_0[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_48__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x)))) % 512)])
      }
    }
  }
}
function fn_broadcast_to_223_substract_224_fused_kernel (_batch_norm_47__tmp_0, _conv2d_100__tmp_0, _substract_Out_110)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (4 + (((245760 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_110[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_100__tmp_0[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_47__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x)))) % 512)])
      }
    }
  }
}
function fn_broadcast_to_269_substract_270_fused_kernel (_batch_norm_46__tmp_0, _conv2d_99__tmp_0, _substract_Out_111)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (13 + (((196608 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_111[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_99__tmp_0[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_46__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x)))) % 2048)])
      }
    }
  }
}
function fn_broadcast_to_305_substract_306_fused_kernel (_batch_norm_45__tmp_0, _conv2d_98__tmp_0, _substract_Out_112)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (13 + (((196608 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_112[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_98__tmp_0[((-1 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + (blockIdx.x + ((3 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((2048 * ((100351 + ((99328 * blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * threadIdx.x)))) / 100352)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] - batch_norm_45__tmp_0[(((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((-2027 * blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * threadIdx.x)))) % 2048)])
      }
    }
  }
}
function fn_broadcast_to_349_substract_350_fused_kernel (_batch_norm_44__tmp_0, _conv2d_97__tmp_0, _substract_Out_113)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, (4 + (((245760 + ((1024 * blockIdx.x) + threadIdx.x)) / 262144) * -1)))
      {
        substract_Out_113[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x)) % 7)] = (conv2d_97__tmp_0[((-1 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + (blockIdx.x + ((11 * i_j_fused_k_fused_a_fused_outer) + threadIdx.x))), ((-1 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + ((512 * ((25087 + ((24064 * blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * threadIdx.x)))) / 25088)) + ((-491 * blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * threadIdx.x))))), ((-1 * ((6 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * threadIdx.x)))) / 7)) + ((7 * ((48 + ((5 * blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * threadIdx.x)))) / 49)) + (-6 * threadIdx.x))), (((1024 * blockIdx.x) + ((262144 *
I1201 03:39:27.400007 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(1024) fn_identity_6_kernel(const float* __restrict__ batch_norm_52__b_0, float* __restrict__ identity_Out_107)
{
  if (((int)blockIdx.x < 2)) {
    if (((int)threadIdx.x < 1024)) {
      identity_Out_107[((1024 * (int)blockIdx.x) + (int)threadIdx.x)] = batch_norm_52__b_0[((1024 * (int)blockIdx.x) + (int)threadIdx.x)];
    };
  };
}__global__
void __launch_bounds__(512) fn_identity_48_kernel(const float* __restrict__ batch_norm_51__b_0, float* __restrict__ identity_Out_108)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_108[(int)threadIdx.x] = batch_norm_51__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_90_kernel(const float* __restrict__ batch_norm_50__b_0, float* __restrict__ identity_Out_109)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_109[(int)threadIdx.x] = batch_norm_50__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_136_kernel(const float* __restrict__ batch_norm_49__b_0, float* __restrict__ identity_Out_110)
{
  if (((int)blockIdx.x < 2)) {
    if (((int)threadIdx.x < 1024)) {
      identity_Out_110[((1024 * (int)blockIdx.x) + (int)threadIdx.x)] = batch_norm_49__b_0[((1024 * (int)blockIdx.x) + (int)threadIdx.x)];
    };
  };
}__global__
void __launch_bounds__(512) fn_identity_178_kernel(const float* __restrict__ batch_norm_48__b_0, float* __restrict__ identity_Out_111)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_111[(int)threadIdx.x] = batch_norm_48__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_220_kernel(const float* __restrict__ batch_norm_47__b_0, float* __restrict__ identity_Out_112)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_112[(int)threadIdx.x] = batch_norm_47__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_266_kernel(const float* __restrict__ batch_norm_46__b_0, float* __restrict__ identity_Out_113)
{
  if (((int)blockIdx.x < 2)) {
    if (((int)threadIdx.x < 1024)) {
      identity_Out_113[((1024 * (int)blockIdx.x) + (int)threadIdx.x)] = batch_norm_46__b_0[((1024 * (int)blockIdx.x) + (int)threadIdx.x)];
    };
  };
}__global__
void __launch_bounds__(1024) fn_identity_302_kernel(const float* __restrict__ batch_norm_45__b_0, float* __restrict__ identity_Out_114)
{
  if (((int)blockIdx.x < 2)) {
    if (((int)threadIdx.x < 1024)) {
      identity_Out_114[((1024 * (int)blockIdx.x) + (int)threadIdx.x)] = batch_norm_45__b_0[((1024 * (int)blockIdx.x) + (int)threadIdx.x)];
    };
  };
}__global__
void __launch_bounds__(512) fn_identity_346_kernel(const float* __restrict__ batch_norm_44__b_0, float* __restrict__ identity_Out_115)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_115[(int)threadIdx.x] = batch_norm_44__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_388_kernel(const float* __restrict__ batch_norm_43__b_0, float* __restrict__ identity_Out_116)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_116[(int)threadIdx.x] = batch_norm_43__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_434_kernel(const float* __restrict__ batch_norm_42__b_0, float* __restrict__ identity_Out_117)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_117[(int)threadIdx.x] = batch_norm_42__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_476_kernel(const float* __restrict__ batch_norm_41__b_0, float* __restrict__ identity_Out_118)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_118[(int)threadIdx.x] = batch_norm_41__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_518_kernel(const float* __restrict__ batch_norm_40__b_0, float* __restrict__ identity_Out_119)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_119[(int)threadIdx.x] = batch_norm_40__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_564_kernel(const float* __restrict__ batch_norm_39__b_0, float* __restrict__ identity_Out_120)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_120[(int)threadIdx.x] = batch_norm_39__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_606_kernel(const float* __restrict__ batch_norm_38__b_0, float* __restrict__ identity_Out_121)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_121[(int)threadIdx.x] = batch_norm_38__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_648_kernel(const float* __restrict__ batch_norm_37__b_0, float* __restrict__ identity_Out_122)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_122[(int)threadIdx.x] = batch_norm_37__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_694_kernel(const float* __restrict__ batch_norm_36__b_0, float* __restrict__ identity_Out_123)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_123[(int)threadIdx.x] = batch_norm_36__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_736_kernel(const float* __restrict__ batch_norm_35__b_0, float* __restrict__ identity_Out_124)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_124[(int)threadIdx.x] = batch_norm_35__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_778_kernel(const float* __restrict__ batch_norm_34__b_0, float* __restrict__ identity_Out_125)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_125[(int)threadIdx.x] = batch_norm_34__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_824_kernel(const float* __restrict__ batch_norm_33__b_0, float* __restrict__ identity_Out_126)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_126[(int)threadIdx.x] = batch_norm_33__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_866_kernel(const float* __restrict__ batch_norm_32__b_0, float* __restrict__ identity_Out_127)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_127[(int)threadIdx.x] = batch_norm_32__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_908_kernel(const float* __restrict__ batch_norm_31__b_0, float* __restrict__ identity_Out_128)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_128[(int)threadIdx.x] = batch_norm_31__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_954_kernel(const float* __restrict__ batch_norm_30__b_0, float* __restrict__ identity_Out_129)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_129[(int)threadIdx.x] = batch_norm_30__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_996_kernel(const float* __restrict__ batch_norm_29__b_0, float* __restrict__ identity_Out_130)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_130[(int)threadIdx.x] = batch_norm_29__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_1038_kernel(const float* __restrict__ batch_norm_28__b_0, float* __restrict__ identity_Out_131)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_131[(int)threadIdx.x] = batch_norm_28__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_1084_kernel(const float* __restrict__ batch_norm_26__b_0, float* __restrict__ identity_Out_132)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_132[(int)threadIdx.x] = batch_norm_26__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_identity_1120_kernel(const float* __restrict__ batch_norm_27__b_0, float* __restrict__ identity_Out_133)
{
  if (((int)threadIdx.x < 1024)) {
    identity_Out_133[(int)threadIdx.x] = batch_norm_27__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_1164_kernel(const float* __restrict__ batch_norm_25__b_0, float* __restrict__ identity_Out_134)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_134[(int)threadIdx.x] = batch_norm_25__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_1206_kernel(const float* __restrict__ batch_norm_24__b_0, float* __restrict__ identity_Out_135)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_135[(int)threadIdx.x] = batch_norm_24__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_1252_kernel(const float* __restrict__ batch_norm_23__b_0, float* __restrict__ identity_Out_136)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_136[(int)threadIdx.x] = batch_norm_23__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1294_kernel(const float* __restrict__ batch_norm_22__b_0, float* __restrict__ identity_Out_137)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_137[(int)threadIdx.x] = batch_norm_22__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1336_kernel(const float* __restrict__ batch_norm_21__b_0, float* __restrict__ identity_Out_138)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_138[(int)threadIdx.x] = batch_norm_21__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_1382_kernel(const float* __restrict__ batch_norm_20__b_0, float* __restrict__ identity_Out_139)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_139[(int)threadIdx.x] = batch_norm_20__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1424_kernel(const float* __restrict__ batch_norm_19__b_0, float* __restrict__ identity_Out_140)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_140[(int)threadIdx.x] = batch_norm_19__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1466_kernel(const float* __restrict__ batch_norm_18__b_0, float* __restrict__ identity_Out_141)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_141[(int)threadIdx.x] = batch_norm_18__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_1512_kernel(const float* __restrict__ batch_norm_17__b_0, float* __restrict__ identity_Out_142)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_142[(int)threadIdx.x] = batch_norm_17__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1554_kernel(const float* __restrict__ batch_norm_16__b_0, float* __restrict__ identity_Out_143)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_143[(int)threadIdx.x] = batch_norm_16__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1596_kernel(const float* __restrict__ batch_norm_15__b_0, float* __restrict__ identity_Out_144)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_144[(int)threadIdx.x] = batch_norm_15__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_1642_kernel(const float* __restrict__ batch_norm_14__b_0, float* __restrict__ identity_Out_145)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_145[(int)threadIdx.x] = batch_norm_14__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(512) fn_identity_1678_kernel(const float* __restrict__ batch_norm_13__b_0, float* __restrict__ identity_Out_146)
{
  if (((int)threadIdx.x < 512)) {
    identity_Out_146[(int)threadIdx.x] = batch_norm_13__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1722_kernel(const float* __restrict__ batch_norm_12__b_0, float* __restrict__ identity_Out_147)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_147[(int)threadIdx.x] = batch_norm_12__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(128) fn_identity_1764_kernel(const float* __restrict__ batch_norm_11__b_0, float* __restrict__ identity_Out_148)
{
  if (((int)threadIdx.x < 128)) {
    identity_Out_148[(int)threadIdx.x] = batch_norm_11__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_1810_kernel(const float* __restrict__ batch_norm_10__b_0, float* __restrict__ identity_Out_149)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_149[(int)threadIdx.x] = batch_norm_10__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(64) fn_identity_1852_kernel(const float* __restrict__ batch_norm_9__b_0, float* __restrict__ identity_Out_150)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_150[(int)threadIdx.x] = batch_norm_9__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(64) fn_identity_1894_kernel(const float* __restrict__ batch_norm_8__b_0, float* __restrict__ identity_Out_151)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_151[(int)threadIdx.x] = batch_norm_8__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_1940_kernel(const float* __restrict__ batch_norm_7__b_0, float* __restrict__ identity_Out_152)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_152[(int)threadIdx.x] = batch_norm_7__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(64) fn_identity_1982_kernel(const float* __restrict__ batch_norm_6__b_0, float* __restrict__ identity_Out_153)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_153[(int)threadIdx.x] = batch_norm_6__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(64) fn_identity_2024_kernel(const float* __restrict__ batch_norm_5__b_0, float* __restrict__ identity_Out_154)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_154[(int)threadIdx.x] = batch_norm_5__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_2070_kernel(const float* __restrict__ batch_norm_4__b_0, float* __restrict__ identity_Out_155)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_155[(int)threadIdx.x] = batch_norm_4__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(256) fn_identity_2106_kernel(const float* __restrict__ batch_norm_3__b_0, float* __restrict__ identity_Out_156)
{
  if (((int)threadIdx.x < 256)) {
    identity_Out_156[(int)threadIdx.x] = batch_norm_3__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(64) fn_identity_2150_kernel(const float* __restrict__ batch_norm_2__b_0, float* __restrict__ identity_Out_157)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_157[(int)threadIdx.x] = batch_norm_2__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(64) fn_identity_2192_kernel(const float* __restrict__ batch_norm_1__b_0, float* __restrict__ identity_Out_158)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_158[(int)threadIdx.x] = batch_norm_1__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_9_substract_10_fused_kernel(const float* __restrict__ batch_norm_52__tmp_0, const float* __restrict__ conv2d_105__tmp_0, float* __restrict__ substract_Out_105)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (13 + (((196608 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_105[((-100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] = (conv2d_105__tmp_0[((-100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] - batch_norm_52__tmp_0[(((-1 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((-2027 * (int)blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * (int)threadIdx.x)))) & 2047)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_51_substract_52_fused_kernel(const float* __restrict__ batch_norm_51__tmp_0, const float* __restrict__ conv2d_104__tmp_0, float* __restrict__ substract_Out_106)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (4 + (((245760 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_106[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] = (conv2d_104__tmp_0[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] - batch_norm_51__tmp_0[(((-1 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((-491 * (int)blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * (int)threadIdx.x)))) & 511)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_93_substract_94_fused_kernel(const float* __restrict__ batch_norm_50__tmp_0, const float* __restrict__ conv2d_103__tmp_0, float* __restrict__ substract_Out_107)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (4 + (((245760 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_107[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] = (conv2d_103__tmp_0[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] - batch_norm_50__tmp_0[(((-1 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((-491 * (int)blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * (int)threadIdx.x)))) & 511)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_139_substract_140_fused_kernel(const float* __restrict__ batch_norm_49__tmp_0, const float* __restrict__ conv2d_102__tmp_0, float* __restrict__ substract_Out_108)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (13 + (((196608 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_108[((-100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] = (conv2d_102__tmp_0[((-100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a_fused_outer) + (100351 * (int)threadIdx.x)))) / 100352)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] - batch_norm_49__tmp_0[(((-1 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((-2027 * (int)blockIdx.x) + ((-794 * i_j_fused_k_fused_a_fused_outer) + (-2047 * (int)threadIdx.x)))) & 2047)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_181_substract_182_fused_kernel(const float* __restrict__ batch_norm_48__tmp_0, const float* __restrict__ conv2d_101__tmp_0, float* __restrict__ substract_Out_109)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (4 + (((245760 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_109[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] = (conv2d_101__tmp_0[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] - batch_norm_48__tmp_0[(((-1 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((-491 * (int)blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * (int)threadIdx.x)))) & 511)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_223_substract_224_fused_kernel(const float* __restrict__ batch_norm_47__tmp_0, const float* __restrict__ conv2d_100__tmp_0, float* __restrict__ substract_Out_110)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (4 + (((245760 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_110[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] = (conv2d_100__tmp_0[((-25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((25088 * ((25087 + ((24064 * (int)blockIdx.x) + ((13824 * i_j_fused_k_fused_a_fused_outer) + (25087 * (int)threadIdx.x)))) / 25088)) + ((-7 * ((6 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (6 * (int)threadIdx.x)))) / 7)) + ((49 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((((1024 * (int)blockIdx.x) + ((262144 * i_j_fused_k_fused_a_fused_outer) + (int)threadIdx.x)) % 7) + ((1029 * (int)blockIdx.x) + ((262150 * i_j_fused_k_fused_a_fused_outer) + (7 * (int)threadIdx.x)))))))))] - batch_norm_47__tmp_0[(((-1 * ((48 + ((5 * (int)blockIdx.x) + ((6 * i_j_fused_k_fused_a_fused_outer) + (48 * (int)threadIdx.x)))) / 49)) + ((-491 * (int)blockIdx.x) + ((-282 * i_j_fused_k_fused_a_fused_outer) + (-511 * (int)threadIdx.x)))) & 511)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_269_substract_270_fused_kernel(const float* __restrict__ batch_norm_46__tmp_0, const float* __restrict__ conv2d_99__tmp_0, float* __restrict__ substract_Out_111)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < (13 + (((196608 + ((1024 * (int)blockIdx.x) + (int)threadIdx.x)) / 262144) * -1)); i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_111[((-100352 * ((100351 + ((99328 * (int)blockIdx.x) + ((38912 * i_j_fused_k_fused_a
I1201 03:39:27.402535 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
I1201 03:39:50.640642 25408 compiler.cc:73] [CUDA] host module:
Module module_5_host {

function fn_identity_4_1 (args__ptr, num_args)
{
  fn_identity_4_1_kernel(args__ptr, num_args)
}
function fn_broadcast_to_7_substract_8_fused (args__ptr, num_args)
{
  fn_broadcast_to_7_substract_8_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_0_broadcast_to_1_greater_2_select_3_fused_1 (args__ptr, num_args)
{
  fn_const_scalar_0_broadcast_to_1_greater_2_select_3_fused_1_kernel(args__ptr, num_args)
}
function fn_reduce_sum_5_1 (args__ptr, num_args)
{
  fn_reduce_sum_5_1_kernel(args__ptr, num_args)
}
function fn_reduce_sum_6 (args__ptr, num_args)
{
  fn_reduce_sum_6_kernel(args__ptr, num_args)
}
function fn_elementwise_mul_9_reduce_sum_10_fused (args__ptr, num_args)
{
  fn_elementwise_mul_9_reduce_sum_10_fused_kernel(args__ptr, num_args)
}
function fn_reduce_sum_11 (args__ptr, num_args)
{
  fn_reduce_sum_11_kernel(args__ptr, num_args)
}
function fn_const_scalar_12_broadcast_to_13_elementwise_add_14_rsqrt_15_elementwise_mul_16_fused (args__ptr, num_args)
{
  fn_const_scalar_12_broadcast_to_13_elementwise_add_14_rsqrt_15_elementwise_mul_16_fused_kernel(args__ptr, num_args)
}
function fn_const_scalar_17_const_scalar_19_const_scalar_26_const_scalar_32_broadcast_to_18_broadcast_to_20_broadcast_to_27_broadcast_to_33_elementwise_add_21_elementwise_add_34_rsqrt_22_broadcast_to_35_elementwise_mul_28_elementwise_mul_23_divide_24_broadcast_to_29_broadcast_to_25_substract_37_broadcast_to_30_elementwise_mul_31_divide_36_substract_38_elementwise_mul_39_fused (args__ptr, num_args)
{
  fn_const_scalar_17_const_scalar_19_const_scalar_26_const_scalar_32_broadcast_to_18_broadcast_to_20_broadcast_to_27_broadcast_to_33_elementwise_add_21_elementwise_add_34_rsqrt_22_broadcast_to_35_elementwise_mul_28_elementwise_mul_23_divide_24_broadcast_to_29_broadcast_to_25_substract_37_broadcast_to_30_elementwise_mul_31_divide_36_substract_38_elementwise_mul_39_fused_kernel(args__ptr, num_args)
}
function fn_conv2d_40 (args__ptr, num_args)
{
  fn_conv2d_40_kernel(args__ptr, num_args)
}


}
I1201 03:39:50.640763 25408 compiler.cc:76] [CUDA] device module:
Module module_5_gpu_device {

function fn_identity_4_1_kernel (_batch_norm_0__b_0, _identity_Out_207)
{
  if ((threadIdx.x < 64)) {
    identity_Out_207[threadIdx.x] = batch_norm_0__b_0[threadIdx.x]
  }
}
function fn_broadcast_to_7_substract_8_fused_kernel (_batch_norm_0__tmp_0, _conv2d_53__tmp_0, _substract_Out_261)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, 98)
      {
        substract_Out_261[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] = (conv2d_53__tmp_0[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] - batch_norm_0__tmp_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)])
      }
    }
  }
}
function fn_const_scalar_0_broadcast_to_1_greater_2_select_3_fused_1_kernel (_relu_0__tmp_0, _relu_0__tmp_0____GRAD, _tensor_47)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, 98)
      {
        tensor_47[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] = select((relu_0__tmp_0[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] > 0), relu_0__tmp_0____GRAD[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))], 0)
      }
    }
  }
}
function fn_reduce_sum_5_1_kernel (_var_8669, _reduce_sum_out_420)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 112)) {
      {
        reduce_sum_out_420__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_630, 0, 32)
        {
          for (kk_631, 0, 112)
          {
            reduce_sum_out_420[blockIdx.x, threadIdx.x] = (reduce_sum_out_420[blockIdx.x, threadIdx.x] + var_8669[kk_630, blockIdx.x, kk_631, threadIdx.x])
          }
        }
      }
    }
  }
}
function fn_reduce_sum_6_kernel (_var_8698, _reduce_sum_out_421)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_421__reduce_init[threadIdx.x] = 0
      for (kk_632, 0, 112)
      {
        reduce_sum_out_421[threadIdx.x] = (reduce_sum_out_421[threadIdx.x] + var_8698[threadIdx.x, kk_632])
      }
    }
  }
}
function fn_elementwise_mul_9_reduce_sum_10_fused_kernel (_var_8669, _var_8703, _reduce_sum_out_422)
{
  if ((blockIdx.x < 64)) {
    if ((threadIdx.x < 112)) {
      {
        reduce_sum_out_422__reduce_init[blockIdx.x, threadIdx.x] = 0
        for (kk_633, 0, 32)
        {
          for (kk_634, 0, 112)
          {
            reduce_sum_out_422[blockIdx.x, threadIdx.x] = (reduce_sum_out_422[blockIdx.x, threadIdx.x] + (var_8669[kk_633, blockIdx.x, kk_634, threadIdx.x] * var_8703[kk_633, blockIdx.x, kk_634, threadIdx.x]))
          }
        }
      }
    }
  }
}
function fn_reduce_sum_11_kernel (_var_8705, _reduce_sum_out_423)
{
  if ((threadIdx.x < 64)) {
    {
      reduce_sum_out_423__reduce_init[threadIdx.x] = 0
      for (kk_635, 0, 112)
      {
        reduce_sum_out_423[threadIdx.x] = (reduce_sum_out_423[threadIdx.x] + var_8705[threadIdx.x, kk_635])
      }
    }
  }
}
function fn_const_scalar_12_broadcast_to_13_elementwise_add_14_rsqrt_15_elementwise_mul_16_fused_kernel (_batch_norm_0__tmp_1, _var_8707, _elementwise_mul_Out_736)
{
  if ((threadIdx.x < 64)) {
    elementwise_mul_Out_736[threadIdx.x] = (var_8707[threadIdx.x] * rsqrt((batch_norm_0__tmp_1[threadIdx.x] + 1e-05)))
  }
}
function fn_const_scalar_17_const_scalar_19_const_scalar_26_const_scalar_32_broadcast_to_18_broadcast_to_20_broadcast_to_27_broadcast_to_33_elementwise_add_21_elementwise_add_34_rsqrt_22_broadcast_to_35_elementwise_mul_28_elementwise_mul_23_divide_24_broadcast_to_29_broadcast_to_25_substract_37_broadcast_to_30_elementwise_mul_31_divide_36_substract_38_elementwise_mul_39_fused_kernel (_batch_norm_0__tmp_1, _var_8669, _batch_norm_0__w_0, _var_8678, _var_8707, _var_8703, _elementwise_mul_Out_740)
{
  if ((blockIdx.x < 256)) {
    if ((threadIdx.x < 1024)) {
      for (i_j_fused_k_fused_a_fused_outer, 0, 98)
      {
        elementwise_mul_Out_740[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] = (((batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * var_8669[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))]) + ((-2.49123e-06 * (batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * var_8678[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)])) + (-2.49123e-06 * (batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (var_8703[((-1 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + (blockIdx.x + (i_j_fused_k_fused_a_fused_outer + threadIdx.x))), ((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((64 * ((802815 + ((801792 * blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * threadIdx.x)))) / 802816)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x))))), ((-1 * ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) / 112)) + ((112 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-102 * blockIdx.x) + ((-11 * i_j_fused_k_fused_a_fused_outer) + (-111 * threadIdx.x))))), (111 - ((111 + ((96 * blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * threadIdx.x)))) % 112))] * (var_8707[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] * (1 / (1e-05 + batch_norm_0__tmp_1[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)])))))))) * rsqrt((batch_norm_0__tmp_1[(((-1 * ((12543 + ((11520 * blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * threadIdx.x)))) / 12544)) + ((-63 * blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * threadIdx.x)))) % 64)] + 1e-05)))
      }
    }
  }
}
function fn_conv2d_40_kernel (_data, _T_Identity_out)
{
  if ((blockIdx.x < 4704)) {
    if ((threadIdx.x < 1024)) {
      T_Identity_out[((-1 * ((150527 + ((149504 * blockIdx.x) + (150527 * threadIdx.x))) / 150528)) + (blockIdx.x + threadIdx.x)), ((-1 * ((50175 + ((49152 * blockIdx.x) + (50175 * threadIdx.x))) / 50176)) + ((3 * ((150527 + ((149504 * blockIdx.x) + (150527 * threadIdx.x))) / 150528)) + ((-2 * blockIdx.x) + (-2 * threadIdx.x)))), ((-1 * ((223 + ((96 * blockIdx.x) + (223 * threadIdx.x))) / 224)) + ((224 * ((50175 + ((49152 * blockIdx.x) + (50175 * threadIdx.x))) / 50176)) + ((-219 * blockIdx.x) + (-223 * threadIdx.x)))), (223 - ((223 + ((96 * blockIdx.x) + (223 * threadIdx.x))) % 224))] = data[((-1 * ((150527 + ((149504 * blockIdx.x) + (150527 * threadIdx.x))) / 150528)) + (blockIdx.x + threadIdx.x)), ((-1 * ((50175 + ((49152 * blockIdx.x) + (50175 * threadIdx.x))) / 50176)) + ((3 * ((150527 + ((149504 * blockIdx.x) + (150527 * threadIdx.x))) / 150528)) + ((-2 * blockIdx.x) + (-2 * threadIdx.x)))), ((-1 * ((223 + ((96 * blockIdx.x) + (223 * threadIdx.x))) / 224)) + ((224 * ((50175 + ((49152 * blockIdx.x) + (50175 * threadIdx.x))) / 50176)) + ((-219 * blockIdx.x) + (-223 * threadIdx.x)))), (223 - ((223 + ((96 * blockIdx.x) + (223 * threadIdx.x))) % 224))]
    }
  }
}


}
I1201 03:39:51.956494 25408 compiler.cc:80] [CUDA] source code:
extern "C" {

#include "cinn_cuda_runtime_source.cuh"

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef char int8_t;
#endif



__global__
void __launch_bounds__(64) fn_identity_4_1_kernel(const float* __restrict__ batch_norm_0__b_0, float* __restrict__ identity_Out_207)
{
  if (((int)threadIdx.x < 64)) {
    identity_Out_207[(int)threadIdx.x] = batch_norm_0__b_0[(int)threadIdx.x];
  };
}__global__
void __launch_bounds__(1024) fn_broadcast_to_7_substract_8_fused_kernel(const float* __restrict__ batch_norm_0__tmp_0, const float* __restrict__ conv2d_53__tmp_0, float* __restrict__ substract_Out_261)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < 98; i_j_fused_k_fused_a_fused_outer += 1) {
        substract_Out_261[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] = (conv2d_53__tmp_0[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] - batch_norm_0__tmp_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)]);
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_const_scalar_0_broadcast_to_1_greater_2_select_3_fused_1_kernel(const float* __restrict__ relu_0__tmp_0, const float* __restrict__ relu_0__tmp_0____GRAD, float* __restrict__ tensor_47)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < 98; i_j_fused_k_fused_a_fused_outer += 1) {
        tensor_47[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] = (((relu_0__tmp_0[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] > 0)) ? relu_0__tmp_0____GRAD[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] : 0);
      };
    };
  };
}__global__
void __launch_bounds__(112) fn_reduce_sum_5_1_kernel(const float* __restrict__ var_8669, float* __restrict__ reduce_sum_out_420)
{
  float* reduce_sum_out_420__reduce_init = reduce_sum_out_420;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 112)) {
    {
      reduce_sum_out_420__reduce_init[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_630 = 0; kk_630 < 32; kk_630 += 1) {
        for (int32_t kk_631 = 0; kk_631 < 112; kk_631 += 1) {
          reduce_sum_out_420[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_420[((112 * (int)blockIdx.x) + (int)threadIdx.x)] + var_8669[((12544 * (int)blockIdx.x) + ((802816 * kk_630) + ((112 * kk_631) + (int)threadIdx.x)))]);
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_6_kernel(const float* __restrict__ var_8698, float* __restrict__ reduce_sum_out_421)
{
  float* reduce_sum_out_421__reduce_init = reduce_sum_out_421;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_421__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_632 = 0; kk_632 < 112; kk_632 += 1) {
      reduce_sum_out_421[(int)threadIdx.x] = (reduce_sum_out_421[(int)threadIdx.x] + var_8698[((112 * (int)threadIdx.x) + kk_632)]);
    };
  }
  };
}__global__
void __launch_bounds__(112) fn_elementwise_mul_9_reduce_sum_10_fused_kernel(const float* __restrict__ var_8669, const float* __restrict__ var_8703, float* __restrict__ reduce_sum_out_422)
{
  float* reduce_sum_out_422__reduce_init = reduce_sum_out_422;
  if (((int)blockIdx.x < 64)) {
    if (((int)threadIdx.x < 112)) {
    {
      reduce_sum_out_422__reduce_init[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = 0;
      for (int32_t kk_633 = 0; kk_633 < 32; kk_633 += 1) {
        for (int32_t kk_634 = 0; kk_634 < 112; kk_634 += 1) {
          reduce_sum_out_422[((112 * (int)blockIdx.x) + (int)threadIdx.x)] = (reduce_sum_out_422[((112 * (int)blockIdx.x) + (int)threadIdx.x)] + (var_8669[((12544 * (int)blockIdx.x) + ((802816 * kk_633) + ((112 * kk_634) + (int)threadIdx.x)))] * var_8703[((12544 * (int)blockIdx.x) + ((802816 * kk_633) + ((112 * kk_634) + (int)threadIdx.x)))]));
        };
      };
    }
    };
  };
}__global__
void __launch_bounds__(64) fn_reduce_sum_11_kernel(const float* __restrict__ var_8705, float* __restrict__ reduce_sum_out_423)
{
  float* reduce_sum_out_423__reduce_init = reduce_sum_out_423;
  if (((int)threadIdx.x < 64)) {
  {
    reduce_sum_out_423__reduce_init[(int)threadIdx.x] = 0;
    for (int32_t kk_635 = 0; kk_635 < 112; kk_635 += 1) {
      reduce_sum_out_423[(int)threadIdx.x] = (reduce_sum_out_423[(int)threadIdx.x] + var_8705[((112 * (int)threadIdx.x) + kk_635)]);
    };
  }
  };
}__global__
void __launch_bounds__(64) fn_const_scalar_12_broadcast_to_13_elementwise_add_14_rsqrt_15_elementwise_mul_16_fused_kernel(const float* __restrict__ batch_norm_0__tmp_1, const float* __restrict__ var_8707, float* __restrict__ elementwise_mul_Out_736)
{
  if (((int)threadIdx.x < 64)) {
    elementwise_mul_Out_736[(int)threadIdx.x] = (var_8707[(int)threadIdx.x] * rsqrt((batch_norm_0__tmp_1[(int)threadIdx.x] + 1e-05)));
  };
}__global__
void __launch_bounds__(1024) fn_const_scalar_17_const_scalar_19_const_scalar_26_const_scalar_32_broadcast_to_18_broadcast_to_20_broadcast_to_27_broadcast_to_33_elementwise_add_21_elementwise_add_34_rsqrt_22_broadcast_to_35_elementwise_mul_28_elementwise_mul_23_divide_24_broadcast_to_29_broadcast_to_25_substract_37_broadcast_to_30_elementwise_mul_31_divide_36_substract_38_elementwise_mul_39_fused_kernel(const float* __restrict__ batch_norm_0__tmp_1, const float* __restrict__ var_8669, const float* __restrict__ batch_norm_0__w_0, const float* __restrict__ var_8678, const float* __restrict__ var_8707, const float* __restrict__ var_8703, float* __restrict__ elementwise_mul_Out_740)
{
  if (((int)blockIdx.x < 256)) {
    if (((int)threadIdx.x < 1024)) {
      for (int32_t i_j_fused_k_fused_a_fused_outer = 0; i_j_fused_k_fused_a_fused_outer < 98; i_j_fused_k_fused_a_fused_outer += 1) {
        elementwise_mul_Out_740[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] = (((batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * var_8669[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))]) + ((-2.49123e-06 * (batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * var_8678[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)])) + (-2.49123e-06 * (batch_norm_0__w_0[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (var_8703[(111 + ((-802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((802816 * ((802815 + ((801792 * (int)blockIdx.x) + ((540672 * i_j_fused_k_fused_a_fused_outer) + (802815 * (int)threadIdx.x)))) / 802816)) + ((-112 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) / 112)) + ((12544 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-1 * ((111 + ((96 * (int)blockIdx.x) + ((48 * i_j_fused_k_fused_a_fused_outer) + (111 * (int)threadIdx.x)))) % 112)) + ((1120 * (int)blockIdx.x) + ((262192 * i_j_fused_k_fused_a_fused_outer) + (112 * (int)threadIdx.x))))))))))] * (var_8707[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] * (1 / (1e-05 + batch_norm_0__tmp_1[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)])))))))) * rsqrt((batch_norm_0__tmp_1[(((-1 * ((12543 + ((11520 * (int)blockIdx.x) + ((1280 * i_j_fused_k_fused_a_fused_outer) + (12543 * (int)threadIdx.x)))) / 12544)) + ((-63 * (int)blockIdx.x) + ((-43 * i_j_fused_k_fused_a_fused_outer) + (-63 * (int)threadIdx.x)))) & 63)] + 1e-05)));
      };
    };
  };
}__global__
void __launch_bounds__(1024) fn_conv2d_40_kernel(const float* __restrict__ data, float* __restrict__ T_Identity_out)
{
  if (((int)blockIdx.x < 4704)) {
    if (((int)threadIdx.x < 1024)) {
      T_Identity_out[(223 + ((-150528 * ((150527 + ((149504 * (int)blockIdx.x) + (150527 * (int)threadIdx.x))) / 150528)) + ((-50176 * ((50175 + ((49152 * (int)blockIdx.x) + (50175 * (int)threadIdx.x))) / 50176)) + ((150528 * ((150527 + ((149504 * (int)blockIdx.x) + (150527 * (int)threadIdx.x))) / 150528)) + ((-224 * ((223 + ((96 * (int)blockIdx.x) + (223 * (int)threadIdx.x))) / 224)) + ((50176 * ((50175 + ((49152 * (int)blockIdx.x) + (50175 * (int)threadIdx.x))) / 50176)) + ((-1 * ((223 + ((96 * (int)blockIdx.x) + (223 * (int)threadIdx.x))) % 224)) + ((1120 * (int)blockIdx.x) + (224 * (int)threadIdx.x)))))))))] = data[(223 + ((-150528 * ((150527 + ((149504 * (int)blockIdx.x) + (150527 * (int)threadIdx.x))) / 150528)) + ((-50176 * ((50175 + ((49152 * (int)blockIdx.x) + (50175 * (int)threadIdx.x))) / 50176)) + ((150528 * ((150527 + ((149504 * (int)blockIdx.x) + (150527 * (int)threadIdx.x))) / 150528)) + ((-224 * ((223 + ((96 * (int)blockIdx.x) + (223 * (int)threadIdx.x))) / 224)) + ((50176 * ((50175 + ((49152 * (int)blockIdx.x) + (50175 * (int)threadIdx.x))) / 50176)) + ((-1 * ((223 + ((96 * (int)blockIdx.x) + (223 * (int)threadIdx.x))) % 224)) + ((1120 * (int)blockIdx.x) + (224 * (int)threadIdx.x)))))))))];
    };
  };
}

}
I1201 03:39:51.957114 25408 nvrtc_util.cc:94] compile options: -arch=compute_70 --include-path=/usr/local/cuda/include --include-path=/Paddle/Paddle/build/third_party/CINN/src/external_cinn/cinn/runtime/cuda/
[2021/12/01 03:39:53] root INFO: epoch:0   train step:10   lr: 0.100000, loss: 11.2993 top1:  0.0000 top5:  0.0000 batch_cost: 0.11700 s, reader_cost: 0.00159 s, ips: 273.50793 images/sec.
[2021/12/01 03:39:55] root INFO: epoch:0   train step:20   lr: 0.100000, loss:  7.4566 top1:  0.0000 top5:  0.0000 batch_cost: 0.11693 s, reader_cost: 0.00161 s, ips: 273.67256 images/sec.
[2021/12/01 03:39:56] root INFO: epoch:0   train step:30   lr: 0.100000, loss:  7.6451 top1:  0.0000 top5:  0.0000 batch_cost: 0.11907 s, reader_cost: 0.00216 s, ips: 268.74052 images/sec.
[2021/12/01 03:39:57] root INFO: epoch:0   train step:40   lr: 0.100000, loss:  7.0963 top1:  0.0000 top5:  0.0000 batch_cost: 0.11866 s, reader_cost: 0.00200 s, ips: 269.67231 images/sec.
[2021/12/01 03:39:59] root INFO: epoch:0   train step:50   lr: 0.100000, loss:  6.9832 top1:  0.0000 top5:  0.0000 batch_cost: 0.14315 s, reader_cost: 0.02546 s, ips: 223.54777 images/sec.

------------------------->     Profiling Report     <-------------------------

Note! This Report merge all thread info into one.
Place: All
Time unit: ms
Sorted by total time in descending order in the same thread

-------------------------     Overhead Summary      -------------------------

Total time: 3604.46
  Computation time       Total: 2237.42     Ratio: 62.0736%
  Framework overhead     Total: 1367.04     Ratio: 37.9264%

-------------------------     GpuMemCpy Summary     -------------------------

GpuMemcpy                Calls: 121         Total: 1065.46     Ratio: 29.5595%
  GpuMemcpyAsync         Calls: 121         Total: 1065.46     Ratio: 29.5595%

-------------------------       Event Summary       -------------------------

Event                                                       Calls       Total       CPU Time (Ratio)        GPU Time (Ratio)        Min.        Max.        Ave.        Ratio.      
cinn_launch                                                 70          2187.27     1163.807214 (0.532082)  1023.464856 (0.467918)  0.144505    132.802     31.2467     0.606824    
  cinn_launch5                                              10          1317.3      629.974337 (0.478232)   687.325232 (0.521768)   130.456     132.777     131.73      0.365464    
    cinn_launch5/compute                                    10          1314.55     627.223049 (0.477140)   687.325232 (0.522860)   130.159     132.502     131.455     0.364701    
    cinn_launch5/infer_shape                                10          0.023378    0.023378 (1.000000)     0.000000 (0.000000)     0.002061    0.002586    0.0023378   6.48586e-06 
    cinn_launch5/prepare_data                               10          0.015259    0.015259 (1.000000)     0.000000 (0.000000)     0.001214    0.001909    0.0015259   4.23337e-06 
  cinn_launch1                                              10          792.994     495.389559 (0.624708)   297.604382 (0.375292)   78.7686     80.2022     79.2994     0.220004    
    cinn_launch1/compute                                    10          789.958     492.353210 (0.623265)   297.604382 (0.376735)   78.4529     79.9307     78.9958     0.219161    
    cinn_launch1/infer_shape                                10          0.020952    0.020952 (1.000000)     0.000000 (0.000000)     0.001848    0.002353    0.0020952   5.8128e-06  
    cinn_launch1/prepare_data                               10          0.018554    0.018554 (1.000000)     0.000000 (0.000000)     0.000933    0.007716    0.0018554   5.14752e-06 
  cinn_launch6                                              10          40.1071     13.895106 (0.346450)    26.211954 (0.653550)    3.91057     4.08749     4.01071     0.0111271   
    cinn_launch6/compute                                    10          39.9744     13.762402 (0.344281)    26.211954 (0.655719)    3.89783     4.07389     3.99744     0.0110903   
    cinn_launch6/prepare_data                               10          0.017337    0.017337 (1.000000)     0.000000 (0.000000)     0.001352    0.003878    0.0017337   4.80988e-06 
    cinn_launch6/infer_shape                                10          0.016893    0.016893 (1.000000)     0.000000 (0.000000)     0.001492    0.002011    0.0016893   4.6867e-06  
  cinn_launch0                                              10          24.7722     12.636389 (0.510105)    12.135768 (0.489895)    2.36682     2.79607     2.47722     0.00687265  
    cinn_launch0/compute                                    10          24.6097     12.473901 (0.506870)    12.135768 (0.493130)    2.35135     2.7698      2.46097     0.00682757  
    cinn_launch0/prepare_data                               10          0.02063     0.020630 (1.000000)     0.000000 (0.000000)     0.001075    0.005477    0.002063    5.72347e-06 
    cinn_launch0/infer_shape                                10          0.019433    0.019433 (1.000000)     0.000000 (0.000000)     0.001686    0.002248    0.0019433   5.39138e-06 
  cinn_launch3                                              10          5.74238     5.631181 (0.980635)     0.111200 (0.019365)     0.299111    0.913227    0.574238    0.00159313  
    cinn_launch3/compute                                    10          5.51012     5.398919 (0.979819)     0.111200 (0.020181)     0.289619    0.902642    0.551012    0.0015287   
    cinn_launch3/prepare_data                               10          0.128577    0.128577 (1.000000)     0.000000 (0.000000)     0.000861    0.119963    0.0128577   3.56717e-05 
    cinn_launch3/infer_shape                                10          0.020379    0.020379 (1.000000)     0.000000 (0.000000)     0.001508    0.003291    0.0020379   5.65383e-06 
  cinn_launch2                                              10          3.74691     3.701082 (0.987770)     0.045824 (0.012230)     0.247041    0.792207    0.374691    0.00103952  
    cinn_launch2/compute                                    10          3.60893     3.563102 (0.987303)     0.045824 (0.012697)     0.234999    0.780948    0.360893    0.00100124  
    cinn_launch2/infer_shape                                10          0.025019    0.025019 (1.000000)     0.000000 (0.000000)     0.002175    0.002989    0.0025019   6.94113e-06 
    cinn_launch2/prepare_data                               10          0.014288    0.014288 (1.000000)     0.000000 (0.000000)     0.000807    0.006676    0.0014288   3.96398e-06 
  cinn_launch4                                              10          1.62351     1.593013 (0.981216)     0.030496 (0.018784)     0.14286     0.208642    0.162351    0.000450417 
    cinn_launch4/compute                                    10          1.54474     1.514241 (0.980258)     0.030496 (0.019742)     0.135918    0.20025     0.154474    0.000428563 
    cinn_launch4/infer_shape                                10          0.019444    0.019444 (1.000000)     0.000000 (0.000000)     0.001384    0.005009    0.0019444   5.39443e-06 
    cinn_launch4/prepare_data                               10          0.009371    0.009371 (1.000000)     0.000000 (0.000000)     0.000728    0.001368    0.0009371   2.59984e-06 
GpuMemcpyAsync:CUDAPinned->GPU                              1           1045.4      1045.402880 (1.000000)  0.000000 (0.000000)     1045.4      1045.4      1045.4      0.290031    
BufferedReader:MemoryCopy                                   10          250.509     234.627218 (0.936600)   15.882199 (0.063400)    17.2275     44.3467     25.0509     0.0694999   
  GpuMemcpyAsync:CUDAPinned->GPU                            20          16.7283     0.846059 (0.050577)     15.882199 (0.949423)    0.020954    1.66126     0.836413    0.00464099  
momentum                                                    1610        58.5373     46.446097 (0.793444)    12.091229 (0.206556)    0.027103    0.470355    0.0363586   0.0162403   
  momentum1                                                 10          0.959404    0.404716 (0.421841)     0.554688 (0.578159)     0.091276    0.112871    0.0959404   0.000266172 
    momentum1/compute                                       10          0.772729    0.218041 (0.282170)     0.554688 (0.717830)     0.074786    0.079863    0.0772729   0.000214382 
    momentum1/infer_shape                                   10          0.05509     0.055090 (1.000000)     0.000000 (0.000000)     0.004736    0.009342    0.005509    1.52839e-05 
    momentum1/prepare_data                                  10          0.031826    0.031826 (1.000000)     0.000000 (0.000000)     0.001174    0.018994    0.0031826   8.82963e-06 
  momentum71                                                10          0.903698    0.273042 (0.302139)     0.630656 (0.697861)     0.087678    0.100672    0.0903698   0.000250717 
    momentum71/compute                                      10          0.76409     0.133434 (0.174631)     0.630656 (0.825369)     0.075316    0.077558    0.076409    0.000211985 
    momentum71/infer_shape                                  10          0.035797    0.035797 (1.000000)     0.000000 (0.000000)     0.0033      0.00451     0.0035797   9.93132e-06 
    momentum71/prepare_data                                 10          0.010768    0.010768 (1.000000)     0.000000 (0.000000)     0.000938    0.001447    0.0010768   2.98741e-06 
  momentum88                                                10          0.901322    0.276042 (0.306263)     0.625280 (0.693737)     0.086156    0.108517    0.0901322   0.000250058 
    momentum88/compute                                      10          0.775949    0.150669 (0.194174)     0.625280 (0.805826)     0.074064    0.096368    0.0775949   0.000215275 
    momentum88/infer_shape                                  10          0.035793    0.035793 (1.000000)     0.000000 (0.000000)     0.003299    0.003873    0.0035793   9.93021e-06 
    momentum88/prepare_data                                 10          0.015928    0.015928 (1.000000)     0.000000 (0.000000)     0.001093    0.003827    0.0015928   4.41897e-06 
  momentum132                                               10          0.869803    0.242283 (0.278549)     0.627520 (0.721451)     0.085286    0.089928    0.0869803   0.000241313 
    momentum132/compute                                     10          0.758306    0.130786 (0.172471)     0.627520 (0.827529)     0.074427    0.077271    0.0758306   0.00021038  
    momentum132/infer_shape                                 10          0.029231    0.029231 (1.000000)     0.000000 (0.000000)     0.002776    0.003212    0.0029231   8.10968e-06 
    momentum132/prepare_data                                10          0.013501    0.013501 (1.000000)     0.000000 (0.000000)     0.000886    0.004361    0.0013501   3.74564e-06 
  momentum20                                                10          0.839798    0.275798 (0.328410)     0.564000 (0.671590)     0.080932    0.092716    0.0839798   0.000232989 
    momentum20/compute                                      10          0.695027    0.131027 (0.188521)     0.564000 (0.811479)     0.06844     0.070687    0.0695027   0.000192824 
    momentum20/infer_shape                                  10          0.03773     0.037730 (1.000000)     0.000000 (0.000000)     0.003145    0.005865    0.003773    1.04676e-05 
    momentum20/prepare_data                                 10          0.013271    0.013271 (1.000000)     0.000000 (0.000000)     0.001102    0.00154     0.0013271   3.68183e-06 
  momentum2                                                 10          0.832178    0.793202 (0.953164)     0.038976 (0.046836)     0.079112    0.088385    0.0832178   0.000230875 
    momentum2/compute                                       10          0.468485    0.429509 (0.916804)     0.038976 (0.083196)     0.043624    0.050151    0.0468485   0.000129974 
    momentum2/infer_shape                                   10          0.129463    0.129463 (1.000000)     0.000000 (0.000000)     0.012059    0.014667    0.0129463   3.59175e-05 
    momentum2/prepare_data                                  10          0.01614     0.016140 (1.000000)     0.000000 (0.000000)     0.001386    0.001853    0.001614    4.47779e-06 
  momentum49                                                10          0.732366    0.701006 (0.957180)     0.031360 (0.042820)     0.028383    0.468479    0.0732366   0.000203183 
    momentum49/compute                                      10          0.600286    0.568926 (0.947758)     0.031360 (0.052242)     0.015544    0.454795    0.0600286   0.00016654  
    momentum49/infer_shape                                  10          0.03428     0.034280 (1.000000)     0.000000 (0.000000)     0.003021    0.003802    0.003428    9.51045e-06 
    momentum49/prepare_data                                 10          0.014438    0.014438 (1.000000)     0.000000 (0.000000)     0.000906    0.004012    0.0014438   4.0056e-06  
  momentum7                                                 10          0.571286    0.273846 (0.479350)     0.297440 (0.520650)     0.054083    0.06469     0.0571286   0.000158494 
    momentum7/compute                                       10          0.426134    0.128694 (0.302004)     0.297440 (0.697996)     0.041966    0.043587    0.0426134   0.000118224 
    momentum7/infer_shape                                   10          0.039842    0.039842 (1.000000)     0.000000 (0.000000)     0.002731    0.012386    0.0039842   1.10535e-05 
    momentum7/prepare_data                                  10          0.013448    0.013448 (1.000000)     0.000000 (0.000000)     0.000982    0.003298    0.0013448   3.73094e-06 
  momentum145                                               10          0.568752    0.256432 (0.450868)     0.312320 (0.549132)     0.05391     0.060621    0.0568752   0.000157791 
    momentum145/compute                                     10          0.447327    0.135007 (0.301808)     0.312320 (0.698192)     0.043057    0.049147    0.0447327   0.000124104 
    momentum145/infer_shape                                 10          0.039191    0.039191 (1.000000)     0.000000 (0.000000)     0.003437    0.006108    0.0039191   1.08729e-05 
    momentum145/prepare_data                                10          0.009539    0.009539 (1.000000)     0.000000 (0.000000)     0.000802    0.00117     0.0009539   2.64645e-06 
  momentum55                                                10          0.560808    0.270952 (0.483146)     0.289856 (0.516854)     0.054827    0.057577    0.0560808   0.000155587 
    momentum55/compute                                      10          0.42159     0.131734 (0.312469)     0.289856 (0.687531)     0.041276    0.04288     0.042159    0.000116964 
    momentum55/infer_shape                                  10          0.040777    0.040777 (1.000000)     0.000000 (0.000000)     0.003379    0.006199    0.0040777   1.13129e-05 
    momentum55/prepare_data                                 10          0.009439    0.009439 (1.000000)     0.000000 (0.000000)     0.000778    0.001405    0.0009439   2.6187e-06  
  momentum81                                                10          0.558533    0.261125 (0.467519)     0.297408 (0.532481)     0.054238    0.058208    0.0558533   0.000154956 
    momentum81/compute                                      10          0.427333    0.129925 (0.304037)     0.297408 (0.695963)     0.042062    0.043719    0.0427333   0.000118557 
    momentum81/infer_shape                                  10          0.044122    0.044122 (1.000000)     0.000000 (0.000000)     0.003791    0.006183    0.0044122   1.2241e-05  
    momentum81/prepare_data                                 10          0.013564    0.013564 (1.000000)     0.000000 (0.000000)     0.001194    0.001578    0.0013564   3.76312e-06 
  momentum111                                               10          0.557831    0.253351 (0.454172)     0.304480 (0.545828)     0.053923    0.057717    0.0557831   0.000154761 
    momentum111/compute                                     10          0.435202    0.130722 (0.300371)     0.304480 (0.699629)     0.042216    0.045638    0.0435202   0.00012074  
    momentum111/infer_shape                                 10          0.033074    0.033074 (1.000000)     0.000000 (0.000000)     0.002846    0.003533    0.0033074   9.17586e-06 
    momentum111/prepare_data                                10          0.016372    0.016372 (1.000000)     0.000000 (0.000000)     0.001055    0.004106    0.0016372   4.54216e-06 
  momentum0                                                 10          0.539067    0.505915 (0.938501)     0.033152 (0.061499)     0.049082    0.063612    0.0539067   0.000149556 
    momentum0/compute                                       10          0.310903    0.277751 (0.893369)     0.033152 (0.106631)     0.029847    0.032667    0.0310903   8.62552e-05 
    momentum0/infer_shape                                   10          0.077918    0.077918 (1.000000)     0.000000 (0.000000)     0.006913    0.008302    0.0077918   2.16171e-05 
    momentum0/prepare_data                                  10          0.011628    0.011628 (1.000000)     0.000000 (0.000000)     0.000988    0.001448    0.0011628   3.22601e-06 
  momentum51                                                10          0.529321    0.495081 (0.935313)     0.034240 (0.064687)     0.047532    0.067862    0.0529321   0.000146852 
    momentum51/compute                                      10          0.31942     0.285180 (0.892806)     0.034240 (0.107194)     0.026306    0.046438    0.031942    8.86181e-05 
    momentum51/infer_shape                                  10          0.063602    0.063602 (1.000000)     0.000000 (0.000000)     0.005723    0.007459    0.0063602   1.76454e-05 
    momentum51/prepare_data                                 10          0.012097    0.012097 (1.000000)     0.000000 (0.000000)     0.001005    0.001298    0.0012097   3.35612e-06 
  momentum78                                                10          0.45589     0.258674 (0.567404)     0.197216 (0.432596)     0.044512    0.048738    0.045589    0.00012648  
    momentum78/compute                                      10          0.32507     0.127854 (0.393312)     0.197216 (0.606688)     0.03187     0.033809    0.032507    9.01856e-05 
    momentum78/infer_shape                                  10          0.034425    0.034425 (1.000000)     0.000000 (0.000000)     0.003073    0.003946    0.0034425   9.55068e-06 
    momentum78/prepare_data                                 10          0.012777    0.012777 (1.000000)     0.000000 (0.000000)     0.001064    0.001634    0.0012777   3.54478e-06 
  momentum87                                                10          0.452022    0.262614 (0.580976)     0.189408 (0.419024)     0.0433      0.047546    0.0452022   0.000125406 
    momentum87/compute                                      10          0.316465    0.127057 (0.401488)     0.189408 (0.598512)     0.030718    0.032481    0.0316465   8.77983e-05 
    momentum87/infer_shape                                  10          0.032827    0.032827 (1.000000)     0.000000 (0.000000)     0.003126    0.003653    0.0032827   9.10734e-06 
    momentum87/prepare_data                                 10          0.015408    0.015408 (1.000000)     0.000000 (0.000000)     0.001119    0.003765    0.0015408   4.27471e-06 
  momentum117                                               10          0.448806    0.257670 (0.574123)     0.191136 (0.425877)     0.043568    0.047558    0.0448806   0.000124514 
    momentum117/compute                                     10          0.321293    0.130157 (0.405104)     0.191136 (0.594896)     0.030903    0.033628    0.0321293   8.91377e-05 
    momentum117/infer_shape                                 10          0.033715    0.033715 (1.000000)     0.000000 (0.000000)     0.00298     0.004016    0.0033715   9.3537e-06  
    momentum117/prepare_data                                10          0.013266    0.013266 (1.000000)     0.000000 (0.000000)     0.001094    0.001555    0.0013266   3.68044e-06 
  momentum76                                                10          0.445961    0.256073 (0.574205)     0.189888 (0.425795)     0.043651    0.04663     0.0445961   0.000123725 
    momentum76/compute                                      10          0.320395    0.130507 (0.407332)     0.189888 (0.592668)     0.031319    0.032799    0.0320395   8.88886e-05 
    momentum76/infer_shape                                  10          0.033584    0.033584 (1.000000)     0.000000 (0.000000)     0.00309     0.003902    0.0033584   9.31736e-06 
    momentum76/prepare_data                                 10          0.013581    0.013581 (1.000000)     0.000000 (0.000000)     0.001145    0.00192     0.0013581   3.76784e-06 
  momentum125                                               10          0.435974    0.248390 (0.569736)     0.187584 (0.430264)     0.041737    0.045758    0.0435974   0.000120954 
    momentum125/compute                                     10          0.317762    0.130178 (0.409671)     0.187584 (0.590329)     0.030725    0.032943    0.0317762   8.81581e-05 
    momentum125/infer_shape                                 10          0.03362     0.033620 (1.000000)     0.000000 (0.000000)     0.003113    0.00366     0.003362    9.32734e-06 
    momentum125/prepare_data                                10          0.011954    0.011954 (1.000000)     0.000000 (0.000000)     0.00095     0.001653    0.0011954   3.31645e-06 
  momentum27                                                10          0.432707    0.253731 (0.586381)     0.178976 (0.413619)     0.041874    0.045344    0.0432707   0.000120048 
    momentum27/compute                                      10          0.306091    0.127115 (0.415285)     0.178976 (0.584715)     0.029648    0.031664    0.0306091   8.49202e-05 
    momentum27/infer_shape                                  10          0.028424    0.028424 (1.000000)     0.000000 (0.000000)     0.002644    0.003323    0.0028424   7.88579e-06 
    momentum27/prepare_data                                 10          0.01576     0.015760 (1.000000)     0.000000 (0.000000)     0.000997    0.00386     0.001576    4.37237e-06 
  momentum122                                               10          0.424438    0.236374 (0.556911)     0.188064 (0.443089)     0.040494    0.043786    0.0424438   0.000117754 
    momentum122/compute                                     10          0.314391    0.126327 (0.401815)     0.188064 (0.598185)     0.030395    0.032769    0.0314391   8.72229e-05 
    momentum122/infer_shape                                 10          0.034458    0.034458 (1.000000)     0.000000 (0.000000)     0.002973    0.005884    0.0034458   9.55983e-06 
    momentum122/prepare_data                                10          0.009781    0.009781 (1.000000)     0.000000 (0.000000)     0.000794    0.001139    0.0009781   2.71359e-06 
  momentum25                                                10          0.422867    0.252179 (0.596355)     0.170688 (0.403645)     0.040806    0.044857    0.0422867   0.000117318 
    momentum25/compute                                      10          0.29955     0.128862 (0.430185)     0.170688 (0.569815)     0.028794    0.03126     0.029955    8.31055e-05 
    momentum25/infer_shape                                  10          0.029262    0.029262 (1.000000)     0.000000 (0.000000)     0.002731    0.003192    0.0029262   8.11828e-06 
    momentum25/prepare_data                                 10          0.015453    0.015453 (1.000000)     0.000000 (0.000000)     0.001006    0.003712    0.0015453   4.28719e-06 
  momentum119                                               10          0.388353    0.282721 (0.728000)     0.105632 (0.272000)     0.033129    0.073225    0.0388353   0.000107742 
    momentum119/compute                                     10          0.271491    0.165859 (0.610919)     0.105632 (0.389081)     0.022356    0.061963    0.0271491   7.53209e-05 
    momentum119/infer_shape                                 10          0.034243    0.034243 (1.000000)     0.000000 (0.000000)     0.002994    0.005135    0.0034243   9.50019e-06 
    momentum119/prepare_data                                10          0.010021    0.010021 (1.000000)     0.000000 (0.000000)     0.000807    0.001213    0.0010021   2.78017e-06 
  momentum16                                                10          0.371246    0.267406 (0.720293)     0.103840 (0.279707)     0.033718    0.046751    0.0371246   0.000102996 
    momentum16/compute                                      10          0.232431    0.128591 (0.553244)     0.103840 (0.446756)     0.021879    0.024318    0.0232431   6.44843e-05 
    momentum16/infer_shape                                  10          0.030433    0.030433 (1.000000)     0.000000 (0.000000)     0.002649    0.003346    0.0030433   8.44316e-06 
    momentum16/prepare_data                                 10          0.013832    0.013832 (1.000000)     0.000000 (0.000000)     0.001161    0.001777    0.0013832   3.83747e-06 
  momentum22                                                10          0.371026    0.263346 (0.709778)     0.107680 (0.290222)     0.034601    0.040856    0.0371026   0.000102935 
    momentum22/compute                                      10          0.235898    0.128218 (0.543532)     0.107680 (0.456468)     0.022772    0.024836    0.0235898   6.54462e-05 
    momentum22/infer_shape                                  10          0.034224    0.034224 (1.000000)     0.000000 (0.000000)     0.002674    0.007845    0.0034224   9.49491e-06 
    momentum22/prepare_data                                 10          0.012147    0.012147 (1.000000)     0.000000 (0.000000)     0.000955    0.001671    0.0012147   3.37e-06    
  momentum107                                               10          0.369097    0.267209 (0.723953)     0.101888 (0.276047)     0.034091    0.045554    0.0369097   0.0001024   
    momentum107/compute                                     10          0.229842    0.127954 (0.556704)     0.101888 (0.443296)     0.022149    0.024661    0.0229842   6.37661e-05 
    momentum107/infer_shape                                 10          0.04545     0.045450 (1.000000)     0.000000 (0.000000)     0.003425    0.01208     0.004545    1.26094e-05 
    momentum107/prepare_data                                10          0.016558    0.016558 (1.000000)     0.000000 (0.000000)     0.001246    0.003655    0.0016558   4.59376e-06 
  momentum84                                                10          0.368969    0.262249 (0.710762)     0.106720 (0.289238)     0.035306    0.039533    0.0368969   0.000102365 
    momentum84/compute                                      10          0.238874    0.132154 (0.553237)     0.106720 (0.446763)     0.022585    0.024963    0.0238874   6.62719e-05 
    momentum84/infer_shape                                  10          0.039086    0.039086 (1.000000)     0.000000 (0.000000)     0.003456    0.005923    0.0039086   1.08438e-05 
    momentum84/prepare_data                                 10          0.013907    0.013907 (1.000000)     0.000000 (0.000000)     0.001167    0.001779    0.0013907   3.85828e-06 
  momentum126                                               10          0.366005    0.254357 (0.694955)     0.111648 (0.305045)     0.034136    0.044712    0.0366005   0.000101542 
    momentum126/compute                                     10          0.248306    0.136658 (0.550361)     0.111648 (0.449639)     0.023276    0.033367    0.0248306   6.88886e-05 
    momentum126/infer_shape                                 10          0.032914    0.032914 (1.000000)     0.000000 (0.000000)     0.002717    0.005221    0.0032914   9.13147e-06 
    momentum126/prepare_data                                10          0.009343    0.009343 (1.000000)     0.000000 (0.000000)     0.000806    0.001131    0.0009343   2.59207e-06 
  momentum63                                                10          0.363781    0.260581 (0.716313)     0.103200 (0.283687)     0.033498    0.046924    0.0363781   0.000100925 
    momentum63/compute                                      10          0.234173    0.130973 (0.559300)     0.103200 (0.440700)     0.022524    0.025805    0.0234173   6.49676e-05 
    momentum63/infer_shape                                  10          0.033084    0.033084 (1.000000)     0.000000 (0.000000)     0.003004    0.003864    0.0033084   9.17864e-06 
    momentum63/prepare_data                                 10          0.009461    0.009461 (1.000000)     0.000000 (0.000000)     0.000827    0.00119     0.0009461   2.62481e-06 
  momentum8                                                 10          0.358421    0.251925 (0.702875)     0.106496 (0.297125)     0.034424    0.038267    0.0358421   9.94383e-05 
    momentum8/compute                                       10          0.232326    0.125830 (0.541610)     0.106496 (0.458390)     0.022288    0.024376    0.0232326   6.44552e-05 
    momentum8/infer_shape                                   10          0.029294    0.029294 (1.000000)     0.000000 (0.000000)     0.002696    0.003367    0.0029294   8.12716e-06 
    momentum8/prepare_data                                  10          0.015046    0.015046 (1.000000)     0.000000 (0.000000)     0.001072    0.003435    0.0015046   4.17428e-06 
  momentum65                                                10          0.356545    0.253505 (0.711004)     0.103040 (0.288996)     0.033347    0.038356    0.0356545   9.89178e-05 
    momentum65/compute                                      10          0.230046    0.127006 (0.552090)     0.103040 (0.447910)     0.021474    0.024111    0.0230046   6.38227e-05 
    momentum65/infer_shape                                  10          0.032841    0.032841 (1.000000)     0.000000 (0.000000)     0.002986    0.003934    0.0032841   9.11122e-06 
    momentum65/prepare_data                                 10          0.01216     0.012160 (1.000000)     0.000000 (0.000000)     0.000804    0.003305    0.001216    3.3736e-06  
  momentum42                                                10          0.353683    0.274675 (0.776614)     0.079008 (0.223386)     0.032525    0.043305    0.0353683   9.81238e-05 
    momentum42/compute                                      10          0.205947    0.126939 (0.616367)     0.079008 (0.383633)     0.019951    0.022269    0.0205947   5.71368e-05 
    momentum42/infer_shape                                  10          0.038657    0.038657 (1.000000)     0.000000 (0.000000)     0.003309    0.006563    0.0038657   1.07248e-05 
    momentum42/prepare_data                                 10          0.013687    0.013687 (1.000000)     0.000000 (0.000000)     0.001053    0.001739    0.0013687   3.79724e-06 
  momentum131                                               10          0.351969    0.250017 (0.710338)     0.101952 (0.289662)     0.033095    0.037032    0.0351969   9.76483e-05 
    momentum131/compute                                     10          0.230781    0.128829 (0.558231)     0.101952 (0.441769)     0.022168    0.025555    0.0230781   6.40266e-05 
    momentum131/infer_shape                                 10          0.03474     0.034740 (1.000000)     0.000000 (0.000000)     0.003098    0.00378     0.003474    9.63807e-06 
    momentum131/prepare_data                                10          0.010939    0.010939 (1.000000)     0.000000 (0.000000)     0.000797    0.002897    0.0010939   3.03485e-06 
  momentum146                                               10          0.350152    0.240168 (0.685896)     0.109984 (0.314104)     0.033534    0.036327    0.0350152   9.71442e-05 
    momentum146/compute                                     10          0.234345    0.124361 (0.530675)     0.109984 (0.469325)     0.022266    0.024549    0.0234345   6.50154e-05 
    momentum146/infer_shape                                 10          0.034152    0.034152 (1.000000)     0.000000 (0.000000)     0.002929    0.005169    0.0034152   9.47494e-06 
    momentum146/prepare_data                                10          0.012052    0.012052 (1.000000)     0.000000 (0.000000)     0.000874    0.00161     0.0012052   3.34364e-06 
  momentum41                                                10          0.345933    0.314637 (0.909532)     0.031296 (0.090468)     0.027345    0.089257    0.0345933   9.59737e-05 
    momentum41/compute                                      10          0.164743    0.133447 (0.810031)     0.031296 (0.189969)     0.01503     0.020051    0.0164743   4.57054e-05 
    momentum41/infer_shape                                  10          0.050909    0.050909 (1.000000)     0.000000 (0.000000)     0.002694    0.024237    0.0050909   1.41239e-05 
    momentum41/prepare_data                                 10          0.009301    0.009301 (1.000000)     0.000000 (0.000000)     0.000733    0.001276    0.0009301   2.58042e-06 
  momentum52                                                10          0.345902    0.308494 (0.891854)     0.037408 (0.108146)     0.030681    0.050659    0.0345902   9.59651e-05 
    momentum52/compute                                      10          0.178843    0.141435 (0.790833)     0.037408 (0.209167)     0.016234    0.020948    0.0178843   4.96172e-05 
    momentum52/infer_shape                                  10          0.033515    0.033515 (1.000000)     0.000000 (0.000000)     0.002905    0.00394     0.0033515   9.29821e-06 
    momentum52/prepare_data                                 10          0.011312    0.011312 (1.000000)     0.000000 (0.000000)     0.001046    0.001419    0.0011312   3.13834e-06 
  momentum53                                                10          0.34494     0.287564 (0.833664)     0.057376 (0.166336)     0.03219     0.038501    0.034494    9.56982e-05 
    momentum53/compute                                      10          0.199351    0.141975 (0.712186)     0.057376 (0.287814)     0.019033    0.02071     0.0199351   5.53068e-05 
    momentum53/infer_shape                                  10          0.034767    0.034767 (1.000000)     0.000000 (0.000000)     0.003153    0.004076    0.0034767   9.64556e-06 
    momentum53/prepare_data                                 10          0.008868    0.008868 (1.000000)     0.000000 (0.000000)     0.000783    0.001186    0.0008868   2.46029e-06 
  momentum97                                                10          0.338199    0.265943 (0.786351)     0.072256 (0.213649)     0.031315    0.042933    0.0338199   9.3828e-05  
    momentum97/compute                                      10          0.211253    0.138997 (0.657965)     0.072256 (0.342035)     0.019134    0.030498    0.0211253   5.86088e-05 
    momentum97/infer_shape                                  10          0.036624    0.036624 (1.000000)     0.000000 (0.000000)     0.003254    0.005639    0.0036624   1.01608e-05 
    momentum97/prepare_data                                 10          0.014213    0.014213 (1.000000)     0.000000 (0.000000)     0.001213    0.001848    0.0014213   3.94317e-06 
  momentum86                                                10          0.336774    0.266694 (0.791908)     0.070080 (0.208092)     0.031055    0.039587    0.0336774   9.34327e-05 
    momentum86/compute                                      10          0.213236    0.143156 (0.671350)     0.070080 (0.328650)     0.019579    0.02677     0.0213236   5.9159e-05  
    momentum86/infer_shape                                  10          0.033607    0.033607 (1.000000)     0.000000 (0.000000)     0.003       0.003696    0.0033607   9.32374e-06 
    momentum86/prepare_data                                 10          0.013178    0.013178 (1.000000)     0.000000 (0.000000)     0.001081    0.001789    0.0013178   3.65603e-06 
  momentum28                                                10          0.330525    0.292605 (0.885273)     0.037920 (0.114727)     0.028427    0.048053    0.0330525   9.1699e-05  
    momentum28/compute                                      10          0.17327     0.135350 (0.781151)     0.037920 (0.218849)     0.015633    0.019821    0.017327    4.80711e-05 
    momentum28/infer_shape                                  10          0.05251     0.052510 (1.000000)     0.000000 (0.000000)     0.003106    0.01959     0.005251    1.45681e-05 
    momentum28/prepare_data                                 10          0.019938    0.019938 (1.000000)     0.000000 (0.000000)     0.000939    0.007038    0.0019938   5.53149e-06 
  momentum40                                                10          0.330306    0.291810 (0.883454)     0.038496 (0.116546)     0.027466    0.046675    0.0330306   9.16382e-05 
    momentum40/compute                                      10          0.199264    0.160768 (0.806809)     0.038496 (0.193191)     0.015456    0.033681    0.0199264   5.52827e-05 
    momentum40/infer_shape                                  10          0.032721    0.032721 (1.000000)     0.000000 (0.000000)     0.00281     0.005174    0.0032721   9.07793e-06 
    momentum40/prepare_data                                 10          0.014345    0.014345 (1.000000)     0.000000 (0.000000)     0.001148    0.001792    0.0014345   3.9798e-06  
  momentum80                                                10          0.329772    0.293772 (0.890834)     0.036000 (0.109166)     0.029182    0.050381    0.0329772   9.14901e-05 
    momentum80/compute                                      10          0.170596    0.134596 (0.788975)     0.036000 (0.211025)     0.015531    0.021685    0.0170596   4.73292e-05 
    momentum80/infer_shape                                  10          0.060719    0.060719 (1.000000)     0.000000 (0.000000)     0.003143    0.018629    0.0060719   1.68455e-05 
    momentum80/prepare_data                                 10          0.014262    0.014262 (1.000000)     0.000000 (0.000000)     0.001303    0.001717    0.0014262   3.95677e-06 
  momentum3                                                 10          0.327682    0.291938 (0.890919)     0.035744 (0.109081)     0.030455    0.038356    0.0327682   9.09102e-05 
    momentum3/compute                                       10          0.179244    0.143500 (0.800585)     0.035744 (0.199415)     0.016622    0.018978    0.0179244   4.97284e-05 
    momentum3/infer_shape                                   10          0.035769    0.035769 (1.000000)     0.000000 (0.000000)     0.00329     0.003947    0.0035769   9.92355e-06 
    momentum3/prepare_data                                  10          0.018211    0.018211 (1.000000)     0.000000 (0.000000)     0.001099    0.007414    0.0018211   5.05236e-06 
  momentum46                                                10          0.326891    0.248619 (0.760556)     0.078272 (0.239444)     0.030726    0.035562    0.0326891   9.06908e-05 
    momentum46/compute                                      10          0.206364    0.128092 (0.620709)     0.078272 (0.379291)     0.019262    0.021921    0.0206364   5.72525e-05 
    momentum46/infer_shape                                  10          0.033935    0.033935 (1.000000)     0.000000 (0.000000)     0.003029    0.00376     0.0033935   9.41474e-06 
    momentum46/prepare_data                                 10          0.012784    0.012784 (1.000000)     0.000000 (0.000000)     0.001091    0.001548    0.0012784   3.54672e-06 
  momentum32                                                10          0.326691    0.275811 (0.844256)     0.050880 (0.155744)     0.029198    0.039674    0.0326691   9.06353e-05 
    momentum32/compute                                      10          0.178255    0.127375 (0.714566)     0.050880 (0.285434)     0.016983    0.019099    0.0178255   4.94541e-05 
    momentum32/infer_shape                                  10          0.038951    0.038951 (1.000000)     0.000000 (0.000000)     0.002724    0.011577    0.0038951   1.08063e-05 
    momentum32/prepare_data                                 10          0.014478    0.014478 (1.000000)     0.000000 (0.000000)     0.001203    0.002113    0.0014478   4.01669e-06 
  momentum5                                                 10          0.326349    0.285070 (0.873513)     0.041279 (0.126487)     0.030273    0.040344    0.0326349   9.05404e-05 
    momentum5/compute                                       10          0.207754    0.166475 (0.801308)     0.041279 (0.198692)     0.01933     0.0288      0.0207754   5.76381e-05 
    momentum5/infer_shape                                   10          0.02852     0.028520 (1.000000)     0.000000 (0.000000)     0.002593    0.003261    0.002852    7.91243e-06 
    momentum5/prepare_data                                  10          0.016095    0.016095 (1.000000)     0.000000 (0.000000)     0.001209    0.003693    0.0016095   4.46531e-06 
  momentum91                                                10          0.324932    0.291332 (0.896594)     0.033600 (0.103406)     0.027999    0.060286    0.0324932   9.01473e-05 
    momentum91/compute                                      10          0.164589    0.130989 (0.795855)     0.033600 (0.204145)     0.014859    0.020492    0.0164589   4.56626e-05 
    momentum91/infer_shape                                  10          0.041295    0.041295 (1.000000)     0.000000 (0.000000)     0.003579    0.004883    0.0041295   1.14567e-05 
    momentum91/prepare_data                                 10          0.01579     0.015790 (1.000000)     0.000000 (0.000000)     0.001375    0.001931    0.001579    4.38069e-06 
  momentum54                                                10          0.324342    0.292950 (0.903213)     0.031392 (0.096787)     0.027667    0.04704     0.0324342   8.99836e-05 
    momentum54/compute                                      10          0.18534     0.153948 (0.830625)     0.031392 (0.169375)     0.015359    0.034162    0.018534    5.14197e-05 
    momentum54/infer_shape                                  10          0.037908    0.037908 (1.000000)     0.000000 (0.000000)     0.003144    0.00661     0.0037908   1.0517e-05  
    momentum54/prepare_data                                 10          0.009467    0.009467 (1.000000)     0.000000 (0.000000)     0.000769    0.00129     0.0009467   2.62647e-06 
  momentum39                                                10          0.323928    0.290329 (0.896276)     0.033599 (0.103724)     0.027426    0.066869    0.0323928   8.98688e-05 
    momentum39/compute                                      10          0.161743    0.128144 (0.792269)     0.033599 (0.207731)     0.015473    0.017255    0.0161743   4.48731e-05 
    momentum39/infer_shape                                  10          0.035895    0.035895 (1.000000)     0.000000 (0.000000)     0.002935    0.006591    0.0035895   9.95851e-06 
    momentum39/prepare_data                                 10          0.011489    0.011489 (1.000000)     0.000000 (0.000000)     0.000912    0.001482    0.0011489   3.18744e-06 
  momentum134                                               10          0.321136    0.249296 (0.776294)     0.071840 (0.223706)     0.030156    0.034586    0.0321136   8.90942e-05 
    momentum134/compute                                     10          0.195841    0.124001 (0.633172)     0.071840 (0.366828)     0.01843     0.020415    0.0195841   5.4333e-05  
    momentum134/infer_shape                                 10          0.032171    0.032171 (1.000000)     0.000000 (0.000000)     0.003035    0.003611    0.0032171   8.92534e-06 
    momentum134/prepare_data                                10          0.012619    0.012619 (1.000000)     0.000000 (0.000000)     0.000923    0.003036    0.0012619   3.50094e-06 
  momentum113                                               10          0.320979    0.266515 (0.830319)     0.054464 (0.169681)     0.030578    0.034177    0.0320979   8.90506e-05 
    momentum113/compute                                     10          0.186982    0.132518 (0.708721)     0.054464 (0.291279)     0.017397    0.020465    0.0186982   5.18752e-05 
    momentum113/infer_shape                                 10          0.035219    0.035219 (1.000000)     0.000000 (0.000000)     0.003218    0.003692    0.0035219   9.77096e-06 
    momentum113/prepare_data                                10          0.010234    0.010234 (1.000000)     0.000000 (0.000000)     0.000824    0.001179    0.0010234   2.83926e-06 
  momentum82                                                10          0.317343    0.278431 (0.877382)     0.038912 (0.122618)     0.028642    0.048774    0.0317343   8.80419e-05 
    momentum82/compute                                      10          0.187038    0.148126 (0.791957)     0.038912 (0.208043)     0.016344    0.034521    0.0187038   5.18908e-05 
    momentum82/infer_shape                                  10          0.036917    0.036917 (1.000000)     0.000000 (0.000000)     0.003125    0.006718    0.0036917   1.0242e-05  
    momentum82/prepare_data                                 10          0.014224    0.014224 (1.000000)     0.000000 (0.000000)     0.001258    0.001583    0.0014224   3.94623e-06 
  momentum156                                               10          0.316777    0.264489 (0.834938)     0.052288 (0.165062)     0.029441    0.040609    0.0316777   8.78848e-05 
    momentum156/compute                                     10          0.18271     0.130422 (0.713820)     0.052288 (0.286180)     0.017599    0.018867    0.018271    5.069e-05   
    momentum156/infer_shape                                 10          0.028922    0.028922 (1.000000)     0.000000 (0.000000)     0.002619    0.003292    0.0028922   8.02396e-06 
    momentum156/prepare_data                                10          0.013222    0.013222 (1.000000)     0.000000 (0.000000)     0.000964    0.002866    0.0013222   3.66824e-06 
  momentum15                                                10          0.313552    0.270864 (0.863857)     0.042688 (0.136143)     0.028356    0.045728    0.0313552   8.69901e-05 
    momentum15/compute                                      10          0.187058    0.144370 (0.771793)     0.042688 (0.228207)     0.016162    0.030963    0.0187058   5.18963e-05 
    momentum15/infer_shape                                  10          0.034392    0.034392 (1.000000)     0.000000 (0.000000)     0.002917    0.005448    0.0034392   9.54152e-06 
    momentum15/prepare_data                                 10          0.013839    0.013839 (1.000000)     0.000000 (0.000000)     0.001124    0.001879    0.0013839   3.83941e-06 
  momentum124                                               10          0.312687    0.274063 (0.876477)     0.038624 (0.123523)     0.027221    0.048811    0.0312687   8.67501e-05 
    momentum124/compute                                     10          0.189965    0.151341 (0.796678)     0.038624 (0.203322)     0.015717    0.037131    0.0189965   5.27028e-05 
    momentum124/infer_shape                                 10          0.034676    0.034676 (1.000000)     0.000000 (0.000000)     0.003067    0.005526    0.0034676   9.62031e-06 
    momentum124/prepare_data                                10          0.011909    0.011909 (1.000000)     0.000000 (0.000000)     0.001011    0.001554    0.0011909   3.30397e-06 
  momentum90                                                10          0.312655    0.267023 (0.854050)     0.045632 (0.145950)     0.029789    0.033986    0.0312655   8.67412e-05 
    momentum90/compute                                      10          0.18041     0.134778 (0.747065)     0.045632 (0.252935)     0.016466    0.020382    0.018041    5.00519e-05 
    momentum90/infer_shape                                  10          0.036171    0.036171 (1.000000)     0.000000 (0.000000)     0.003359    0.003821    0.0036171   1.00351e-05 
    momentum90/prepare_data                                 10          0.01732     0.017320 (1.000000)     0.000000 (0.000000)     0.00133     0.00381     0.001732    4.80516e-06 
  momentum72                                                10          0.312585    0.276489 (0.884524)     0.036096 (0.115476)     0.02778     0.043817    0.0312585   8.67218e-05 
    momentum72/compute                                      10          0.179651    0.143555 (0.799077)     0.036096 (0.200923)     0.015692    0.029069    0.0179651   4.98414e-05 
    momentum72/infer_shape                                  10          0.034823    0.034823 (1.000000)     0.000000 (0.000000)     0.003008    0.003846    0.0034823   9.6611e-06  
    momentum72/prepare_data                                 10          0.021397    0.021397 (1.000000)     0.000000 (0.000000)     0.001231    0.00801     0.0021397   5.93626e-06 
  momentum83                                                10          0.312526    0.276494 (0.884707)     0.036032 (0.115293)     0.02799     0.040772    0.0312526   8.67054e-05 
    momentum83/compute                                      10          0.174428    0.138396 (0.793428)     0.036032 (0.206572)     0.015427    0.026814    0.0174428   4.83923e-05 
    momentum83/infer_shape                                  10          0.04081     0.040810 (1.000000)     0.000000 (0.000000)     0.002918    0.010436    0.004081    1.13221e-05 
    momentum83/prepare_data                                 10          0.012636    0.012636 (1.000000)     0.000000 (0.000000)     0.001075    0.001485    0.0012636   3.50566e-06 
  momentum29                                                10          0.311906    0.258114 (0.827538)     0.053792 (0.172462)     0.029417    0.037212    0.0311906   8.65334e-05 
    momentum29/compute                                      10          0.188336    0.134544 (0.714383)     0.053792 (0.285617)     0.017405    0.02518     0.0188336   5.22509e-05 
    momentum29/infer_shape                                  10          0.031864    0.031864 (1.000000)     0.000000 (0.000000)     0.002978    0.003731    0.0031864   8.84017e-06 
    momentum29/prepare_data                                 10          0.014806    0.014806 (1.000000)     0.000000 (0.000000)     0.001116    0.001796    0.0014806   4.10769e-06 
  momentum12                                                10          0.311749    0.278693 (0.893966)     0.033056 (0.106034)     0.027402    0.040929    0.0311749   8.64899e-05 
    momentum12/compute                                      10          0.170927    0.137871 (0.806607)     0.033056 (0.193393)     0.015123    0.026182    0.0170927   4.7421e-05  
    momentum12/infer_shape                                  10          0.026733    0.026733 (1.000000)     0.000000 (0.000000)     0.002507    0.002845    0.0026733   7.41665e-06 
    momentum12/prepare_data                                 10          0.02486     0.024860 (1.000000)     0.000000 (0.000000)     0.001286    0.01144     0.002486    6.89702e-06 
  momentum10                                                10          0.311429    0.268581 (0.862415)     0.042848 (0.137585)     0.02864     0.040326    0.0311429   8.64011e-05 
    momentum10/compute                                      10          0.179748    0.136900 (0.761622)     0.042848 (0.238378)     0.016114    0.027602    0.0179748   4.98683e-05 
    momentum10/infer_shape                                  10          0.028604    0.028604 (1.000000)     0.000000 (0.000000)     0.002571    0.003313    0.0028604   7.93573e-06 
    momentum10/prepare_data                                 10          0.015463    0.015463 (1.000000)     0.000000 (0.000000)     0.001108    0.003475    0.0015463   4.28997e-06 
  momentum158                                               10          0.309838    0.279150 (0.900955)     0.030688 (0.099045)     0.026241    0.050074    0.0309838   8.59597e-05 
    momentum158/compute                                     10          0.190417    0.159729 (0.838838)     0.030688 (0.161162)     0.015258    0.03748     0.0190417   5.28282e-05 
    momentum158/infer_shape                                 10          0.033376    0.033376 (1.000000)     0.000000 (0.000000)     0.002905    0.004108    0.0033376   9.25965e-06 
    momentum158/prepare_data                                10          0.011311    0.011311 (1.000000)     0.000000 (0.000000)     0.001023    0.001516    0.0011311   3.13806e-06 
  momentum24                                                10          0.308615    0.273799 (0.887186)     0.034816 (0.112814)     0.027589    0.044125    0.0308615   8.56204e-05 
    momentum24/compute                                      10          0.174912    0.140096 (0.800951)     0.034816 (0.199049)     0.015654    0.027357    0.0174912   4.85266e-05 
    momentum24/infer_shape                                  10          0.029731    0.029731 (1.000000)     0.000000 (0.000000)     0.002452    0.003429    0.0029731   8.2484e-06  
    momentum24/prepare_data                                 10          0.013337    0.013337 (1.000000)     0.000000 (0.000000)     0.000991    0.001616    0.0013337   3.70014e-06 
  momentum11                                                10          0.308233    0.277417 (0.900024)     0.030816 (0.099976)     0.028232    0.047341    0.0308233   8.55144e-05 
    momentum11/compute                                      10          0.182643    0.151827 (0.831277)     0.030816 (0.168723)     0.015972    0.035331    0.0182643   5.06714e-05 
    momentum11/infer_shape                                  10          0.031874    0.031874 (1.000000)     0.000000 (0.000000)     0.003013    0.003518    0.0031874   8.84294e-06 
    momentum11/prepare_data                                 10          0.014771    0.014771 (1.000000)     0.000000 (0.000000)     0.00134     0.001794    0.0014771   4.09798e-06 
  momentum37                                                10          0.307747    0.276451 (0.898306)     0.031296 (0.101694)     0.02799     0.03905     0.0307747   8.53796e-05 
    momentum37/compute                                      10          0.158819    0.127523 (0.802945)     0.031296 (0.197055)     0.015458    0.017421    0.0158819   4.40618e-05 
    momentum37/infer_shape                                  10          0.039643    0.039643 (1.000000)     0.000000 (0.000000)     0.003115    0.006013    0.0039643   1.09983e-05 
    momentum37/prepare_data                                 10          0.011437    0.011437 (1.000000)     0.000000 (0.000000)     0.000916    0.001512    0.0011437   3.17302e-06 
  momentum14                                                10          0.305851    0.252859 (0.826739)     0.052992 (0.173261)     0.029113    0.034483    0.0305851   8.48536e-05 
    momentum14/compute                                      10          0.180593    0.127601 (0.706567)     0.052992 (0.293433)     0.017413    0.019471    0.0180593   5.01027e-05 
    momentum14/infer_shape                                  10          0.027422    0.027422 (1.000000)     0.000000 (0.000000)     0.002562    0.003006    0.0027422   7.6078e-06  
    momentum14/prepare_data                                 10          0.014063    0.014063 (1.000000)     0.000000 (0.000000)     0.001131    0.001651    0.0014063   3.90156e-06 
  momentum100                                               10          0.305724    0.264700 (0.865814)     0.041024 (0.134186)     0.028589    0.034614    0.0305724   8.48183e-05 
    momentum100/compute                                     10          0.171904    0.130880 (0.761355)     0.041024 (0.238645)     0.016326    0.018149    0.0171904   4.76921e-05 
    momentum100/infer_shape                                 10          0.040097    0.040097 (1.000000)     0.000000 (0.000000)     0.003255    0.008201    0.0040097   1.11243e-05 
    momentum100/prepare_data                                10          0.013731    0.013731 (1.000000)     0.000000 (0.000000)     0.001204    0.001624    0.0013731   3.80945e-06 
  momentum112                                               10          0.305306    0.265082 (0.868250)     0.040224 (0.131750)     0.027082    0.039458    0.0305306   8.47024e-05 
    momentum112/compute                                     10          0.175228    0.135004 (0.770448)     0.040224 (0.229552)     0.015917    0.021536    0.0175228   4.86143e-05 
    momentum112/infer_shape                                 10          0.03476     0.034760 (1.000000)     0.000000 (0.000000)     0.003119    0.004126    0.003476    9.64362e-06 
    momentum112/prepare_data                                10          0.013538    0.013538 (1.000000)     0.000000 (0.000000)     0.001014    0.003177    0.0013538   3.75591e-06 
  momentum19                                                10          0.304791    0.268727 (0.881676)     0.036064 (0.118324)     0.028022    0.03855     0.0304791   8.45595e-05 
    momentum19/compute                                      10          0.16387     0.127806 (0.779923)     0.036064 (0.220077)     0.01538     0.017666    0.016387    4.54632e-05 
    momentum19/infer_shape                                  10          0.036737    0.036737 (1.000000)     0.000000 (0.000000)     0.003069    0.006416    0.0036737   1.01921e-05 
    momentum19/prepare_data                                 10          0.013325    0.013325 (1.000000)     0.000000 (0.000000)     0.001013    0.00162     0.0013325   3.69681e-06 
  momentum99                                                10          0.304398    0.252366 (0.829066)     0.052032 (0.170934)     0.029037    0.033533    0.0304398   8.44505e-05 
    momentum99/compute                                      10          0.178177    0.126145 (0.707976)     0.052032 (0.292024)     0.017205    0.01885     0.0178177   4.94324e-05 
    momentum99/infer_shape                                  10          0.034234    0.034234 (1.000000)     0.000000 (0.000000)     0.003034    0.005328    0.0034234   9.49769e-06 
    momentum99/prepare_data                                 10          0.011077    0.011077 (1.000000)     0.000000 (0.000000)     0.000969    0.001298    0.0011077   3.07314e-06 
  momentum67                                                10          0.304293    0.271173 (0.891158)     0.033120 (0.108842)     0.029223    0.032955    0.0304293   8.44213e-05 
    momentum67/compute                                      10          0.169442    0.136322 (0.804535)     0.033120 (0.195465)     0.015547    0.019385    0.0169442   4.7009e-05  
    momentum67/infer_shape                                  10          0.032289    0.032289 (1.000000)     0.000000 (0.000000)     0.003086    0.003773    0.0032289   8.95808e-06 
    momentum67/prepare_data                                 10          0.01391     0.013910 (1.000000)     0.000000 (0.000000)     0.000993    0.003212    0.001391    3.85911e-06 
  momentum48                                                10          0.304234    0.267914 (0.880618)     0.036320 (0.119382)     0.027986    0.039332    0.0304234   8.4405e-05  
    momentum48/compute                                      10          0.168707    0.132387 (0.784716)     0.036320 (0.215284)     0.015916    0.018249    0.0168707   4.68051e-05 
    momentum48/infer_shape                                  10          0.039798    0.039798 (1.000000)     0.000000 (0.000000)     0.002555    0.013616    0.0039798   1.10413e-05 
    momentum48/prepare_data                                 10          0.017825    0.017825 (1.000000)     0.000000 (0.000000)     0.001278    0.004316    0.0017825   4.94527e-06 
  momentum23                                                10          0.304118    0.264822 (0.870787)     0.039296 (0.129213)     0.028493    0.033739    0.0304118   8.43728e-05 
    momentum23/compute                                      10          0.168633    0.129337 (0.766973)     0.039296 (0.233027)     0.015732    0.018145    0.0168633   4.67846e-05 
    momentum23/infer_shape                                  10          0.0334      0.033400 (1.000000)     0.000000 (0.000000)     0.002874    0.003809    0.00334     9.26631e-06 
    momentum23/prepare_data                                 10          0.016015    0.016015 (1.000000)     0.000000 (0.000000)     0.00093     0.004068    0.0016015   4.44311e-06 
  momentum108                                               10          0.30264     0.267888 (0.885170)     0.034752 (0.114830)     0.027727    0.038294    0.030264    8.39627e-05 
    momentum108/compute                                     10          0.166595    0.131843 (0.791398)     0.034752 (0.208602)     0.015524    0.017897    0.0166595   4.62192e-05 
    momentum108/infer_shape                                 10          0.047739    0.047739 (1.000000)     0.000000 (0.000000)     0.003679    0.013137    0.0047739   1.32444e-05 
    momentum108/prepare_data                                10          0.013855    0.013855 (1.000000)     0.000000 (0.000000)     0.001184    0.001637    0.0013855   3.84385e-06 
  momentum103                                               10          0.302085    0.260933 (0.863773)     0.041152 (0.136227)     0.028486    0.032707    0.0302085   8.38088e-05 
    momentum103/compute                                     10          0.17343     0.132278 (0.762717)     0.041152 (0.237283)     0.016754    0.018025    0.017343    4.81154e-05 
    momentum103/infer_shape                                 10          0.039537    0.039537 (1.000000)     0.000000 (0.000000)     0.003755    0.00434     0.0039537   1.09689e-05 
    momentum103/prepare_data                                10          0.011372    0.011372 (1.000000)     0.000000 (0.000000)     0.001026    0.001354    0.0011372   3.15498e-06 
  momentum79                                                10          0.300957    0.263325 (0.874959)     0.037632 (0.125041)     0.028105    0.040647    0.0300957   8.34958e-05 
    momentum79/compute                                      10          0.173758    0.136126 (0.783423)     0.037632 (0.216577)     0.015557    0.025696    0.0173758   4.82064e-05 
    momentum79/infer_shape                                  10          0.034233    0.034233 (1.000000)     0.000000 (0.000000)     0.002997    0.005456    0.0034233   9.49741e-06 
    momentum79/prepare_data                                 10          0.011967    0.011967 (1.000000)     0.000000 (0.000000)     0.001067    0.001324    0.0011967   3.32006e-06 
  momentum69                                                10          0.299828    0.259956 (0.867017)     0.039872 (0.132983)     0.027212    0.033527    0.0299828   8.31826e-05 
    momentum69/compute                                      10          0.173544    0.133672 (0.770248)     0.039872 (0.229752)     0.015696    0.018853    0.0173544   4.81471e-05 
    momentum69/infer_shape                                  10          0.032691    0.032691 (1.000000)     0.000000 (0.000000)     0.002859    0.004064    0.0032691   9.06961e-06 
    momentum69/prepare_data                                 10          0.013739    0.013739 (1.000000)     0.000000 (0.000000)     0.000861    0.004098    0.0013739   3.81167e-06 
  momentum50                                                10          0.298959    0.265359 (0.887610)     0.033600 (0.112390)     0.027304    0.040586    0.0298959   8.29415e-05 
    momentum50/compute                                      10          0.166448    0.132848 (0.798135)     0.033600 (0.201865)     0.015248    0.0212      0.0166448   4.61784e-05 
    momentum50/infer_shape                                  10          0.033102    0.033102 (1.000000)     0.000000 (0.000000)     0.002734    0.004929    0.0033102   9.18363e-06 
    momentum50/prepare_data                                 10          0.014688    0.014688 (1.000000)     0.000000 (0.000000)     0.000912    0.003795    0.0014688   4.07496e-06 
  momentum17                                                10          0.298586    0.261114 (0.874502)     0.037472 (0.125498)     0.027428    0.031323    0.0298586   8.2838e-05  
    momentum17/compute                                      10          0.168406    0.130934 (0.777490)     0.037472 (0.222510)     0.015167    0.017586    0.0168406   4.67216e-05 
    momentum17/infer_shape                                  10          0.030745    0.030745 (1.000000)     0.000000 (0.000000)     0.002692    0.004806    0.0030745   8.52972e-06 
    momentum17/prepare_data                                 10          0.014862    0.014862 (1.000000)     0.000000 (0.000000)     0.000974    0.00177     0.0014862   4.12323e-06 
  momentum68                                                10          0.298008    0.258520 (0.867493)     0.039488 (0.132507)     0.027025    0.034405    0.0298008   8.26777e-05 
    momentum68/compute                                      10          0.176703    0.137215 (0.776529)     0.039488 (0.223471)     0.015369    0.022086    0.0176703   4.90235e-05 
    momentum68/infer_shape                                  10          0.032088    0.032088 (1.000000)     0.000000 (0.000000)     0.002911    0.003529    0.0032088   8.90231e-06 
    momentum68/prepare_data                                 10          0.013573    0.013573 (1.000000)     0.000000 (0.000000)     0.000837    0.003093    0.0013573   3.76562e-06 
  momentum139                                               10          0.297491    0.262899 (0.883721)     0.034592 (0.116279)     0.026766    0.047471    0.0297491   8.25342e-05 
    momentum139/compute                                     10          0.165417    0.130825 (0.790880)     0.034592 (0.209120)     0.015258    0.019128    0.0165417   4.58924e-05 
    momentum139/infer_shape                                 10          0.032564    0.032564 (1.000000)     0.000000 (0.000000)     0.0031      0.003443    0.0032564   9.03437e-06 
    momentum139/prepare_data                                10          0.013728    0.013728 (1.000000)     0.000000 (0.000000)     0.001076    0.001698    0.0013728   3.80862e-06 
  momentum44                                                10          0.296483    0.264131 (0.890881)     0.032352 (0.109119)     0.02721     0.032635    0.0296483   8.22546e-05 
    momentum44/compute                                      10          0.1601      0.127748 (0.797926)     0.032352 (0.202074)     0.015233    0.018122    0.01601     4.44172e-05 
    momentum44/infer_shape                                  10          0.030811    0.030811 (1.000000)     0.000000 (0.000000)     0.002856    0.003378    0.0030811   8.54803e-06 
    momentum44/prepare_data                                 10          0.018834    0.018834 (1.000000)     0.000000 (0.000000)     0.001246    0.005588    0.0018834   5.2252e-06  
  momentum89                                                10          0.295978    0.251786 (0.850692)     0.044192 (0.149308)     0.027763    0.035313    0.0295978   8.21145e-05 
    momentum89/compute                                      10          0.174057    0.129865 (0.746106)     0.044192 (0.253894)     0.016334    0.021745    0.0174057   4.82894e-05 
    momentum89/infer_shape                                  10          0.034737    0.034737 (1.000000)     0.000000 (0.000000)     0.002917    0.004491    0.0034737   9.63724e-06 
    momentum89/prepare_data                                 10          0.015962    0.015962 (1.000000)     0.000000 (0.000000)     0.001262    0.003452    0.0015962   4.42841e-06 
  momentum4                                                 10          0.295859    0.264627 (0.894436)     0.031232 (0.105564)     0.027377    0.034753    0.0295859   8.20815e-05 
    momentum4/compute                                       10          0.165458    0.134226 (0.811239)     0.031232 (0.188761)     0.014818    0.021197    0.0165458   4.59037e-05 
    momentum4/infer_shape                                   10          0.028923    0.028923 (1.000000)     0.000000 (0.000000)     0.002728    0.00315     0.0028923   8.02423e-06 
    momentum4/prepare_data                                  10          0.013726    0.013726 (1.000000)     0.000000 (0.000000)     0.001229    0.001512    0.0013726   3.80806e-06 
  momentum21                                                10          0.295628    0.257292 (0.870324)     0.038336 (0.129676)     0.027658    0.032066    0.0295628   8.20174e-05 
    momentum21/compute                                      10          0.165027    0.126691 (0.767699)     0.038336 (0.232301)     0.015277    0.01751     0.0165027   4.57842e-05 
    momentum21/infer_shape                                  10          0.030201    0.030201 (1.000000)     0.000000 (0.000000)     0.002727    0.003943    0.0030201   8.3788e-06  
    momentum21/prepare_data                                 10          0.014848    0.014848 (1.000000)     0.000000 (0.000000)     0.000958    0.001634    0.0014848   4.11935e-06 
  momentum73                                                10          0.29557     0.259986 (0.879609)     0.035584 (0.120391)     0.028573    0.0311      0.029557    8.20013e-05 
    momentum73/compute                                      10          0.166172    0.130588 (0.785860)     0.035584 (0.214140)     0.015722    0.017499    0.0166172   4.61018e-05 
    momentum73/infer_shape                                  10          0.03403     0.034030 (1.000000)     0.000000 (0.000000)     0.003078    0.003672    0.003403    9.44109e-06 
    momentum73/prepare_data                                 10          0.014578    0.014578 (1.000000)     0.000000 (0.000000)     0.001087    0.001692    0.0014578   4.04444e-06 
  momentum123                                               10          0.295507    0.247603 (0.837892)     0.047904 (0.162108)     0.027484    0.032098    0.0295507   8.19838e-05 
    momentum123/compute                                     10          0.174167    0.126263 (0.724954)     0.047904 (0.275046)     0.0162      0.019335    0.0174167   4.83199e-05 
    momentum123/infer_shape                                 10          0.032236    0.032236 (1.000000)     0.000000 (0.000000)     0.002788    0.005202    0.0032236   8.94337e-06 
    momentum123/prepare_data                                10          0.011847    0.011847 (1.000000)     0.000000 (0.000000)     0.00103     0.001355    0.0011847   3.28676e-06 
  momentum75                                                10          0.295422    0.262686 (0.889189)     0.032736 (0.110811)     0.028249    0.031681    0.0295422   8.19602e-05 
    momentum75/compute                                      10          0.163243    0.130507 (0.799465)     0.032736 (0.200535)     0.015776    0.017372    0.0163243   4.52892e-05 
    momentum75/infer_shape                                  10          0.03464     0.034640 (1.000000)     0.000000 (0.000000)     0.003054    0.003741    0.003464    9.61033e-06 
    momentum75/prepare_data                                 10          0.01329     0.013290 (1.000000)     0.000000 (0.000000)     0.001024    0.001756    0.001329    3.6871e-06  
  momentum66                                                10          0.295354    0.257338 (0.871287)     0.038016 (0.128713)     0.026774    0.033118    0.0295354   8.19413e-05 
    momentum66/compute                                      10          0.177323    0.139307 (0.785612)     0.038016 (0.214388)     0.015489    0.021337    0.0177323   4.91955e-05 
    momentum66/infer_shape                                  10          0.031555    0.031555 (1.000000)     0.000000 (0.000000)     0.002773    0.003499    0.0031555   8.75444e-06 
    momentum66/prepare_data                                 10          0.01017     0.010170 (1.000000)     0.000000 (0.000000)     0.00081     0.001579    0.001017    2.82151e-06 
  momentum57                                                10          0.295288    0.260184 (0.881119)     0.035104 (0.118881)     0.027444    0.032932    0.0295288   8.1923e-05  
    momentum57/compute                                      10          0.162145    0.127041 (0.783502)     0.035104 (0.216498)     0.015297    0.018272    0.0162145   4.49846e-05 
    momentum57/infer_shape                                  10          0.032688    0.032688 (1.000000)     0.000000 (0.000000)     0.002612    0.005065    0.0032688   9.06877e-06 
    momentum57/prepare_data                                 10          0.010742    0.010742 (1.000000)     0.000000 (0.000000)     0.000994    0.001192    0.0010742   2.9802e-06  
  momentum109                                               10          0.295122    0.262482 (0.889402)     0.032640 (0.110598)     0.027404    0.037123    0.0295122   8.1877e-05  
    momentum109/compute                                     10          0.1705      0.137860 (0.808563)     0.032640 (0.191437)     0.015383    0.024682    0.01705     4.73026e-05 
    momentum109/infer_shape                                 10          0.031229    0.031229 (1.000000)     0.000000 (0.000000)     0.002786    0.003334    0.0031229   8.664e-06   
    momentum109/prepare_data                                10          0.017833    0.017833 (1.000000)     0.000000 (0.000000)     0.00113     0.003563    0.0017833   4.94749e-06 
  momentum70                                                10          0.294658    0.262146 (0.889662)     0.032512 (0.110338)     0.027934    0.032113    0.0294658   8.17483e-05 
    momentum70/compute                                      10          0.165962    0.133450 (0.804100)     0.032512 (0.195900)     0.015453    0.017548    0.0165962   4.60436e-05 
    momentum70/infer_shape                                  10          0.032412    0.032412 (1.000000)     0.000000 (0.000000)     0.002808    0.003991    0.0032412   8.9922e-06  
    momentum70/prepare_data                                 10          0.013879    0.013879 (1.000000)     0.000000 (0.000000)     0.000934    0.00304     0.0013879   3.85051e-06 
  momentum9                                                 10          0.294611    0.256563 (0.870853)     0.038048 (0.129147)     0.027203    0.031102    0.0294611   8.17352e-05 
    momentum9/compute                                       10          0.171527    0.133479 (0.778181)     0.038048 (0.221819)     0.015671    0.019022    0.0171527   4.75875e-05 
    momentum9/infer_shape                                   10          0.028846    0.028846 (1.000000)     0.000000 (0.000000)     0.002573    0.003151    0.0028846   8.00287e-06 
    momentum9/prepare_data                                  10          0.014633    0.014633 (1.000000)     0.000000 (0.000000)     0.001241    0.001871    0.0014633   4.0597e-06  
  momentum128                                               10          0.294125    0.253229 (0.860957)     0.040896 (0.139043)     0.028603    0.030252    0.0294125   8.16004e-05 
    momentum128/compute                                     10          0.171634    0.130738 (0.761726)     0.040896 (0.238274)     0.016827    0.018014    0.0171634   4.76172e-05 
    momentum128/infer_shape                                 10          0.033657    0.033657 (1.000000)     0.000000 (0.000000)     0.003134    0.003709    0.0033657   9.33761e-06 
    momentum128/prepare_data                                10          0.011305    0.011305 (1.000000)     0.000000 (0.000000)     0.000985    0.0013      0.0011305   3.1364e-06  
  momentum18                                                10          0.293703    0.256519 (0.873396)     0.037184 (0.126604)     0.026027    0.044911    0.0293703   8.14833e-05 
    momentum18/compute                                      10          0.161242    0.124058 (0.769390)     0.037184 (0.230610)     0.015089    0.018247    0.0161242   4.47341e-05 
    momentum18/infer_shape                                  10          0.03286     0.032860 (1.000000)     0.000000 (0.000000)     0.002671    0.005764    0.003286    9.11649e-06 
    momentum18/prepare_data                                 10          0.014261    0.014261 (1.000000)     0.000000 (0.000000)     0.00093     0.002131    0.0014261   3.95649e-06 
  momentum95                                                10          0.292967    0.262887 (0.897326)     0.030080 (0.102674)     0.027897    0.031485    0.0292967   8.12791e-05 
    momentum95/compute                                      10          0.160211    0.130131 (0.812248)     0.030080 (0.187752)     0.015465    0.016843    0.0160211   4.4448e-05  
    momentum95/infer_shape                                  10          0.03784     0.037840 (1.000000)     0.000000 (0.000000)     0.003408    0.004186    0.003784    1.04981e-05 
    momentum95/prepare_data                                 10          0.011011    0.011011 (1.000000)     0.000000 (0.000000)     0.000928    0.001199    0.0011011   3.05483e-06 
  momentum115                                               10          0.292586    0.256458 (0.876522)     0.036128 (0.123478)     0.026767    0.034379    0.0292586   8.11734e-05 
    momentum115/compute                                     10          0.172189    0.136061 (0.790184)     0.036128 (0.209816)     0.015652    0.023218    0.0172189   4.77711e-05 
    momentum115/infer_shape                                 10          0.033578    0.033578 (1.000000)     0.000000 (0.000000)     0.002929    0.003933    0.0033578   9.31569e-06 
    momentum115/prepare_data                                10          0.012826    0.012826 (1.000000)     0.000000 (0.000000)     0.001023    0.001563    0.0012826   3.55837e-06 
  momentum140                                               10          0.292423    0.263143 (0.899871)     0.029280 (0.100129)     0.027487    0.03562     0.0292423   8.11282e-05 
    momentum140/compute                                     10          0.16321     0.133930 (0.820599)     0.029280 (0.179401)     0.015338    0.017421    0.016321    4.52801e-05 
    momentum140/infer_shape                                 10          0.028869    0.028869 (1.000000)     0.000000 (0.000000)     0.002614    0.003041    0.0028869   8.00925e-06 
    momentum140/prepare_data                                10          0.009071    0.009071 (1.000000)     0.000000 (0.000000)     0.000811    0.000976    0.0009071   2.51661e-06 
  momentum13                                                10          0.291703    0.259927 (0.891067)     0.031776 (0.108933)     0.027815    0.031805    0.0291703   8.09284e-05 
    momentum13/compute                                      10          0.161921    0.130145 (0.803756)     0.031776 (0.196244)     0.015601    0.016579    0.0161921   4.49224e-05 
    momentum13/infer_shape                                  10          0.02989     0.029890 (1.000000)     0.000000 (0.000000)     0.002766    0.003302    0.002989    8.29251e-06 
    momentum13/prepare_data                                 10          0.016385    0.016385 (1.000000)     0.000000 (0.000000)     0.001356    0.002117    0.0016385   4.54576e-06 
  momentum43                                                10          0.291644    0.260444 (0.893020)     0.031200 (0.106980)     0.028404    0.031786    0.0291644   8.09121e-05 
    momentum43/compute                                      10          0.161744    0.130544 (0.807103)     0.031200 (0.192897)     0.015545    0.017538    0.0161744   4.48733e-05 
    momentum43/infer_shape                                  10          0.032176    0.032176 (1.000000)     0.000000 (0.000000)     0.002973    0.003472    0.0032176   8.92673e-06 
    momentum43/prepare_data                                 10          0.015806    0.015806 (1.000000)     0.000000 (0.000000)     0.001379    0.001982    0.0015806   4.38513e-06 
  momentum130                                               10          0.291433    0.255241 (0.875814)     0.036192 (0.124186)     0.026824    0.036747    0.0291433   8.08535e-05 
    momentum130/compute                                     10          0.169972    0.133780 (0.787071)     0.036192 (0.212929)     0.015125    0.023969    0.0169972   4.71561e-05 
    momentum130/infer_shape                                 10          0.033875    0.033875 (1.000000)     0.000000 (0.000000)     0.003016    0.004061    0.0033875   9.39809e-06 
    momentum130/prepare_data                                10          0.014604    0.014604 (1.000000)     0.000000 (0.000000)     0.001263    0.001862    0.0014604   4.05165e-06 
  momentum77                                                10          0.291379    0.253075 (0.868542)     0.038304 (0.131458)     0.027368    0.030981    0.0291379   8.08385e-05 
    momentum77/compute                                      10          0.166228    0.127924 (0.769570)     0.038304 (0.230430)     0.01546     0.017742    0.0166228   4.61174e-05 
    momentum77/infer_shape                                  10          0.033922    0.033922 (1.000000)     0.000000 (0.000000)     0.002953    0.005229    0.0033922   9.41113e-06 
    momentum77/prepare_data                                 10          0.015032    0.015032 (1.000000)     0.000000 (0.000000)     0.00139     0.001788    0.0015032   4.17039e-06 
  momentum36                                                10          0.291315    0.259955 (0.892350)     0.031360 (0.107650)     0.027194    0.040445    0.0291315   8.08208e-05 
    momentum36/compute                                      10          0.156641    0.125281 (0.799797)     0.031360 (0.200203)     0.015348    0.016222    0.0156641   4.34576e-05 
    momentum36/infer_shape                                  10          0.031585    0.031585 (1.000000)     0.000000 (0.000000)     0.002627    0.003592    0.0031585   8.76276e-06 
    momentum36/prepare_data                                 10          0.01261     0.012610 (1.000000)     0.000000 (0.000000)     0.000981    0.001589    0.001261    3.49845e-06 
  momentum59                                                10          0.29108     0.257768 (0.885557)     0.033312 (0.114443)     0.027828    0.030958    0.029108    8.07556e-05 
    momentum59/compute                                      10          0.159842    0.126530 (0.791594)     0.033312 (0.208406)     0.015193    0.016996    0.0159842   4.43457e-05 
    momentum59/infer_shape                                  10          0.03152     0.031520 (1.000000)     0.000000 (0.000000)     0.002746    0.005275    0.003152    8.74473e-06 
    momentum59/prepare_data                                 10          0.010644    0.010644 (1.000000)     0.000000 (0.000000)     0.000956    0.001135    0.0010644   2.95301e-06 
  momentum56                                                10          0.290287    0.254223 (0.875764)     0.036064 (0.124236)     0.027567    0.032564    0.0290287   8.05356e-05 
    momentum56/compute                                      10          0.164083    0.128019 (0.780209)     0.036064 (0.219791)     0.01565     0.017239    0.0164083   4.55223e-05 
    momentum56/infer_shape                                  10          0.029308    0.029308 (1.000000)     0.000000 (0.000000)     0.002655    0.003427    0.0029308   8.13105e-06 
    momentum56/prepare_data                                 10          0.009297    0.009297 (1.000000)     0.000000 (0.000000)     0.000789    0.001101    0.0009297   2.57931e-06 
  momentum38                                                10          0.290161    0.260209 (0.896775)     0.029952 (0.103225)     0.02819     0.031138    0.0290161   8.05006e-05 
    momentum38/compute                                      10          0.160395    0.130443 (0.813261)     0.029952 (0.186739)     0.015307    0.01725     0.0160395   4.44991e-05 
    momentum38/infer_shape                                  10          0.032688    0.032688 (1.000000)     0.000000 (0.000000)     0.002855    0.004908    0.0032688   9.06877e-06 
    momentum38/prepare_data                                 10          0.011701    0.011701 (1.000000)     0.000000 (0.000000)     0.000938    0.00166     0.0011701   3.24626e-06 
  momentum35                                                10          0.289992    0.259304 (0.894176)     0.030688 (0.105824)     0.027774    0.030526    0.0289992   8.04537e-05 
    momentum35/compute                                      10          0.159518    0.128830 (0.807620)     0.030688 (0.192380)     0.015353    0.017345    0.0159518   4.42558e-05 
    momentum35/infer_shape                                  10          0.032578    0.032578 (1.000000)     0.000000 (0.000000)     0.002549    0.004723    0.0032578   9.03826e-06 
    momentum35/prepare_data                                 10          0.012656    0.012656 (1.000000)     0.000000 (0.000000)     0.000945    0.001879    0.0012656   3.51121e-06 
  momentum127                                               10          0.289652    0.253108 (0.873835)     0.036544 (0.126165)     0.026335    0.036868    0.0289652   8.03594e-05 
    momentum127/compute                                     10          0.169893    0.133349 (0.784900)     0.036544 (0.215100)     0.015309    0.021283    0.0169893   4.71342e-05 
    momentum127/infer_shape                                 10          0.033319    0.033319 (1.000000)     0.000000 (0.000000)     0.003078    0.00373     0.0033319   9.24384e-06 
    momentum127/prepare_data                                10          0.014283    0.014283 (1.000000)     0.000000 (0.000000)     0.00127     0.001765    0.0014283   3.96259e-06 
  momentum94                                                10          0.28917     0.257490 (0.890445)     0.031680 (0.109555)     0.027828    0.031777    0.028917    8.02257e-05 
    momentum94/compute                                      10          0.161857    0.130177 (0.804272)     0.031680 (0.195728)     0.015536    0.017008    0.0161857   4.49047e-05 
    momentum94/infer_shape                                  10          0.033466    0.033466 (1.000000)     0.000000 (0.000000)     0.003111    0.003713    0.0033466   9.28462e-06 
    momentum94/prepare_data                                 10          0.01158     0.011580 (1.000000)     0.000000 (0.000000)     0.000989    0.001313    0.001158    3.21269e-06 
  momentum33                                                10          0.289102    0.257230 (0.889755)     0.031872 (0.110245)     0.027703    0.03186     0.0289102   8.02068e-05 
    momentum33/compute                                      10          0.159756    0.127884 (0.800496)     0.031872 (0.199504)     0.015151    0.016964    0.0159756   4.43218e-05 
    momentum33/infer_shape                                  10          0.028708    0.028708 (1.000000)     0.000000 (0.000000)     0.002579    0.003311    0.0028708   7.96459e-06 
    momentum33/prepare_data                                 10          0.014932    0.014932 (1.000000)     0.000000 (0.000000)     0.001335    0.001717    0.0014932   4.14265e-06 
  momentum129                                               10          0.28873     0.237050 (0.821009)     0.051680 (0.178991)     0.027798    0.030726    0.028873    8.01036e-05 
    momentum129/compute                                     10          0.177118    0.125438 (0.708217)     0.051680 (0.291783)     0.017312    0.018414    0.0177118   4.91386e-05 
    momentum129/infer_shape                                 10          0.031098    0.031098 (1.000000)     0.000000 (0.000000)     0.002973    0.003335    0.0031098   8.62765e-06 
    momentum129/prepare_data                                10          0.011383    0.011383 (1.000000)     0.000000 (0.000000)     0.000783    0.002996    0.0011383   3.15804e-06 
  momentum141                                               10          0.288435    0.257715 (0.893494)     0.030720 (0.106506)     0.027439    0.031231    0.0288435   8.00218e-05 
    momentum141/compute                                     10          0.158531    0.127811 (0.806221)     0.030720 (0.193779)     0.015361    0.016802    0.0158531   4.39819e-05 
    momentum141/infer_shape                                 10          0.034125    0.034125 (1.000000)     0.000000 (0.000000)     0.002922    0.005733    0.0034125   9.46745e-06 
    momentum141/prepare_data                                10          0.013445    0.013445 (1.000000)     0.000000 (0.000000)     0.001014    0.001746    0.0013445   3.7301e-06  
  momentum106                                               10          0.288294    0.253702 (0.880011)     0.034592 (0.119989)     0.027191    0.032151    0.0288294   7.99827e-05 
    momentum106/compute                                     10          0.162274    0.127682 (0.786830)     0.034592 (0.213170)     0.015129    0.018092    0.0162274   4.50204e-05 
    momentum106/infer_shape                                 10          0.03437     0.034370 (1.000000)     0.000000 (0.000000)     0.003113    0.003788    0.003437    9.53542e-06 
    momentum106/prepare_data                                10          0.013979    0.013979 (1.000000)     0.000000 (0.000000)     0.001057    0.00329     0.0013979   3.87826e-06 
  momentum101                                               10          0.288101    0.255909 (0.888261)     0.032192 (0.111739)     0.027038    0.037313    0.0288101   7.99291e-05 
    momentum101/compute                                     10          0.159602    0.127410 (0.798298)     0.032192 (0.201702)     0.015308    0.016379    0.0159602   4.42791e-05 
    momentum101/infer_shape                                 10          0.042712    0.042712 (1.000000)     0.000000 (0.000000)     0.003007    0.011373    0.0042712   1.18498e-05 
    momentum101/prepare_data                                10          0.013129    0.013129 (1.000000)     0.000000 (0.000000)     0.00117     0.001464    0.0013129   3.64244e-06 
  momentum114                                               10          0.287638    0.252118 (0.876511)     0.035520 (0.123489)     0.026589    0.030825    0.0287638   7.98007e-05 
    momentum114/compute                                     10          0.165072    0.129552 (0.784821)     0.035520 (0.215179)     0.015125    0.017537    0.0165072   4.57966e-05 
    momentum114/infer_shape                                 10          0.031025    0.031025 (1.000000)     0.000000 (0.000000)     0.002874    0.003505    0.0031025   8.6074e-06  
    momentum114/prepare_data                                10          0.013807    0.013807 (1.000000)     0.000000 (0.000000)     0.001013    0.003469    0.0013807   3.83054e-06 
  momentum137                                               10          0.287505    0.252881 (0.879571)     0.034624 (0.120429)     0.027641    0.031241    0.0287505   7.97638e-05 
    momentum137/compute                                     10          0.164232    0.129608 (0.789176)     0.034624 (0.210824)     0.015472    0.017034    0.0164232   4.55636e-05 
    momentum137/infer_shape                                 10          0.030522    0.030522 (1.000000)     0.000000 (0.000000)     0.002546    0.003382    0.0030522   8.46785e-06 
    momentum137/prepare_data                                10          0.014409    0.014409 (1.000000)     0.000000 (0.000000)     0.00105     0.004123    0.0014409   3.99755e-06 
  momentum64                                                10          0.287361    0.252641 (0.879176)     0.034720 (0.120824)     0.027411    0.030708    0.0287361   7.97238e-05 
    momentum64/compute                                      10          0.167705    0.132985 (0.792970)     0.034720 (0.207030)     0.015666    0.019185    0.0167705   4.65271e-05 
    momentum64/infer_shape                                  10          0.029011    0.029011 (1.000000)     0.000000 (0.000000)     0.002674    0.00316     0.0029011   8.04865e-06 
    momentum64/prepare_data                                 10          0.010752    0.010752 (1.000000)     0.000000 (0.000000)     0.00095     0.001345    0.0010752   2.98297e-06 
  momentum152                                               10          0.287346    0.249714 (0.869036)     0.037632 (0.130964)     0.027423    0.030901    0.0287346   7.97197e-05 
    momentum152/compute                                     10          0.165607    0.127975 (0.772763)     0.037632 (0.227237)     0.015186    0.018511    0.0165607   4.59451e-05 
    momentum152/infer_shape                                 10          0.031238    0.031238 (1.000000)     0.000000 (0.000000)     0.00277     0.003563    0.0031238   8.66649e-06 
    momentum152/prepare_data                                10          0.013312    0.013312 (1.000000)     0.000000 (0.000000)     0.001045    0.003331    0.0013312   3.69321e-06 
  momentum148                                               10          0.287254    0.253078 (0.881025)     0.034176 (0.118975)     0.025973    0.035287    0.0287254   7.96941e-05 
    momentum148/compute                                     10          0.170295    0.136119 (0.799313)     0.034176 (0.200687)     0.01535     0.023048    0.0170295   4.72457e-05 
    momentum148/infer_shape                                 10          0.035789    0.035789 (1.000000)     0.000000 (0.000000)     0.002919    0.006246    0.0035789   9.9291e-06  
    momentum148/prepare_data                                10          0.014141    0.014141 (1.000000)     0.000000 (0.000000)     0.001029    0.002363    0.0014141   3.9232e-06  
  momentum104                                               10          0.287063    0.256215 (0.892539)     0.030848 (0.107461)     0.027394    0.030507    0.0287063   7.96411e-05 
    momentum104/compute                                     10          0.15968     0.128832 (0.806814)     0.030848 (0.193186)     0.015417    0.016804    0.015968    4.43007e-05 
    momentum104/infer_shape                                 10          0.038536    0.038536 (1.000000)     0.000000 (0.000000)     0.003083    0.005792    0.0038536   1.06912e-05 
    momentum104/prepare_data                                10          0.014335    0.014335 (1.000000)     0.000000 (0.000000)     0.001293    0.001577    0.0014335   3.97702e-06 
  momentum62                                                10          0.286558    0.255710 (0.892350)     0.030848 (0.107650)     0.027421    0.030926    0.0286558   7.9501e-05  
    momentum62/compute                                      10          0.157883    0.127035 (0.804615)     0.030848 (0.195385)     0.015245    0.016427    0.0157883   4.38022e-05 
    momentum62/infer_shape                                  10          0.033681    0.033681 (1.000000)     0.000000 (0.000000)     0.002932    0.005755    0.0033681   9.34427e-06 
    momentum62/prepare_data                                 10          0.010442    0.010442 (1.000000)     0.000000 (0.000000)     0.000949    0.001182    0.0010442   2.89697e-06 
  momentum47                                                10          0.286429    0.253885 (0.886380)     0.032544 (0.113620)     0.026601    0.030903    0.0286429   7.94652e-05 
    momentum47/compute                                      10          0.15926     0.126716 (0.795655)     0.032544 (0.204345)     0.014983    0.018474    0.015926    4.41842e-05 
    momentum47/infer_shape                                  10          0.028651    0.028651 (1.000000)     0.000000 (0.000000)     0.002683    0.003253    0.0028651   7.94877e-06 
    momentum47/prepare_data                                 10          0.019962    0.019962 (1.000000)     0.000000 (0.000000)     0.001279    0.003685    0.0019962   5.53814e-06 
  momentum45                                                10          0.286108    0.255932 (0.894529)     0.030176 (0.105471)     0.026187    0.035012    0.0286108   7.93762e-05 
    momentum45/compute                                      10          0.155869    0.125693 (0.806402)     0.030176 (0.193598)     0.014534    0.016505    0.0155869   4.32434e-05 
    momentum45/infer_shape                                  10          0.03139     0.031390 (1.000000)     0.000000 (0.000000)     0.002851    0.003515    0.003139    8.70866e-06 
    momentum45/prepare_data                                 10          0.010827    0.010827 (1.000000)     0.000000 (0.000000)     0.000711    0.003161    0.0010827   3.00378e-06 
  momentum92                                                10          0.285876    0.252660 (0.883810)     0.033216 (0.116190)     0.027309    0.031149    0.0285876   7.93118e-05 
    momentum92/compute                                      10          0.161886    0.128670 (0.794819)     0.033216 (0.205181)     0.015363    0.017316    0.0161886   4.49127e-05 
    momentum92/infer_shape                                  10          0.03365     0.033650 (1.000000)     0.000000 (0.000000)     0.003117    0.003712    0.003365    9.33567e-06 
    momentum92/prepare_data                                 10          0.014266    0.014266 (1.000000)     0.000000 (0.000000)     0.001189    0.003114    0.0014266   3.95788e-06 
  momentum133                                               10          0.285843    0.248819 (0.870474)     0.037024 (0.129526)     0.026322    0.031084    0.0285843   7.93027e-05 
    momentum133/compute                                     10          0.166031    0.129007 (0.777005)     0.037024 (0.222995)     0.015421    0.01924     0.0166031   4.60627e-05 
    momentum133/infer_shape                                 10          0.033472    0.033472 (1.000000)     0.000000 (0.000000)     0.003139    0.004077    0.0033472   9.28628e-06 
    momentum133/prepare_data                                10          0.015169    0.015169 (1.000000)     0.000000 (0.000000)     0.001185    0.003463    0.0015169   4.2084e-06  
  momentum26                                                10          0.285815    0.249399 (0.872589)     0.036416 (0.127411)     0.027077    0.030244    0.0285815   7.92949e-05 
    momentum26/compute                                      10          0.165238    0.128822 (0.779615)     0.036416 (0.220385)     0.015224    0.018275    0.0165238   4.58427e-05 
    momentum26/infer_shape                                  10          0.029162    0.029162 (1.000000)     0.000000 (0.000000)     0.002765    0.003155    0.0029162   8.09054e-06 
    momentum26/prepare_data                                 10          0.01862     0.018620 (1.000000)     0.000000 (0.000000)     0.001149    0.003599    0.001862    5.16583e-06 
  momentum159                                               10          0.285683    0.247059 (0.864801)     0.038624 (0.135199)     0.026577    0.031626    0.0285683   7.92583e-05 
    momentum159/compute                                     10          0.169727    0.131103 (0.772435)     0.038624 (0.227565)     0.01566     0.020228    0.0169727   4.70881e-05 
    momentum159/infer_shape                                 10          0.031638    0.031638 (1.000000)     0.000000 (0.000000)     0.002894    0.003598    0.0031638   8.77747e-06 
    momentum159/prepare_data                                10          0.009686    0.009686 (1.000000)     0.000000 (0.000000)     0.000869    0.001311    0.0009686   2.68723e-06 
  momentum157                                               10          0.285131    0.249451 (0.874865)     0.035680 (0.125135)     0.026646    0.030236    0.0285131   7.91051e-05 
    momentum157/compute                                     10          0.162222    0.126542 (0.780054)     0.035680 (0.219946)     0.015146    0.017763    0.0162222   4.5006e-05  
    momentum157/infer_shape                                 10          0.032808    0.032808 (1.000000)     0.000000 (0.000000)     0.003139    0.003417    0.0032808   9.10207e-06 
    momentum157/prepare_data                                10          0.012423    0.012423 (1.000000)     0.000000 (0.000000)     0.001073    0.00138     0.0012423   3.44657e-06 
  momentum96                                                10          0.284884    0.253428 (0.889583)     0.031456 (0.110417)     0.026633    0.030359    0.0284884   7.90366e-05 
    momentum96/compute                                      10          0.159585    0.128129 (0.802889)     0.031456 (0.197111)     0.014982    0.017259    0.0159585   4.42744e-05 
    momentum96/infer_shape                                  10          0.035413    0.035413 (1.000000)     0.000000 (0.000000)     0.003189    0.004197    0.0035413   9.82478e-06 
    momentum96/prepare_data                                 10          0.012629    0.012629 (1.000000)     0.000000 (0.000000)     0.001048    0.001798    0.0012629   3.50372e-06 
  momentum120                                               10          0.284778    0.248234 (0.871675)     0.036544 (0.128325)     0.027626    0.029798    0.0284778   7.90072e-05 
    momentum120/compute                                     10          0.164459    0.127915 (0.777793)     0.036544 (0.222207)     0.015532    0.017846    0.0164459   4.56266e-05 
    momentum120/infer_shape                                 10          0.030942    0.030942 (1.000000)     0.000000 (0.000000)     0.002724    0.003335    0.0030942   8.58437e-06 
    momentum120/prepare_data                                10          0.015012    0.015012 (1.000000)     0.000000 (0.000000)     0.001256    0.001678    0.0015012   4.16484e-06 
  momentum153                                               10          0.284729    0.247065 (0.867720)     0.037664 (0.132280)     0.027027    0.030825    0.0284729   7.89936e-05 
    momentum153/compute                                     10          0.167045    0.129381 (0.774528)     0.037664 (0.225472)     0.015525    0.018424    0.0167045   4.6344e-05  
    momentum153/infer_shape                                 10          0.030443    0.030443 (1.000000)     0.000000 (0.000000)     0.002852    0.003587    0.0030443   8.44593e-06 
    momentum153/prepare_data                                10          0.015084    0.015084 (1.000000)     0.000000 (0.000000)     0.001214    0.003063    0.0015084   4.18482e-06 
  momentum85                                                10          0.284527    0.251023 (0.882247)     0.033504 (0.117753)     0.027323    0.03011     0.0284527   7.89376e-05 
    momentum85/compute                                      10          0.161746    0.128242 (0.792860)     0.033504 (0.207140)     0.015448    0.017948    0.0161746   4.48739e-05 
    momentum85/infer_shape                                  10          0.032179    0.032179 (1.000000)     0.000000 (0.000000)     0.003004    0.003575    0.0032179   8.92756e-06 
    momentum85/prepare_data                                 10          0.014665    0.014665 (1.000000)     0.000000 (0.000000)     0.001149    0.003275    0.0014665   4.06857e-06 
  momentum30                                                10          0.284358    0.249830 (0.878576)     0.034528 (0.121424)     0.027065    0.030224    0.0284358   7.88907e-05 
    momentum30/compute                                      10          0.162243    0.127715 (0.787183)     0.034528 (0.212817)     0.015346    0.017412    0.0162243   4.50118e-05 
    momentum30/infer_shape                                  10          0.030058    0.030058 (1.000000)     0.000000 (0.000000)     0.002684    0.003708    0.0030058   8.33912e-06 
    momentum30/prepare_data                                 10          0.017391    0.017391 (1.000000)     0.000000 (0.000000)     0.000997    0.003959    0.0017391   4.82486e-06 
  momentum60                                                10          0.283758    0.250766 (0.883732)     0.032992 (0.116268)     0.026869    0.031235    0.0283758   7.87242e-05 
    momentum60/compute                                      10          0.159809    0.126817 (0.793554)     0.032992 (0.206446)     0.015302    0.017059    0.0159809   4.43365e-05 
    momentum60/infer_shape                                  10          0.028778    0.028778 (1.000000)     0.000000 (0.000000)     0.002482    0.004666    0.0028778   7.98401e-06 
    momentum60/prepare_data                                 10          0.011013    0.011013 (1.000000)     0.000000 (0.000000)     0.000952    0.001296    0.0011013   3.05538e-06 
  momentum151                                               10          0.283635    0.248787 (0.877138)     0.034848 (0.122862)     0.026026    0.032002    0.0283635   7.86901e-05 
    momentum151/compute                                     10          0.162126    0.127278 (0.785056)     0.034848 (0.214944)     0.015238    0.017294    0.0162126   4.49793e-05 
    momentum151/infer_shape                                 10          0.02973     0.029730 (1.000000)     0.000000 (0.000000)     0.002786    0.003268    0.002973    8.24812e-06 
    momentum151/prepare_data                                10          0.015163    0.015163 (1.000000)     0.000000 (0.000000)     0.001123    0.003575    0.0015163   4.20674e-06 
  momentum116                                               10          0.282711    0.251767 (0.890545)     0.030944 (0.109455)     0.025499    0.030159    0.0282711   7.84337e-05 
    momentum116/compute                                     10          0.160234    0.129290 (0.806882)     0.030944 (0.193118)     0.014888    0.016853    0.0160234   4.44544e-05 
    momentum116/infer_shape                                 10          0.034273    0.034273 (1.000000)     0.000000 (0.000000)     0.002948    0.003822    0.0034273   9.50851e-06 
    momentum116/prepare_data                                10          0.00986     0.009860 (1.000000)     0.000000 (0.000000)     0.000813    0.001253    0.000986    2.7355e-06  
  momentum74                                                10          0.282591    0.250207 (0.885403)     0.032384 (0.114597)     0.026671    0.031834    0.0282591   7.84005e-05 
    momentum74/compute                                      10          0.161596    0.129212 (0.799599)     0.032384 (0.200401)     0.014889    0.017844    0.0161596   4.48323e-05 
    momentum74/infer_shape                                  10          0.032801    0.032801 (1.000000)     0.000000 (0.000000)     0.002928    0.003863    0.0032801   9.10012e-06 
    momentum74/prepare_data                                 10          0.013917    0.013917 (1.000000)     0.000000 (0.000000)     0.001019    0.001903    0.0013917   3.86105e-06 
  momentum110                                               10          0.282494    0.247934 (0.877661)     0.034560 (0.122339)     0.027172    0.029806    0.0282494   7.83735e-05 
    momentum110/compute                                     10          0.163685    0.129125 (0.788863)     0.034560 (0.211137)     0.015592    0.017705    0.0163685   4.54118e-05 
    momentum110/infer_shape                                 10          0.032469    0.032469 (1.000000)     0.000000 (0.000000)     0.002935    0.003748    0.0032469   9.00802e-06 
    momentum110/prepare_data                                10          0.013481    0.013481 (1.000000)     0.000000 (0.000000)     0.000994    0.003286    0.0013481   3.74009e-06 
  momentum58                                                10          0.282308    0.250277 (0.886539)     0.032031 (0.113461)     0.025641    0.037559    0.0282308   7.83219e-05 
    momentum58/compute                                      10          0.16619     0.134159 (0.807263)     0.032031 (0.192737)     0.014545    0.025819    0.016619    4.61068e-05 
    momentum58/infer_shape                                  10          0.033105    0.033105 (1.000000)     0.000000 (0.000000)     0.002821    0.005163    0.0033105   9.18446e-06 
    momentum58/prepare_data                                 10          0.008949    0.008949 (1.000000)     0.000000 (0.000000)     0.000785    0.001095    0.0008949   2.48276e-06 
  momentum6                                                 10          0.282206    0.249758 (0.885020)     0.032448 (0.114980)     0.026216    0.033273    0.0282206   7.82936e-05 
    momentum6/compute                                       10          0.16423     0.131782 (0.802423)     0.032448 (0.197577)     0.014908    0.019288    0.016423    4.5563e-05  
    momentum6/infer_shape                                   10          0.025686    0.025686 (1.000000)     0.000000 (0.000000)     0.00227     0.002937    0.0025686   7.12618e-06 
    momentum6/prepare_data                                  10          0.018825    0.018825 (1.000000)     0.000000 (0.000000)     0.00121     0.00682     0.0018825   5.2227e-06  
  momentum136                                               10          0.2821      0.248372 (0.880440)     0.033728 (0.119560)     0.02639     0.030173    0.02821     7.82642e-05 
    momentum136/compute                                     10          0.160513    0.126785 (0.789874)     0.033728 (0.210126)     0.015279    0.017191    0.0160513   4.45318e-05 
    momentum136/infer_shape                                 10          0.031945    0.031945 (1.000000)     0.000000 (0.000000)     0.002998    0.003417    0.0031945   8.86264e-06 
    momentum136/prepare_data                                10          0.013788    0.013788 (1.000000)     0.000000 (0.000000)     0.001027    0.003678    0.0013788   3.82527e-06 
  momentum135                                               10          0.282086    0.246438 (0.873627)     0.035648 (0.126373)     0.026741    0.030911    0.0282086   7.82603e-05 
    momentum135/compute                                     10          0.165694    0.130046 (0.784856)     0.035648 (0.215144)     0.015407    0.018198    0.0165694   4.59692e-05 
    momentum135/infer_shape                                 10          0.03541     0.035410 (1.000000)     0.000000 (0.000000)     0.003357    0.003734    0.003541    9.82395e-06 
    momentum135/prepare_data                                10          0.011215    0.011215 (1.000000)     0.000000 (0.000000)     0.001006    0.001423    0.0011215   3.11143e-06 
  momentum149                                               10          0.281977    0.251865 (0.893211)     0.030112 (0.106789)     0.026115    0.029906    0.0281977   7.82301e-05 
    momentum149/compute                                     10          0.159808    0.129696 (0.811574)     0.030112 (0.188426)     0.015086    0.017134    0.0159808   4.43362e-05 
    momentum149/infer_shape                                 10          0.028621    0.028621 (1.000000)     0.000000 (0.000000)     0.002624    0.003193    0.0028621   7.94045e-06 
    momentum149/prepare_data                                10          0.011785    0.011785 (1.000000)     0.000000 (0.000000)     0.000831    0.002919    0.0011785   3.26956e-06 
  momentum155                                               10          0.2818      0.247272 (0.877473)     0.034528 (0.122527)     0.026173    0.030426    0.02818     7.8181e-05  
    momentum155/compute                                     10          0.166621    0.132093 (0.792775)     0.034528 (0.207225)     0.015448    0.019163    0.0166621   4.62264e-05 
    momentum155/infer_shape                                 10          0.032331    0.032331 (1.000000)     0.000000 (0.000000)     0.003018    0.003474    0.0032331   8.96973e-06 
    momentum155/prepare_data                                10          0.011587    0.011587 (1.000000)     0.000000 (0.000000)     0.001001    0.001526    0.0011587   3.21463e-06 
  momentum102                                               10          0.281532    0.249820 (0.887359)     0.031712 (0.112641)     0.027092    0.030541    0.0281532   7.81066e-05 
    momentum102/compute                                     10          0.161638    0.129926 (0.803809)     0.031712 (0.196191)     0.01561     0.017103    0.0161638   4.48439e-05 
    momentum102/infer_shape                                 10          0.033401    0.033401 (1.000000)     0.000000 (0.000000)     0.002911    0.005218    0.0033401   9.26659e-06 
    momentum102/prepare_data                                10          0.013648    0.013648 (1.000000)     0.000000 (0.000000)     0.00123     0.001585    0.0013648   3.78642e-06 
  momentum93                                                10          0.280475    0.249339 (0.888988)     0.031136 (0.111012)     0.026919    0.029798    0.0280475   7.78134e-05 
    momentum93/compute                                      10          0.160664    0.129528 (0.806204)     0.031136 (0.193796)     0.015222    0.017882    0.0160664   4.45737e-05 
    momentum93/infer_shape                                  10          0.035203    0.035203 (1.000000)     0.000000 (0.000000)     0.003157    0.003745    0.0035203   9.76652e-06 
    momentum93/prepare_data                                 10          0.014295    0.014295 (1.000000)     0.000000 (0.000000)     0.001213    0.001663    0.0014295   3.96592e-06 
  momentum150                                               10          0.280021    0.248437 (0.887208)     0.031584 (0.112792)     0.026727    0.029722    0.0280021   7.76874e-05 
    momentum150/compute                                     10          0.161795    0.130211 (0.804790)     0.031584 (0.195210)     0.015319    0.018066    0.0161795   4.48875e-05 
    momentum150/infer_shape                                 10          0.031783    0.031783 (1.000000)     0.000000 (0.000000)     0.002943    0.003602    0.0031783   8.8177e-06  
    momentum150/prepare_data                                10          0.011314    0.011314 (1.000000)     0.000000 (0.000000)     0.001005    0.001566    0.0011314   3.13889e-06 
  momentum118                                               10          0.279706    0.243130 (0.869234)     0.036576 (0.130766)     0.025958    0.030728    0.0279706   7.76001e-05 
    momentum118/compute                                     10          0.164599    0.128023 (0.777787)     0.036576 (0.222213)     0.015259    0.017666    0.0164599   4.56654e-05 
    momentum118/infer_shape                                 10          0.038919    0.038919 (1.000000)     0.000000 (0.000000)     0.003381    0.005589    0.0038919   1.07975e-05 
    momentum118/prepare_data                                10          0.011487    0.011487 (1.000000)     0.000000 (0.000000)     0.000998    0.001495    0.0011487   3.18689e-06 
  momentum31                                                10          0.27967     0.247414 (0.884664)     0.032256 (0.115336)     0.026582    0.029999    0.027967    7.75901e-05 
    momentum31/compute                                      10          0.161288    0.129032 (0.800010)     0.032256 (0.199990)     0.015197    0.017812    0.0161288   4.47468e-05 
    momentum31/infer_shape                                  10          0.03014     0.030140 (1.000000)     0.000000 (0.000000)     0.002661    0.003392    0.003014    8.36187e-06 
    momentum31/prepare_data                                 10          0.015407    0.015407 (1.000000)     0.000000 (0.000000)     0.001103    0.001757    0.0015407   4.27443e-06 
  momentum98                                                10          0.279582    0.248222 (0.887833)     0.031360 (0.112167)     0.026713    0.030974    0.0279582   7.75657e-05 
    momentum98/compute                                      10          0.157618    0.126258 (0.801038)     0.031360 (0.198962)     0.015066    0.017005    0.0157618   4.37286e-05 
    momentum98/infer_shape                                  10          0.032292    0.032292 (1.000000)     0.000000 (0.000000)     0.003026    0.003417    0.0032292   8.95891e-06 
    momentum98/prepare_data                                 10          0.012087    0.012087 (1.000000)     0.000000 (0.000000)     0.001049    0.001638    0.0012087   3.35335e-06 
  momentum121                                               10          0.279355    0.244955 (0.876859)     0.034400 (0.123141)     0.025815    0.030213    0.0279355   7.75027e-05 
    momentum121/compute                                     10          0.161463    0.127063 (0.786948)     0.034400 (0.213052)     0.015142    0.016808    0.0161463   4.47954e-05 
    momentum121/infer_shape                                 10          0.034142    0.034142 (1.000000)     0.000000 (0.000000)     0.002753    0.00513     0.0034142   9.47216e-06 
    momentum121/prepare_data                                10          0.011096    0.011096 (1.000000)     0.000000 (0.000000)     0.000975    0.001176    0.0011096   3.07841e-06 
  momentum61                                                10          0.278839    0.244631 (0.877320)     0.034208 (0.122680)     0.02702     0.029684    0.0278839   7.73595e-05 
    momentum61/compute                                      10          0.160543    0.126335 (0.786923)     0.034208 (0.213077)     0.015288    0.016443    0.0160543   4.45401e-05 
    momentum61/infer_shape                                  10          0.026983    0.026983 (1.000000)     0.000000 (0.000000)     0.002453    0.002861    0.0026983   7.48601e-06 
    momentum61/prepare_data                                 10          0.010946    0.010946 (1.000000)     0.000000 (0.000000)     0.000969    0.001479    0.0010946   3.0368e-06  
  momentum34                                                10          0.27713     0.243594 (0.878988)     0.033536 (0.121012)     0.026379    0.029965    0.027713    7.68854e-05 
    momentum34/compute                                      10          0.158178    0.124642 (0.787986)     0.033536 (0.212014)     0.015034    0.016445    0.0158178   4.3884e-05  
    momentum34/infer_shape                                  10          0.029495    0.029495 (1.000000)     0.000000 (0.000000)     0.00282     0.003367    0.0029495   8.18293e-06 
    momentum34/prepare_data                                 10          0.015069    0.015069 (1.000000)     0.000000 (0.000000)     0.001252    0.001982    0.0015069   4.18066e-06 
  momentum143                                               10          0.27663     0.246646 (0.891610)     0.029984 (0.108390)     0.026688    0.029653    0.027663    7.67467e-05 
    momentum143/compute                                     10          0.1599      0.129916 (0.812483)     0.029984 (0.187517)     0.015251    0.016724    0.01599     4.43618e-05 
    momentum143/infer_shape                                 10          0.031099    0.031099 (1.000000)     0.000000 (0.000000)     0.00257     0.004776    0.0031099   8.62793e-06 
    momentum143/prepare_data                                10          0.009454    0.009454 (1.000000)     0.000000 (0.000000)     0.000808    0.001155    0.0009454   2.62286e-06 
  momentum138                                               10          0.275649    0.243809 (0.884491)     0.031840 (0.115509)     0.025572    0.029947    0.0275649   7.64745e-05 
    momentum138/compute                                     10          0.157036    0.125196 (0.797244)     0.031840 (0.202756)     0.015181    0.016714    0.0157036   4.35672e-05 
    momentum138/infer_shape                                 10          0.032587    0.032587 (1.000000)     0.000000 (0.000000)     0.00294     0.00363     0.0032587   9.04075e-06 
    momentum138/prepare_data                                10          0.010623    0.010623 (1.000000)     0.000000 (0.000000)     0.000837    0.001297    0.0010623   2.94719e-06 
  momentum105                                               10          0.275213    0.243757 (0.885703)     0.031456 (0.114297)     0.026797    0.03008     0.0275213   7.63535e-05 
    momentum105/compute                                     10          0.156648    0.125192 (0.799193)     0.031456 (0.200807)     0.015092    0.016308    0.0156648   4.34595e-05 
    momentum105/infer_shape                                 10          0.032549    0.032549 (1.000000)     0.000000 (0.000000)     0.00295     0.003627    0.0032549   9.03021e-06 
    momentum105/prepare_data                                10          0.01156     0.011560 (1.000000)     0.000000 (0.000000)     0.001085    0.001279    0.001156    3.20714e-06 
  momentum147                                               10          0.272178    0.237554 (0.872789)     0.034624 (0.127211)     0.026247    0.02906     0.0272178   7.55115e-05 
    momentum147/compute                                     10          0.163064    0.128440 (0.787666)     0.034624 (0.212334)     0.015767    0.017256    0.0163064   4.52396e-05 
    momentum147/infer_shape                                 10          0.029559    0.029559 (1.000000)     0.000000 (0.000000)     0.002768    0.00327     0.0029559   8.20068e-06 
    momentum147/prepare_data                                10          0.011527    0.011527 (1.000000)     0.000000 (0.000000)     0.001014    0.001434    0.0011527   3.19799e-06 
  momentum144                                               10          0.270113    0.239393 (0.886270)     0.030720 (0.113730)     0.025865    0.029209    0.0270113   7.49386e-05 
    momentum144/compute                                     10          0.15688     0.126160 (0.804182)     0.030720 (0.195818)     0.01506     0.016396    0.015688    4.35239e-05 
    momentum144/infer_shape                                 10          0.033396    0.033396 (1.000000)     0.000000 (0.000000)     0.002756    0.005188    0.0033396   9.2652e-06  
    momentum144/prepare_data                                10          0.010676    0.010676 (1.000000)     0.000000 (0.000000)     0.000927    0.00129     0.0010676   2.96189e-06 
  momentum154                                               10          0.269124    0.238884 (0.887635)     0.030240 (0.112365)     0.025652    0.030483    0.0269124   7.46642e-05 
    momentum154/compute                                     10          0.160353    0.130113 (0.811416)     0.030240 (0.188584)     0.015176    0.017961    0.0160353   4.44874e-05 
    momentum154/infer_shape                                 10          0.029997    0.029997 (1.000000)     0.000000 (0.000000)     0.00258     0.003348    0.0029997   8.3222e-06  
    momentum154/prepare_data                                10          0.014172    0.014172 (1.000000)     0.000000 (0.000000)     0.001027    0.004275    0.0014172   3.9318e-06  
  momentum160                                               10          0.267401    0.237577 (0.888467)     0.029824 (0.111533)     0.02563     0.029045    0.0267401   7.41862e-05 
    momentum160/compute                                     10          0.159398    0.129574 (0.812896)     0.029824 (0.187104)     0.015416    0.016302    0.0159398   4.42225e-05 
    momentum160/infer_shape                                 10          0.028372    0.028372 (1.000000)     0.000000 (0.000000)     0.002533    0.003014    0.0028372   7.87137e-06 
    momentum160/prepare_data                                10          0.01127     0.011270 (1.000000)     0.000000 (0.000000)     0.000873    0.001476    0.001127    3.12669e-06 
  momentum142                                               10          0.263861    0.232949 (0.882847)     0.030912 (0.117153)     0.025279    0.02916     0.0263861   7.32041e-05 
    momentum142/compute                                     10          0.154646    0.123734 (0.800111)     0.030912 (0.199889)     0.015063    0.016309    0.0154646   4.29041e-05 
    momentum142/infer_shape                                 10          0.030386    0.030386 (1.000000)     0.000000 (0.000000)     0.002812    0.003245    0.0030386   8.43012e-06 
    momentum142/prepare_data                                10          0.012086    0.012086 (1.000000)     0.000000 (0.000000)     0.001031    0.001566    0.0012086   3.35307e-06 
DropLocalExeScopes                                          1           22.8683     22.868292 (1.000000)    0.000000 (0.000000)     22.8683     22.8683     22.8683     0.00634445  
pool2d_grad                                                 20          8.72527     1.641435 (0.188124)     7.083835 (0.811876)     0.139231    0.75109     0.436264    0.00242069  
  pool2d_grad1                                              10          7.22537     0.961248 (0.133038)     6.264126 (0.866962)     0.710751    0.749318    0.722537    0.00200457  
    pool2d_grad1/compute                                    10          7.10141     0.837280 (0.117903)     6.264126 (0.882097)     0.698298    0.737182    0.710141    0.00197017  
    pool2d_grad1/infer_shape                                10          0.047446    0.047446 (1.000000)     0.000000 (0.000000)     0.004389    0.00523     0.0047446   1.31632e-05 
    pool2d_grad1/prepare_data                               10          0.007863    0.007863 (1.000000)     0.000000 (0.000000)     0.000695    0.000906    0.0007863   2.18147e-06 
  pool2d_grad0                                              10          1.46289     0.643177 (0.439663)     0.819709 (0.560337)     0.137393    0.165609    0.146289    0.000405855 
    pool2d_grad0/compute                                    10          1.33137     0.511665 (0.384313)     0.819709 (0.615687)     0.127089    0.154036    0.133137    0.000369369 
    pool2d_grad0/infer_shape                                10          0.033232    0.033232 (1.000000)     0.000000 (0.000000)     0.002607    0.005377    0.0033232   9.2197e-06  
    pool2d_grad0/prepare_data                               10          0.028431    0.028431 (1.000000)     0.000000 (0.000000)     0.001172    0.01318     0.0028431   7.88774e-06 
eager_deletion                                              1810        8.01594     8.015944 (1.000000)     0.000000 (0.000000)     0.000719    0.331789    0.0044287   0.0022239   
pool2d                                                      20          5.62698     2.668161 (0.474173)     2.958815 (0.525827)     0.158932    0.492713    0.281349    0.00156112  
  pool2d0                                                   10          3.36047     0.839323 (0.249763)     2.521151 (0.750237)     0.325205    0.36992     0.336047    0.000932311 
    pool2d0/compute                                         10          3.02301     0.501861 (0.166014)     2.521151 (0.833986)     0.295365    0.31441     0.302301    0.000838687 
    pool2d0/infer_shape                                     10          0.234195    0.234195 (1.000000)     0.000000 (0.000000)     0.019404    0.04381     0.0234195   6.49737e-05 
    pool2d0/prepare_data                                    10          0.009753    0.009753 (1.000000)     0.000000 (0.000000)     0.000895    0.00108     0.0009753   2.70582e-06 
  pool2d1                                                   10          2.16973     1.732061 (0.798286)     0.437664 (0.201714)     0.153895    0.487722    0.216973    0.000601956 
    pool2d1/compute                                         10          1.57262     1.134954 (0.721697)     0.437664 (0.278303)     0.103511    0.418838    0.157262    0.000436298 
    pool2d1/infer_shape                                     10          0.311116    0.311116 (1.000000)     0.000000 (0.000000)     0.026703    0.037646    0.0311116   8.63143e-05 
    pool2d1/prepare_data                                    10          0.029201    0.029201 (1.000000)     0.000000 (0.000000)     0.001315    0.00706     0.0029201   8.10136e-06 
matmul_v2_grad                                              10          1.96888     1.436527 (0.729617)     0.532352 (0.270383)     0.190886    0.200177    0.196888    0.000546235 
  matmul_v2_grad0                                           10          1.93755     1.405194 (0.725244)     0.532352 (0.274756)     0.187026    0.197143    0.193755    0.000537542 
    matmul_v2_grad0/compute                                 10          1.77474     1.242387 (0.700039)     0.532352 (0.299961)     0.172564    0.181941    0.177474    0.000492374 
    matmul_v2_grad0/infer_shape                             10          0.056007    0.056007 (1.000000)     0.000000 (0.000000)     0.004695    0.00817     0.0056007   1.55383e-05 
    matmul_v2_grad0/prepare_data                            10          0.014146    0.014146 (1.000000)     0.000000 (0.000000)     0.000958    0.001863    0.0014146   3.92459e-06 
matmul_v2                                                   10          1.74352     1.435523 (0.823346)     0.308000 (0.176654)     0.162007    0.20274     0.174352    0.000483713 
  matmul_v20                                                10          1.71048     1.402475 (0.819933)     0.308000 (0.180067)     0.159413    0.200219    0.171048    0.000474545 
    matmul_v20/compute                                      10          1.52552     1.217520 (0.798102)     0.308000 (0.201898)     0.134283    0.184443    0.152552    0.000423232 
    matmul_v20/infer_shape                                  10          0.076505    0.076505 (1.000000)     0.000000 (0.000000)     0.005962    0.012789    0.0076505   2.12251e-05 
    matmul_v20/prepare_data                                 10          0.009062    0.009062 (1.000000)     0.000000 (0.000000)     0.000834    0.00104     0.0009062   2.51411e-06 
flatten_contiguous_range                                    10          1.628       1.584190 (0.973091)     0.043808 (0.026909)     0.074235    0.472363    0.1628      0.000451663 
  flatten_contiguous_range0                                 10          1.5969      1.553088 (0.972567)     0.043808 (0.027433)     0.071947    0.470031    0.15969     0.000443034 
    flatten_contiguous_range0/compute                       10          1.4345      1.390690 (0.969461)     0.043808 (0.030539)     0.057727    0.455247    0.14345     0.000397979 
      GpuMemcpyAsync(same_gpu):GPU->GPU                     10          0.381014    0.337206 (0.885023)     0.043808 (0.114977)     0.031292    0.048694    0.0381014   0.000105706 
    flatten_contiguous_range0/infer_shape                   10          0.061895    0.061895 (1.000000)     0.000000 (0.000000)     0.005718    0.006542    0.0061895   1.71718e-05 
    flatten_contiguous_range0/prepare_data                  10          0.024705    0.024705 (1.000000)     0.000000 (0.000000)     0.001172    0.010523    0.0024705   6.85402e-06 
accuracy                                                    20          1.29942     1.116606 (0.859310)     0.182816 (0.140690)     0.050164    0.08712     0.0649711   0.000360504 
  accuracy0                                                 10          0.730271    0.628671 (0.860874)     0.101600 (0.139126)     0.066713    0.081337    0.0730271   0.000202602 
    accuracy0/compute                                       10          0.541965    0.440365 (0.812534)     0.101600 (0.187466)     0.049281    0.062269    0.0541965   0.00015036  
    accuracy0/infer_shape                                   10          0.068821    0.068821 (1.000000)     0.000000 (0.000000)     0.006357    0.007376    0.0068821   1.90933e-05 
    accuracy0/prepare_data                                  10          0.00801     0.008010 (1.000000)     0.000000 (0.000000)     0.000696    0.001011    0.000801    2.22225e-06 
  accuracy1                                                 10          0.524837    0.443621 (0.845255)     0.081216 (0.154745)     0.048356    0.056907    0.0524837   0.000145608 
    accuracy1/compute                                       10          0.382189    0.300973 (0.787498)     0.081216 (0.212502)     0.034964    0.040596    0.0382189   0.000106032 
    accuracy1/infer_shape                                   10          0.048497    0.048497 (1.000000)     0.000000 (0.000000)     0.004246    0.007807    0.0048497   1.34547e-05 
    accuracy1/prepare_data                                  10          0.011458    0.011458 (1.000000)     0.000000 (0.000000)     0.000953    0.001412    0.0011458   3.17884e-06 
reduce_mean                                                 20          1.23052     1.156916 (0.940188)     0.073600 (0.059812)     0.055965    0.076752    0.0615258   0.000341387 
  reduce_mean0                                              10          0.630092    0.586316 (0.930524)     0.043776 (0.069476)     0.057471    0.075023    0.0630092   0.000174809 
    reduce_mean0/compute                                    10          0.485718    0.441942 (0.909874)     0.043776 (0.090126)     0.045249    0.056889    0.0485718   0.000134755 
    reduce_mean0/infer_shape                                10          0.073955    0.073955 (1.000000)     0.000000 (0.000000)     0.006459    0.010565    0.0073955   2.05177e-05 
    reduce_mean0/prepare_data                               10          0.014935    0.014935 (1.000000)     0.000000 (0.000000)     0.00076     0.006505    0.0014935   4.14348e-06 
  reduce_mean1                                              10          0.560178    0.530354 (0.946760)     0.029824 (0.053240)     0.052755    0.060501    0.0560178   0.000155413 
    reduce_mean1/compute                                    10          0.445017    0.415193 (0.932982)     0.029824 (0.067018)     0.041977    0.048801    0.0445017   0.000123463 
      GpuMemcpyAsync(same_gpu):GPU->GPU                     10          0.259944    0.230120 (0.885268)     0.029824 (0.114732)     0.023825    0.030835    0.0259944   7.21174e-05 
    reduce_mean1/infer_shape                                10          0.051102    0.051102 (1.000000)     0.000000 (0.000000)     0.004635    0.005781    0.0051102   1.41775e-05 
    reduce_mean1/prepare_data                               10          0.010902    0.010902 (1.000000)     0.000000 (0.000000)     0.0008      0.001453    0.0010902   3.02459e-06 
FetchAsync                                                  30          1.17956     1.094440 (0.927837)     0.085120 (0.072163)     0.020777    0.074542    0.0393187   0.00032725  
  GpuMemcpyAsync:GPU->CUDAPinned                            30          0.65539     0.570270 (0.870123)     0.085120 (0.129877)     0.012167    0.038614    0.0218463   0.000181828 
GpuMemcpyAsync(same_gpu):GPU->GPU                           20          1.17424     0.601982 (0.512658)     0.572255 (0.487342)     0.038453    0.087841    0.0587119   0.000325774 
scale                                                       30          1.15176     1.052334 (0.913677)     0.099423 (0.086323)     0.024454    0.060752    0.0383919   0.000319537 
  scale0                                                    10          0.471113    0.438665 (0.931125)     0.032448 (0.068875)     0.043918    0.058712    0.0471113   0.000130703 
    scale0/compute                                          10          0.355682    0.323234 (0.908772)     0.032448 (0.091228)     0.034165    0.041175    0.0355682   9.86784e-05 
    scale0/infer_shape                                      10          0.037471    0.037471 (1.000000)     0.000000 (0.000000)     0.002979    0.006564    0.0037471   1.03957e-05 
    scale0/prepare_data                                     10          0.020155    0.020155 (1.000000)     0.000000 (0.000000)     0.000797    0.009068    0.0020155   5.59169e-06 
  scale2                                                    10          0.348762    0.319579 (0.916324)     0.029183 (0.083676)     0.030048    0.043217    0.0348762   9.67586e-05 
    scale2/compute                                          10          0.255967    0.226784 (0.885989)     0.029183 (0.114011)     0.021742    0.031565    0.0255967   7.1014e-05  
    scale2/infer_shape                                      10          0.024138    0.024138 (1.000000)     0.000000 (0.000000)     0.002145    0.002762    0.0024138   6.69671e-06 
    scale2/prepare_data                                     10          0.016398    0.016398 (1.000000)     0.000000 (0.000000)     0.000844    0.004172    0.0016398   4.54937e-06 
  scale1                                                    10          0.267844    0.230052 (0.858903)     0.037792 (0.141097)     0.022777    0.040711    0.0267844   7.43091e-05 
    scale1/compute                                          10          0.184895    0.147103 (0.795603)     0.037792 (0.204397)     0.015066    0.031897    0.0184895   5.12962e-05 
    scale1/infer_shape                                      10          0.023923    0.023923 (1.000000)     0.000000 (0.000000)     0.001385    0.005506    0.0023923   6.63706e-06 
    scale1/prepare_data                                     10          0.012742    0.012742 (1.000000)     0.000000 (0.000000)     0.000744    0.003659    0.0012742   3.53507e-06 
top_k_v2                                                    20          1.06059     0.833392 (0.785780)     0.227200 (0.214220)     0.034387    0.078834    0.0530296   0.000294245 
  top_k_v20                                                 10          0.650996    0.479956 (0.737264)     0.171040 (0.262736)     0.060984    0.075642    0.0650996   0.000180609 
    top_k_v20/compute                                       10          0.47851     0.307470 (0.642557)     0.171040 (0.357443)     0.045499    0.057886    0.047851    0.000132755 
    top_k_v20/infer_shape                                   10          0.079914    0.079914 (1.000000)     0.000000 (0.000000)     0.007181    0.011616    0.0079914   2.21709e-05 
    top_k_v20/prepare_data                                  10          0.008743    0.008743 (1.000000)     0.000000 (0.000000)     0.000738    0.001427    0.0008743   2.42561e-06 
  top_k_v21                                                 10          0.361794    0.305634 (0.844774)     0.056160 (0.155226)     0.032448    0.039786    0.0361794   0.000100374 
    top_k_v21/compute                                       10          0.234822    0.178662 (0.760840)     0.056160 (0.239160)     0.021673    0.025766    0.0234822   6.51477e-05 
    top_k_v21/infer_shape                                   10          0.04999     0.049990 (1.000000)     0.000000 (0.000000)     0.00411     0.007169    0.004999    1.38689e-05 
    top_k_v21/prepare_data                                  10          0.011047    0.011047 (1.000000)     0.000000 (0.000000)     0.000865    0.001463    0.0011047   3.06482e-06 
FastThreadedSSAGraphExecutorPrepare                         10          1.01591     1.015911 (1.000000)     0.000000 (0.000000)     0.043918    0.184       0.101591    0.000281849 
reshape2                                                    10          0.923536    0.888944 (0.962544)     0.034592 (0.037456)     0.0768      0.125537    0.0923536   0.000256221 
  reshape20                                                 10          0.888009    0.853417 (0.961045)     0.034592 (0.038955)     0.073656    0.121932    0.0888009   0.000246364 
    reshape20/compute                                       10          0.637534    0.602942 (0.945741)     0.034592 (0.054259)     0.052087    0.096617    0.0637534   0.000176874 
      GpuMemcpyAsync(same_gpu):GPU->GPU                     10          0.297844    0.263252 (0.883859)     0.034592 (0.116141)     0.022108    0.04782     0.0297844   8.26322e-05 
    reshape20/infer_shape                                   10          0.132319    0.132319 (1.000000)     0.000000 (0.000000)     0.0113      0.01763     0.0132319   3.67098e-05 
    reshape20/prepare_data                                  10          0.016687    0.016687 (1.000000)     0.000000 (0.000000)     0.001452    0.002001    0.0016687   4.62955e-06 
softmax_with_cross_entropy                                  10          0.862615    0.747479 (0.866527)     0.115136 (0.133473)     0.075162    0.152028    0.0862615   0.000239319 
  softmax_with_cross_entropy0                               10          0.82787     0.712734 (0.860925)     0.115136 (0.139075)     0.072524    0.149285    0.082787    0.00022968  
    softmax_with_cross_entropy0/compute                     10          0.667773    0.552637 (0.827582)     0.115136 (0.172418)     0.056851    0.134061    0.0667773   0.000185263 
    softmax_with_cross_entropy0/infer_shape                 10          0.080028    0.080028 (1.000000)     0.000000 (0.000000)     0.00711     0.011344    0.0080028   2.22025e-05 
    softmax_with_cross_entropy0/prepare_data                10          0.014741    0.014741 (1.000000)     0.000000 (0.000000)     0.000975    0.00359     0.0014741   4.08966e-06 
reduce_mean_grad                                            20          0.736037    0.667077 (0.906309)     0.068960 (0.093691)     0.02603     0.064879    0.0368019   0.000204202 
  reduce_mean_grad0                                         10          0.397189    0.359781 (0.905818)     0.037408 (0.094182)     0.036444    0.04873     0.0397189   0.000110194 
    reduce_mean_grad0/compute                               10          0.276244    0.238836 (0.864583)     0.037408 (0.135417)     0.024638    0.036476    0.0276244   7.66396e-05 
    reduce_mean_grad0/infer_shape                           10          0.04568     0.045680 (1.000000)     0.000000 (0.000000)     0.004167    0.005002    0.004568    1.26732e-05 
    reduce_mean_grad0/prepare_data                          10          0.015725    0.015725 (1.000000)     0.000000 (0.000000)     0.000796    0.002117    0.0015725   4.36266e-06 
  reduce_mean_grad1                                         10          0.296593    0.265041 (0.893619)     0.031552 (0.106381)     0.024298    0.062753    0.0296593   8.22851e-05 
    reduce_mean_grad1/compute                               10          0.199728    0.168176 (0.842025)     0.031552 (0.157975)     0.015788    0.051503    0.0199728   5.54114e-05 
    reduce_mean_grad1/infer_shape                           10          0.024959    0.024959 (1.000000)     0.000000 (0.000000)     0.002295    0.002672    0.0024959   6.92448e-06 
    reduce_mean_grad1/prepare_data                          10          0.01702     0.017020 (1.000000)     0.000000 (0.000000)     0.001071    0.00405     0.001702    4.72193e-06 
ScaleLossGrad                                               10          0.393141    0.375733 (0.955721)     0.017408 (0.044279)     0.037164    0.042198    0.0393141   0.000109071 
  GpuMemcpyAsync:CPU->GPU                                   10          0.286248    0.268840 (0.939186)     0.017408 (0.060814)     0.026611    0.031607    0.0286248   7.9415e-05  
softmax_with_cross_entropy_grad                             10          0.373665    0.332097 (0.888756)     0.041568 (0.111244)     0.03372     0.0404      0.0373665   0.000103668 
  softmax_with_cross_entropy_grad0                          10          0.346983    0.305415 (0.880202)     0.041568 (0.119798)     0.031705    0.037504    0.0346983   9.6265e-05  
    softmax_with_cross_entropy_grad0/compute                10          0.224126    0.182558 (0.814533)     0.041568 (0.185467)     0.021137    0.023812    0.0224126   6.21803e-05 
    softmax_with_cross_entropy_grad0/infer_shape            10          0.038347    0.038347 (1.000000)     0.000000 (0.000000)     0.003219    0.004213    0.0038347   1.06388e-05 
    softmax_with_cross_entropy_grad0/prepare_data           10          0.015663    0.015663 (1.000000)     0.000000 (0.000000)     0.001103    0.00232     0.0015663   4.34545e-06 
GpuMemcpyAsync:CPU->GPU                                     10          0.273682    0.258322 (0.943876)     0.015360 (0.056124)     0.023964    0.03668     0.0273682   7.59288e-05 
InitLocalVars                                               1           0.202196    0.202196 (1.000000)     0.000000 (0.000000)     0.202196    0.202196    0.202196    5.60961e-05 
flatten_contiguous_range_grad                               10          0.196442    0.196442 (1.000000)     0.000000 (0.000000)     0.017499    0.023256    0.0196442   5.44998e-05 
  flatten_contiguous_range_grad0                            10          0.16535     0.165350 (1.000000)     0.000000 (0.000000)     0.01483     0.020817    0.016535    4.58738e-05 
    flatten_contiguous_range_grad0/compute                  10          0.05041     0.050410 (1.000000)     0.000000 (0.000000)     0.004474    0.00578     0.005041    1.39855e-05 
    flatten_contiguous_range_grad0/infer_shape              10          0.036354    0.036354 (1.000000)     0.000000 (0.000000)     0.003303    0.00413     0.0036354   1.00858e-05 
    flatten_contiguous_range_grad0/prepare_data             10          0.016318    0.016318 (1.000000)     0.000000 (0.000000)     0.001114    0.002167    0.0016318   4.52717e-06 
ScopeBufferedMonitor::post_local_exec_scopes_process        10          0.057485    0.057485 (1.000000)     0.000000 (0.000000)     0.004675    0.008794    0.0057485   1.59483e-05 
ScopeBufferedMonitor::pre_local_exec_scopes_process         10          0.026853    0.026853 (1.000000)     0.000000 (0.000000)     0.001763    0.005659    0.0026853   7.44994e-06 

------------------------->     Profiling Report     <-------------------------

Place: All
Time unit: ms
Sorted by total time in descending order in the same thread


-------------------------       Event Summary       -------------------------

Event                                                                Calls       Total       CPU Time (Ratio)        GPU Time (Ratio)        Min.        Max.        Ave.        Ratio.      
thread1::cinn_launch                                                 70          2187.27     1163.807214 (0.532082)  1023.464856 (0.467918)  0.144505    132.802     31.2467     0.947468    
  cinn_launch5                                                       10          1317.3      629.974337 (0.478232)   687.325232 (0.521768)   130.456     132.777     131.73      0.570619    
    cinn_launch5/compute                                             10          1314.55     627.223049 (0.477140)   687.325232 (0.522860)   130.159     132.502     131.455     0.569427    
    cinn_launch5/infer_shape                                         10          0.023378    0.023378 (1.000000)     0.000000 (0.000000)     0.002061    0.002586    0.0023378   1.01267e-05 
    cinn_launch5/prepare_data                                        10          0.015259    0.015259 (1.000000)     0.000000 (0.000000)     0.001214    0.001909    0.0015259   6.60979e-06 
  cinn_launch1                                                       10          792.994     495.389559 (0.624708)   297.604382 (0.375292)   78.7686     80.2022     79.2994     0.343504    
    cinn_launch1/compute                                             10          789.958     492.353210 (0.623265)   297.604382 (0.376735)   78.4529     79.9307     78.9958     0.342189    
    cinn_launch1/infer_shape                                         10          0.020952    0.020952 (1.000000)     0.000000 (0.000000)     0.001848    0.002353    0.0020952   9.07585e-06 
    cinn_launch1/prepare_data                                        10          0.018554    0.018554 (1.000000)     0.000000 (0.000000)     0.000933    0.007716    0.0018554   8.0371e-06  
  cinn_launch6                                                       10          40.1071     13.895106 (0.346450)    26.211954 (0.653550)    3.91057     4.08749     4.01071     0.0173733   
    cinn_launch6/compute                                             10          39.9744     13.762402 (0.344281)    26.211954 (0.655719)    3.89783     4.07389     3.99744     0.0173158   
    cinn_launch6/prepare_data                                        10          0.017337    0.017337 (1.000000)     0.000000 (0.000000)     0.001352    0.003878    0.0017337   7.50993e-06 
    cinn_launch6/infer_shape                                         10          0.016893    0.016893 (1.000000)     0.000000 (0.000000)     0.001492    0.002011    0.0016893   7.3176e-06  
  cinn_launch0                                                       10          24.7722     12.636389 (0.510105)    12.135768 (0.489895)    2.36682     2.79607     2.47722     0.0107306   
    cinn_launch0/compute                                             10          24.6097     12.473901 (0.506870)    12.135768 (0.493130)    2.35135     2.7698      2.46097     0.0106603   
    cinn_launch0/prepare_data                                        10          0.02063     0.020630 (1.000000)     0.000000 (0.000000)     0.001075    0.005477    0.002063    8.93637e-06 
    cinn_launch0/infer_shape                                         10          0.019433    0.019433 (1.000000)     0.000000 (0.000000)     0.001686    0.002248    0.0019433   8.41786e-06 
  cinn_launch3                                                       10          5.74238     5.631181 (0.980635)     0.111200 (0.019365)     0.299111    0.913227    0.574238    0.00248745  
    cinn_launch3/compute                                             10          5.51012     5.398919 (0.979819)     0.111200 (0.020181)     0.289619    0.902642    0.551012    0.00238684  
    cinn_launch3/prepare_data                                        10          0.128577    0.128577 (1.000000)     0.000000 (0.000000)     0.000861    0.119963    0.0128577   5.56961e-05 
    cinn_launch3/infer_shape                                         10          0.020379    0.020379 (1.000000)     0.000000 (0.000000)     0.001508    0.003291    0.0020379   8.82764e-06 
  cinn_launch2                                                       10          3.74691     3.701082 (0.987770)     0.045824 (0.012230)     0.247041    0.792207    0.374691    0.00162306  
    cinn_launch2/compute                                             10          3.60893     3.563102 (0.987303)     0.045824 (0.012697)     0.234999    0.780948    0.360893    0.00156329  
    cinn_launch2/infer_shape                                         10          0.025019    0.025019 (1.000000)     0.000000 (0.000000)     0.002175    0.002989    0.0025019   1.08376e-05 
    cinn_launch2/prepare_data                                        10          0.014288    0.014288 (1.000000)     0.000000 (0.000000)     0.000807    0.006676    0.0014288   6.18918e-06 
  cinn_launch4                                                       10          1.62351     1.593013 (0.981216)     0.030496 (0.018784)     0.14286     0.208642    0.162351    0.000703261 
    cinn_launch4/compute                                             10          1.54474     1.514241 (0.980258)     0.030496 (0.019742)     0.135918    0.20025     0.154474    0.000669139 
    cinn_launch4/infer_shape                                         10          0.019444    0.019444 (1.000000)     0.000000 (0.000000)     0.001384    0.005009    0.0019444   8.42262e-06 
    cinn_launch4/prepare_data                                        10          0.009371    0.009371 (1.000000)     0.000000 (0.000000)     0.000728    0.001368    0.0009371   4.05927e-06 
thread1::momentum                                                    1610        58.5373     46.446097 (0.793444)    12.091229 (0.206556)    0.027103    0.470355    0.0363586   0.0253568   
  momentum1                                                          10          0.959404    0.404716 (0.421841)     0.554688 (0.578159)     0.091276    0.112871    0.0959404   0.000415588 
    momentum1/compute                                                10          0.772729    0.218041 (0.282170)     0.554688 (0.717830)     0.074786    0.079863    0.0772729   0.000334726 
    momentum1/infer_shape                                            10          0.05509     0.055090 (1.000000)     0.000000 (0.000000)     0.004736    0.009342    0.005509    2.38635e-05 
    momentum1/prepare_data                                           10          0.031826    0.031826 (1.000000)     0.000000 (0.000000)     0.001174    0.018994    0.0031826   1.37862e-05 
  momentum71                                                         10          0.903698    0.273042 (0.302139)     0.630656 (0.697861)     0.087678    0.100672    0.0903698   0.000391458 
    momentum71/compute                                               10          0.76409     0.133434 (0.174631)     0.630656 (0.825369)     0.075316    0.077558    0.076409    0.000330984 
    momentum71/infer_shape                                           10          0.035797    0.035797 (1.000000)     0.000000 (0.000000)     0.0033      0.00451     0.0035797   1.55063e-05 
    momentum71/prepare_data                                          10          0.010768    0.010768 (1.000000)     0.000000 (0.000000)     0.000938    0.001447    0.0010768   4.66441e-06 
  momentum88                                                         10          0.901322    0.276042 (0.306263)     0.625280 (0.693737)     0.086156    0.108517    0.0901322   0.000390429 
    momentum88/compute                                               10          0.775949    0.150669 (0.194174)     0.625280 (0.805826)     0.074064    0.096368    0.0775949   0.000336121 
    momentum88/infer_shape                                           10          0.035793    0.035793 (1.000000)     0.000000 (0.000000)     0.003299    0.003873    0.0035793   1.55046e-05 
    momentum88/prepare_data                                          10          0.015928    0.015928 (1.000000)     0.000000 (0.000000)     0.001093    0.003827    0.0015928   6.89959e-06 
  momentum132                                                        10          0.869803    0.242283 (0.278549)     0.627520 (0.721451)     0.085286    0.089928    0.0869803   0.000376776 
    momentum132/compute                                              10          0.758306    0.130786 (0.172471)     0.627520 (0.827529)     0.074427    0.077271    0.0758306   0.000328478 
    momentum132/infer_shape                                          10          0.029231    0.029231 (1.000000)     0.000000 (0.000000)     0.002776    0.003212    0.0029231   1.26621e-05 
    momentum132/prepare_data                                         10          0.013501    0.013501 (1.000000)     0.000000 (0.000000)     0.000886    0.004361    0.0013501   5.84828e-06 
  momentum20                                                         10          0.839798    0.275798 (0.328410)     0.564000 (0.671590)     0.080932    0.092716    0.0839798   0.000363778 
    momentum20/compute                                               10          0.695027    0.131027 (0.188521)     0.564000 (0.811479)     0.06844     0.070687    0.0695027   0.000301067 
    momentum20/infer_shape                                           10          0.03773     0.037730 (1.000000)     0.000000 (0.000000)     0.003145    0.005865    0.003773    1.63436e-05 
    momentum20/prepare_data                                          10          0.013271    0.013271 (1.000000)     0.000000 (0.000000)     0.001102    0.00154     0.0013271   5.74865e-06 
  momentum2                                                          10          0.832178    0.793202 (0.953164)     0.038976 (0.046836)     0.079112    0.088385    0.0832178   0.000360477 
    momentum2/compute                                                10          0.468485    0.429509 (0.916804)     0.038976 (0.083196)     0.043624    0.050151    0.0468485   0.000202935 
    momentum2/infer_shape                                            10          0.129463    0.129463 (1.000000)     0.000000 (0.000000)     0.012059    0.014667    0.0129463   5.60799e-05 
    momentum2/prepare_data                                           10          0.01614     0.016140 (1.000000)     0.000000 (0.000000)     0.001386    0.001853    0.001614    6.99142e-06 
  momentum49                                                         10          0.732366    0.701006 (0.957180)     0.031360 (0.042820)     0.028383    0.468479    0.0732366   0.000317242 
    momentum49/compute                                               10          0.600286    0.568926 (0.947758)     0.031360 (0.052242)     0.015544    0.454795    0.0600286   0.000260028 
    momentum49/infer_shape                                           10          0.03428     0.034280 (1.000000)     0.000000 (0.000000)     0.003021    0.003802    0.003428    1.48492e-05 
    momentum49/prepare_data                                          10          0.014438    0.014438 (1.000000)     0.000000 (0.000000)     0.000906    0.004012    0.0014438   6.25416e-06 
  momentum7                                                          10          0.571286    0.273846 (0.479350)     0.297440 (0.520650)     0.054083    0.06469     0.0571286   0.000247466 
    momentum7/compute                                                10          0.426134    0.128694 (0.302004)     0.297440 (0.697996)     0.041966    0.043587    0.0426134   0.00018459  
    momentum7/infer_shape                                            10          0.039842    0.039842 (1.000000)     0.000000 (0.000000)     0.002731    0.012386    0.0039842   1.72585e-05 
    momentum7/prepare_data                                           10          0.013448    0.013448 (1.000000)     0.000000 (0.000000)     0.000982    0.003298    0.0013448   5.82532e-06 
  momentum145                                                        10          0.568752    0.256432 (0.450868)     0.312320 (0.549132)     0.05391     0.060621    0.0568752   0.000246368 
    momentum145/compute                                              10          0.447327    0.135007 (0.301808)     0.312320 (0.698192)     0.043057    0.049147    0.0447327   0.00019377  
    momentum145/infer_shape                                          10          0.039191    0.039191 (1.000000)     0.000000 (0.000000)     0.003437    0.006108    0.0039191   1.69765e-05 
    momentum145/prepare_data                                         10          0.009539    0.009539 (1.000000)     0.000000 (0.000000)     0.000802    0.00117     0.0009539   4.13204e-06 
  momentum55                                                         10          0.560808    0.270952 (0.483146)     0.289856 (0.516854)     0.054827    0.057577    0.0560808   0.000242927 
    momentum55/compute                                               10          0.42159     0.131734 (0.312469)     0.289856 (0.687531)     0.041276    0.04288     0.042159    0.000182622 
    momentum55/infer_shape                                           10          0.040777    0.040777 (1.000000)     0.000000 (0.000000)     0.003379    0.006199    0.0040777   1.76635e-05 
    momentum55/prepare_data                                          10          0.009439    0.009439 (1.000000)     0.000000 (0.000000)     0.000778    0.001405    0.0009439   4.08872e-06 
  momentum81                                                         10          0.558533    0.261125 (0.467519)     0.297408 (0.532481)     0.054238    0.058208    0.0558533   0.000241942 
    momentum81/compute                                               10          0.427333    0.129925 (0.304037)     0.297408 (0.695963)     0.042062    0.043719    0.0427333   0.000185109 
    momentum81/infer_shape                                           10          0.044122    0.044122 (1.000000)     0.000000 (0.000000)     0.003791    0.006183    0.0044122   1.91125e-05 
    momentum81/prepare_data                                          10          0.013564    0.013564 (1.000000)     0.000000 (0.000000)     0.001194    0.001578    0.0013564   5.87556e-06 
  momentum111                                                        10          0.557831    0.253351 (0.454172)     0.304480 (0.545828)     0.053923    0.057717    0.0557831   0.000241638 
    momentum111/compute                                              10          0.435202    0.130722 (0.300371)     0.304480 (0.699629)     0.042216    0.045638    0.0435202   0.000188518 
    momentum111/infer_shape                                          10          0.033074    0.033074 (1.000000)     0.000000 (0.000000)     0.002846    0.003533    0.0033074   1.43268e-05 
    momentum111/prepare_data                                         10          0.016372    0.016372 (1.000000)     0.000000 (0.000000)     0.001055    0.004106    0.0016372   7.09192e-06 
  momentum0                                                          10          0.539067    0.505915 (0.938501)     0.033152 (0.061499)     0.049082    0.063612    0.0539067   0.00023351  
    momentum0/compute                                                10          0.310903    0.277751 (0.893369)     0.033152 (0.106631)     0.029847    0.032667    0.0310903   0.000134675 
    momentum0/infer_shape                                            10          0.077918    0.077918 (1.000000)     0.000000 (0.000000)     0.006913    0.008302    0.0077918   3.3752e-05  
    momentum0/prepare_data                                           10          0.011628    0.011628 (1.000000)     0.000000 (0.000000)     0.000988    0.001448    0.0011628   5.03694e-06 
  momentum51                                                         10          0.529321    0.495081 (0.935313)     0.034240 (0.064687)     0.047532    0.067862    0.0529321   0.000229288 
    momentum51/compute                                               10          0.31942     0.285180 (0.892806)     0.034240 (0.107194)     0.026306    0.046438    0.031942    0.000138364 
    momentum51/infer_shape                                           10          0.063602    0.063602 (1.000000)     0.000000 (0.000000)     0.005723    0.007459    0.0063602   2.75507e-05 
    momentum51/prepare_data                                          10          0.012097    0.012097 (1.000000)     0.000000 (0.000000)     0.001005    0.001298    0.0012097   5.2401e-06  
  momentum78                                                         10          0.45589     0.258674 (0.567404)     0.197216 (0.432596)     0.044512    0.048738    0.045589    0.000197479 
    momentum78/compute                                               10          0.32507     0.127854 (0.393312)     0.197216 (0.606688)     0.03187     0.033809    0.032507    0.000140812 
    momentum78/infer_shape                                           10          0.034425    0.034425 (1.000000)     0.000000 (0.000000)     0.003073    0.003946    0.0034425   1.4912e-05  
    momentum78/prepare_data                                          10          0.012777    0.012777 (1.000000)     0.000000 (0.000000)     0.001064    0.001634    0.0012777   5.53466e-06 
  momentum87                                                         10          0.452022    0.262614 (0.580976)     0.189408 (0.419024)     0.0433      0.047546    0.0452022   0.000195804 
    momentum87/compute                                               10          0.316465    0.127057 (0.401488)     0.189408 (0.598512)     0.030718    0.032481    0.0316465   0.000137084 
    momentum87/infer_shape                                           10          0.032827    0.032827 (1.000000)     0.000000 (0.000000)     0.003126    0.003653    0.0032827   1.42198e-05 
    momentum87/prepare_data                                          10          0.015408    0.015408 (1.000000)     0.000000 (0.000000)     0.001119    0.003765    0.0015408   6.67434e-06 
  momentum117                                                        10          0.448806    0.257670 (0.574123)     0.191136 (0.425877)     0.043568    0.047558    0.0448806   0.000194411 
    momentum117/compute                                              10          0.321293    0.130157 (0.405104)     0.191136 (0.594896)     0.030903    0.033628    0.0321293   0.000139176 
    momentum117/infer_shape                                          10          0.033715    0.033715 (1.000000)     0.000000 (0.000000)     0.00298     0.004016    0.0033715   1.46044e-05 
    momentum117/prepare_data                                         10          0.013266    0.013266 (1.000000)     0.000000 (0.000000)     0.001094    0.001555    0.0013266   5.74648e-06 
  momentum76                                                         10          0.445961    0.256073 (0.574205)     0.189888 (0.425795)     0.043651    0.04663     0.0445961   0.000193178 
    momentum76/compute                                               10          0.320395    0.130507 (0.407332)     0.189888 (0.592668)     0.031319    0.032799    0.0320395   0.000138787 
    momentum76/infer_shape                                           10          0.033584    0.033584 (1.000000)     0.000000 (0.000000)     0.00309     0.003902    0.0033584   1.45477e-05 
    momentum76/prepare_data                                          10          0.013581    0.013581 (1.000000)     0.000000 (0.000000)     0.001145    0.00192     0.0013581   5.88293e-06 
  momentum125                                                        10          0.435974    0.248390 (0.569736)     0.187584 (0.430264)     0.041737    0.045758    0.0435974   0.000188852 
    momentum125/compute                                              10          0.317762    0.130178 (0.409671)     0.187584 (0.590329)     0.030725    0.032943    0.0317762   0.000137646 
    momentum125/infer_shape                                          10          0.03362     0.033620 (1.000000)     0.000000 (0.000000)     0.003113    0.00366     0.003362    1.45633e-05 
    momentum125/prepare_data                                         10          0.011954    0.011954 (1.000000)     0.000000 (0.000000)     0.00095     0.001653    0.0011954   5.17816e-06 
  momentum27                                                         10          0.432707    0.253731 (0.586381)     0.178976 (0.413619)     0.041874    0.045344    0.0432707   0.000187437 
    momentum27/compute                                               10          0.306091    0.127115 (0.415285)     0.178976 (0.584715)     0.029648    0.031664    0.0306091   0.000132591 
    momentum27/infer_shape                                           10          0.028424    0.028424 (1.000000)     0.000000 (0.000000)     0.002644    0.003323    0.0028424   1.23125e-05 
    momentum27/prepare_data                                          10          0.01576     0.015760 (1.000000)     0.000000 (0.000000)     0.000997    0.00386     0.001576    6.82681e-06 
  momentum122                                                        10          0.424438    0.236374 (0.556911)     0.188064 (0.443089)     0.040494    0.043786    0.0424438   0.000183855 
    momentum122/compute                                              10          0.314391    0.126327 (0.401815)     0.188064 (0.598185)     0.030395    0.032769    0.0314391   0.000136186 
    momentum122/infer_shape                                          10          0.034458    0.034458 (1.000000)     0.000000 (0.000000)     0.002973    0.005884    0.0034458   1.49263e-05 
    momentum122/prepare_data                                         10          0.009781    0.009781 (1.000000)     0.000000 (0.000000)     0.000794    0.001139    0.0009781   4.23687e-06 
  momentum25                                                         10          0.422867    0.252179 (0.596355)     0.170688 (0.403645)     0.040806    0.044857    0.0422867   0.000183175 
    momentum25/compute                                               10          0.29955     0.128862 (0.430185)     0.170688 (0.569815)     0.028794    0.03126     0.029955    0.000129757 
    momentum25/infer_shape                                           10          0.029262    0.029262 (1.000000)     0.000000 (0.000000)     0.002731    0.003192    0.0029262   1.26755e-05 
    momentum25/prepare_data                                          10          0.015453    0.015453 (1.000000)     0.000000 (0.000000)     0.001006    0.003712    0.0015453   6.69383e-06 
  momentum119                                                        10          0.388353    0.282721 (0.728000)     0.105632 (0.272000)     0.033129    0.073225    0.0388353   0.000168224 
    momentum119/compute                                              10          0.271491    0.165859 (0.610919)     0.105632 (0.389081)     0.022356    0.061963    0.0271491   0.000117603 
    momentum119/infer_shape                                          10          0.034243    0.034243 (1.000000)     0.000000 (0.000000)     0.002994    0.005135    0.0034243   1.48332e-05 
    momentum119/prepare_data                                         10          0.010021    0.010021 (1.000000)     0.000000 (0.000000)     0.000807    0.001213    0.0010021   4.34083e-06 
  momentum16                                                         10          0.371246    0.267406 (0.720293)     0.103840 (0.279707)     0.033718    0.046751    0.0371246   0.000160814 
    momentum16/compute                                               10          0.232431    0.128591 (0.553244)     0.103840 (0.446756)     0.021879    0.024318    0.0232431   0.000100683 
    momentum16/infer_shape                                           10          0.030433    0.030433 (1.000000)     0.000000 (0.000000)     0.002649    0.003346    0.0030433   1.31828e-05 
    momentum16/prepare_data                                          10          0.013832    0.013832 (1.000000)     0.000000 (0.000000)     0.001161    0.001777    0.0013832   5.99166e-06 
  momentum22                                                         10          0.371026    0.263346 (0.709778)     0.107680 (0.290222)     0.034601    0.040856    0.0371026   0.000160719 
    momentum22/compute                                               10          0.235898    0.128218 (0.543532)     0.107680 (0.456468)     0.022772    0.024836    0.0235898   0.000102185 
    momentum22/infer_shape                                           10          0.034224    0.034224 (1.000000)     0.000000 (0.000000)     0.002674    0.007845    0.0034224   1.48249e-05 
    momentum22/prepare_data                                          10          0.012147    0.012147 (1.000000)     0.000000 (0.000000)     0.000955    0.001671    0.0012147   5.26176e-06 
  momentum107                                                        10          0.369097    0.267209 (0.723953)     0.101888 (0.276047)     0.034091    0.045554    0.0369097   0.000159883 
    momentum107/compute                                              10          0.229842    0.127954 (0.556704)     0.101888 (0.443296)     0.022149    0.024661    0.0229842   9.95615e-05 
    momentum107/infer_shape                                          10          0.04545     0.045450 (1.000000)     0.000000 (0.000000)     0.003425    0.01208     0.004545    1.96877e-05 
    momentum107/prepare_data                                         10          0.016558    0.016558 (1.000000)     0.000000 (0.000000)     0.001246    0.003655    0.0016558   7.17249e-06 
  momentum84                                                         10          0.368969    0.262249 (0.710762)     0.106720 (0.289238)     0.035306    0.039533    0.0368969   0.000159828 
    momentum84/compute                                               10          0.238874    0.132154 (0.553237)     0.106720 (0.446763)     0.022585    0.024963    0.0238874   0.000103474 
    momentum84/infer_shape                                           10          0.039086    0.039086 (1.000000)     0.000000 (0.000000)     0.003456    0.005923    0.0039086   1.6931e-05  
    momentum84/prepare_data                                          10          0.013907    0.013907 (1.000000)     0.000000 (0.000000)     0.001167    0.001779    0.0013907   6.02414e-06 
  momentum126                                                        10          0.366005    0.254357 (0.694955)     0.111648 (0.305045)     0.034136    0.044712    0.0366005   0.000158544 
    momentum126/compute                                              10          0.248306    0.136658 (0.550361)     0.111648 (0.449639)     0.023276    0.033367    0.0248306   0.00010756  
    momentum126/infer_shape                                          10          0.032914    0.032914 (1.000000)     0.000000 (0.000000)     0.002717    0.005221    0.0032914   1.42575e-05 
    momentum126/prepare_data                                         10          0.009343    0.009343 (1.000000)     0.000000 (0.000000)     0.000806    0.001131    0.0009343   4.04714e-06 
  momentum63                                                         10          0.363781    0.260581 (0.716313)     0.103200 (0.283687)     0.033498    0.046924    0.0363781   0.00015758  
    momentum63/compute                                               10          0.234173    0.130973 (0.559300)     0.103200 (0.440700)     0.022524    0.025805    0.0234173   0.000101438 
    momentum63/infer_shape                                           10          0.033084    0.033084 (1.000000)     0.000000 (0.000000)     0.003004    0.003864    0.0033084   1.43311e-05 
    momentum63/prepare_data                                          10          0.009461    0.009461 (1.000000)     0.000000 (0.000000)     0.000827    0.00119     0.0009461   4.09825e-06 
  momentum8                                                          10          0.358421    0.251925 (0.702875)     0.106496 (0.297125)     0.034424    0.038267    0.0358421   0.000155258 
    momentum8/compute                                                10          0.232326    0.125830 (0.541610)     0.106496 (0.458390)     0.022288    0.024376    0.0232326   0.000100637 
    momentum8/infer_shape                                            10          0.029294    0.029294 (1.000000)     0.000000 (0.000000)     0.002696    0.003367    0.0029294   1.26894e-05 
    momentum8/prepare_data                                           10          0.015046    0.015046 (1.000000)     0.000000 (0.000000)     0.001072    0.003435    0.0015046   6.51753e-06 
  momentum65                                                         10          0.356545    0.253505 (0.711004)     0.103040 (0.288996)     0.033347    0.038356    0.0356545   0.000154446 
    momentum65/compute                                               10          0.230046    0.127006 (0.552090)     0.103040 (0.447910)     0.021474    0.024111    0.0230046   9.96498e-05 
    momentum65/infer_shape                                           10          0.032841    0.032841 (1.000000)     0.000000 (0.000000)     0.002986    0.003934    0.0032841   1.42259e-05 
    momentum65/prepare_data                                          10          0.01216     0.012160 (1.000000)     0.000000 (0.000000)     0.000804    0.003305    0.001216    5.26739e-06 
  momentum42                                                         10          0.353683    0.274675 (0.776614)     0.079008 (0.223386)     0.032525    0.043305    0.0353683   0.000153206 
    momentum42/compute                                               10          0.205947    0.126939 (0.616367)     0.079008 (0.383633)     0.019951    0.022269    0.0205947   8.92108e-05 
    momentum42/infer_shape                                           10          0.038657    0.038657 (1.000000)     0.000000 (0.000000)     0.003309    0.006563    0.0038657   1.67452e-05 
    momentum42/prepare_data                                          10          0.013687    0.013687 (1.000000)     0.000000 (0.000000)     0.001053    0.001739    0.0013687   5.92885e-06 
  momentum131                                                        10          0.351969    0.250017 (0.710338)     0.101952 (0.289662)     0.033095    0.037032    0.0351969   0.000152464 
    momentum131/compute                                              10          0.230781    0.128829 (0.558231)     0.101952 (0.441769)     0.022168    0.025555    0.0230781   9.99682e-05 
    momentum131/infer_shape                                          10          0.03474     0.034740 (1.000000)     0.000000 (0.000000)     0.003098    0.00378     0.003474    1.50484e-05 
    momentum131/prepare_data                                         10          0.010939    0.010939 (1.000000)     0.000000 (0.000000)     0.000797    0.002897    0.0010939   4.73848e-06 
  momentum146                                                        10          0.350152    0.240168 (0.685896)     0.109984 (0.314104)     0.033534    0.036327    0.0350152   0.000151677 
    momentum146/compute                                              10          0.234345    0.124361 (0.530675)     0.109984 (0.469325)     0.022266    0.024549    0.0234345   0.000101512 
    momentum146/infer_shape                                          10          0.034152    0.034152 (1.000000)     0.000000 (0.000000)     0.002929    0.005169    0.0034152   1.47937e-05 
    momentum146/prepare_data                                         10          0.012052    0.012052 (1.000000)     0.000000 (0.000000)     0.000874    0.00161     0.0012052   5.22061e-06 
  momentum41                                                         10          0.345933    0.314637 (0.909532)     0.031296 (0.090468)     0.027345    0.089257    0.0345933   0.000149849 
    momentum41/compute                                               10          0.164743    0.133447 (0.810031)     0.031296 (0.189969)     0.01503     0.020051    0.0164743   7.13623e-05 
    momentum41/infer_shape                                           10          0.050909    0.050909 (1.000000)     0.000000 (0.000000)     0.002694    0.024237    0.0050909   2.20524e-05 
    momentum41/prepare_data                                          10          0.009301    0.009301 (1.000000)     0.000000 (0.000000)     0.000733    0.001276    0.0009301   4.02895e-06 
  momentum52                                                         10          0.345902    0.308494 (0.891854)     0.037408 (0.108146)     0.030681    0.050659    0.0345902   0.000149836 
    momentum52/compute                                               10          0.178843    0.141435 (0.790833)     0.037408 (0.209167)     0.016234    0.020948    0.0178843   7.747e-05   
    momentum52/infer_shape                                           10          0.033515    0.033515 (1.000000)     0.000000 (0.000000)     0.002905    0.00394     0.0033515   1.45178e-05 
    momentum52/prepare_data                                          10          0.011312    0.011312 (1.000000)     0.000000 (0.000000)     0.001046    0.001419    0.0011312   4.90006e-06 
  momentum53                                                         10          0.34494     0.287564 (0.833664)     0.057376 (0.166336)     0.03219     0.038501    0.034494    0.000149419 
    momentum53/compute                                               10          0.199351    0.141975 (0.712186)     0.057376 (0.287814)     0.019033    0.02071     0.0199351   8.63536e-05 
    momentum53/infer_shape                                           10          0.034767    0.034767 (1.000000)     0.000000 (0.000000)     0.003153    0.004076    0.0034767   1.50601e-05 
    momentum53/prepare_data                                          10          0.008868    0.008868 (1.000000)     0.000000 (0.000000)     0.000783    0.001186    0.0008868   3.84138e-06 
  momentum97                                                         10          0.338199    0.265943 (0.786351)     0.072256 (0.213649)     0.031315    0.042933    0.0338199   0.000146499 
    momentum97/compute                                               10          0.211253    0.138997 (0.657965)     0.072256 (0.342035)     0.019134    0.030498    0.0211253   9.15092e-05 
    momentum97/infer_shape                                           10          0.036624    0.036624 (1.000000)     0.000000 (0.000000)     0.003254    0.005639    0.0036624   1.58645e-05 
    momentum97/prepare_data                                          10          0.014213    0.014213 (1.000000)     0.000000 (0.000000)     0.001213    0.001848    0.0014213   6.15669e-06 
  momentum86                                                         10          0.336774    0.266694 (0.791908)     0.070080 (0.208092)     0.031055    0.039587    0.0336774   0.000145882 
    momentum86/compute                                               10          0.213236    0.143156 (0.671350)     0.070080 (0.328650)     0.019579    0.02677     0.0213236   9.23682e-05 
    momentum86/infer_shape                                           10          0.033607    0.033607 (1.000000)     0.000000 (0.000000)     0.003       0.003696    0.0033607   1.45577e-05 
    momentum86/prepare_data                                          10          0.013178    0.013178 (1.000000)     0.000000 (0.000000)     0.001081    0.001789    0.0013178   5.70836e-06 
  momentum28                                                         10          0.330525    0.292605 (0.885273)     0.037920 (0.114727)     0.028427    0.048053    0.0330525   0.000143175 
    momentum28/compute                                               10          0.17327     0.135350 (0.781151)     0.037920 (0.218849)     0.015633    0.019821    0.017327    7.5056e-05  
    momentum28/infer_shape                                           10          0.05251     0.052510 (1.000000)     0.000000 (0.000000)     0.003106    0.01959     0.005251    2.27459e-05 
    momentum28/prepare_data                                          10          0.019938    0.019938 (1.000000)     0.000000 (0.000000)     0.000939    0.007038    0.0019938   8.63661e-06 
  momentum40                                                         10          0.330306    0.291810 (0.883454)     0.038496 (0.116546)     0.027466    0.046675    0.0330306   0.00014308  
    momentum40/compute                                               10          0.199264    0.160768 (0.806809)     0.038496 (0.193191)     0.015456    0.033681    0.0199264   8.63159e-05 
    momentum40/infer_shape                                           10          0.032721    0.032721 (1.000000)     0.000000 (0.000000)     0.00281     0.005174    0.0032721   1.41739e-05 
    momentum40/prepare_data                                          10          0.014345    0.014345 (1.000000)     0.000000 (0.000000)     0.001148    0.001792    0.0014345   6.21387e-06 
  momentum80                                                         10          0.329772    0.293772 (0.890834)     0.036000 (0.109166)     0.029182    0.050381    0.0329772   0.000142848 
    momentum80/compute                                               10          0.170596    0.134596 (0.788975)     0.036000 (0.211025)     0.015531    0.021685    0.0170596   7.38977e-05 
    momentum80/infer_shape                                           10          0.060719    0.060719 (1.000000)     0.000000 (0.000000)     0.003143    0.018629    0.0060719   2.63019e-05 
    momentum80/prepare_data                                          10          0.014262    0.014262 (1.000000)     0.000000 (0.000000)     0.001303    0.001717    0.0014262   6.17792e-06 
  momentum3                                                          10          0.327682    0.291938 (0.890919)     0.035744 (0.109081)     0.030455    0.038356    0.0327682   0.000141943 
    momentum3/compute                                                10          0.179244    0.143500 (0.800585)     0.035744 (0.199415)     0.016622    0.018978    0.0179244   7.76437e-05 
    momentum3/infer_shape                                            10          0.035769    0.035769 (1.000000)     0.000000 (0.000000)     0.00329     0.003947    0.0035769   1.54942e-05 
    momentum3/prepare_data                                           10          0.018211    0.018211 (1.000000)     0.000000 (0.000000)     0.001099    0.007414    0.0018211   7.88852e-06 
  momentum46                                                         10          0.326891    0.248619 (0.760556)     0.078272 (0.239444)     0.030726    0.035562    0.0326891   0.000141601 
    momentum46/compute                                               10          0.206364    0.128092 (0.620709)     0.078272 (0.379291)     0.019262    0.021921    0.0206364   8.93914e-05 
    momentum46/infer_shape                                           10          0.033935    0.033935 (1.000000)     0.000000 (0.000000)     0.003029    0.00376     0.0033935   1.46997e-05 
    momentum46/prepare_data                                          10          0.012784    0.012784 (1.000000)     0.000000 (0.000000)     0.001091    0.001548    0.0012784   5.53769e-06 
  momentum32                                                         10          0.326691    0.275811 (0.844256)     0.050880 (0.155744)     0.029198    0.039674    0.0326691   0.000141514 
    momentum32/compute                                               10          0.178255    0.127375 (0.714566)     0.050880 (0.285434)     0.016983    0.019099    0.0178255   7.72153e-05 
    momentum32/infer_shape                                           10          0.038951    0.038951 (1.000000)     0.000000 (0.000000)     0.002724    0.011577    0.0038951   1.68725e-05 
    momentum32/prepare_data                                          10          0.014478    0.014478 (1.000000)     0.000000 (0.000000)     0.001203    0.002113    0.0014478   6.27149e-06 
  momentum5                                                          10          0.326349    0.285070 (0.873513)     0.041279 (0.126487)     0.030273    0.040344    0.0326349   0.000141366 
    momentum5/compute                                                10          0.207754    0.166475 (0.801308)     0.041279 (0.198692)     0.01933     0.0288      0.0207754   8.99935e-05 
    momentum5/infer_shape                                            10          0.02852     0.028520 (1.000000)     0.000000 (0.000000)     0.002593    0.003261    0.002852    1.23541e-05 
    momentum5/prepare_data                                           10          0.016095    0.016095 (1.000000)     0.000000 (0.000000)     0.001209    0.003693    0.0016095   6.97193e-06 
  momentum91                                                         10          0.324932    0.291332 (0.896594)     0.033600 (0.103406)     0.027999    0.060286    0.0324932   0.000140752 
    momentum91/compute                                               10          0.164589    0.130989 (0.795855)     0.033600 (0.204145)     0.014859    0.020492    0.0164589   7.12956e-05 
    momentum91/infer_shape                                           10          0.041295    0.041295 (1.000000)     0.000000 (0.000000)     0.003579    0.004883    0.0041295   1.78879e-05 
    momentum91/prepare_data                                          10          0.01579     0.015790 (1.000000)     0.000000 (0.000000)     0.001375    0.001931    0.001579    6.83981e-06 
  momentum54                                                         10          0.324342    0.292950 (0.903213)     0.031392 (0.096787)     0.027667    0.04704     0.0324342   0.000140496 
    momentum54/compute                                               10          0.18534     0.153948 (0.830625)     0.031392 (0.169375)     0.015359    0.034162    0.018534    8.02844e-05 
    momentum54/infer_shape                                           10          0.037908    0.037908 (1.000000)     0.000000 (0.000000)     0.003144    0.00661     0.0037908   1.64207e-05 
    momentum54/prepare_data                                          10          0.009467    0.009467 (1.000000)     0.000000 (0.000000)     0.000769    0.00129     0.0009467   4.10085e-06 
  momentum39                                                         10          0.323928    0.290329 (0.896276)     0.033599 (0.103724)     0.027426    0.066869    0.0323928   0.000140317 
    momentum39/compute                                               10          0.161743    0.128144 (0.792269)     0.033599 (0.207731)     0.015473    0.017255    0.0161743   7.00628e-05 
    momentum39/infer_shape                                           10          0.035895    0.035895 (1.000000)     0.000000 (0.000000)     0.002935    0.006591    0.0035895   1.55488e-05 
    momentum39/prepare_data                                          10          0.011489    0.011489 (1.000000)     0.000000 (0.000000)     0.000912    0.001482    0.0011489   4.97673e-06 
  momentum134                                                        10          0.321136    0.249296 (0.776294)     0.071840 (0.223706)     0.030156    0.034586    0.0321136   0.000139108 
    momentum134/compute                                              10          0.195841    0.124001 (0.633172)     0.071840 (0.366828)     0.01843     0.020415    0.0195841   8.48331e-05 
    momentum134/infer_shape                                          10          0.032171    0.032171 (1.000000)     0.000000 (0.000000)     0.003035    0.003611    0.0032171   1.39356e-05 
    momentum134/prepare_data                                         10          0.012619    0.012619 (1.000000)     0.000000 (0.000000)     0.000923    0.003036    0.0012619   5.46622e-06 
  momentum113                                                        10          0.320979    0.266515 (0.830319)     0.054464 (0.169681)     0.030578    0.034177    0.0320979   0.00013904  
    momentum113/compute                                              10          0.186982    0.132518 (0.708721)     0.054464 (0.291279)     0.017397    0.020465    0.0186982   8.09956e-05 
    momentum113/infer_shape                                          10          0.035219    0.035219 (1.000000)     0.000000 (0.000000)     0.003218    0.003692    0.0035219   1.52559e-05 
    momentum113/prepare_data                                         10          0.010234    0.010234 (1.000000)     0.000000 (0.000000)     0.000824    0.001179    0.0010234   4.4331e-06  
  momentum82                                                         10          0.317343    0.278431 (0.877382)     0.038912 (0.122618)     0.028642    0.048774    0.0317343   0.000137465 
    momentum82/compute                                               10          0.187038    0.148126 (0.791957)     0.038912 (0.208043)     0.016344    0.034521    0.0187038   8.10199e-05 
    momentum82/infer_shape                                           10          0.036917    0.036917 (1.000000)     0.000000 (0.000000)     0.003125    0.006718    0.0036917   1.59915e-05 
    momentum82/prepare_data                                          10          0.014224    0.014224 (1.000000)     0.000000 (0.000000)     0.001258    0.001583    0.0014224   6.16146e-06 
  momentum156                                                        10          0.316777    0.264489 (0.834938)     0.052288 (0.165062)     0.029441    0.040609    0.0316777   0.000137219 
    momentum156/compute                                              10          0.18271     0.130422 (0.713820)     0.052288 (0.286180)     0.017599    0.018867    0.018271    7.91451e-05 
    momentum156/infer_shape                                          10          0.028922    0.028922 (1.000000)     0.000000 (0.000000)     0.002619    0.003292    0.0028922   1.25282e-05 
    momentum156/prepare_data                                         10          0.013222    0.013222 (1.000000)     0.000000 (0.000000)     0.000964    0.002866    0.0013222   5.72742e-06 
  momentum15                                                         10          0.313552    0.270864 (0.863857)     0.042688 (0.136143)     0.028356    0.045728    0.0313552   0.000135822 
    momentum15/compute                                               10          0.187058    0.144370 (0.771793)     0.042688 (0.228207)     0.016162    0.030963    0.0187058   8.10286e-05 
    momentum15/infer_shape                                           10          0.034392    0.034392 (1.000000)     0.000000 (0.000000)     0.002917    0.005448    0.0034392   1.48977e-05 
    momentum15/prepare_data                                          10          0.013839    0.013839 (1.000000)     0.000000 (0.000000)     0.001124    0.001879    0.0013839   5.99469e-06 
  momentum124                                                        10          0.312687    0.274063 (0.876477)     0.038624 (0.123523)     0.027221    0.048811    0.0312687   0.000135448 
    momentum124/compute                                              10          0.189965    0.151341 (0.796678)     0.038624 (0.203322)     0.015717    0.037131    0.0189965   8.22878e-05 
    momentum124/infer_shape                                          10          0.034676    0.034676 (1.000000)     0.000000 (0.000000)     0.003067    0.005526    0.0034676   1.50207e-05 
    momentum124/prepare_data                                         10          0.011909    0.011909 (1.000000)     0.000000 (0.000000)     0.001011    0.001554    0.0011909   5.15866e-06 
  momentum90                                                         10          0.312655    0.267023 (0.854050)     0.045632 (0.145950)     0.029789    0.033986    0.0312655   0.000135434 
    momentum90/compute                                               10          0.18041     0.134778 (0.747065)     0.045632 (0.252935)     0.016466    0.020382    0.018041    7.81488e-05 
    momentum90/infer_shape                                           10          0.036171    0.036171 (1.000000)     0.000000 (0.000000)     0.003359    0.003821    0.0036171   1.56683e-05 
    momentum90/prepare_data                                          10          0.01732     0.017320 (1.000000)     0.000000 (0.000000)     0.00133     0.00381     0.001732    7.50256e-06 
  momentum72                                                         10          0.312585    0.276489 (0.884524)     0.036096 (0.115476)     0.02778     0.043817    0.0312585   0.000135404 
    momentum72/compute                                               10          0.179651    0.143555 (0.799077)     0.036096 (0.200923)     0.015692    0.029069    0.0179651   7.782e-05   
    momentum72/infer_shape                                           10          0.034823    0.034823 (1.000000)     0.000000 (0.000000)     0.003008    0.003846    0.0034823   1.50844e-05 
    momentum72/prepare_data                                          10          0.021397    0.021397 (1.000000)     0.000000 (0.000000)     0.001231    0.00801     0.0021397   9.26861e-06 
  momentum83                                                         10          0.312526    0.276494 (0.884707)     0.036032 (0.115293)     0.02799     0.040772    0.0312526   0.000135378 
    momentum83/compute                                               10          0.174428    0.138396 (0.793428)     0.036032 (0.206572)     0.015427    0.026814    0.0174428   7.55576e-05 
    momentum83/infer_shape                                           10          0.04081     0.040810 (1.000000)     0.000000 (0.000000)     0.002918    0.010436    0.004081    1.76778e-05 
    momentum83/prepare_data                                          10          0.012636    0.012636 (1.000000)     0.000000 (0.000000)     0.001075    0.001485    0.0012636   5.47358e-06 
  momentum29                                                         10          0.311906    0.258114 (0.827538)     0.053792 (0.172462)     0.029417    0.037212    0.0311906   0.000135109 
    momentum29/compute                                               10          0.188336    0.134544 (0.714383)     0.053792 (0.285617)     0.017405    0.02518     0.0188336   8.15822e-05 
    momentum29/infer_shape                                           10          0.031864    0.031864 (1.000000)     0.000000 (0.000000)     0.002978    0.003731    0.0031864   1.38026e-05 
    momentum29/prepare_data                                          10          0.014806    0.014806 (1.000000)     0.000000 (0.000000)     0.001116    0.001796    0.0014806   6.41357e-06 
  momentum12                                                         10          0.311749    0.278693 (0.893966)     0.033056 (0.106034)     0.027402    0.040929    0.0311749   0.000135041 
    momentum12/compute                                               10          0.170927    0.137871 (0.806607)     0.033056 (0.193393)     0.015123    0.026182    0.0170927   7.4041e-05  
    momentum12/infer_shape                                           10          0.026733    0.026733 (1.000000)     0.000000 (0.000000)     0.002507    0.002845    0.0026733   1.158e-05   
    momentum12/prepare_data                                          10          0.02486     0.024860 (1.000000)     0.000000 (0.000000)     0.001286    0.01144     0.002486    1.07687e-05 
  momentum10                                                         10          0.311429    0.268581 (0.862415)     0.042848 (0.137585)     0.02864     0.040326    0.0311429   0.000134903 
    momentum10/compute                                               10          0.179748    0.136900 (0.761622)     0.042848 (0.238378)     0.016114    0.027602    0.0179748   7.78621e-05 
    momentum10/infer_shape                                           10          0.028604    0.028604 (1.000000)     0.000000 (0.000000)     0.002571    0.003313    0.0028604   1.23905e-05 
    momentum10/prepare_data                                          10          0.015463    0.015463 (1.000000)     0.000000 (0.000000)     0.001108    0.003475    0.0015463   6.69816e-06 
  momentum158                                                        10          0.309838    0.279150 (0.900955)     0.030688 (0.099045)     0.026241    0.050074    0.0309838   0.000134214 
    momentum158/compute                                              10          0.190417    0.159729 (0.838838)     0.030688 (0.161162)     0.015258    0.03748     0.0190417   8.24836e-05 
    momentum158/infer_shape                                          10          0.033376    0.033376 (1.000000)     0.000000 (0.000000)     0.002905    0.004108    0.0033376   1.44576e-05 
    momentum158/prepare_data                                         10          0.011311    0.011311 (1.000000)     0.000000 (0.000000)     0.001023    0.001516    0.0011311   4.89963e-06 
  momentum24                                                         10          0.308615    0.273799 (0.887186)     0.034816 (0.112814)     0.027589    0.044125    0.0308615   0.000133684 
    momentum24/compute                                               10          0.174912    0.140096 (0.800951)     0.034816 (0.199049)     0.015654    0.027357    0.0174912   7.57672e-05 
    momentum24/infer_shape                                           10          0.029731    0.029731 (1.000000)     0.000000 (0.000000)     0.002452    0.003429    0.0029731   1.28787e-05 
    momentum24/prepare_data                                          10          0.013337    0.013337 (1.000000)     0.000000 (0.000000)     0.000991    0.001616    0.0013337   5.77723e-06 
  momentum11                                                         10          0.308233    0.277417 (0.900024)     0.030816 (0.099976)     0.028232    0.047341    0.0308233   0.000133518 
    momentum11/compute                                               10          0.182643    0.151827 (0.831277)     0.030816 (0.168723)     0.015972    0.035331    0.0182643   7.91161e-05 
    momentum11/infer_shape                                           10          0.031874    0.031874 (1.000000)     0.000000 (0.000000)     0.003013    0.003518    0.0031874   1.3807e-05  
    momentum11/prepare_data                                          10          0.014771    0.014771 (1.000000)     0.000000 (0.000000)     0.00134     0.001794    0.0014771   6.39841e-06 
  momentum37                                                         10          0.307747    0.276451 (0.898306)     0.031296 (0.101694)     0.02799     0.03905     0.0307747   0.000133308 
    momentum37/compute                                               10          0.158819    0.127523 (0.802945)     0.031296 (0.197055)     0.015458    0.017421    0.0158819   6.87962e-05 
    momentum37/infer_shape                                           10          0.039643    0.039643 (1.000000)     0.000000 (0.000000)     0.003115    0.006013    0.0039643   1.71723e-05 
    momentum37/prepare_data                                          10          0.011437    0.011437 (1.000000)     0.000000 (0.000000)     0.000916    0.001512    0.0011437   4.9542e-06  
  momentum14                                                         10          0.305851    0.252859 (0.826739)     0.052992 (0.173261)     0.029113    0.034483    0.0305851   0.000132487 
    momentum14/compute                                               10          0.180593    0.127601 (0.706567)     0.052992 (0.293433)     0.017413    0.019471    0.0180593   7.82281e-05 
    momentum14/infer_shape                                           10          0.027422    0.027422 (1.000000)     0.000000 (0.000000)     0.002562    0.003006    0.0027422   1.18785e-05 
    momentum14/prepare_data                                          10          0.014063    0.014063 (1.000000)     0.000000 (0.000000)     0.001131    0.001651    0.0014063   6.09172e-06 
  momentum100                                                        10          0.305724    0.264700 (0.865814)     0.041024 (0.134186)     0.028589    0.034614    0.0305724   0.000132432 
    momentum100/compute                                              10          0.171904    0.130880 (0.761355)     0.041024 (0.238645)     0.016326    0.018149    0.0171904   7.44643e-05 
    momentum100/infer_shape                                          10          0.040097    0.040097 (1.000000)     0.000000 (0.000000)     0.003255    0.008201    0.0040097   1.7369e-05  
    momentum100/prepare_data                                         10          0.013731    0.013731 (1.000000)     0.000000 (0.000000)     0.001204    0.001624    0.0013731   5.9479e-06  
  momentum112                                                        10          0.305306    0.265082 (0.868250)     0.040224 (0.131750)     0.027082    0.039458    0.0305306   0.00013225  
    momentum112/compute                                              10          0.175228    0.135004 (0.770448)     0.040224 (0.229552)     0.015917    0.021536    0.0175228   7.59041e-05 
    momentum112/infer_shape                                          10          0.03476     0.034760 (1.000000)     0.000000 (0.000000)     0.003119    0.004126    0.003476    1.50571e-05 
    momentum112/prepare_data                                         10          0.013538    0.013538 (1.000000)     0.000000 (0.000000)     0.001014    0.003177    0.0013538   5.8643e-06  
  momentum19                                                         10          0.304791    0.268727 (0.881676)     0.036064 (0.118324)     0.028022    0.03855     0.0304791   0.000132027 
    momentum19/compute                                               10          0.16387     0.127806 (0.779923)     0.036064 (0.220077)     0.01538     0.017666    0.016387    7.09841e-05 
    momentum19/infer_shape                                           10          0.036737    0.036737 (1.000000)     0.000000 (0.000000)     0.003069    0.006416    0.0036737   1.59135e-05 
    momentum19/prepare_data                                          10          0.013325    0.013325 (1.000000)     0.000000 (0.000000)     0.001013    0.00162     0.0013325   5.77204e-06 
  momentum99                                                         10          0.304398    0.252366 (0.829066)     0.052032 (0.170934)     0.029037    0.033533    0.0304398   0.000131857 
    momentum99/compute                                               10          0.178177    0.126145 (0.707976)     0.052032 (0.292024)     0.017205    0.01885     0.0178177   7.71816e-05 
    momentum99/infer_shape                                           10          0.034234    0.034234 (1.000000)     0.000000 (0.000000)     0.003034    0.005328    0.0034234   1.48293e-05 
    momentum99/prepare_data                                          10          0.011077    0.011077 (1.000000)     0.000000 (0.000000)     0.000969    0.001298    0.0011077   4.79826e-06 
  momentum67                                                         10          0.304293    0.271173 (0.891158)     0.033120 (0.108842)     0.029223    0.032955    0.0304293   0.000131812 
    momentum67/compute                                               10          0.169442    0.136322 (0.804535)     0.033120 (0.195465)     0.015547    0.019385    0.0169442   7.33978e-05 
    momentum67/infer_shape                                           10          0.032289    0.032289 (1.000000)     0.000000 (0.000000)     0.003086    0.003773    0.0032289   1.39867e-05 
    momentum67/prepare_data                                          10          0.01391     0.013910 (1.000000)     0.000000 (0.000000)     0.000993    0.003212    0.001391    6.02544e-06 
  momentum48                                                         10          0.304234    0.267914 (0.880618)     0.036320 (0.119382)     0.027986    0.039332    0.0304234   0.000131786 
    momentum48/compute                                               10          0.168707    0.132387 (0.784716)     0.036320 (0.215284)     0.015916    0.018249    0.0168707   7.30794e-05 
    momentum48/infer_shape                                           10          0.039798    0.039798 (1.000000)     0.000000 (0.000000)     0.002555    0.013616    0.0039798   1.72394e-05 
    momentum48/prepare_data                                          10          0.017825    0.017825 (1.000000)     0.000000 (0.000000)     0.001278    0.004316    0.0017825   7.72132e-06 
  momentum23                                                         10          0.304118    0.264822 (0.870787)     0.039296 (0.129213)     0.028493    0.033739    0.0304118   0.000131736 
    momentum23/compute                                               10          0.168633    0.129337 (0.766973)     0.039296 (0.233027)     0.015732    0.018145    0.0168633   7.30473e-05 
    momentum23/infer_shape                                           10          0.0334      0.033400 (1.000000)     0.000000 (0.000000)     0.002874    0.003809    0.00334     1.4468e-05  
    momentum23/prepare_data                                          10          0.016015    0.016015 (1.000000)     0.000000 (0.000000)     0.00093     0.004068    0.0016015   6.93727e-06 
  momentum108                                                        10          0.30264     0.267888 (0.885170)     0.034752 (0.114830)     0.027727    0.038294    0.030264    0.000131096 
    momentum108/compute                                              10          0.166595    0.131843 (0.791398)     0.034752 (0.208602)     0.015524    0.017897    0.0166595   7.21645e-05 
    momentum108/infer_shape                                          10          0.047739    0.047739 (1.000000)     0.000000 (0.000000)     0.003679    0.013137    0.0047739   2.06793e-05 
    momentum108/prepare_data                                         10          0.013855    0.013855 (1.000000)     0.000000 (0.000000)     0.001184    0.001637    0.0013855   6.00162e-06 
  momentum103                                                        10          0.302085    0.260933 (0.863773)     0.041152 (0.136227)     0.028486    0.032707    0.0302085   0.000130855 
    momentum103/compute                                              10          0.17343     0.132278 (0.762717)     0.041152 (0.237283)     0.016754    0.018025    0.017343    7.51253e-05 
    momentum103/infer_shape                                          10          0.039537    0.039537 (1.000000)     0.000000 (0.000000)     0.003755    0.00434     0.0039537   1.71264e-05 
    momentum103/prepare_data                                         10          0.011372    0.011372 (1.000000)     0.000000 (0.000000)     0.001026    0.001354    0.0011372   4.92605e-06 
  momentum79                                                         10          0.300957    0.263325 (0.874959)     0.037632 (0.125041)     0.028105    0.040647    0.0300957   0.000130367 
    momentum79/compute                                               10          0.173758    0.136126 (0.783423)     0.037632 (0.216577)     0.015557    0.025696    0.0173758   7.52674e-05 
    momentum79/infer_shape                                           10          0.034233    0.034233 (1.000000)     0.000000 (0.000000)     0.002997    0.005456    0.0034233   1.48288e-05 
    momentum79/prepare_data                                          10          0.011967    0.011967 (1.000000)     0.000000 (0.000000)     0.001067    0.001324    0.0011967   5.18379e-06 
  momentum69                                                         10          0.299828    0.259956 (0.867017)     0.039872 (0.132983)     0.027212    0.033527    0.0299828   0.000129878 
    momentum69/compute                                               10          0.173544    0.133672 (0.770248)     0.039872 (0.229752)     0.015696    0.018853    0.0173544   7.51747e-05 
    momentum69/infer_shape                                           10          0.032691    0.032691 (1.000000)     0.000000 (0.000000)     0.002859    0.004064    0.0032691   1.41609e-05 
    momentum69/prepare_data                                          10          0.013739    0.013739 (1.000000)     0.000000 (0.000000)     0.000861    0.004098    0.0013739   5.95137e-06 
  momentum50                                                         10          0.298959    0.265359 (0.887610)     0.033600 (0.112390)     0.027304    0.040586    0.0298959   0.000129501 
    momentum50/compute                                               10          0.166448    0.132848 (0.798135)     0.033600 (0.201865)     0.015248    0.0212      0.0166448   7.21009e-05 
    momentum50/infer_shape                                           10          0.033102    0.033102 (1.000000)     0.000000 (0.000000)     0.002734    0.004929    0.0033102   1.43389e-05 
    momentum50/prepare_data                                          10          0.014688    0.014688 (1.000000)     0.000000 (0.000000)     0.000912    0.003795    0.0014688   6.36245e-06 
  momentum17                                                         10          0.298586    0.261114 (0.874502)     0.037472 (0.125498)     0.027428    0.031323    0.0298586   0.00012934  
    momentum17/compute                                               10          0.168406    0.130934 (0.777490)     0.037472 (0.222510)     0.015167    0.017586    0.0168406   7.2949e-05  
    momentum17/infer_shape                                           10          0.030745    0.030745 (1.000000)     0.000000 (0.000000)     0.002692    0.004806    0.0030745   1.33179e-05 
    momentum17/prepare_data                                          10          0.014862    0.014862 (1.000000)     0.000000 (0.000000)     0.000974    0.00177     0.0014862   6.43782e-06 
  momentum68                                                         10          0.298008    0.258520 (0.867493)     0.039488 (0.132507)     0.027025    0.034405    0.0298008   0.000129089 
    momentum68/compute                                               10          0.176703    0.137215 (0.776529)     0.039488 (0.223471)     0.015369    0.022086    0.0176703   7.65431e-05 
    momentum68/infer_shape                                           10          0.032088    0.032088 (1.000000)     0.000000 (0.000000)     0.002911    0.003529    0.0032088   1.38997e-05 
    momentum68/prepare_data                                          10          0.013573    0.013573 (1.000000)     0.000000 (0.000000)     0.000837    0.003093    0.0013573   5.87946e-06 
  momentum139                                                        10          0.297491    0.262899 (0.883721)     0.034592 (0.116279)     0.026766    0.047471    0.0297491   0.000128865 
    momentum139/compute                                              10          0.165417    0.130825 (0.790880)     0.034592 (0.209120)     0.015258    0.019128    0.0165417   7.16543e-05 
    momentum139/infer_shape                                          10          0.032564    0.032564 (1.000000)     0.000000 (0.000000)     0.0031      0.003443    0.0032564   1.41059e-05 
    momentum139/prepare_data                                         10          0.013728    0.013728 (1.000000)     0.000000 (0.000000)     0.001076    0.001698    0.0013728   5.94661e-06 
  momentum44                                                         10          0.296483    0.264131 (0.890881)     0.032352 (0.109119)     0.02721     0.032635    0.0296483   0.000128429 
    momentum44/compute                                               10          0.1601      0.127748 (0.797926)     0.032352 (0.202074)     0.015233    0.018122    0.01601     6.93511e-05 
    momentum44/infer_shape                                           10          0.030811    0.030811 (1.000000)     0.000000 (0.000000)     0.002856    0.003378    0.0030811   1.33465e-05 
    momentum44/prepare_data                                          10          0.018834    0.018834 (1.000000)     0.000000 (0.000000)     0.001246    0.005588    0.0018834   8.15839e-06 
  momentum89                                                         10          0.295978    0.251786 (0.850692)     0.044192 (0.149308)     0.027763    0.035313    0.0295978   0.00012821  
    momentum89/compute                                               10          0.174057    0.129865 (0.746106)     0.044192 (0.253894)     0.016334    0.021745    0.0174057   7.53969e-05 
    momentum89/infer_shape                                           10          0.034737    0.034737 (1.000000)     0.000000 (0.000000)     0.002917    0.004491    0.0034737   1.50471e-05 
    momentum89/prepare_data                                          10          0.015962    0.015962 (1.000000)     0.000000 (0.000000)     0.001262    0.003452    0.0015962   6.91432e-06 
  momentum4                                                          10          0.295859    0.264627 (0.894436)     0.031232 (0.105564)     0.027377    0.034753    0.0295859   0.000128158 
    momentum4/compute                                                10          0.165458    0.134226 (0.811239)     0.031232 (0.188761)     0.014818    0.021197    0.0165458   7.1672e-05  
    momentum4/infer_shape                                            10          0.028923    0.028923 (1.000000)     0.000000 (0.000000)     0.002728    0.00315     0.0028923   1.25287e-05 
    momentum4/prepare_data                                           10          0.013726    0.013726 (1.000000)     0.000000 (0.000000)     0.001229    0.001512    0.0013726   5.94574e-06 
  momentum21                                                         10          0.295628    0.257292 (0.870324)     0.038336 (0.129676)     0.027658    0.032066    0.0295628   0.000128058 
    momentum21/compute                                               10          0.165027    0.126691 (0.767699)     0.038336 (0.232301)     0.015277    0.01751     0.0165027   7.14853e-05 
    momentum21/infer_shape                                           10          0.030201    0.030201 (1.000000)     0.000000 (0.000000)     0.002727    0.003943    0.0030201   1.30823e-05 
    momentum21/prepare_data                                          10          0.014848    0.014848 (1.000000)     0.000000 (0.000000)     0.000958    0.001634    0.0014848   6.43176e-06 
  momentum73                                                         10          0.29557     0.259986 (0.879609)     0.035584 (0.120391)     0.028573    0.0311      0.029557    0.000128033 
    momentum73/compute                                               10          0.166172    0.130588 (0.785860)     0.035584 (0.214140)     0.015722    0.017499    0.0166172   7.19813e-05 
    momentum73/infer_shape                                           10          0.03403     0.034030 (1.000000)     0.000000 (0.000000)     0.003078    0.003672    0.003403    1.47409e-05 
    momentum73/prepare_data                                          10          0.014578    0.014578 (1.000000)     0.000000 (0.000000)     0.001087    0.001692    0.0014578   6.3148e-06  
  momentum123                                                        10          0.295507    0.247603 (0.837892)     0.047904 (0.162108)     0.027484    0.032098    0.0295507   0.000128006 
    momentum123/compute                                              10          0.174167    0.126263 (0.724954)     0.047904 (0.275046)     0.0162      0.019335    0.0174167   7.54445e-05 
    momentum123/infer_shape                                          10          0.032236    0.032236 (1.000000)     0.000000 (0.000000)     0.002788    0.005202    0.0032236   1.39638e-05 
    momentum123/prepare_data                                         10          0.011847    0.011847 (1.000000)     0.000000 (0.000000)     0.00103     0.001355    0.0011847   5.13181e-06 
  momentum75                                                         10          0.295422    0.262686 (0.889189)     0.032736 (0.110811)     0.028249    0.031681    0.0295422   0.000127969 
    momentum75/compute                                               10          0.163243    0.130507 (0.799465)     0.032736 (0.200535)     0.015776    0.017372    0.0163243   7.07125e-05 
    momentum75/infer_shape                                           10          0.03464     0.034640 (1.000000)     0.000000 (0.000000)     0.003054    0.003741    0.003464    1.50051e-05 
    momentum75/prepare_data                                          10          0.01329     0.013290 (1.000000)     0.000000 (0.000000)     0.001024    0.001756    0.001329    5.75688e-06 
  momentum66                                                         10          0.295354    0.257338 (0.871287)     0.038016 (0.128713)     0.026774    0.033118    0.0295354   0.00012794  
    momentum66/compute                                               10          0.177323    0.139307 (0.785612)     0.038016 (0.214388)     0.015489    0.021337    0.0177323   7.68116e-05 
    momentum66/infer_shape                                           10          0.031555    0.031555 (1.000000)     0.000000 (0.000000)     0.002773    0.003499    0.0031555   1.36688e-05 
    momentum66/prepare_data                                          10          0.01017     0.010170 (1.000000)     0.000000 (0.000000)     0.00081     0.001579    0.001017    4.40537e-06 
  momentum57                                                         10          0.295288    0.260184 (0.881119)     0.035104 (0.118881)     0.027444    0.032932    0.0295288   0.000127911 
    momentum57/compute                                               10          0.162145    0.127041 (0.783502)     0.035104 (0.216498)     0.015297    0.018272    0.0162145   7.02369e-05 
    momentum57/infer_shape                                           10          0.032688    0.032688 (1.000000)     0.000000 (0.000000)     0.002612    0.005065    0.0032688   1.41596e-05 
    momentum57/prepare_data                                          10          0.010742    0.010742 (1.000000)     0.000000 (0.000000)     0.000994    0.001192    0.0010742   4.65315e-06 
  momentum109                                                        10          0.295122    0.262482 (0.889402)     0.032640 (0.110598)     0.027404    0.037123    0.0295122   0.000127839 
    momentum109/compute                                              10          0.1705      0.137860 (0.808563)     0.032640 (0.191437)     0.015383    0.024682    0.01705     7.38561e-05 
    momentum109/infer_shape                                          10          0.031229    0.031229 (1.000000)     0.000000 (0.000000)     0.002786    0.003334    0.0031229   1.35276e-05 
    momentum109/prepare_data                                         10          0.017833    0.017833 (1.000000)     0.000000 (0.000000)     0.00113     0.003563    0.0017833   7.72478e-06 
  momentum70                                                         10          0.294658    0.262146 (0.889662)     0.032512 (0.110338)     0.027934    0.032113    0.0294658   0.000127638 
    momentum70/compute                                               10          0.165962    0.133450 (0.804100)     0.032512 (0.195900)     0.015453    0.017548    0.0165962   7.18903e-05 
    momentum70/infer_shape                                           10          0.032412    0.032412 (1.000000)     0.000000 (0.000000)     0.002808    0.003991    0.0032412   1.404e-05   
    momentum70/prepare_data                                          10          0.013879    0.013879 (1.000000)     0.000000 (0.000000)     0.000934    0.00304     0.0013879   6.01201e-06 
  momentum9                                                          10          0.294611    0.256563 (0.870853)     0.038048 (0.129147)     0.027203    0.031102    0.0294611   0.000127618 
    momentum9/compute                                                10          0.171527    0.133479 (0.778181)     0.038048 (0.221819)     0.015671    0.019022    0.0171527   7.43009e-05 
    momentum9/infer_shape                                            10          0.028846    0.028846 (1.000000)     0.000000 (0.000000)     0.002573    0.003151    0.0028846   1.24953e-05 
    momentum9/prepare_data                                           10          0.014633    0.014633 (1.000000)     0.000000 (0.000000)     0.001241    0.001871    0.0014633   6.33863e-06 
  momentum128                                                        10          0.294125    0.253229 (0.860957)     0.040896 (0.139043)     0.028603    0.030252    0.0294125   0.000127407 
    momentum128/compute                                              10          0.171634    0.130738 (0.761726)     0.040896 (0.238274)     0.016827    0.018014    0.0171634   7.43473e-05 
    momentum128/infer_shape                                          10          0.033657    0.033657 (1.000000)     0.000000 (0.000000)     0.003134    0.003709    0.0033657   1.45793e-05 
    momentum128/prepare_data                                         10          0.011305    0.011305 (1.000000)     0.000000 (0.000000)     0.000985    0.0013      0.0011305   4.89703e-06 
  momentum18                                                         10          0.293703    0.256519 (0.873396)     0.037184 (0.126604)     0.026027    0.044911    0.0293703   0.000127224 
    momentum18/compute                                               10          0.161242    0.124058 (0.769390)     0.037184 (0.230610)     0.015089    0.018247    0.0161242   6.98458e-05 
    momentum18/infer_shape                                           10          0.03286     0.032860 (1.000000)     0.000000 (0.000000)     0.002671    0.005764    0.003286    1.42341e-05 
    momentum18/prepare_data                                          10          0.014261    0.014261 (1.000000)     0.000000 (0.000000)     0.00093     0.002131    0.0014261   6.17749e-06 
  momentum95                                                         10          0.292967    0.262887 (0.897326)     0.030080 (0.102674)     0.027897    0.031485    0.0292967   0.000126906 
    momentum95/compute                                               10          0.160211    0.130131 (0.812248)     0.030080 (0.187752)     0.015465    0.016843    0.0160211   6.93992e-05 
    momentum95/infer_shape                                           10          0.03784     0.037840 (1.000000)     0.000000 (0.000000)     0.003408    0.004186    0.003784    1.63913e-05 
    momentum95/prepare_data                                          10          0.011011    0.011011 (1.000000)     0.000000 (0.000000)     0.000928    0.001199    0.0011011   4.76967e-06 
  momentum115                                                        10          0.292586    0.256458 (0.876522)     0.036128 (0.123478)     0.026767    0.034379    0.0292586   0.000126741 
    momentum115/compute                                              10          0.172189    0.136061 (0.790184)     0.036128 (0.209816)     0.015652    0.023218    0.0172189   7.45877e-05 
    momentum115/infer_shape                                          10          0.033578    0.033578 (1.000000)     0.000000 (0.000000)     0.002929    0.003933    0.0033578   1.45451e-05 
    momentum115/prepare_data                                         10          0.012826    0.012826 (1.000000)     0.000000 (0.000000)     0.001023    0.001563    0.0012826   5.55588e-06 
  momentum140                                                        10          0.292423    0.263143 (0.899871)     0.029280 (0.100129)     0.027487    0.03562     0.0292423   0.00012667  
    momentum140/compute                                              10          0.16321     0.133930 (0.820599)     0.029280 (0.179401)     0.015338    0.017421    0.016321    7.06982e-05 
    momentum140/infer_shape                                          10          0.028869    0.028869 (1.000000)     0.000000 (0.000000)     0.002614    0.003041    0.0028869   1.25053e-05 
    momentum140/prepare_data                                         10          0.009071    0.009071 (1.000000)     0.000000 (0.000000)     0.000811    0.000976    0.0009071   3.92932e-06 
  momentum13                                                         10          0.291703    0.259927 (0.891067)     0.031776 (0.108933)     0.027815    0.031805    0.0291703   0.000126358 
    momentum13/compute                                               10          0.161921    0.130145 (0.803756)     0.031776 (0.196244)     0.015601    0.016579    0.0161921   7.01399e-05 
    momentum13/infer_shape                                           10          0.02989     0.029890 (1.000000)     0.000000 (0.000000)     0.002766    0.003302    0.002989    1.29476e-05 
    momentum13/prepare_data                                          10          0.016385    0.016385 (1.000000)     0.000000 (0.000000)     0.001356    0.002117    0.0016385   7.09755e-06 
  momentum43                                                         10          0.291644    0.260444 (0.893020)     0.031200 (0.106980)     0.028404    0.031786    0.0291644   0.000126332 
    momentum43/compute                                               10          0.161744    0.130544 (0.807103)     0.031200 (0.192897)     0.015545    0.017538    0.0161744   7.00632e-05 
    momentum43/infer_shape                                           10          0.032176    0.032176 (1.000000)     0.000000 (0.000000)     0.002973    0.003472    0.0032176   1.39378e-05 
    momentum43/prepare_data                                          10          0.015806    0.015806 (1.000000)     0.000000 (0.000000)     0.001379    0.001982    0.0015806   6.84674e-06 
  momentum130                                                        10          0.291433    0.255241 (0.875814)     0.036192 (0.124186)     0.026824    0.036747    0.0291433   0.000126241 
    momentum130/compute                                              10          0.169972    0.133780 (0.787071)     0.036192 (0.212929)     0.015125    0.023969    0.0169972   7.36274e-05 
    momentum130/infer_shape                                          10          0.033875    0.033875 (1.000000)     0.000000 (0.000000)     0.003016    0.004061    0.0033875   1.46738e-05 
    momentum130/prepare_data                                         10          0.014604    0.014604 (1.000000)     0.000000 (0.000000)     0.001263    0.001862    0.0014604   6.32607e-06 
  momentum77                                                         10          0.291379    0.253075 (0.868542)     0.038304 (0.131458)     0.027368    0.030981    0.0291379   0.000126218 
    momentum77/compute                                               10          0.166228    0.127924 (0.769570)     0.038304 (0.230430)     0.01546     0.017742    0.0166228   7.20056e-05 
    momentum77/infer_shape                                           10          0.033922    0.033922 (1.000000)     0.000000 (0.000000)     0.002953    0.005229    0.0033922   1.46941e-05 
    momentum77/prepare_data                                          10          0.015032    0.015032 (1.000000)     0.000000 (0.000000)     0.00139     0.001788    0.0015032   6.51146e-06 
  momentum36                                                         10          0.291315    0.259955 (0.892350)     0.031360 (0.107650)     0.027194    0.040445    0.0291315   0.00012619  
    momentum36/compute                                               10          0.156641    0.125281 (0.799797)     0.031360 (0.200203)     0.015348    0.016222    0.0156641   6.78527e-05 
    momentum36/infer_shape                                           10          0.031585    0.031585 (1.000000)     0.000000 (0.000000)     0.002627    0.003592    0.0031585   1.36818e-05 
    momentum36/prepare_data                                          10          0.01261     0.012610 (1.000000)     0.000000 (0.000000)     0.000981    0.001589    0.001261    5.46232e-06 
  momentum59                                                         10          0.29108     0.257768 (0.885557)     0.033312 (0.114443)     0.027828    0.030958    0.029108    0.000126088 
    momentum59/compute                                               10          0.159842    0.126530 (0.791594)     0.033312 (0.208406)     0.015193    0.016996    0.0159842   6.92393e-05 
    momentum59/infer_shape                                           10          0.03152     0.031520 (1.000000)     0.000000 (0.000000)     0.002746    0.005275    0.003152    1.36536e-05 
    momentum59/prepare_data                                          10          0.010644    0.010644 (1.000000)     0.000000 (0.000000)     0.000956    0.001135    0.0010644   4.6107e-06  
  momentum56                                                         10          0.290287    0.254223 (0.875764)     0.036064 (0.124236)     0.027567    0.032564    0.0290287   0.000125745 
    momentum56/compute                                               10          0.164083    0.128019 (0.780209)     0.036064 (0.219791)     0.01565     0.017239    0.0164083   7.10764e-05 
    momentum56/infer_shape                                           10          0.029308    0.029308 (1.000000)     0.000000 (0.000000)     0.002655    0.003427    0.0029308   1.26954e-05 
    momentum56/prepare_data                                          10          0.009297    0.009297 (1.000000)     0.000000 (0.000000)     0.000789    0.001101    0.0009297   4.02721e-06 
  momentum38                                                         10          0.290161    0.260209 (0.896775)     0.029952 (0.103225)     0.02819     0.031138    0.0290161   0.00012569  
    momentum38/compute                                               10          0.160395    0.130443 (0.813261)     0.029952 (0.186739)     0.015307    0.01725     0.0160395   6.94789e-05 
    momentum38/infer_shape                                           10          0.032688    0.032688 (1.000000)     0.000000 (0.000000)     0.002855    0.004908    0.0032688   1.41596e-05 
    momentum38/prepare_data                                          10          0.011701    0.011701 (1.000000)     0.000000 (0.000000)     0.000938    0.00166     0.0011701   5.06856e-06 
  momentum35                                                         10          0.289992    0.259304 (0.894176)     0.030688 (0.105824)     0.027774    0.030526    0.0289992   0.000125617 
    momentum35/compute                                               10          0.159518    0.128830 (0.807620)     0.030688 (0.192380)     0.015353    0.017345    0.0159518   6.9099e-05  
    momentum35/infer_shape                                           10          0.032578    0.032578 (1.000000)     0.000000 (0.000000)     0.002549    0.004723    0.0032578   1.41119e-05 
    momentum35/prepare_data                                          10          0.012656    0.012656 (1.000000)     0.000000 (0.000000)     0.000945    0.001879    0.0012656   5.48224e-06 
  momentum127                                                        10          0.289652    0.253108 (0.873835)     0.036544 (0.126165)     0.026335    0.036868    0.0289652   0.00012547  
    momentum127/compute                                              10          0.169893    0.133349 (0.784900)     0.036544 (0.215100)     0.015309    0.021283    0.0169893   7.35931e-05 
    momentum127/infer_shape                                          10          0.033319    0.033319 (1.000000)     0.000000 (0.000000)     0.003078    0.00373     0.0033319   1.44329e-05 
    momentum127/prepare_data                                         10          0.014283    0.014283 (1.000000)     0.000000 (0.000000)     0.00127     0.001765    0.0014283   6.18702e-06 
  momentum94                                                         10          0.28917     0.257490 (0.890445)     0.031680 (0.109555)     0.027828    0.031777    0.028917    0.000125261 
    momentum94/compute                                               10          0.161857    0.130177 (0.804272)     0.031680 (0.195728)     0.015536    0.017008    0.0161857   7.01122e-05 
    momentum94/infer_shape                                           10          0.033466    0.033466 (1.000000)     0.000000 (0.000000)     0.003111    0.003713    0.0033466   1.44966e-05 
    momentum94/prepare_data                                          10          0.01158     0.011580 (1.000000)     0.000000 (0.000000)     0.000989    0.001313    0.001158    5.01615e-06 
  momentum33                                                         10          0.289102    0.257230 (0.889755)     0.031872 (0.110245)     0.027703    0.03186     0.0289102   0.000125231 
    momentum33/compute                                               10          0.159756    0.127884 (0.800496)     0.031872 (0.199504)     0.015151    0.016964    0.0159756   6.92021e-05 
    momentum33/infer_shape                                           10          0.028708    0.028708 (1.000000)     0.000000 (0.000000)     0.002579    0.003311    0.0028708   1.24355e-05 
    momentum33/prepare_data                                          10          0.014932    0.014932 (1.000000)     0.000000 (0.000000)     0.001335    0.001717    0.0014932   6.46815e-06 
  momentum129                                                        10          0.28873     0.237050 (0.821009)     0.051680 (0.178991)     0.027798    0.030726    0.028873    0.00012507  
    momentum129/compute                                              10          0.177118    0.125438 (0.708217)     0.051680 (0.291783)     0.017312    0.018414    0.0177118   7.67228e-05 
    momentum129/infer_shape                                          10          0.031098    0.031098 (1.000000)     0.000000 (0.000000)     0.002973    0.003335    0.0031098   1.34708e-05 
    momentum129/prepare_data                                         10          0.011383    0.011383 (1.000000)     0.000000 (0.000000)     0.000783    0.002996    0.0011383   4.93081e-06 
  momentum141                                                        10          0.288435    0.257715 (0.893494)     0.030720 (0.106506)     0.027439    0.031231    0.0288435   0.000124942 
    momentum141/compute                                              10          0.158531    0.127811 (0.806221)     0.030720 (0.193779)     0.015361    0.016802    0.0158531   6.86714e-05 
    momentum141/infer_shape                                          10          0.034125    0.034125 (1.000000)     0.000000 (0.000000)     0.002922    0.005733    0.0034125   1.4782e-05  
    momentum141/prepare_data                                         10          0.013445    0.013445 (1.000000)     0.000000 (0.000000)     0.001014    0.001746    0.0013445   5.82402e-06 
  momentum106                                                        10          0.288294    0.253702 (0.880011)     0.034592 (0.119989)     0.027191    0.032151    0.0288294   0.000124881 
    momentum106/compute                                              10          0.162274    0.127682 (0.786830)     0.034592 (0.213170)     0.015129    0.018092    0.0162274   7.02928e-05 
    momentum106/infer_shape                                          10          0.03437     0.034370 (1.000000)     0.000000 (0.000000)     0.003113    0.003788    0.003437    1.48882e-05 
    momentum106/prepare_data                                         10          0.013979    0.013979 (1.000000)     0.000000 (0.000000)     0.001057    0.00329     0.0013979   6.05533e-06 
  momentum101                                                        10          0.288101    0.255909 (0.888261)     0.032192 (0.111739)     0.027038    0.037313    0.0288101   0.000124798 
    momentum101/compute                                              10          0.159602    0.127410 (0.798298)     0.032192 (0.201702)     0.015308    0.016379    0.0159602   6.91354e-05 
    momentum101/infer_shape                                          10          0.042712    0.042712 (1.000000)     0.000000 (0.000000)     0.003007    0.011373    0.0042712   1.85017e-05 
    momentum101/prepare_data                                         10          0.013129    0.013129 (1.000000)     0.000000 (0.000000)     0.00117     0.001464    0.0013129   5.68713e-06 
  momentum114                                                        10          0.287638    0.252118 (0.876511)     0.035520 (0.123489)     0.026589    0.030825    0.0287638   0.000124597 
    momentum114/compute                                              10          0.165072    0.129552 (0.784821)     0.035520 (0.215179)     0.015125    0.017537    0.0165072   7.15048e-05 
    momentum114/infer_shape                                          10          0.031025    0.031025 (1.000000)     0.000000 (0.000000)     0.002874    0.003505    0.0031025   1.34392e-05 
    momentum114/prepare_data                                         10          0.013807    0.013807 (1.000000)     0.000000 (0.000000)     0.001013    0.003469    0.0013807   5.98083e-06 
  momentum137                                                        10          0.287505    0.252881 (0.879571)     0.034624 (0.120429)     0.027641    0.031241    0.0287505   0.00012454  
    momentum137/compute                                              10          0.164232    0.129608 (0.789176)     0.034624 (0.210824)     0.015472    0.017034    0.0164232   7.11409e-05 
    momentum137/infer_shape                                          10          0.030522    0.030522 (1.000000)     0.000000 (0.000000)     0.002546    0.003382    0.0030522   1.32213e-05 
    momentum137/prepare_data                                         10          0.014409    0.014409 (1.000000)     0.000000 (0.000000)     0.00105     0.004123    0.0014409   6.2416e-06  
  momentum64                                                         10          0.287361    0.252641 (0.879176)     0.034720 (0.120824)     0.027411    0.030708    0.0287361   0.000124477 
    momentum64/compute                                               10          0.167705    0.132985 (0.792970)     0.034720 (0.207030)     0.015666    0.019185    0.0167705   7.26454e-05 
    momentum64/infer_shape                                           10          0.029011    0.029011 (1.000000)     0.000000 (0.000000)     0.002674    0.00316     0.0029011   1.25668e-05 
    momentum64/prepare_data                                          10          0.010752    0.010752 (1.000000)     0.000000 (0.000000)     0.00095     0.001345    0.0010752   4.65748e-06 
  momentum152                                                        10          0.287346    0.249714 (0.869036)     0.037632 (0.130964)     0.027423    0.030901    0.0287346   0.000124471 
    momentum152/compute                                              10          0.165607    0.127975 (0.772763)     0.037632 (0.227237)     0.015186    0.018511    0.0165607   7.17366e-05 
    momentum152/infer_shape                                          10          0.031238    0.031238 (1.000000)     0.000000 (0.000000)     0.00277     0.003563    0.0031238   1.35315e-05 
    momentum152/prepare_data                                         10          0.013312    0.013312 (1.000000)     0.000000 (0.000000)     0.001045    0.003331    0.0013312   5.76641e-06 
  momentum148                                                        10          0.287254    0.253078 (0.881025)     0.034176 (0.118975)     0.025973    0.035287    0.0287254   0.000124431 
    momentum148/compute                                              10          0.170295    0.136119 (0.799313)     0.034176 (0.200687)     0.01535     0.023048    0.0170295   7.37673e-05 
    momentum148/infer_shape                                          10          0.035789    0.035789 (1.000000)     0.000000 (0.000000)     0.002919    0.006246    0.0035789   1.55028e-05 
    momentum148/prepare_data                                         10          0.014141    0.014141 (1.000000)     0.000000 (0.000000)     0.001029    0.002363    0.0014141   6.12551e-06 
  momentum104                                                        10          0.287063    0.256215 (0.892539)     0.030848 (0.107461)     0.027394    0.030507    0.0287063   0.000124348 
    momentum104/compute                                              10          0.15968     0.128832 (0.806814)     0.030848 (0.193186)     0.015417    0.016804    0.015968    6.91691e-05 
    momentum104/infer_shape                                          10          0.038536    0.038536 (1.000000)     0.000000 (0.000000)     0.003083    0.005792    0.0038536   1.66928e-05 
    momentum104/prepare_data                                         10          0.014335    0.014335 (1.000000)     0.000000 (0.000000)     0.001293    0.001577    0.0014335   6.20954e-06 
  momentum62                                                         10          0.286558    0.255710 (0.892350)     0.030848 (0.107650)     0.027421    0.030926    0.0286558   0.000124129 
    momentum62/compute                                               10          0.157883    0.127035 (0.804615)     0.030848 (0.195385)     0.015245    0.016427    0.0157883   6.83907e-05 
    momentum62/infer_shape                                           10          0.033681    0.033681 (1.000000)     0.000000 (0.000000)     0.002932    0.005755    0.0033681   1.45897e-05 
    momentum62/prepare_data                                          10          0.010442    0.010442 (1.000000)     0.000000 (0.000000)     0.000949    0.001182    0.0010442   4.5232e-06  
  momentum47                                                         10          0.286429    0.253885 (0.886380)     0.032544 (0.113620)     0.026601    0.030903    0.0286429   0.000124073 
    momentum47/compute                                               10          0.15926     0.126716 (0.795655)     0.032544 (0.204345)     0.014983    0.018474    0.015926    6.89872e-05 
    momentum47/infer_shape                                           10          0.028651    0.028651 (1.000000)     0.000000 (0.000000)     0.002683    0.003253    0.0028651   1.24109e-05 
    momentum47/prepare_data                                          10          0.019962    0.019962 (1.000000)     0.000000 (0.000000)     0.001279    0.003685    0.0019962   8.64701e-06 
  momentum45                                                         10          0.286108    0.255932 (0.894529)     0.030176 (0.105471)     0.026187    0.035012    0.0286108   0.000123934 
    momentum45/compute                                               10          0.155869    0.125693 (0.806402)     0.030176 (0.193598)     0.014534    0.016505    0.0155869   6.75183e-05 
    momentum45/infer_shape                                           10          0.03139     0.031390 (1.000000)     0.000000 (0.000000)     0.002851    0.003515    0.003139    1.35973e-05 
    momentum45/prepare_data                                          10          0.010827    0.010827 (1.000000)     0.000000 (0.000000)     0.000711    0.003161    0.0010827   4.68997e-06 
  momentum92                                                         10          0.285876    0.252660 (0.883810)     0.033216 (0.116190)     0.027309    0.031149    0.0285876   0.000123834 
    momentum92/compute                                               10          0.161886    0.128670 (0.794819)     0.033216 (0.205181)     0.015363    0.017316    0.0161886   7.01247e-05 
    momentum92/infer_shape                                           10          0.03365     0.033650 (1.000000)     0.000000 (0.000000)     0.003117    0.003712    0.003365    1.45763e-05 
    momentum92/prepare_data                                          10          0.014266    0.014266 (1.000000)     0.000000 (0.000000)     0.001189    0.003114    0.0014266   6.17965e-06 
  momentum133                                                        10          0.285843    0.248819 (0.870474)     0.037024 (0.129526)     0.026322    0.031084    0.0285843   0.00012382  
    momentum133/compute                                              10          0.166031    0.129007 (0.777005)     0.037024 (0.222995)     0.015421    0.01924     0.0166031   7.19202e-05 
    momentum133/infer_shape                                          10          0.033472    0.033472 (1.000000)     0.000000 (0.000000)     0.003139    0.004077    0.0033472   1.44992e-05 
    momentum133/prepare_data                                         10          0.015169    0.015169 (1.000000)     0.000000 (0.000000)     0.001185    0.003463    0.0015169   6.57081e-06 
  momentum26                                                         10          0.285815    0.249399 (0.872589)     0.036416 (0.127411)     0.027077    0.030244    0.0285815   0.000123807 
    momentum26/compute                                               10          0.165238    0.128822 (0.779615)     0.036416 (0.220385)     0.015224    0.018275    0.0165238   7.15767e-05 
    momentum26/infer_shape                                           10          0.029162    0.029162 (1.000000)     0.000000 (0.000000)     0.002765    0.003155    0.0029162   1.26322e-05 
    momentum26/prepare_data                                          10          0.01862     0.018620 (1.000000)     0.000000 (0.000000)     0.001149    0.003599    0.001862    8.06569e-06 
  momentum159                                                        10          0.285683    0.247059 (0.864801)     0.038624 (0.135199)     0.026577    0.031626    0.0285683   0.00012375  
    momentum159/compute                                              10          0.169727    0.131103 (0.772435)     0.038624 (0.227565)     0.01566     0.020228    0.0169727   7.35212e-05 
    momentum159/infer_shape                                          10          0.031638    0.031638 (1.000000)     0.000000 (0.000000)     0.002894    0.003598    0.0031638   1.37047e-05 
    momentum159/prepare_data                                         10          0.009686    0.009686 (1.000000)     0.000000 (0.000000)     0.000869    0.001311    0.0009686   4.19572e-06 
  momentum157                                                        10          0.285131    0.249451 (0.874865)     0.035680 (0.125135)     0.026646    0.030236    0.0285131   0.000123511 
    momentum157/compute                                              10          0.162222    0.126542 (0.780054)     0.035680 (0.219946)     0.015146    0.017763    0.0162222   7.02703e-05 
    momentum157/infer_shape                                          10          0.032808    0.032808 (1.000000)     0.000000 (0.000000)     0.003139    0.003417    0.0032808   1.42116e-05 
    momentum157/prepare_data                                         10          0.012423    0.012423 (1.000000)     0.000000 (0.000000)     0.001073    0.00138     0.0012423   5.38131e-06 
  momentum96                                                         10          0.284884    0.253428 (0.889583)     0.031456 (0.110417)     0.026633    0.030359    0.0284884   0.000123404 
    momentum96/compute                                               10          0.159585    0.128129 (0.802889)     0.031456 (0.197111)     0.014982    0.017259    0.0159585   6.9128e-05  
    momentum96/infer_shape                                           10          0.035413    0.035413 (1.000000)     0.000000 (0.000000)     0.003189    0.004197    0.0035413   1.534e-05   
    momentum96/prepare_data                                          10          0.012629    0.012629 (1.000000)     0.000000 (0.000000)     0.001048    0.001798    0.0012629   5.47055e-06 
  momentum120                                                        10          0.284778    0.248234 (0.871675)     0.036544 (0.128325)     0.027626    0.029798    0.0284778   0.000123358 
    momentum120/compute                                              10          0.164459    0.127915 (0.777793)     0.036544 (0.222207)     0.015532    0.017846    0.0164459   7.12393e-05 
    momentum120/infer_shape                                          10          0.030942    0.030942 (1.000000)     0.000000 (0.000000)     0.002724    0.003335    0.0030942   1.34033e-05 
    momentum120/prepare_data                                         10          0.015012    0.015012 (1.000000)     0.000000 (0.000000)     0.001256    0.001678    0.0015012   6.5028e-06  
  momentum153                                                        10          0.284729    0.247065 (0.867720)     0.037664 (0.132280)     0.027027    0.030825    0.0284729   0.000123337 
    momentum153/compute                                              10          0.167045    0.129381 (0.774528)     0.037664 (0.225472)     0.015525    0.018424    0.0167045   7.23595e-05 
    momentum153/infer_shape                                          10          0.030443    0.030443 (1.000000)     0.000000 (0.000000)     0.002852    0.003587    0.0030443   1.31871e-05 
    momentum153/prepare_data                                         10          0.015084    0.015084 (1.000000)     0.000000 (0.000000)     0.001214    0.003063    0.0015084   6.53399e-06 
  momentum85                                                         10          0.284527    0.251023 (0.882247)     0.033504 (0.117753)     0.027323    0.03011     0.0284527   0.00012325  
    momentum85/compute                                               10          0.161746    0.128242 (0.792860)     0.033504 (0.207140)     0.015448    0.017948    0.0161746   7.00641e-05 
    momentum85/infer_shape                                           10          0.032179    0.032179 (1.000000)     0.000000 (0.000000)     0.003004    0.003575    0.0032179   1.39391e-05 
    momentum85/prepare_data                                          10          0.014665    0.014665 (1.000000)     0.000000 (0.000000)     0.001149    0.003275    0.0014665   6.35249e-06 
  momentum30                                                         10          0.284358    0.249830 (0.878576)     0.034528 (0.121424)     0.027065    0.030224    0.0284358   0.000123176 
    momentum30/compute                                               10          0.162243    0.127715 (0.787183)     0.034528 (0.212817)     0.015346    0.017412    0.0162243   7.02794e-05 
    momentum30/infer_shape                                           10          0.030058    0.030058 (1.000000)     0.000000 (0.000000)     0.002684    0.003708    0.0030058   1.30203e-05 
    momentum30/prepare_data                                          10          0.017391    0.017391 (1.000000)     0.000000 (0.000000)     0.000997    0.003959    0.0017391   7.53332e-06 
  momentum60                                                         10          0.283758    0.250766 (0.883732)     0.032992 (0.116268)     0.026869    0.031235    0.0283758   0.000122916 
    momentum60/compute                                               10          0.159809    0.126817 (0.793554)     0.032992 (0.206446)     0.015302    0.017059    0.0159809   6.9225e-05  
    momentum60/infer_shape                                           10          0.028778    0.028778 (1.000000)     0.000000 (0.000000)     0.002482    0.004666    0.0028778   1.24659e-05 
    momentum60/prepare_data                                          10          0.011013    0.011013 (1.000000)     0.000000 (0.000000)     0.000952    0.001296    0.0011013   4.77054e-06 
  momentum151                                                        10          0.283635    0.248787 (0.877138)     0.034848 (0.122862)     0.026026    0.032002    0.0283635   0.000122863 
    momentum151/compute                                              10          0.162126    0.127278 (0.785056)     0.034848 (0.214944)     0.015238    0.017294    0.0162126   7.02287e-05 
    momentum151/infer_shape                                          10          0.02973     0.029730 (1.000000)     0.000000 (0.000000)     0.002786    0.003268    0.002973    1.28782e-05 
    momentum151/prepare_data                                         10          0.015163    0.015163 (1.000000)     0.000000 (0.000000)     0.001123    0.003575    0.0015163   6.56821e-06 
  momentum116                                                        10          0.282711    0.251767 (0.890545)     0.030944 (0.109455)     0.025499    0.030159    0.0282711   0.000122463 
    momentum116/compute                                              10          0.160234    0.129290 (0.806882)     0.030944 (0.193118)     0.014888    0.016853    0.0160234   6.94091e-05 
    momentum116/infer_shape                                          10          0.034273    0.034273 (1.000000)     0.000000 (0.000000)     0.002948    0.003822    0.0034273   1.48462e-05 
    momentum116/prepare_data                                         10          0.00986     0.009860 (1.000000)     0.000000 (0.000000)     0.000813    0.001253    0.000986    4.27109e-06 
  momentum74                                                         10          0.282591    0.250207 (0.885403)     0.032384 (0.114597)     0.026671    0.031834    0.0282591   0.000122411 
    momentum74/compute                                               10          0.161596    0.129212 (0.799599)     0.032384 (0.200401)     0.014889    0.017844    0.0161596   6.99991e-05 
    momentum74/infer_shape                                           10          0.032801    0.032801 (1.000000)     0.000000 (0.000000)     0.002928    0.003863    0.0032801   1.42085e-05 
    momentum74/prepare_data                                          10          0.013917    0.013917 (1.000000)     0.000000 (0.000000)     0.001019    0.001903    0.0013917   6.02848e-06 
  momentum110                                                        10          0.282494    0.247934 (0.877661)     0.034560 (0.122339)     0.027172    0.029806    0.0282494   0.000122369 
    momentum110/compute                                              10          0.163685    0.129125 (0.788863)     0.034560 (0.211137)     0.015592    0.017705    0.0163685   7.0904e-05  
    momentum110/infer_shape                                          10          0.032469    0.032469 (1.000000)     0.000000 (0.000000)     0.002935    0.003748    0.0032469   1.40647e-05 
    momentum110/prepare_data                                         10          0.013481    0.013481 (1.000000)     0.000000 (0.000000)     0.000994    0.003286    0.0013481   5.83961e-06 
  momentum58                                                         10          0.282308    0.250277 (0.886539)     0.032031 (0.113461)     0.025641    0.037559    0.0282308   0.000122288 
    momentum58/compute                                               10          0.16619     0.134159 (0.807263)     0.032031 (0.192737)     0.014545    0.025819    0.016619    7.19891e-05 
    momentum58/infer_shape                                           10          0.033105    0.033105 (1.000000)     0.000000 (0.000000)     0.002821    0.005163    0.0033105   1.43402e-05 
    momentum58/prepare_data                                          10          0.008949    0.008949 (1.000000)     0.000000 (0.000000)     0.000785    0.001095    0.0008949   3.87647e-06 
  momentum6                                                          10          0.282206    0.249758 (0.885020)     0.032448 (0.114980)     0.026216    0.033273    0.0282206   0.000122244 
    momentum6/compute                                                10          0.16423     0.131782 (0.802423)     0.032448 (0.197577)     0.014908    0.019288    0.016423    7.11401e-05 
    momentum6/infer_shape                                            10          0.025686    0.025686 (1.000000)     0.000000 (0.000000)     0.00227     0.002937    0.0025686   1.11265e-05 
    momentum6/prepare_data                                           10          0.018825    0.018825 (1.000000)     0.000000 (0.000000)     0.00121     0.00682     0.0018825   8.15449e-06 
  momentum136                                                        10          0.2821      0.248372 (0.880440)     0.033728 (0.119560)     0.02639     0.030173    0.02821     0.000122198 
    momentum136/compute                                              10          0.160513    0.126785 (0.789874)     0.033728 (0.210126)     0.015279    0.017191    0.0160513   6.953e-05   
    momentum136/infer_shape                                          10          0.031945    0.031945 (1.000000)     0.000000 (0.000000)     0.002998    0.003417    0.0031945   1.38377e-05 
    momentum136/prepare_data                                         10          0.013788    0.013788 (1.000000)     0.000000 (0.000000)     0.001027    0.003678    0.0013788   5.9726e-06  
  momentum135                                                        10          0.282086    0.246438 (0.873627)     0.035648 (0.126373)     0.026741    0.030911    0.0282086   0.000122192 
    momentum135/compute                                              10          0.165694    0.130046 (0.784856)     0.035648 (0.215144)     0.015407    0.018198    0.0165694   7.17742e-05 
    momentum135/infer_shape                                          10          0.03541     0.035410 (1.000000)     0.000000 (0.000000)     0.003357    0.003734    0.003541    1.53387e-05 
    momentum135/prepare_data                                         10          0.011215    0.011215 (1.000000)     0.000000 (0.000000)     0.001006    0.001423    0.0011215   4.85804e-06 
  momentum149                                                        10          0.281977    0.251865 (0.893211)     0.030112 (0.106789)     0.026115    0.029906    0.0281977   0.000122145 
    momentum149/compute                                              10          0.159808    0.129696 (0.811574)     0.030112 (0.188426)     0.015086    0.017134    0.0159808   6.92246e-05 
    momentum149/infer_shape                                          10          0.028621    0.028621 (1.000000)     0.000000 (0.000000)     0.002624    0.003193    0.0028621   1.23979e-05 
    momentum149/prepare_data                                         10          0.011785    0.011785 (1.000000)     0.000000 (0.000000)     0.000831    0.002919    0.0011785   5.10495e-06 
  momentum155                                                        10          0.2818      0.247272 (0.877473)     0.034528 (0.122527)     0.026173    0.030426    0.02818     0.000122068 
    momentum155/compute                                              10          0.166621    0.132093 (0.792775)     0.034528 (0.207225)     0.015448    0.019163    0.0166621   7.21758e-05 
    momentum155/infer_shape                                          10          0.032331    0.032331 (1.000000)     0.000000 (0.000000)     0.003018    0.003474    0.0032331   1.40049e-05 
    momentum155/prepare_data                                         10          0.011587    0.011587 (1.000000)     0.000000 (0.000000)     0.001001    0.001526    0.0011587   5.01918e-06 
  momentum102                                                        10          0.281532    0.249820 (0.887359)     0.031712 (0.112641)     0.027092    0.030541    0.0281532   0.000121952 
    momentum102/compute                                              10          0.161638    0.129926 (0.803809)     0.031712 (0.196191)     0.01561     0.017103    0.0161638   7.00173e-05 
    momentum102/infer_shape                                          10          0.033401    0.033401 (1.000000)     0.000000 (0.000000)     0.002911    0.005218    0.0033401   1.44684e-05 
    momentum102/prepare_data                                         10          0.013648    0.013648 (1.000000)     0.000000 (0.000000)     0.00123     0.001585    0.0013648   5.91195e-06 
  momentum93                                                         10          0.280475    0.249339 (0.888988)     0.031136 (0.111012)     0.026919    0.029798    0.0280475   0.000121494 
    momentum93/compute                                               10          0.160664    0.129528 (0.806204)     0.031136 (0.193796)     0.015222    0.017882    0.0160664   6.95954e-05 
    momentum93/infer_shape                                           10          0.035203    0.035203 (1.000000)     0.000000 (0.000000)     0.003157    0.003745    0.0035203   1.5249e-05  
    momentum93/prepare_data                                          10          0.014295    0.014295 (1.000000)     0.000000 (0.000000)     0.001213    0.001663    0.0014295   6.19221e-06 
  momentum150                                                        10          0.280021    0.248437 (0.887208)     0.031584 (0.112792)     0.026727    0.029722    0.0280021   0.000121298 
    momentum150/compute                                              10          0.161795    0.130211 (0.804790)     0.031584 (0.195210)     0.015319    0.018066    0.0161795   7.00853e-05 
    momentum150/infer_shape                                          10          0.031783    0.031783 (1.000000)     0.000000 (0.000000)     0.002943    0.003602    0.0031783   1.37676e-05 
    momentum150/prepare_data                                         10          0.011314    0.011314 (1.000000)     0.000000 (0.000000)     0.001005    0.001566    0.0011314   4.90092e-06 
  momentum118                                                        10          0.279706    0.243130 (0.869234)     0.036576 (0.130766)     0.025958    0.030728    0.0279706   0.000121161 
    momentum118/compute                                              10          0.164599    0.128023 (0.777787)     0.036576 (0.222213)     0.015259    0.017666    0.0164599   7.12999e-05 
    momentum118/infer_shape                                          10          0.038919    0.038919 (1.000000)     0.000000 (0.000000)     0.003381    0.005589    0.0038919   1.68587e-05 
    momentum118/prepare_data                                         10          0.011487    0.011487 (1.000000)     0.000000 (0.000000)     0.000998    0.001495    0.0011487   4.97586e-06 
  momentum31                                                         10          0.27967     0.247414 (0.884664)     0.032256 (0.115336)     0.026582    0.029999    0.027967    0.000121146 
    momentum31/compute                                               10          0.161288    0.129032 (0.800010)     0.032256 (0.199990)     0.015197    0.017812    0.0161288   6.98657e-05 
    momentum31/infer_shape                                           10          0.03014     0.030140 (1.000000)     0.000000 (0.000000)     0.002661    0.003392    0.003014    1.30558e-05 
    momentum31/prepare_data                                          10          0.015407    0.015407 (1.000000)     0.000000 (0.000000)     0.001103    0.001757    0.0015407   6.6739e-06  
  momentum98                                                         10          0.279582    0.248222 (0.887833)     0.031360 (0.112167)     0.026713    0.030974    0.0279582   0.000121108 
    momentum98/compute                                               10          0.157618    0.126258 (0.801038)     0.031360 (0.198962)     0.015066    0.017005    0.0157618   6.82759e-05 
    momentum98/infer_shape                                           10          0.032292    0.032292 (1.000000)     0.000000 (0.000000)     0.003026    0.003417    0.0032292   1.3988e-05  
    momentum98/prepare_data                                          10          0.012087    0.012087 (1.000000)     0.000000 (0.000000)     0.001049    0.001638    0.0012087   5.23577e-06 
  momentum121                                                        10          0.279355    0.244955 (0.876859)     0.034400 (0.123141)     0.025815    0.030213    0.0279355   0.000121009 
    momentum121/compute                                              10          0.161463    0.127063 (0.786948)     0.034400 (0.213052)     0.015142    0.016808    0.0161463   6.99415e-05 
    momentum121/infer_shape                                          10          0.034142    0.034142 (1.000000)     0.000000 (0.000000)     0.002753    0.00513     0.0034142   1.47894e-05 
    momentum121/prepare_data                                         10          0.011096    0.011096 (1.000000)     0.000000 (0.000000)     0.000975    0.001176    0.0011096   4.80649e-06 
  momentum61                                                         10          0.278839    0.244631 (0.877320)     0.034208 (0.122680)     0.02702     0.029684    0.0278839   0.000120786 
    momentum61/compute                                               10          0.160543    0.126335 (0.786923)     0.034208 (0.213077)     0.015288    0.016443    0.0160543   6.9543e-05  
    momentum61/infer_shape                                           10          0.026983    0.026983 (1.000000)     0.000000 (0.000000)     0.002453    0.002861    0.0026983   1.16883e-05 
    momentum61/prepare_data                                          10          0.010946    0.010946 (1.000000)     0.000000 (0.000000)     0.000969    0.001479    0.0010946   4.74152e-06 
  momentum34                                                         10          0.27713     0.243594 (0.878988)     0.033536 (0.121012)     0.026379    0.029965    0.027713    0.000120045 
    momentum34/compute                                               10          0.158178    0.124642 (0.787986)     0.033536 (0.212014)     0.015034    0.016445    0.0158178   6.85185e-05 
    momentum34/infer_shape                                           10          0.029495    0.029495 (1.000000)     0.000000 (0.000000)     0.00282     0.003367    0.0029495   1.27765e-05 
    momentum34/prepare_data                                          10          0.015069    0.015069 (1.000000)     0.000000 (0.000000)     0.001252    0.001982    0.0015069   6.52749e-06 
  momentum143                                                        10          0.27663     0.246646 (0.891610)     0.029984 (0.108390)     0.026688    0.029653    0.027663    0.000119829 
    momentum143/compute                                              10          0.1599      0.129916 (0.812483)     0.029984 (0.187517)     0.015251    0.016724    0.01599     6.92644e-05 
    momentum143/infer_shape                                          10          0.031099    0.031099 (1.000000)     0.000000 (0.000000)     0.00257     0.004776    0.0031099   1.34713e-05 
    momentum143/prepare_data                                         10          0.009454    0.009454 (1.000000)     0.000000 (0.000000)     0.000808    0.001155    0.0009454   4.09522e-06 
  momentum138                                                        10          0.275649    0.243809 (0.884491)     0.031840 (0.115509)     0.025572    0.029947    0.0275649   0.000119404 
    momentum138/compute                                              10          0.157036    0.125196 (0.797244)     0.031840 (0.202756)     0.015181    0.016714    0.0157036   6.80238e-05 
    momentum138/infer_shape                                          10          0.032587    0.032587 (1.000000)     0.000000 (0.000000)     0.00294     0.00363     0.0032587   1.41158e-05 
    momentum138/prepare_data                                         10          0.010623    0.010623 (1.000000)     0.000000 (0.000000)     0.000837    0.001297    0.0010623   4.6016e-06  
  momentum105                                                        10          0.275213    0.243757 (0.885703)     0.031456 (0.114297)     0.026797    0.03008     0.0275213   0.000119215 
    momentum105/compute                                              10          0.156648    0.125192 (0.799193)     0.031456 (0.200807)     0.015092    0.016308    0.0156648   6.78558e-05 
    momentum105/infer_shape                                          10          0.032549    0.032549 (1.000000)     0.000000 (0.000000)     0.00295     0.003627    0.0032549   1.40994e-05 
    momentum105/prepare_data                                         10          0.01156     0.011560 (1.000000)     0.000000 (0.000000)     0.001085    0.001279    0.001156    5.00749e-06 
  momentum147                                                        10          0.272178    0.237554 (0.872789)     0.034624 (0.127211)     0.026247    0.02906     0.0272178   0.0001179   
    momentum147/compute                                              10          0.163064    0.128440 (0.787666)     0.034624 (0.212334)     0.015767    0.017256    0.0163064   7.0635e-05  
    momentum147/infer_shape                                          10          0.029559    0.029559 (1.000000)     0.000000 (0.000000)     0.002768    0.00327     0.0029559   1.28042e-05 
    momentum147/prepare_data                                         10          0.011527    0.011527 (1.000000)     0.000000 (0.000000)     0.001014    0.001434    0.0011527   4.99319e-06 
  momentum144                                                        10          0.270113    0.239393 (0.886270)     0.030720 (0.113730)     0.025865    0.029209    0.0270113   0.000117006 
    momentum144/compute                                              10          0.15688     0.126160 (0.804182)     0.030720 (0.195818)     0.01506     0.016396    0.015688    6.79563e-05 
    momentum144/infer_shape                                          10          0.033396    0.033396 (1.000000)     0.000000 (0.000000)     0.002756    0.005188    0.0033396   1.44663e-05 
    momentum144/prepare_data                                         10          0.010676    0.010676 (1.000000)     0.000000 (0.000000)     0.000927    0.00129     0.0010676   4.62456e-06 
  momentum154                                                        10          0.269124    0.238884 (0.887635)     0.030240 (0.112365)     0.025652    0.030483    0.0269124   0.000116577 
    momentum154/compute                                              10          0.160353    0.130113 (0.811416)     0.030240 (0.188584)     0.015176    0.017961    0.0160353   6.94607e-05 
    momentum154/infer_shape                                          10          0.029997    0.029997 (1.000000)     0.000000 (0.000000)     0.00258     0.003348    0.0029997   1.29939e-05 
    momentum154/prepare_data                                         10          0.014172    0.014172 (1.000000)     0.000000 (0.000000)     0.001027    0.004275    0.0014172   6.13893e-06 
  momentum160                                                        10          0.267401    0.237577 (0.888467)     0.029824 (0.111533)     0.02563     0.029045    0.0267401   0.000115831 
    momentum160/compute                                              10          0.159398    0.129574 (0.812896)     0.029824 (0.187104)     0.015416    0.016302    0.0159398   6.9047e-05  
    momentum160/infer_shape                                          10          0.028372    0.028372 (1.000000)     0.000000 (0.000000)     0.002533    0.003014    0.0028372   1.229e-05   
    momentum160/prepare_data                                         10          0.01127     0.011270 (1.000000)     0.000000 (0.000000)     0.000873    0.001476    0.001127    4.88187e-06 
  momentum142                                                        10          0.263861    0.232949 (0.882847)     0.030912 (0.117153)     0.025279    0.02916     0.0263861   0.000114298 
    momentum142/compute                                              10          0.154646    0.123734 (0.800111)     0.030912 (0.199889)     0.015063    0.016309    0.0154646   6.69885e-05 
    momentum142/infer_shape                                          10          0.030386    0.030386 (1.000000)     0.000000 (0.000000)     0.002812    0.003245    0.0030386   1.31624e-05 
    momentum142/prepare_data                                         10          0.012086    0.012086 (1.000000)     0.000000 (0.000000)     0.001031    0.001566    0.0012086   5.23533e-06 
thread1::DropLocalExeScopes                                          1           22.8683     22.868292 (1.000000)    0.000000 (0.000000)     22.8683     22.8683     22.8683     0.00990594  
thread1::pool2d_grad                                                 20          8.72527     1.641435 (0.188124)     7.083835 (0.811876)     0.139231    0.75109     0.436264    0.00377956  
  pool2d_grad1                                                       10          7.22537     0.961248 (0.133038)     6.264126 (0.866962)     0.710751    0.749318    0.722537    0.00312984  
    pool2d_grad1/compute                                             10          7.10141     0.837280 (0.117903)     6.264126 (0.882097)     0.698298    0.737182    0.710141    0.00307614  
    pool2d_grad1/infer_shape                                         10          0.047446    0.047446 (1.000000)     0.000000 (0.000000)     0.004389    0.00523     0.0047446   2.05523e-05 
    pool2d_grad1/prepare_data                                        10          0.007863    0.007863 (1.000000)     0.000000 (0.000000)     0.000695    0.000906    0.0007863   3.40604e-06 
  pool2d_grad0                                                       10          1.46289     0.643177 (0.439663)     0.819709 (0.560337)     0.137393    0.165609    0.146289    0.000633683 
    pool2d_grad0/compute                                             10          1.33137     0.511665 (0.384313)     0.819709 (0.615687)     0.127089    0.154036    0.133137    0.000576716 
    pool2d_grad0/infer_shape                                         10          0.033232    0.033232 (1.000000)     0.000000 (0.000000)     0.002607    0.005377    0.0033232   1.43952e-05 
    pool2d_grad0/prepare_data                                        10          0.028431    0.028431 (1.000000)     0.000000 (0.000000)     0.001172    0.01318     0.0028431   1.23156e-05 
thread1::eager_deletion                                              1810        8.01594     8.015944 (1.000000)     0.000000 (0.000000)     0.000719    0.331789    0.0044287   0.00347229  
thread1::pool2d                                                      20          5.62698     2.668161 (0.474173)     2.958815 (0.525827)     0.158932    0.492713    0.281349    0.00243746  
  pool2d0                                                            10          3.36047     0.839323 (0.249763)     2.521151 (0.750237)     0.325205    0.36992     0.336047    0.00145567  
    pool2d0/compute                                                  10          3.02301     0.501861 (0.166014)     2.521151 (0.833986)     0.295365    0.31441     0.302301    0.00130949  
    pool2d0/infer_shape                                              10          0.234195    0.234195 (1.000000)     0.000000 (0.000000)     0.019404    0.04381     0.0234195   0.000101447 
    pool2d0/prepare_data                                             10          0.009753    0.009753 (1.000000)     0.000000 (0.000000)     0.000895    0.00108     0.0009753   4.22474e-06 
  pool2d1                                                            10          2.16973     1.732061 (0.798286)     0.437664 (0.201714)     0.153895    0.487722    0.216973    0.000939867 
    pool2d1/compute                                                  10          1.57262     1.134954 (0.721697)     0.437664 (0.278303)     0.103511    0.418838    0.157262    0.000681216 
    pool2d1/infer_shape                                              10          0.311116    0.311116 (1.000000)     0.000000 (0.000000)     0.026703    0.037646    0.0311116   0.000134767 
    pool2d1/prepare_data                                             10          0.029201    0.029201 (1.000000)     0.000000 (0.000000)     0.001315    0.00706     0.0029201   1.26491e-05 
thread1::matmul_v2_grad                                              10          1.96888     1.436527 (0.729617)     0.532352 (0.270383)     0.190886    0.200177    0.196888    0.000852866 
  matmul_v2_grad0                                                    10          1.93755     1.405194 (0.725244)     0.532352 (0.274756)     0.187026    0.197143    0.193755    0.000839294 
    matmul_v2_grad0/compute                                          10          1.77474     1.242387 (0.700039)     0.532352 (0.299961)     0.172564    0.181941    0.177474    0.00076877  
    matmul_v2_grad0/infer_shape                                      10          0.056007    0.056007 (1.000000)     0.000000 (0.000000)     0.004695    0.00817     0.0056007   2.42607e-05 
    matmul_v2_grad0/prepare_data                                     10          0.014146    0.014146 (1.000000)     0.000000 (0.000000)     0.000958    0.001863    0.0014146   6.12767e-06 
thread1::matmul_v2                                                   10          1.74352     1.435523 (0.823346)     0.308000 (0.176654)     0.162007    0.20274     0.174352    0.000755248 
  matmul_v20                                                         10          1.71048     1.402475 (0.819933)     0.308000 (0.180067)     0.159413    0.200219    0.171048    0.000740932 
    matmul_v20/compute                                               10          1.52552     1.217520 (0.798102)     0.308000 (0.201898)     0.134283    0.184443    0.152552    0.000660815 
    matmul_v20/infer_shape                                           10          0.076505    0.076505 (1.000000)     0.000000 (0.000000)     0.005962    0.012789    0.0076505   3.31399e-05 
    matmul_v20/prepare_data                                          10          0.009062    0.009062 (1.000000)     0.000000 (0.000000)     0.000834    0.00104     0.0009062   3.92542e-06 
thread1::flatten_contiguous_range                                    10          1.628       1.584190 (0.973091)     0.043808 (0.026909)     0.074235    0.472363    0.1628      0.000705206 
  flatten_contiguous_range0                                          10          1.5969      1.553088 (0.972567)     0.043808 (0.027433)     0.071947    0.470031    0.15969     0.000691733 
    flatten_contiguous_range0/compute                                10          1.4345      1.390690 (0.969461)     0.043808 (0.030539)     0.057727    0.455247    0.14345     0.000621387 
      GpuMemcpyAsync(same_gpu):GPU->GPU                              10          0.381014    0.337206 (0.885023)     0.043808 (0.114977)     0.031292    0.048694    0.0381014   0.000165045 
    flatten_contiguous_range0/infer_shape                            10          0.061895    0.061895 (1.000000)     0.000000 (0.000000)     0.005718    0.006542    0.0061895   2.68113e-05 
    flatten_contiguous_range0/prepare_data                           10          0.024705    0.024705 (1.000000)     0.000000 (0.000000)     0.001172    0.010523    0.0024705   1.07016e-05 
thread1::accuracy                                                    20          1.29942     1.116606 (0.859310)     0.182816 (0.140690)     0.050164    0.08712     0.0649711   0.000562875 
  accuracy0                                                          10          0.730271    0.628671 (0.860874)     0.101600 (0.139126)     0.066713    0.081337    0.0730271   0.000316334 
    accuracy0/compute                                                10          0.541965    0.440365 (0.812534)     0.101600 (0.187466)     0.049281    0.062269    0.0541965   0.000234765 
    accuracy0/infer_shape                                            10          0.068821    0.068821 (1.000000)     0.000000 (0.000000)     0.006357    0.007376    0.0068821   2.98114e-05 
    accuracy0/prepare_data                                           10          0.00801     0.008010 (1.000000)     0.000000 (0.000000)     0.000696    0.001011    0.000801    3.46972e-06 
  accuracy1                                                          10          0.524837    0.443621 (0.845255)     0.081216 (0.154745)     0.048356    0.056907    0.0524837   0.000227345 
    accuracy1/compute                                                10          0.382189    0.300973 (0.787498)     0.081216 (0.212502)     0.034964    0.040596    0.0382189   0.000165554 
    accuracy1/infer_shape                                            10          0.048497    0.048497 (1.000000)     0.000000 (0.000000)     0.004246    0.007807    0.0048497   2.10076e-05 
    accuracy1/prepare_data                                           10          0.011458    0.011458 (1.000000)     0.000000 (0.000000)     0.000953    0.001412    0.0011458   4.9633e-06  
thread1::reduce_mean                                                 20          1.23052     1.156916 (0.940188)     0.073600 (0.059812)     0.055965    0.076752    0.0615258   0.000533027 
  reduce_mean0                                                       10          0.630092    0.586316 (0.930524)     0.043776 (0.069476)     0.057471    0.075023    0.0630092   0.000272939 
    reduce_mean0/compute                                             10          0.485718    0.441942 (0.909874)     0.043776 (0.090126)     0.045249    0.056889    0.0485718   0.0002104   
    reduce_mean0/infer_shape                                         10          0.073955    0.073955 (1.000000)     0.000000 (0.000000)     0.006459    0.010565    0.0073955   3.20353e-05 
    reduce_mean0/prepare_data                                        10          0.014935    0.014935 (1.000000)     0.000000 (0.000000)     0.00076     0.006505    0.0014935   6.46945e-06 
  reduce_mean1                                                       10          0.560178    0.530354 (0.946760)     0.029824 (0.053240)     0.052755    0.060501    0.0560178   0.000242654 
    reduce_mean1/compute                                             10          0.445017    0.415193 (0.932982)     0.029824 (0.067018)     0.041977    0.048801    0.0445017   0.00019277  
      GpuMemcpyAsync(same_gpu):GPU->GPU                              10          0.259944    0.230120 (0.885268)     0.029824 (0.114732)     0.023825    0.030835    0.0259944   0.000112601 
    reduce_mean1/infer_shape                                         10          0.051102    0.051102 (1.000000)     0.000000 (0.000000)     0.004635    0.005781    0.0051102   2.2136e-05  
    reduce_mean1/prepare_data                                        10          0.010902    0.010902 (1.000000)     0.000000 (0.000000)     0.0008      0.001453    0.0010902   4.72246e-06 
thread1::FetchAsync                                                  30          1.17956     1.094440 (0.927837)     0.085120 (0.072163)     0.020777    0.074542    0.0393187   0.000510954 
  GpuMemcpyAsync:GPU->CUDAPinned                                     30          0.65539     0.570270 (0.870123)     0.085120 (0.129877)     0.012167    0.038614    0.0218463   0.000283898 
thread1::GpuMemcpyAsync(same_gpu):GPU->GPU                           20          1.17424     0.601982 (0.512658)     0.572255 (0.487342)     0.038453    0.087841    0.0587119   0.000508648 
thread1::scale                                                       30          1.15176     1.052334 (0.913677)     0.099423 (0.086323)     0.024454    0.060752    0.0383919   0.000498911 
  scale0                                                             10          0.471113    0.438665 (0.931125)     0.032448 (0.068875)     0.043918    0.058712    0.0471113   0.000204074 
    scale0/compute                                                   10          0.355682    0.323234 (0.908772)     0.032448 (0.091228)     0.034165    0.041175    0.0355682   0.000154072 
    scale0/infer_shape                                               10          0.037471    0.037471 (1.000000)     0.000000 (0.000000)     0.002979    0.006564    0.0037471   1.62314e-05 
    scale0/prepare_data                                              10          0.020155    0.020155 (1.000000)     0.000000 (0.000000)     0.000797    0.009068    0.0020155   8.73061e-06 
  scale2                                                             10          0.348762    0.319579 (0.916324)     0.029183 (0.083676)     0.030048    0.043217    0.0348762   0.000151074 
    scale2/compute                                                   10          0.255967    0.226784 (0.885989)     0.029183 (0.114011)     0.021742    0.031565    0.0255967   0.000110878 
    scale2/infer_shape                                               10          0.024138    0.024138 (1.000000)     0.000000 (0.000000)     0.002145    0.002762    0.0024138   1.04559e-05 
    scale2/prepare_data                                              10          0.016398    0.016398 (1.000000)     0.000000 (0.000000)     0.000844    0.004172    0.0016398   7.10318e-06 
  scale1                                                             10          0.267844    0.230052 (0.858903)     0.037792 (0.141097)     0.022777    0.040711    0.0267844   0.000116023 
    scale1/compute                                                   10          0.184895    0.147103 (0.795603)     0.037792 (0.204397)     0.015066    0.031897    0.0184895   8.00916e-05 
    scale1/infer_shape                                               10          0.023923    0.023923 (1.000000)     0.000000 (0.000000)     0.001385    0.005506    0.0023923   1.03628e-05 
    scale1/prepare_data                                              10          0.012742    0.012742 (1.000000)     0.000000 (0.000000)     0.000744    0.003659    0.0012742   5.5195e-06  
thread1::top_k_v2                                                    20          1.06059     0.833392 (0.785780)     0.227200 (0.214220)     0.034387    0.078834    0.0530296   0.00045942  
  top_k_v20                                                          10          0.650996    0.479956 (0.737264)     0.171040 (0.262736)     0.060984    0.075642    0.0650996   0.000281994 
    top_k_v20/compute                                                10          0.47851     0.307470 (0.642557)     0.171040 (0.357443)     0.045499    0.057886    0.047851    0.000207278 
    top_k_v20/infer_shape                                            10          0.079914    0.079914 (1.000000)     0.000000 (0.000000)     0.007181    0.011616    0.0079914   3.46166e-05 
    top_k_v20/prepare_data                                           10          0.008743    0.008743 (1.000000)     0.000000 (0.000000)     0.000738    0.001427    0.0008743   3.78724e-06 
  top_k_v21                                                          10          0.361794    0.305634 (0.844774)     0.056160 (0.155226)     0.032448    0.039786    0.0361794   0.00015672  
    top_k_v21/compute                                                10          0.234822    0.178662 (0.760840)     0.056160 (0.239160)     0.021673    0.025766    0.0234822   0.000101719 
    top_k_v21/infer_shape                                            10          0.04999     0.049990 (1.000000)     0.000000 (0.000000)     0.00411     0.007169    0.004999    2.16543e-05 
    top_k_v21/prepare_data                                           10          0.011047    0.011047 (1.000000)     0.000000 (0.000000)     0.000865    0.001463    0.0011047   4.78527e-06 
thread1::FastThreadedSSAGraphExecutorPrepare                         10          1.01591     1.015911 (1.000000)     0.000000 (0.000000)     0.043918    0.184       0.101591    0.000440066 
thread1::reshape2                                                    10          0.923536    0.888944 (0.962544)     0.034592 (0.037456)     0.0768      0.125537    0.0923536   0.000400051 
  reshape20                                                          10          0.888009    0.853417 (0.961045)     0.034592 (0.038955)     0.073656    0.121932    0.0888009   0.000384662 
    reshape20/compute                                                10          0.637534    0.602942 (0.945741)     0.034592 (0.054259)     0.052087    0.096617    0.0637534   0.000276163 
      GpuMemcpyAsync(same_gpu):GPU->GPU                              10          0.297844    0.263252 (0.883859)     0.034592 (0.116141)     0.022108    0.04782     0.0297844   0.000129018 
    reshape20/infer_shape                                            10          0.132319    0.132319 (1.000000)     0.000000 (0.000000)     0.0113      0.01763     0.0132319   5.73171e-05 
    reshape20/prepare_data                                           10          0.016687    0.016687 (1.000000)     0.000000 (0.000000)     0.001452    0.002001    0.0016687   7.22837e-06 
thread1::softmax_with_cross_entropy                                  10          0.862615    0.747479 (0.866527)     0.115136 (0.133473)     0.075162    0.152028    0.0862615   0.000373662 
  softmax_with_cross_entropy0                                        10          0.82787     0.712734 (0.860925)     0.115136 (0.139075)     0.072524    0.149285    0.082787    0.000358611 
    softmax_with_cross_entropy0/compute                              10          0.667773    0.552637 (0.827582)     0.115136 (0.172418)     0.056851    0.134061    0.0667773   0.000289262 
    softmax_with_cross_entropy0/infer_shape                          10          0.080028    0.080028 (1.000000)     0.000000 (0.000000)     0.00711     0.011344    0.0080028   3.4666e-05  
    softmax_with_cross_entropy0/prepare_data                         10          0.014741    0.014741 (1.000000)     0.000000 (0.000000)     0.000975    0.00359     0.0014741   6.38541e-06 
thread1::reduce_mean_grad                                            20          0.736037    0.667077 (0.906309)     0.068960 (0.093691)     0.02603     0.064879    0.0368019   0.000318832 
  reduce_mean_grad0                                                  10          0.397189    0.359781 (0.905818)     0.037408 (0.094182)     0.036444    0.04873     0.0397189   0.000172052 
    reduce_mean_grad0/compute                                        10          0.276244    0.238836 (0.864583)     0.037408 (0.135417)     0.024638    0.036476    0.0276244   0.000119662 
    reduce_mean_grad0/infer_shape                                    10          0.04568     0.045680 (1.000000)     0.000000 (0.000000)     0.004167    0.005002    0.004568    1.97874e-05 
    reduce_mean_grad0/prepare_data                                   10          0.015725    0.015725 (1.000000)     0.000000 (0.000000)     0.000796    0.002117    0.0015725   6.81165e-06 
  reduce_mean_grad1                                                  10          0.296593    0.265041 (0.893619)     0.031552 (0.106381)     0.024298    0.062753    0.0296593   0.000128476 
    reduce_mean_grad1/compute                                        10          0.199728    0.168176 (0.842025)     0.031552 (0.157975)     0.015788    0.051503    0.0199728   8.65169e-05 
    reduce_mean_grad1/infer_shape                                    10          0.024959    0.024959 (1.000000)     0.000000 (0.000000)     0.002295    0.002672    0.0024959   1.08116e-05 
    reduce_mean_grad1/prepare_data                                   10          0.01702     0.017020 (1.000000)     0.000000 (0.000000)     0.001071    0.00405     0.001702    7.37261e-06 
thread1::ScaleLossGrad                                               10          0.393141    0.375733 (0.955721)     0.017408 (0.044279)     0.037164    0.042198    0.0393141   0.000170298 
  GpuMemcpyAsync:CPU->GPU                                            10          0.286248    0.268840 (0.939186)     0.017408 (0.060814)     0.026611    0.031607    0.0286248   0.000123995 
thread1::softmax_with_cross_entropy_grad                             10          0.373665    0.332097 (0.888756)     0.041568 (0.111244)     0.03372     0.0404      0.0373665   0.000161862 
  softmax_with_cross_entropy_grad0                                   10          0.346983    0.305415 (0.880202)     0.041568 (0.119798)     0.031705    0.037504    0.0346983   0.000150304 
    softmax_with_cross_entropy_grad0/compute                         10          0.224126    0.182558 (0.814533)     0.041568 (0.185467)     0.021137    0.023812    0.0224126   9.70854e-05 
    softmax_with_cross_entropy_grad0/infer_shape                     10          0.038347    0.038347 (1.000000)     0.000000 (0.000000)     0.003219    0.004213    0.0038347   1.66109e-05 
    softmax_with_cross_entropy_grad0/prepare_data                    10          0.015663    0.015663 (1.000000)     0.000000 (0.000000)     0.001103    0.00232     0.0015663   6.7848e-06  
thread1::GpuMemcpyAsync:CPU->GPU                                     10          0.273682    0.258322 (0.943876)     0.015360 (0.056124)     0.023964    0.03668     0.0273682   0.000118552 
thread1::InitLocalVars                                               1           0.202196    0.202196 (1.000000)     0.000000 (0.000000)     0.202196    0.202196    0.202196    8.75859e-05 
thread1::flatten_contiguous_range_grad                               10          0.196442    0.196442 (1.000000)     0.000000 (0.000000)     0.017499    0.023256    0.0196442   8.50935e-05 
  flatten_contiguous_range_grad0                                     10          0.16535     0.165350 (1.000000)     0.000000 (0.000000)     0.01483     0.020817    0.016535    7.16252e-05 
    flatten_contiguous_range_grad0/compute                           10          0.05041     0.050410 (1.000000)     0.000000 (0.000000)     0.004474    0.00578     0.005041    2.18363e-05 
    flatten_contiguous_range_grad0/infer_shape                       10          0.036354    0.036354 (1.000000)     0.000000 (0.000000)     0.003303    0.00413     0.0036354   1.57476e-05 
    flatten_contiguous_range_grad0/prepare_data                      10          0.016318    0.016318 (1.000000)     0.000000 (0.000000)     0.001114    0.002167    0.0016318   7.06852e-06 
thread1::ScopeBufferedMonitor::post_local_exec_scopes_process        10          0.057485    0.057485 (1.000000)     0.000000 (0.000000)     0.004675    0.008794    0.0057485   2.4901e-05  
thread1::ScopeBufferedMonitor::pre_local_exec_scopes_process         10          0.026853    0.026853 (1.000000)     0.000000 (0.000000)     0.001763    0.005659    0.0026853   1.1632e-05  
thread0::GpuMemcpyAsync:CUDAPinned->GPU                              1           1045.4      1045.402880 (1.000000)  0.000000 (0.000000)     1045.4      1045.4      1045.4      0.806693    
thread0::BufferedReader:MemoryCopy                                   10          250.509     234.627218 (0.936600)   15.882199 (0.063400)    17.2275     44.3467     25.0509     0.193307    
  GpuMemcpyAsync:CUDAPinned->GPU                                     20          16.7283     0.846059 (0.050577)     15.882199 (0.949423)    0.020954    1.66126     0.836413    0.0129085   

------------------------->    Memory Profiling Report     <-------------------------

Event                                                  Alloc Calls       Size(MB)          Free Calls        Size(MB)          
CPUPlace:Unknown                                       10                3.8147e-05        10                3.8147e-05        
CUDAPinnedPlace:BufferedReader:MemoryCopy              20                183.752           19                165.377           
CUDAPinnedPlace:FetchAsync                             30                0.000114441       0                 0                 
CUDAPinnedPlace:Unknown                                1                 0.000244141       32                18.3754           

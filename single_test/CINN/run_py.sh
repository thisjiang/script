#!/bin/bash -ex
echo $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')

rm -f core.*
export CUDA_VISIBLE_DEVICES=1
export CPU_NUM=1
export LD_LIBRARY_PATH=/Paddle/Paddle/build/third_party/CINN/src/external_cinn-build/:${LD_LIBRARY_PATH}
export PYTHONPATH=/Paddle/CINN/build/python:${PYTHONPATH}
export FLAGS_cinn_cudnn_deterministic=1
python3.7 $1

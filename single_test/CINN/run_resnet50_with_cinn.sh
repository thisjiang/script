#!/bin/bash -ex
echo $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')

rm -f core.*
export CUDA_VISIBLE_DEVICES=3
export CPU_NUM=1
export LD_LIBRARY_PATH=/Paddle/Paddle/build/third_party/CINN/src/external_cinn-build/:${LD_LIBRARY_PATH}
export PYTHONPATH=/Paddle/CINN/build/python:${PYTHONPATH}

python3.7 /Paddle/Paddle/python/paddle/fluid/tests/unittests/test_resnet50_with_cinn.py

if [ $? -ne 0 ];then
  echo "Run Failed"
  exit
fi

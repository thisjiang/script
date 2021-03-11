#!/bin/bash -ex

CUR=$(pwd)

rm -f core.*

export CUDA_VISIBLE_DEVICES=4
PROFILE_PATH=/Paddle/profile/ernie-doc/slice_grad

nvcc -arch=sm_70 -lcudnn --expt-relaxed-constexpr softmax_forward.cu -o test

#nsys profile --stats=true -t cuda --cuda-memory-usage=true -o ${PROFILE_PATH}/$1.qdrep  \
ncu --target-processes all --set full -o ${PROFILE_PATH}/$1.ncu-rep \
./test
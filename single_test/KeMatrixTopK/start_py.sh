#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
PROFILE_PATH=/Paddle/profile/topk/

KERNEL=LookupTableV2Grad
if [ $# -ge 3 ];then
    KERNEL=$3
fi

#nvprof --kernels "${KERNEL}"  -f -o ${PROFILE_PATH}/$1.nvvp \
#ncu -k ${KERNEL} --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$1-${KERNEL}.ncu-rep \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true -o ${PROFILE_PATH}/$2.qdrep \
python3.7 $1

#!/bin/bash -ex

rm -f core.*

if [ $# -le 0 ]; then
    exit
fi

export CUDA_VISIBLE_DEVICES=4
PROFILE_PATH=/Paddle/profile/ernie-doc/transpose/
KERNEL=TilingSwapDim1And2Diagonal

mkdir -p ${PROFILE_PATH}
if [ $# -ge 3 ];then
    KERNEL=$3
fi

#ncu --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$2.ncu-rep \
#nvprof -f -o ${PROFILE_PATH}/$2.nvvp \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true --force-overwrite true -o ${PROFILE_PATH}/$2.qdrep \
python3.7 $1

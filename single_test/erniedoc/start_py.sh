#!/bin/bash -ex

rm -f core.*

if [ $# -le 0 ]; then
    exit
fi

export CUDA_VISIBLE_DEVICES=3
PROFILE_PATH=/Paddle/profile/ernie-doc/transpose/
KERNEL=TilingSwapDim1And2Diagonal

mkdir -p ${PROFILE_PATH}
if [ $# -ge 3 ];then
    KERNEL=$3
fi

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_memory_fraction_of_eager_deletion=1
export FLAGS_fraction_of_gpu_memory_to_use=0.5



#ncu --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$2.ncu-rep \
#nvprof -f -o ${PROFILE_PATH}/$2.nvvp \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true --force-overwrite true -o ${PROFILE_PATH}/$2.qdrep \
nvprof \
python3.7 $1

#!/bin/bash -ex
echo $(date '+%Y-%m-%d %H:%M:%S')

export CUDA_VISIBLE_DEVICES=2

OP_NAME=$1
KERNEL=LookupTableV2Grad
FILE_SUFFIX=fp16


PROFILE_PATH=/Paddle/profile/${OP_NAME}/
PROFILE_FILE=${PROFILE_PATH}/${OP_NAME}-${FILE_SUFFIX}

mkdir -p ${PROFILE_PATH}

cd build
#ncu -k ${KERNEL} --target-processes all --set full -o ${PROFILE_FILE}.ncu-rep \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true -f true -o ${PROFILE_FILE}.qdrep \
#nvprof --profile-child-processes -f -o ${PROFILE_FILE}_%p.nvvp \
make test ARGS="-R test_${OP_NAME} -V"
#ctest -R test_${OP_NAME}

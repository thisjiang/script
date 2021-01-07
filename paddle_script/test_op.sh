#!/bin/bash -ex

export CUDA_VISIBLE_DEVICES=3

OP_NAME=lookup_table_v2
KERNEL=LookupTableV2Grad
FILE_SUFFIX=fp32


PROFILE_PATH=/Paddle/profile/${OP_NAME}/
PROFILE_FILE=${PROFILE_PATH}/${OP_NAME}-${FILE_SUFFIX}

mkdir -p ${PROFILE_PATH}

cd build
#ncu -k ${KERNEL} --target-processes all --set full -o ${PROFILE_FILE}.ncu-rep \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true -f true -o ${PROFILE_FILE}.qdrep \
#nvprof --profile-child-processes -f -o ${PROFILE_FILE}_%p.nvvp \
make test ARGS="-R test_${OP_NAME} -V"
#ctest -R test_${OP_NAME}

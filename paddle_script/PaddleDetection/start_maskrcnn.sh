#!/bin/bash -ex

export CUDA_VISIBLE_DEVICES=4

PROFILE_PATH=/Paddle/profile/maskrcnn/where_index/

KERNEL=LookupTableV2Grad
if [ $# -ge 2 ];then
    KERNEL=$2
fi

#export FLAGS_conv_workspace_size_limit=4000 #MB
#export FLAGS_cudnn_exhaustive_search=0

#export CUDNN_LOGINFO_DBG=1
#export CUDNN_LOGDEST_DBG=/Paddle/profile/maskrcnn-cuDNN/cudnn8-search1-limit1000.txt

cd dygraph/
rm -f core.*

#nvprof --kernels "${KERNEL}"  -f -o ${PROFILE_PATH}/$1.nvvp \
#ncu -k ${KERNEL} --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$1-${KERNEL}.ncu-rep \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true -o ${PROFILE_PATH}/$1.qdrep \
python3.7 -u tools/train.py -c configs/mask_rcnn_r50_fpn_1x_coco.yml

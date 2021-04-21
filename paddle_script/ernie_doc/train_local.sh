#!/bin/bash -ex
echo $(date '+%Y-%m-%d %H:%M:%S')
set -eu

#bash -x ./env.sh
source ./slurm/env.sh
source ./slurm/utils.sh

source ./model_conf

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1
#export GLOG_v=4

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    export FLAGS_fuse_parameter_memory_size=131072
    export FLAGS_fuse_parameter_groups_size=10
fi

BATCH_SIZE=2048

#pack output
pidof sh | xargs kill -9
pidof sleep | xargs kill -9
nohup sh ./slurm/pack_model.sh ./output > log/pack_model.log 2>&1 &

# check
check_iplist

PROFILE_PATH=/Paddle/profile/ernie-doc/fleet_py3/
mkdir -p ${PROFILE_PATH}

PROFILE_NAME=${1:-"$(date '+%Y%m%d%H%M%S')"}

KERNEL=EigenMetaKernel
if [ $# -ge 2 ];then
    KERNEL=$2
fi

rm -f core.*
export CUDA_VISIBLE_DEVICES="2"

#FLAGS_conv_workspace_size_limit=4000 #MB 
#export FLAGS_cudnn_exhaustive_search=1

#nvprof --kernels "${KERNEL}"  -f -o ${PROFILE_PATH}/$1.nvvp \
#ncu -k ${KERNEL} -c 10 --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$1-${KERNEL}.ncu-rep \
#ncu -k ${KERNEL} --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$1-${KERNEL}.ncu-rep \
# nsys profile --stats=true -t cuda --cuda-memory-usage=true -o ${PROFILE_PATH}/$1.qdrep \
# cuda-memcheck \
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o ${PROFILE_PATH}/${PROFILE_NAME} --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true --cudabacktrace-threshold=10000 --osrt-threshold=10000 -x true  \
python3.7 -u ./train.py --use_cuda "True" \
                --is_distributed "False" \
                --weight_sharing "True" \
                --use_fast_executor ${e_executor-"True"} \
                --use_fuse ${use_fuse-"False"} \
                --nccl_comm_num ${nccl_comm_num:-"1"} \
                --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
                --in_tokens "True" \
                --batch_size ${BATCH_SIZE} \
                --vocab_path ${vocab_path} \
                --task_group_json ${task_group_json} \
                --hack_old_data ${hack_old_data-"False"} \
                --generate_neg_sample ${generate_neg_sample-"True"} \
                --lr_scheduler ${lr_scheduler} \
                --num_train_steps ${num_train_steps} \
                --checkpoints ./output \
                --use_amp "True" \
                --use_recompute "False" \
                --use_dynamic_loss_scaling ${use_fp16} \
                --init_loss_scaling ${loss_scaling:-128} \
                --save_steps ${SAVE_STEPS} \
                --init_checkpoint ${init_model:-""} \
                --ernie_config_path ${CONFIG_PATH} \
                --learning_rate ${LR_RATE} \
                --validation_steps ${VALIDATION_STEPS:-4000} \
                --warmup_steps ${WARMUP_STEPS:-0} \
                --weight_decay ${WEIGHT_DECAY:-0} \
                --max_seq_len ${MAX_LEN} \
                --mem_len ${MEM_LEN:-384} \
                --skip_steps 10

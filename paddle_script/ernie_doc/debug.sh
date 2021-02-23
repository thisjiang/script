#!/bin/bash

#source ./slurm/env.sh
TrainFilelist=./data/train_filelist
EvalFilelist=./data/eval_filelist
MODEL_DIR=./pretrained_model/
VOCAB_PATH=./data/vocab.txt
CON_PATH=./config/


N_LAYER=12
D_MODEL=768
D_EMBED=768
N_HEAD=12
D_HEAD=64
D_INNER=3072
FF_ACTIVATION="gelu"
UNTIE_R="True"
N_TOKEN=24984


TGT_LEN=512
MEN_LEN=384
REUSE_LEN=256
PERM_SIZE=256
bi_data="True"
NUM_PREDICT=85


BSZ=4
NUM_CORE=4


export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES="2"
export FLAGS_fraction_of_gpu_memory_to_use=1.0
#export FLAGS_initial_gpu_memory_in_mb=1
#export GLOG_v=4


node_ips="10.255.129.37"
current_node_ip="10.255.129.37"
selected_gpus="0"
#export FLAGS_benchmark=True
#export GLOG_vmodule=fast_threaded_ssa_graph_executor=10

export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export PADDLE_IS_LOCAL=0
#export GLOG_v=4
export CPU_NUM=20

#init_model="./pretrained_model/step_192000"

distributed_args="--node_ips ${node_ips} --node_id 0 --current_node_ip
${current_node_ip} --nproc_per_node 1 --selected_gpus ${selected_gpus}"

export CUDA_VISIBLE_DEVICES=4
rm -f core.*

PROFILE_PATH=/Paddle/profile/ernie-doc/slice_grad/
mkdir -p ${PROFILE_PATH}

KERNEL=EigenMetaKernel
if [ $# -ge 2 ];then
    KERNEL=$2
fi

#FLAGS_conv_workspace_size_limit=4000 #MB 
#export FLAGS_cudnn_exhaustive_search=1

#nvprof --kernels "${KERNEL}"  -f -o ${PROFILE_PATH}/$1.nvvp \
#ncu -k ${KERNEL} -c 10 --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$1-${KERNEL}.ncu-rep \
#cuda-memcheck \
#nsys profile --stats=true -t cuda --cuda-memory-usage=true -o ${PROFILE_PATH}/$1.qdrep \
#ncu -k ${KERNEL} --details-all --print-summary=per-kernel --target-processes all --set full -o ${PROFILE_PATH}/$1-${KERNEL}.ncu-rep \
python3.7 -u ./train_gpu_paddle.py  \
                --model_dir ${CON_PATH} \
                --train_filelist $TrainFilelist \
                --eval_filelist $EvalFilelist \
                --vocab_path $VOCAB_PATH \
                --untie_r ${UNTIE_R:-"True"} \
                --use_fast_executor "True" \
                --ff_activation ${FF_ACTIVATION} \
                --epoch 1000 \
                --n_token ${N_TOKEN} \
                --n_layer ${N_LAYER} \
                --d_model ${D_MODEL} \
                --d_embed ${D_EMBED} \
                --n_head ${N_HEAD} \
                --d_head ${D_HEAD} \
                --d_inner ${D_INNER} \
                --dropout 0.1 \
                --dropatt 0.0 \
                --learning_rate 0.00004 \
                --warmup_steps 4000 \
                --train_steps 10000000 \
                --tgt_len ${TGT_LEN} \
                --mem_len ${MEN_LEN} \
                --reuse_len ${REUSE_LEN} \
                --perm_size ${PERM_SIZE} \
                --train_batch_size ${BSZ} \
                --num_predict ${NUM_PREDICT} \
                --bi_data ${bi_data:-"True"} \
                --use_cuda "True" \
                --is_distributed "False" \
                --use_fuse "False" \
                --nccl_comm_num ${nccl_comm_num:-"1"} \
                --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
                --lr_scheduler "linear_warmup_decay" \
                --checkpoints ${MODEL_DIR}/checkpoint \
                --use_amp "False" \
                --use_dynamic_loss_scaling "False" \
                --init_loss_scaling ${loss_scaling:-128} \
                --save_steps 8000 \
                --validation_steps 1000 \
                --init_checkpoint ${init_model:-""} \
                --weight_decay 0.01 \
                --skip_steps 10 #1> log/lanch.log 2> log/err.log


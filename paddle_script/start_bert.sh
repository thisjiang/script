#!/bin/bash -ex

export DATA_DIR=data/
export PYTHONPATH=/Paddle/models/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python3.7 -u ./run_pretrain_single.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 64   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir $DATA_DIR \
    --output_dir ./tmp2/ \
    --logging_steps 10 \
    --save_steps 20000 \
    --max_steps 500 \
    --use_amp true \
    --scale_loss 128.0 \
    --use_pure_fp16 true

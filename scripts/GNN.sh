#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

DATA_DIR="./dataset/SAHU/direct_5_working"
BATCH_SIZE=16
HIDDEN_DIM=64
NUM_EPOCHS=100
LEARNING_RATE=0.001
MODEL_NAME="GNN"

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path "$DATA_DIR" \
    --model_id SAHU \
    --model GNN \
    --data Graph \
    --step 72 \
    --seq_len 72 \
    --num_class 8 \
    --batch_size "$BATCH_SIZE" \
    --hidden_dim "$HIDDEN_DIM" \
    --train_epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" 
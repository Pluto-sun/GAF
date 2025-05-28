#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 设置Python路径（如果需要）
# export PYTHONPATH=$(dirname "$0")/..

# # 直接模式训练（不使用通道分组）
# echo "====== Training Inception in Direct Mode ======"
# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SAHU/ \
#   --model_id SAHU \
#   --model ClusteredInception \
#   --data SAHU \
#   --step 72 \
#   --seq_len 72 \
#   --num_class 3 \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 512 \
#   --d_ff 2048 \
#   --n_heads 8 \
#   --d_layers 1 \
#   --dropout 0.1 \
#   --des "Inception_Direct" \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 50 \
#   --patience 10 \
#   --use_gpu True \
#   --gpu 0 \
#   --gpu_type cuda \
#   --gaf_method summation

# # 清理GPU内存
# python -c "import torch; torch.cuda.empty_cache()"

# 分组模式训练（使用通道分组）
echo "====== Training Inception in Clustered Mode ======"
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SAHU/ \
  --model_id SAHU \
  --model ClusteredInception \
  --data SAHU \
  --step 72 \
  --seq_len 72 \
  --num_class 3 \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --d_layers 1 \
  --dropout 0.1 \
  --des "Inception_Clustered" \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 1000 \
  --patience 10 \
  --use_gpu True \
  --gpu 0 \
  --gpu_type cuda \
  --gaf_method summation \
  --channel_groups "0,1|2|3,4,5,6|7,8,9,10|11,12,13,14|15,16,17,18,19|20,21,22,23|24|25,26,27,28,29"

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"

echo "Training completed!" 
#!/bin/bash

# 允许用户通过第一个参数指定GPU编号，默认用0号卡
export CUDA_VISIBLE_DEVICES=1

echo "当前使用的GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

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
  --root_path ./dataset/SAHU/direct_5_working \
  --model_id SAHU_group \
  --model ClusteredInception \
  --data SAHU \
  --data_type_method uint8 \
  --step 96 \
  --seq_len 96 \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 512 \
  --d_ff 2048 \
  --feature_dim 32 \
  --n_heads 8 \
  --d_layers 1 \
  --dropout 0.1 \
  --des "Inception_Clustered" \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --patience 10 \
  --use_gpu True \
  --gpu 0 \
  --gpu_type cuda \
  --gaf_method summation \
  --channel_groups "0,1|2|3,4,5,6|7,8,9,10|11,12,13|14,15,16,17|18,19,20|21|22,23,24,25"


echo "Training completed!" 
#!/bin/bash

# 安全运行DualGAF网络脚本 - 修复内存双重释放错误
echo "🛡️ 安全运行DualGAF网络 - 内存错误修复版"
echo "=========================================="

# 设置环境变量防止内存问题
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1

# 基础配置
model="DualGAFNet"
data="DualGAF"
root_path="./dataset/SAHU"
seq_len=96
step=4
epochs=20
batch_size=4  # 使用小batch size
learning_rate=0.001
large_feature_dim=64
# 梯度累积配置：小batch_size + 梯度累积 = 更大的有效batch_size
gradient_accumulation_steps=4  # 有效batch_size = 4 * 4 = 16

echo "实验1: 安全配置测试 (batch_size=${batch_size}, GA_steps=${gradient_accumulation_steps}, 有效batch_size=$((batch_size * gradient_accumulation_steps)))"
python run.py \
    --model $model \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --feature_dim $large_feature_dim \
    --extractor_type resnet18_gaf \
    --fusion_type adaptive \
    --attention_type channel \
    --classifier_type mlp \
    --use_statistical_features \
    --stat_type basic \
    --multimodal_fusion_strategy concat \
    --train_epochs $epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --loss_preset standard \
    --ablation_mode none \
    --lr_scheduler_type f1_based \
    --num_workers 0 \
    --safe_mode \
    --des "safe_run_with_GA"

echo "安全实验完成"

echo ""
echo "💡 梯度累积说明:"
echo "   - 物理batch_size: ${batch_size}"
echo "   - 梯度累积步数: ${gradient_accumulation_steps}"
echo "   - 有效batch_size: $((batch_size * gradient_accumulation_steps))"
echo "   - 内存使用: 仅占用${batch_size}个样本的内存"
echo "   - 训练效果: 等效于${batch_size}x${gradient_accumulation_steps}的大batch训练" 
#!/bin/bash

eval "$(conda shell.bash hook)"  # 初始化 conda 支持
conda activate gaf     # 替换为你的环境名，例如 test_env
# 模块配置说明:
# extractor_type: large_kernel (大核卷积), inception (Inception网络), dilated/optimized_dilated (膨胀卷积), multiscale (多尺度)
# fusion_type: adaptive (自适应), concat (拼接), bidirectional (双向注意力), gated (门控融合), add (相加), mul (相乘), weighted_add (加权相加)
# attention_type: channel (通道注意力), spatial (空间注意力), cbam (CBAM), self (自注意力), none (无注意力)
# classifier_type: mlp, simple, residual, residual_bottleneck, residual_dense
# use_statistical_features: 是否使用统计特征
# stat_type: 'basic', 'comprehensive', 'correlation_focused'
# multimodal_fusion_strategy: 'concat', 'attention', 'gated', 'adaptive'
# ablation_mode 消融实验模式快捷设置: none(完整模型), no_diff(无差分分支), no_stat(无统计特征), no_attention(无注意力), minimal(最简模型)
# 增强版双路GAF网络运行示例脚本
export CUDA_VISIBLE_DEVICES=0
echo "增强版双路GAF网络实验脚本"
echo "================================"

# 基础配置
model=OneDCNN
data=DualGAF_DDAHU
root_path="./dataset/DDAHU/direct_5_working"
epochs=300
batch_size=4
learning_rate=0.0001
step=96
seq_len=96
feature_dim=64
large_feature_dim=128

# echo "实验0: 基础增强版网络+concat"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $large_feature_dim \
#     --extractor_type dilated \
#     --fusion_type adaptive \
#     --attention_type channel \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy adaptive \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --enable_auto_gradient_accumulation \
#     --ablation_mode none \
#     --drop_last_batch \
#     --lr_scheduler_type f1_based \
#     --safe_mode \
#     --des "dilated_enhanced"

# echo "实验0完成"
# echo "--------------------------------"

# 实验1: 基础增强版网络（使用统计特征）
# echo "实验1: 基础增强版网络+concat"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $large_feature_dim \
#     --extractor_type dilated \
#     --fusion_type adaptive \
#     --attention_type channel \
#     --classifier_type residual_bottleneck \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy concat \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --enable_auto_gradient_accumulation \
#     --ablation_mode none \
#     --drop_last_batch \
#     --lr_scheduler_type f1_based \
#     --safe_mode \
#     --des "静态融合缩放+残差分类器"

# echo "实验1完成"
# echo "--------------------------------"

# 实验2: 使用相关性聚焦的统计特征
# echo "实验2: 相关性聚焦统计特征 + 注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $large_feature_dim \
#     --extractor_type dilated \
#     --fusion_type adaptive \
#     --attention_type self \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy gated \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --enable_auto_gradient_accumulation \
#     --ablation_mode none \
#     --drop_last_batch \
#     --lr_scheduler_type f1_based \
#     --safe_mode \
#     --des "静态融合缩放+门控融合+自注意力"

# echo "实验2完成"
# echo "--------------------------------"

# 实验3: 使用相关性聚焦的统计特征
echo "实验3: comprehensive统计特征 + 注意力融合"
python run.py \
    --model $model \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --feature_dim $large_feature_dim \
    --extractor_type dilated \
    --fusion_type concat \
    --attention_type channel \
    --classifier_type mlp \
    --use_statistical_features \
    --stat_type basic \
    --multimodal_fusion_strategy concat \
    --train_epochs $epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --enable_auto_gradient_accumulation \
    --ablation_mode none \
    --drop_last_batch \
    --lr_scheduler_type f1_based \
    --safe_mode \
    --des "oneCNN"

echo "实验3完成"
echo "--------------------------------"

# 实验4: 使用相关性聚焦的统计特征

# echo "实验4: 相关性聚焦统计特征 + 门控融合"
# python run.py \
#     --model SimpleGAFNet \
#     --backbone_type resnet34 \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $large_feature_dim \
#     --extractor_type dilated \
#     --fusion_type adaptive \
#     --attention_type channel \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy concat \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --enable_auto_gradient_accumulation \
#     --ablation_mode none \
#     --drop_last_batch \
#     --lr_scheduler_type f1_based \
#     --safe_mode \
#     --des "消融-直接卷积-resnet"

# echo "实验4完成"
# echo "--------------------------------"

# echo "实验5: 基础统计特征 + 注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $large_feature_dim \
#     --extractor_type dilated \
#     --fusion_type concat \
#     --attention_type channel \
#     --classifier_type residual_bottleneck \
#     --use_signal_level_stats \
#     --signal_stat_type extended \
#     --signal_stat_fusion_strategy concat_project \
#     --signal_stat_feature_dim 64 \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --enable_auto_gradient_accumulation \
#     --drop_last_batch \
#     --lr_scheduler_type f1_based \
#     --safe_mode \
#     --des "dilated_enhanced"

# echo "实验5完成"
# echo "--------------------------------"



echo "所有实验完成！"
echo "结果保存在 ./result/ 目录下" 
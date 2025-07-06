#!/bin/bash

eval "$(conda shell.bash hook)"  # 初始化 conda 支持
conda activate test_env     # 替换为你的环境名，例如 test_env
# 模块配置说明:
# extractor_type: large_kernel (大核卷积), inception (Inception网络), dilated (膨胀卷积), multiscale (多尺度)
# fusion_type: adaptive (自适应), concat (拼接), bidirectional (双向注意力), gated (门控融合), add (相加), mul (相乘), weighted_add (加权相加)
# attention_type: channel (通道注意力), spatial (空间注意力), cbam (CBAM), self (自注意力), none (无注意力)
# classifier_type: mlp (多层感知机), simple (简单分类器) 
# use_statistical_features: 是否使用统计特征
# stat_type: 'basic', 'comprehensive', 'correlation_focused'
# multimodal_fusion_strategy: 'concat', 'attention', 'gated', 'adaptive'

# 增强版双路GAF网络运行示例脚本
export CUDA_VISIBLE_DEVICES=1
echo "增强版双路GAF网络实验脚本"
echo "================================"

# 基础配置
model=DualGAFNet
data=DualGAF_DDAHU
root_path="./dataset/DDAHU/direct_5_working"
epochs=1000
batch_size=8
learning_rate=0.001
step=96
seq_len=96
feature_dim=64
large_feature_dim=128
# 实验1: 基础增强版网络（使用统计特征）
echo "实验1: 基础增强版网络+拼接+双向注意力"
python run.py \
    --model $model \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --feature_dim $large_feature_dim \
    --extractor_type inception \
    --fusion_type bidirectional \
    --attention_type cbam \
    --classifier_type mlp \
    --use_statistical_features \
    --stat_type basic \
    --multimodal_fusion_strategy concat \
    --train_epochs $epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --loss_preset hvac_hard_samples \
    --des "inception_enhanced"

echo "实验1完成"
echo "--------------------------------"

# # 实验2: 使用相关性聚焦的统计特征
# echo "实验2: 基础统计特征 + 双向注意力融合+cbam"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $large_feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type cbam \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy concat \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# echo "实验2完成"
# echo "--------------------------------"

# # 实验3: 使用相关性聚焦的统计特征
# echo "实验3: 基础统计特征 + 双向注意力融合+cbam+attention"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type cbam \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy attention \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# echo "实验3完成"
# echo "--------------------------------"

# # 实验4: 使用相关性聚焦的统计特征

# echo "实验4: 基础统计特征 + 双向注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type cbam \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy gated \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# # 实验5: 使用相关性聚焦的统计特征
# echo "实验5: 基础统计特征 + 双向注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type cbam \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy adaptive \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# echo "实验5完成"
# echo "--------------------------------"

# echo "实验6: 基础统计特征 + 双向注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type cbam \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type correlation_focused \
#     --multimodal_fusion_strategy concat \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# echo "实验6完成"

# echo "实验7: 基础统计特征 + 双向注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type spatial \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy concat \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# echo "实验7完成"   

# echo "实验8: 基础统计特征 + 双向注意力融合"
# python run.py \
#     --model $model \
#     --data $data \
#     --root_path $root_path \
#     --seq_len $seq_len \
#     --step $step \
#     --feature_dim $feature_dim \
#     --extractor_type inception \
#     --fusion_type bidirectional \
#     --attention_type self \
#     --classifier_type mlp \
#     --use_statistical_features \
#     --stat_type basic \
#     --multimodal_fusion_strategy concat \
#     --train_epochs $epochs \
#     --batch_size $batch_size \
#     --learning_rate $learning_rate \
#     --des "inception_enhanced"

# echo "实验8完成"   
# echo "--------------------------------"
echo "所有实验完成！"
echo "结果保存在 ./result/ 目录下" 
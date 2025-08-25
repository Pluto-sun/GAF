#!/bin/bash

# 双路GAF网络训练脚本
# 该脚本演示如何使用不同的模块配置运行双路GAF网络

# 基本配置
export CUDA_VISIBLE_DEVICES=0

model_name=DualGAFNet
data=DualGAF  # 使用双路GAF数据


# 数据配置
root_path="./dataset/SAHU/direct_5_working"
seq_len=96
step=96
batch_size=8
test_size=0.2
data_type_method=uint8
feature_dim=64
large_feature_dim=128

# 训练配置
train_epochs=1000
learning_rate=0.0001
patience=10

# HVAC信号分组配置 (可选)
hvac_groups="SA_TEMP,OA_TEMP,MA_TEMP,RA_TEMP,ZONE_TEMP_1,ZONE_TEMP_2,ZONE_TEMP_3,ZONE_TEMP_4,ZONE_TEMP_5|OA_CFM,RA_CFM,SA_CFM|SA_SP,SA_SPSPT|SF_WAT,RF_WAT|SF_SPD,RF_SPD,SF_CS,RF_CS|CHWC_VLV_DM,CHWC_VLV|OA_DMPR_DM,RA_DMPR_DM,OA_DMPR,RA_DMPR"

echo "开始运行双路GAF网络实验..."

# 实验1: 基础配置 - Inception提取器 + 自适应融合 + 通道注意力 + MLP分类器
echo "=== 实验1: Inception通道注意力配置 (inception + bidirectional + channel + mlp) ==="
python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type bidirectional \
    --attention_type channel \
    --classifier_type mlp \
    --des "inception-bidirectional-channel-mlp-large-dim"

# 实验2: Inception特征提取器 + 拼接融合 + 空间注意力
echo "=== 实验2: Inception拼接融合 + 空间注意力 (inception + bidirectional + spatial + mlp) ==="
python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type bidirectional \
    --attention_type spatial \
    --classifier_type mlp \
    --des "inception-bidirectional-spatial-mlp-large-dim"

# 实验3: 膨胀卷积特征提取器 + 加权相加融合 + CBAM注意力
echo "=== 实验3: Inception加权相加融合 + CBAM注意力 (inception + gated + channel + simple) ==="
python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type gated \
    --attention_type cbam \
    --classifier_type mlp \
    --des "inception-gated-cbam-mlp-large-dim"

# 实验4: 多尺度特征提取器 + 元素相乘融合 + 自注意力
echo "=== 实验4: Inception元素相乘融合 + 自注意力 (inception + bidirectional + self + mlp) ==="
python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type bidirectional \
    --attention_type self \
    --classifier_type mlp \
    --des "inception-bidirectional-self-mlp-large-dim"

# 实验5: 无分组 + 简单相加融合 + 无注意力机制
echo "=== 实验5: Inception简单相加融合 + 无注意力机制 (inception + bidirectional + none + simple) - 无HVAC分组 ==="
python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type bidirectional \
    --attention_type none \
    --classifier_type mlp \
    --des "inception-bidirectional-none-mlp-large-dim"

# 实验6: 最优配置组合测试 (更大的特征维度)
echo "=== 实验6: Inception配置 (inception + bidirectional + cbam + mlp, feature_dim=128) ==="
python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type bidirectional \
    --attention_type cbam \
    --classifier_type mlp \
    --des "inception-bidirectional-cbam-mlp-large-dim"

echo "=== 实验7: Inception配置 (inception + gated + cbam + mlp, feature_dim=128) ==="

python run.py \
    --is_training 1 \
    --model $model_name \
    --data $data \
    --root_path $root_path \
    --seq_len $seq_len \
    --step $step \
    --batch_size $batch_size \
    --test_size $test_size \
    --data_type_method $data_type_method \
    --feature_dim $large_feature_dim \
    --train_epochs $train_epochs \
    --learning_rate $learning_rate \
    --patience $patience \
    --extractor_type inception \
    --fusion_type gated \
    --attention_type cbam \
    --classifier_type mlp \
    --des "inception-gated-cbam-mlp-large-dim"


echo "所有双路GAF网络实验完成！"

# 使用说明:
# 1. 确保数据集路径正确
# 2. 根据GPU内存调整batch_size
# 3. 根据需要修改hvac_groups分组配置
# 4. 可以注释掉不需要的实验来节省时间
# 5. 结果将保存在result/目录下

# 模块配置说明:
# extractor_type: large_kernel (大核卷积), inception (Inception网络), dilated (膨胀卷积), multiscale (多尺度)
# fusion_type: adaptive (自适应), concat (拼接),bidirectional (双向注意力), gated (门控融合), add (相加), mul (相乘), weighted_add (加权相加)
# attention_type: channel (通道注意力), spatial (空间注意力), cbam (CBAM), self (自注意力), none (无注意力)
# classifier_type: mlp (多层感知机), simple (简单分类器) 
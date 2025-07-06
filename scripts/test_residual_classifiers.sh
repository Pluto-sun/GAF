#!/bin/bash

# 残差分类器性能测试脚本
# 比较不同分类器类型在HVAC异常检测中的表现

echo "🧪 开始残差分类器性能测试"
echo "================================"

# 公共参数设置
COMMON_ARGS="--model DualGAFNet --data SAHU --step 72 --seq_len 72 --train_epochs 10 --batch_size 4 --feature_dim 64 --loss_preset hvac_similar_optimized --use_statistical_features"

# 基本配置
echo "📋 测试配置:"
echo "  - 模型: DualGAFNet"  
echo "  - 数据集: SAHU"
echo "  - 序列长度: 72"
echo "  - 批次大小: 4"
echo "  - 特征维度: 64"
echo "  - 训练轮数: 10 (测试用)"
echo "  - 启用统计特征"
echo ""

# 1. 传统MLP分类器（基线）
echo "🏁 实验1: 传统MLP分类器（基线）"
echo "------------------------------------"
python run.py $COMMON_ARGS \
    --classifier_type mlp \
    --des "baseline_mlp_classifier" \
    --itr 1

echo ""

# 2. 简单分类器
echo "📊 实验2: 简单分类器"
echo "------------------------------------"
python run.py $COMMON_ARGS \
    --classifier_type simple \
    --des "simple_classifier" \
    --itr 1

echo ""

# 3. 基础残差分类器
echo "🏗️ 实验3: 基础残差分类器"
echo "------------------------------------"
python run.py $COMMON_ARGS \
    --classifier_type residual \
    --des "residual_classifier" \
    --itr 1

echo ""

# 4. 瓶颈残差分类器
echo "🎯 实验4: 瓶颈残差分类器"
echo "------------------------------------"
python run.py $COMMON_ARGS \
    --classifier_type residual_bottleneck \
    --des "residual_bottleneck_classifier" \
    --itr 1

echo ""

# 5. 密集残差分类器
echo "🚀 实验5: 密集残差分类器"
echo "------------------------------------"
python run.py $COMMON_ARGS \
    --classifier_type residual_dense \
    --des "residual_dense_classifier" \
    --itr 1

echo ""

# 6. 残差分类器 + 消融实验示例
echo "🔬 实验6: 残差分类器 + 消融实验"
echo "------------------------------------"
echo "6a. 残差分类器 + 无差分分支"
python run.py $COMMON_ARGS \
    --classifier_type residual \
    --ablation_mode no_diff \
    --des "residual_ablation_no_diff" \
    --itr 1

echo ""
echo "6b. 残差分类器 + 无注意力机制"
python run.py $COMMON_ARGS \
    --classifier_type residual_dense \
    --ablation_mode no_attention \
    --des "residual_dense_ablation_no_attention" \
    --itr 1

echo ""
echo "✅ 残差分类器测试完成！"
echo ""
echo "📊 分析结果："
echo "  - 对比不同分类器的准确率和F1分数"
echo "  - 评估参数数量和推理时间的权衡"
echo "  - 观察残差连接对收敛速度的影响"
echo "  - 验证残差分类器在消融实验中的表现"
echo ""
echo "💡 预期发现："
echo "  1. 残差分类器通常有更好的收敛性"
echo "  2. 密集残差分类器具有最强的表达能力"
echo "  3. 瓶颈残差分类器在高维特征时更高效"
echo "  4. 残差连接有助于梯度流动和特征重用"
echo "  5. 在相似类别的HVAC异常检测中表现更优" 
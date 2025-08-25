#!/bin/bash

# 消融实验运行脚本
# 用于测试双路GAF网络各组件的有效性

echo "🔬 开始消融实验测试"
echo "================================"

# 公共参数设置
COMMON_ARGS="--model DualGAFNet --data SAHU --step 72 --seq_len 72 --train_epochs 5 --batch_size 4 --feature_dim 32 --loss_preset hvac_similar_optimized"

# 基本配置
echo "📋 基本实验配置:"
echo "  - 模型: DualGAFNet"  
echo "  - 数据集: SAHU"
echo "  - 序列长度: 72"
echo "  - 批次大小: 4"
echo "  - 特征维度: 32"
echo "  - 训练轮数: 5 (测试用)"
echo ""

# 1. 完整模型（基线）
echo "🏁 实验1: 完整模型（基线）"
echo "-----------------------------"
python run.py $COMMON_ARGS \
    --use_statistical_features \
    --use_diff_branch \
    --attention_type channel \
    --des "baseline_full_model" \
    --itr 1

echo ""

# 2. 消融实验1：移除差分分支
echo "🔬 实验2: 消融差分分支"
echo "-----------------------------"
python run.py $COMMON_ARGS \
    --ablation_mode no_diff \
    --use_statistical_features \
    --des "ablation_no_diff_branch" \
    --itr 1

echo ""

# 3. 消融实验2：移除统计特征
echo "🔬 实验3: 消融统计特征"
echo "-----------------------------"
python run.py $COMMON_ARGS \
    --ablation_mode no_stat \
    --use_diff_branch \
    --des "ablation_no_statistical_features" \
    --itr 1

echo ""

# 4. 消融实验3：移除注意力机制
echo "🔬 实验4: 消融注意力机制"
echo "-----------------------------"
python run.py $COMMON_ARGS \
    --ablation_mode no_attention \
    --use_statistical_features \
    --use_diff_branch \
    --des "ablation_no_attention" \
    --itr 1

echo ""

# 5. 最简化模型
echo "🔬 实验5: 最简化模型"
echo "-----------------------------"
python run.py $COMMON_ARGS \
    --ablation_mode minimal \
    --des "ablation_minimal_model" \
    --itr 1

echo ""
echo "✅ 消融实验完成！"
echo ""
echo "📊 查看结果："
echo "  - 训练日志：查看每个实验的输出"
echo "  - 模型性能：对比各实验的准确率和F1分数"
echo "  - 模型复杂度：对比参数数量和推理时间"
echo ""
echo "💡 建议分析："
echo "  1. 基线模型 vs 各消融实验的性能差异"
echo "  2. 差分分支对融合效果的贡献"
echo "  3. 统计特征对多模态学习的价值"
echo "  4. 注意力机制对信号建模的重要性"
echo "  5. 模型复杂度与性能的权衡" 
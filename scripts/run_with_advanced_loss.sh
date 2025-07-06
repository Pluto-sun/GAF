#!/bin/bash

# =============================================================================
# 高级损失函数使用示例脚本
# 
# 用于解决类别相似性问题的多种损失函数配置示例
# =============================================================================

echo "🎯 高级损失函数配置示例"
echo "========================================="

# 基础配置
BASE_ARGS="--task_name classification --model DualGAFNet --data DualGAF \
           --root_path ./dataset/SAHU/ --seq_len 96 --step 96 --num_class 9 \
           --feature_dim 128 --batch_size 32 --train_epochs 50 \
           --extractor_type resnet18_gaf --fusion_type adaptive --attention_type channel \
           --use_statistical_features --stat_type basic \
           --checkpoints ./checkpoints/ --result_path ./result/"

# =============================================================================
# 方案1: 针对HVAC相似类别问题的推荐配置
# =============================================================================
echo ""
echo "🔥 方案1: HVAC相似类别问题 - 标签平滑"
echo "适用于: 故障模式相似、容易混淆的类别"
echo "特点: 防止过度自信，提高泛化能力"

python run.py $BASE_ARGS \
    --loss_preset hvac_similar \
    --des "HVAC_similar_classes_label_smoothing" \
    --model_id "hvac_similar_ls_0.15"

# 或者手动配置
echo ""
echo "📝 手动配置示例 (等效于上面的预设):"
echo "python run.py \$BASE_ARGS \\"
echo "    --loss_type label_smoothing \\"
echo "    --label_smoothing 0.15 \\"
echo "    --des \"manual_label_smoothing\""

# =============================================================================
# 方案2: 针对难分类样本的Focal Loss配置
# =============================================================================
echo ""
echo "🎯 方案2: 难分类样本聚焦 - Focal Loss"
echo "适用于: 有些样本特别难分类，需要更多关注"
echo "特点: 动态调整损失权重，专注于困难样本"

python run.py $BASE_ARGS \
    --loss_preset hard_samples \
    --des "hard_samples_focal_loss" \
    --model_id "hard_samples_focal_a0.25_g3.0"

# =============================================================================
# 方案3: 类别不平衡问题的Focal Loss配置
# =============================================================================
echo ""
echo "⚖️ 方案3: 类别不平衡问题 - 平衡Focal Loss"
echo "适用于: 某些类别样本数量显著少于其他类别"
echo "特点: 自动平衡不同类别的重要性"

python run.py $BASE_ARGS \
    --loss_preset imbalanced_focus \
    --des "imbalanced_classes_focal" \
    --model_id "imbalanced_focal_a1.0_g2.0"

# =============================================================================
# 方案4: 防止过度自信的组合损失
# =============================================================================
echo ""
echo "🛡️ 方案4: 防止过度自信 - 组合损失"
echo "适用于: 模型在训练集上过度自信，验证性能不佳"
echo "特点: 标签平滑 + 置信度惩罚"

python run.py $BASE_ARGS \
    --loss_preset overconfidence_prevention \
    --des "overconfidence_prevention_combined" \
    --model_id "combined_ls0.1_cp0.1"

# =============================================================================
# 方案5: 自定义配置示例
# =============================================================================
echo ""
echo "🔧 方案5: 自定义高级配置"
echo "适用于: 需要精细调节参数的情况"

# 5.1 强化标签平滑（相似度极高的类别）
echo "5.1 强化标签平滑 (smoothing=0.2)"
python run.py $BASE_ARGS \
    --loss_type label_smoothing \
    --label_smoothing 0.2 \
    --des "strong_label_smoothing" \
    --model_id "strong_ls_0.2"

# 5.2 极端Focal Loss（严重类别不平衡）
echo "5.2 极端Focal Loss (gamma=4.0)"
python run.py $BASE_ARGS \
    --loss_type focal \
    --focal_alpha 0.5 \
    --focal_gamma 4.0 \
    --des "extreme_focal_loss" \
    --model_id "extreme_focal_a0.5_g4.0"

# 5.3 类别权重示例（仅用于不平衡数据）
echo "5.3 类别权重示例（如果数据不平衡时使用）"
echo "# 注意：对于平衡数据集，默认不启用类别权重"
echo "# 如需启用，添加 --enable_class_weights 参数"
echo "python run.py \$BASE_ARGS \\"
echo "    --loss_type ce \\"
echo "    --enable_class_weights \\"
echo "    --class_weights \"1.0,2.0,1.5,3.0,1.0,2.5,1.8,1.2,2.2\" \\"
echo "    --des \"custom_class_weights\" \\"
echo "    --model_id \"custom_weights\""

# 5.4 组合损失自定义参数
echo "5.4 自定义组合损失"
python run.py $BASE_ARGS \
    --loss_type combined \
    --label_smoothing 0.12 \
    --confidence_penalty_beta 0.08 \
    --des "custom_combined_loss" \
    --model_id "custom_combined_ls0.12_cp0.08"

# =============================================================================
# 比较实验：不同损失函数对比
# =============================================================================
echo ""
echo "📊 比较实验: 损失函数性能对比"
echo "========================================="

# 创建对比实验的基础配置（较少的epoch用于快速对比）
COMPARE_ARGS="--task_name classification --model DualGAFNet --data DualGAF \
              --root_path ./dataset/SAHU/ --seq_len 96 --step 96 --num_class 9 \
              --feature_dim 64 --batch_size 32 --train_epochs 20 \
              --extractor_type resnet18_gaf_light --fusion_type adaptive \
              --checkpoints ./checkpoints/loss_comparison/ --result_path ./result/loss_comparison/"

echo "运行对比实验..."

# 基线：标准交叉熵
python run.py $COMPARE_ARGS \
    --loss_type ce \
    --des "baseline_cross_entropy" \
    --model_id "baseline_ce"

# 标签平滑
python run.py $COMPARE_ARGS \
    --loss_type label_smoothing \
    --label_smoothing 0.1 \
    --des "label_smoothing_0.1" \
    --model_id "ls_0.1"

# Focal Loss
python run.py $COMPARE_ARGS \
    --loss_type focal \
    --focal_gamma 2.0 \
    --des "focal_loss_gamma_2.0" \
    --model_id "focal_g2.0"

# 组合损失
python run.py $COMPARE_ARGS \
    --loss_type combined \
    --label_smoothing 0.1 \
    --confidence_penalty_beta 0.05 \
    --des "combined_ls_cp" \
    --model_id "combined_ls0.1_cp0.05"

echo ""
echo "✅ 所有实验完成！"
echo ""
echo "📋 结果分析建议："
echo "1. 检查 ./result/ 目录下的混淆矩阵，观察误分类模式"
echo "2. 比较不同损失函数的验证准确率和F1分数"
echo "3. 关注容易混淆的类别对，选择最适合的损失函数"
echo "4. 如果某些类别持续被误分类，考虑："
echo "   - 增加标签平滑因子 (0.15-0.25)"
echo "   - 使用更强的Focal Loss (gamma=3.0-4.0)"
echo "   - 结合特征层面的改进（更好的特征提取器）"
echo "   - 检查数据质量和特征工程"
echo "   - 仅当数据不平衡时考虑类别权重"

echo ""
echo "🔧 调参建议："
echo "标签平滑因子选择："
echo "  - 高度相似类别: 0.15-0.25"
echo "  - 中等相似类别: 0.08-0.15"
echo "  - 差异较大类别: 0.05-0.08"
echo ""
echo "Focal Loss参数选择："
echo "  - gamma=1.0: 轻微聚焦难样本"
echo "  - gamma=2.0: 标准聚焦 (推荐起始值)"
echo "  - gamma=3.0-4.0: 强烈聚焦极难样本"
echo "  - alpha=0.25: 降低易分类样本权重"
echo "  - alpha=1.0: 平衡权重" 
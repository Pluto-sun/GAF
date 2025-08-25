#!/bin/bash

# 梯度累积功能使用示例脚本
# 适用于小batch_size训练场景，减少梯度噪声

echo "🚀 梯度累积功能使用示例"
echo "======================================================"

# 确保在test_env环境中运行
if [[ "$CONDA_DEFAULT_ENV" != "test_env" ]]; then
    echo "⚠️  请先激活test_env环境: conda activate test_env"
    exit 1
fi

echo "📋 当前配置:"
echo "   环境: $CONDA_DEFAULT_ENV"
echo "   项目路径: $(pwd)"

# 示例1: 手动设置梯度累积
echo ""
echo "🔧 示例1: 手动设置梯度累积步数为2"
echo "   实际batch_size: 4"
echo "   有效batch_size: 4 × 2 = 8"
echo "   命令: python run.py --model DualGAFNet --data DualGAF --batch_size 4 --gradient_accumulation_steps 2 --loss_preset hvac_similar_optimized --train_epochs 2"
echo "------------------------------------------------------"

python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 2 \
    --des "gradient_accumulation_manual"

echo ""
echo "✅ 示例1完成"

# 示例2: 自动梯度累积
echo ""
echo "🤖 示例2: 自动梯度累积（当batch_size<8时自动设为2）"
echo "   实际batch_size: 4"
echo "   自动设置累积步数: 2"
echo "   有效batch_size: 4 × 2 = 8"
echo "   命令: python run.py --model DualGAFNet --data DualGAF --batch_size 4 --enable_auto_gradient_accumulation --loss_preset hvac_similar_optimized --train_epochs 2"
echo "------------------------------------------------------"

python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 4 \
    --enable_auto_gradient_accumulation \
    --loss_preset hvac_similar_optimized \
    --train_epochs 2 \
    --des "gradient_accumulation_auto"

echo ""
echo "✅ 示例2完成"

# 示例3: 大batch_size，不使用梯度累积
echo ""
echo "📊 示例3: 大batch_size对比（不使用梯度累积）"
echo "   实际batch_size: 8"
echo "   梯度累积: 禁用"
echo "   有效batch_size: 8"
echo "   命令: python run.py --model DualGAFNet --data DualGAF --batch_size 8 --gradient_accumulation_steps 1 --loss_preset hvac_similar_optimized --train_epochs 2"
echo "------------------------------------------------------"

python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 2 \
    --des "no_gradient_accumulation"

echo ""
echo "✅ 示例3完成"

# 示例4: 非常小的batch_size，使用更大的累积步数
echo ""
echo "🔥 示例4: 极小batch_size，累积步数为4"
echo "   实际batch_size: 2"
echo "   累积步数: 4"
echo "   有效batch_size: 2 × 4 = 8"
echo "   命令: python run.py --model DualGAFNet --data DualGAF --batch_size 2 --gradient_accumulation_steps 4 --loss_preset hvac_similar_optimized --train_epochs 2"
echo "------------------------------------------------------"

python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 2 \
    --des "gradient_accumulation_large"

echo ""
echo "✅ 示例4完成"

echo ""
echo "🎉 所有示例完成！"
echo "======================================================"
echo "📊 结果总结:"
echo "   - 手动梯度累积: 精确控制累积步数"
echo "   - 自动梯度累积: 智能判断是否需要累积"
echo "   - 对比实验: 验证不同batch_size的效果"
echo ""
echo "💡 建议:"
echo "   1. batch_size=4时，推荐gradient_accumulation_steps=2"
echo "   2. batch_size=2时，推荐gradient_accumulation_steps=4"
echo "   3. batch_size>=8时，通常不需要梯度累积"
echo "   4. 结合优化损失函数(hvac_similar_optimized)和RAdam优化器效果更佳"
echo ""
echo "📁 查看结果: ls -la result/" 
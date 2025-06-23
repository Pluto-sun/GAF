#!/bin/bash

# HVAC异常检测GNN训练脚本
# 作者：HVAC研究生
# 用途：基于图神经网络的HVAC异常检测

echo "========================================"
echo "HVAC异常检测GNN训练系统"
echo "========================================"

# 检查Python环境
echo "检查Python环境..."
python --version
echo ""

# 检查必要的包
echo "检查必要的Python包..."
python -c "
import torch
import torch_geometric
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
print('所有必要的包都已安装')
"

if [ $? -ne 0 ]; then
    echo "错误：缺少必要的Python包"
    echo "请安装以下包："
    echo "pip install torch torch-geometric pandas numpy scikit-learn matplotlib tqdm"
    exit 1
fi

# 设置默认参数
DATA_DIR="./dataset/SAHU/5"  # 数据文件夹路径
WINDOW_SIZE=60     # 时间窗口大小
OVERLAP=0.5        # 窗口重叠率
BATCH_SIZE=32      # 批次大小
HIDDEN_DIM=64      # 隐藏层维度
NUM_EPOCHS=100     # 训练轮数
LEARNING_RATE=0.001 # 学习率
TEST_SIZE=0.2      # 测试集比例
MODEL_NAME="hvac_gnn_model.pth"  # 模型保存名称

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --test_size)
            TEST_SIZE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --data_dir      数据文件夹路径 (默认: ./data)"
            echo "  --window_size   时间窗口大小 (默认: 60)"
            echo "  --overlap       窗口重叠率 (默认: 0.5)"
            echo "  --batch_size    批次大小 (默认: 32)"
            echo "  --hidden_dim    隐藏层维度 (默认: 64)"
            echo "  --num_epochs    训练轮数 (默认: 100)"
            echo "  --lr            学习率 (默认: 0.001)"
            echo "  --test_size     测试集比例 (默认: 0.2)"
            echo "  --model_name    模型保存名称 (默认: hvac_gnn_model.pth)"
            echo "  --help          显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --data_dir ./my_data --num_epochs 200 --lr 0.01"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查数据文件夹是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误：数据文件夹 '$DATA_DIR' 不存在"
    echo "请确保数据文件夹路径正确，或使用 --data_dir 指定正确路径"
    exit 1
fi

# 检查CSV文件
CSV_COUNT=$(find "$DATA_DIR" -name "*.csv" | wc -l)
if [ $CSV_COUNT -eq 0 ]; then
    echo "错误：在 '$DATA_DIR' 中没有找到CSV文件"
    exit 1
elif [ $CSV_COUNT -ne 5 ]; then
    echo "警告：找到 $CSV_COUNT 个CSV文件，期望5个文件（对应5种异常类型）"
    echo "继续执行..."
fi

echo "找到 $CSV_COUNT 个CSV文件"
echo "数据文件列表："
find "$DATA_DIR" -name "*.csv" | sort

# 显示训练参数
echo ""
echo "========================================"
echo "训练参数配置"
echo "========================================"
echo "数据文件夹: $DATA_DIR"
echo "时间窗口大小: $WINDOW_SIZE"
echo "窗口重叠率: $OVERLAP"
echo "批次大小: $BATCH_SIZE"
echo "隐藏层维度: $HIDDEN_DIM"
echo "训练轮数: $NUM_EPOCHS"
echo "学习率: $LEARNING_RATE"
echo "测试集比例: $TEST_SIZE"
echo "模型保存名称: $MODEL_NAME"
echo ""

# 创建结果文件夹
RESULTS_DIR="./result/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "结果将保存到: $RESULTS_DIR"

# 询问是否继续
read -p "是否开始训练？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

echo ""
echo "========================================"
echo "开始训练..."
echo "========================================"

# 执行训练
python GNN/train_hvac_gnn.py \
    --data_dir "$DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --hidden_dim "$HIDDEN_DIM" \
    --num_epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --test_size "$TEST_SIZE" \
    --save_model "$MODEL_NAME" \
    --use_attention \
    --results_dir "$RESULTS_DIR"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "训练成功完成！"
    echo "========================================"
    
    echo "结果文件已保存到: $RESULTS_DIR"
    echo "包含文件："
    ls -la "$RESULTS_DIR"
    
    echo ""
    echo "训练完成！你可以在 $RESULTS_DIR 中查看结果。"
else
    echo ""
    echo "========================================"
    echo "训练失败！"
    echo "========================================"
    echo "请检查错误信息并重试"
    exit 1
fi
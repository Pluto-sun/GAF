#!/bin/bash

# 设置PYTHONPATH为项目根目录
export PYTHONPATH=$(cd "$(dirname "$0")/.."; pwd)

# 设置参数
ROOT_PATH="./dataset/SAHU/"      # 数据根目录
WIN_SIZE=1                       # 不用窗口，设为1即可
STEP=1                           # 步长，设为1即可
NUM_CLASS=5                      # 类别数，根据你的数据实际情况填写
VAL_RATIO=0.3                    # 验证集比例

# 运行基线实验
python exp/exp_baseline.py \
    --root_path $ROOT_PATH \
    --win_size $WIN_SIZE \
    --step $STEP \
    --num_class $NUM_CLASS \
    --val_ratio $VAL_RATIO
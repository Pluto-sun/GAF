#!/bin/bash
eval "$(conda shell.bash hook)"  # 初始化 conda 支持
conda activate gaf     # 替换为你的环境名，例如 test_env
# 设置PYTHONPATH为项目根目录
export PYTHONPATH=$(cd "$(dirname "$0")/.."; pwd)

# 设置参数
ROOT_PATH="./dataset/DDAHU/direct_5_working"      # 数据根目录
WIN_SIZE=1                       # 不用窗口，设为1即可
STEP=1                           # 步长，设为1即可
NUM_CLASS=29                      # 类别数，根据你的数据实际情况填写
VAL_RATIO=0.2                    # 验证集比例

# 运行基线实验
python -m exp.exp_baseline \
    --root_path $ROOT_PATH \
    --win_size $WIN_SIZE \
    --step $STEP \
    --num_class $NUM_CLASS \
    --val_ratio $VAL_RATIO \
    --max_samples 10000
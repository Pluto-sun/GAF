# 双路GAF网络 (DualGAFNet) 实现

## 概述

本项目实现了一个双路GAF（Gramian Angular Field）网络架构，能够同时处理summation和difference两种GAF变换方法的特征，通过特征融合和信号注意力机制提升HVAC异常检测的性能。

## 架构设计

### 1. 网络结构
```
输入: Summation GAF + Difference GAF
    ↓                    ↓
特征提取器1              特征提取器2
    ↓                    ↓
        特征融合模块
            ↓
        信号注意力模块
            ↓
          分类器
            ↓
          输出
```

### 2. 核心模块

#### 特征提取器 (Extractor)
- **large_kernel**: 大核无填充卷积特征提取器（默认）
- **inception**: Inception网络特征提取器
- **dilated**: 膨胀卷积特征提取器
- **multiscale**: 多尺度特征提取器

#### 特征融合模块 (Fusion)
- **adaptive**: 自适应加权融合（学习融合权重）
- **concat**: 简单拼接融合
- **bidirectional**: 双向注意力融合（使用多头注意力机制，特征维度翻倍）
- **gated**: 门控机制融合（使用门控单元控制信息流）
- **add**: 逐元素相加
- **mul**: 逐元素相乘
- **weighted_add**: 固定权重相加

#### 注意力模块 (Attention)
- **channel**: 通道注意力机制
- **spatial**: 空间注意力机制
- **cbam**: CBAM注意力（通道+空间）
- **self**: 自注意力机制
- **none**: 无注意力机制

#### 分类器 (Classifier)
- **mlp**: 多层感知机分类器（包含Dropout）
- **simple**: 简单线性分类器

## 文件结构

```
├── models/
│   ├── DualGAFNet.py              # 双路GAF网络实现
│   └── MultiImageFeatureNet.py    # 基础特征提取器
├── data_provider/
│   ├── data_loader/
│   │   └── DualGAFDataLoader.py   # 双路GAF数据加载器
│   └── data_factory.py            # 数据工厂（已支持DualGAF）
├── exp/
│   └── exp.py                     # 实验类（已支持双路数据）
├── scripts/
│   └── DualGAF.sh                # 双路网络实验脚本
└── run.py                         # 主运行文件（已添加参数）
```

## 使用方法

### 1. 基础使用

```bash
python run.py \
    --task_name classification \
    --is_training 1 \
    --model_id SAHU_dual \
    --model DualGAFNet \
    --data DualGAF \
    --root_path "./dataset/SAHU/" \
    --seq_len 72 \
    --step 12 \
    --batch_size 32 \
    --feature_dim 64 \
    --extractor_type large_kernel \
    --fusion_type adaptive \
    --attention_type channel \
    --classifier_type mlp \
    --des "dual-gaf-test"
```

### 2. 使用HVAC信号分组

```bash
python run.py \
    ... # 其他参数
    --hvac_groups "SA_TEMP,OA_TEMP,MA_TEMP|OA_CFM,RA_CFM,SA_CFM|SA_SP,SA_SPSPT" \
    --des "with-hvac-groups"
```

### 3. 批量实验

```bash
# 运行预定义的6个实验配置
bash scripts/DualGAF.sh
```

## 参数说明

### 核心参数
- `--model DualGAFNet`: 使用双路GAF网络
- `--data DualGAF`: 使用双路GAF数据加载器
- `--feature_dim`: 特征维度（默认32）

### 模块配置参数
- `--extractor_type`: 特征提取器类型
- `--fusion_type`: 特征融合方式
- `--attention_type`: 注意力机制类型
- `--classifier_type`: 分类器类型

### HVAC分组参数
- `--hvac_groups`: HVAC信号分组配置，格式："group1_signals|group2_signals|..."

### 数据参数
- `--data_type_method`: 数据类型转换方法（float32/uint8/uint16）
- `--test_size`: 验证集比例（默认0.2）

## 数据格式

### 输入数据
双路GAF数据加载器返回三元组：
```python
(summation_gaf, difference_gaf, label)
```

- `summation_gaf`: [batch_size, channels, height, width] - Summation GAF图像
- `difference_gaf`: [batch_size, channels, height, width] - Difference GAF图像
- `label`: [batch_size] - 分类标签

### 数据预处理
1. 自动检测和加载持久化数据（如果存在）
2. 通道级别归一化到[-1, 1]
3. 双GAF转换（summation和difference）
4. 数据类型转换（支持内存优化）
5. 数据集划分（训练/验证）

## 实验配置示例

脚本`scripts/DualGAF.sh`提供了6种不同的配置组合：

1. **基础配置**: large_kernel + adaptive + channel + mlp
2. **Inception配置**: inception + concat + spatial + mlp
3. **膨胀卷积配置**: dilated + weighted_add + cbam + simple
4. **多尺度配置**: multiscale + mul + self + mlp
5. **简化配置**: large_kernel + add + none + simple（无分组，无注意力）
6. **高维配置**: large_kernel + adaptive + channel + mlp（feature_dim=128）

## 性能优化

### 内存优化
- 支持uint8/uint16数据类型存储，减少内存占用
- 分批处理大型数据集
- 持久化预处理结果

### 训练优化
- 早停机制（基于F1分数）
- 梯度裁剪（max_norm=4.0）
- 学习率调度
- 多种评估指标（准确率、F1、精确率、召回率）

## 结果分析

训练完成后会生成：
- 混淆矩阵可视化
- 综合训练结果图表（包含多种指标曲线）
- 详细的CSV指标记录
- 模型检查点

## 扩展性

### 添加新的特征提取器
在`DualGAFNet.py`中的`_create_extractor`方法中添加新类型：

```python
elif self.extractor_type == 'your_new_type':
    return YourNewExtractor(self.feature_dim)
```

### 添加新的融合方式
在`DualGAFNet.py`中创建新的融合类并在`_build_fusion_module`中注册。

### 添加新的注意力机制
在`SignalAttention`类中添加新的注意力类型。

## 注意事项

1. **数据集路径**: 确保`root_path`指向正确的数据集目录
2. **GPU内存**: 根据GPU内存调整`batch_size`和`feature_dim`
3. **分组配置**: HVAC分组需要与实际特征列名匹配
4. **数据类型**: uint8模式可节省内存但可能影响精度
5. **实验记录**: 结果保存在`result/`目录下，包含时间戳

## 常见问题

### Q: 如何添加新的数据集？
A: 参考`DualGAFDataLoader.py`实现新的数据加载器，并在`data_factory.py`中注册。

### Q: 如何调整网络结构？
A: 修改`DualGAFNet.py`中相应的模块构建方法，或者通过参数控制现有模块的行为。

### Q: 内存不足怎么办？
A: 减小`batch_size`、使用`uint8`数据类型、或减小`feature_dim`。

### Q: 如何获得最佳性能？
A: 运行多种配置组合，使用网格搜索或贝叶斯优化调参。

## 依赖项

- PyTorch
- PyTorch Geometric (用于图网络支持)
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- pyts (用于GAF转换)

确保所有依赖项已正确安装并且版本兼容。 
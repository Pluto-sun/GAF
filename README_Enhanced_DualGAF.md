# 增强版双路GAF网络 (Enhanced Dual-GAF Network)

## 概述

增强版双路GAF网络是在原有双路GAF网络基础上添加了统计特征提取器和多模态融合功能的改进版本。该网络能够同时利用GAF图像的时间依赖特征和原始时序数据的统计特征，显著提升分类性能。

## 主要特性

### 1. 多模态架构
- **GAF图像分支**: 处理Summation和Difference GAF图像，捕捉时间序列的时间依赖关系
- **统计特征分支**: 从原始时序数据提取统计特征和信号间关联，关注静态关系
- **多模态融合**: 将两种特征进行有效融合，互补增强

### 2. 统计特征提取器
提供三种统计特征类型：

#### Basic (基础统计)
- 均值、标准差、最大值、最小值、中位数
- 特征维度: `num_signals × 5`

#### Comprehensive (综合统计)
- 基础统计 + 高阶统计（偏度、峰度、变异系数）
- 信号间相关性矩阵
- 特征维度: `num_signals × 8 + correlation_pairs`

#### Correlation-focused (相关性聚焦)
- 基础统计 + 相关性矩阵 + 交叉统计特征
- 专注于信号间的关系分析
- 特征维度: `num_signals × 4 + correlation_pairs + cross_features`

### 3. 多模态融合策略
- **Concat**: 简单拼接后线性投影
- **Attention**: 使用注意力机制融合
- **Gated**: 门控机制控制融合权重
- **Adaptive**: 自适应学习融合权重

### 4. 数据兼容性
- **向后兼容**: 支持原有的三元组数据格式 `(sum_gaf, diff_gaf, label)`
- **增强格式**: 支持新的四元组数据格式 `(sum_gaf, diff_gaf, time_series, label)`
- **自动检测**: 根据数据格式自动选择处理方式

## 网络架构

```
原始时序数据 [B, C, T] ──┐
                         │
                         ├─→ 统计特征提取器 ──┐
                         │                    │
Summation GAF [B,C,H,W] ─┼─→ GAF特征提取器 ──┼─→ 多模态融合 ─→ 信号注意力 ─→ 分类器
                         │                    │
Difference GAF [B,C,H,W] ─┘                  │
                                             │
                                             ↓
                                        融合特征 [B,C,D]
```

## 使用方法

### 1. 基础使用

```python
from models.DualGAFNet import Model

# 创建配置
configs = type("cfg", (), {
    "feature_dim": 64,
    "num_class": 4,
    "enc_in": 26,
    "seq_len": 96,
    "use_statistical_features": True,
    "stat_type": "comprehensive",
    "multimodal_fusion_strategy": "concat"
})()

# 创建模型
model = Model(configs)

# 前向传播
sum_gaf = torch.randn(4, 26, 32, 32)      # Summation GAF
diff_gaf = torch.randn(4, 26, 32, 32)     # Difference GAF  
time_series = torch.randn(4, 26, 96)      # 原始时序数据

output = model(sum_gaf, diff_gaf, time_series)
```

### 2. 命令行使用

```bash
# 基础增强版网络
python run_enhanced_dual_gaf.py \
    --model DualGAFNet \
    --data DualGAF \
    --use_statistical_features \
    --stat_type comprehensive \
    --multimodal_fusion_strategy concat

# 使用HVAC信号分组
python run_enhanced_dual_gaf.py \
    --model DualGAFNet \
    --data DualGAF \
    --use_statistical_features \
    --use_hvac_groups \
    --stat_type correlation_focused \
    --multimodal_fusion_strategy attention
```

### 3. 批量实验

```bash
# 运行多个对比实验
bash scripts/run_enhanced_dual_gaf_example.sh
```

## 配置参数

### 统计特征相关
- `--use_statistical_features`: 是否启用统计特征（默认: True）
- `--stat_type`: 统计特征类型 [basic|comprehensive|correlation_focused]
- `--multimodal_fusion_strategy`: 多模态融合策略 [concat|attention|gated|adaptive]

### 模型架构相关
- `--extractor_type`: 特征提取器类型 [large_kernel|inception|dilated|multiscale]
- `--fusion_type`: GAF融合类型 [adaptive|concat|add|mul|weighted_add|bidirectional|gated]
- `--attention_type`: 注意力类型 [channel|spatial|cbam|self|none]
- `--classifier_type`: 分类器类型 [mlp|simple]

### HVAC分组相关
- `--use_hvac_groups`: 是否使用HVAC信号分组

## 技术优势

### 1. 多模态学习
- **互补特征**: GAF图像捕捉时间依赖，统计特征捕捉静态关系
- **信息融合**: 有效结合两种模态的优势
- **性能提升**: 特别适合混淆类别的区分

### 2. 灵活配置
- **模块化设计**: 各组件可独立配置和替换
- **多种策略**: 提供多种融合和注意力策略选择
- **向后兼容**: 支持原有数据格式，平滑升级

### 3. 统计特征丰富
- **多层次统计**: 从基础到高阶统计特征
- **关系建模**: 专门建模信号间的相关性
- **自适应投影**: 自动学习最优特征表示

## 实验建议

### 1. 消融实验
- 对比有无统计特征的性能差异
- 测试不同统计特征类型的效果
- 评估不同融合策略的性能

### 2. 参数调优
- 调整特征维度 (32, 64, 128)
- 尝试不同的融合策略组合
- 测试不同的注意力机制

### 3. 数据特异性
- 针对HVAC数据使用信号分组
- 根据数据特点选择统计特征类型
- 考虑数据不平衡的影响

## 文件结构

```
models/
├── DualGAFNet.py                    # 增强版双路GAF网络
│   ├── TimeSeriesStatisticalExtractor   # 统计特征提取器
│   ├── MultiModalFusion                 # 多模态融合模块
│   └── DualGAFNet                       # 主网络架构

data_provider/data_loader/
└── DualGAFDataLoader.py             # 支持四元组数据的加载器

exp/
└── exp.py                           # 支持新数据格式的实验框架

scripts/
└── run_enhanced_dual_gaf_example.sh # 示例运行脚本

run_enhanced_dual_gaf.py            # 主运行脚本
test_enhanced_dual_gaf.py           # 功能测试脚本
```

## 注意事项

1. **内存使用**: 统计特征提取会增加一定的内存开销
2. **计算复杂度**: 相关性计算的复杂度为O(C²)，C为信号数量
3. **数据格式**: 确保原始时序数据的格式正确 [B, C, T] 或 [B, T, C]
4. **特征维度**: 统计特征维度会根据信号数量和统计类型自动调整

## 性能预期

基于设计思路，增强版网络在以下场景中表现更好：
- **混淆类别区分**: 统计特征有助于区分GAF图像相似的类别
- **信号关联性**: 能够捕捉信号间的复杂关系
- **多模态优势**: 结合时间和静态特征的优势

建议在实际应用中进行充分的对比实验，验证增强效果。 
# 双路GAF网络消融实验指南

## 🔬 概述

消融实验是验证模型各组件有效性的重要方法。本指南详细介绍如何使用双路GAF网络的消融实验功能来证明每个模块的贡献。

## 📋 支持的消融实验

### 1. GAF差分分支消融
- **控制参数**: `--use_diff_branch False`
- **作用**: 禁用差分分支，仅使用sum分支进行特征提取
- **目的**: 验证GAF融合（sum + diff）相比单一GAF的优势

### 2. 统计特征消融
- **控制参数**: `--use_statistical_features False`
- **作用**: 禁用时序统计特征提取和多模态融合
- **目的**: 验证统计特征对多模态学习的贡献

### 3. 注意力机制消融
- **控制参数**: `--attention_type none`
- **作用**: 禁用信号注意力模块
- **目的**: 验证注意力机制对信号建模的重要性

## 🚀 快速开始

### 方法1：使用消融模式快捷设置

```bash
# 完整模型（基线）
python run.py --model DualGAFNet --data SAHU --ablation_mode none

# 移除差分分支
python run.py --model DualGAFNet --data SAHU --ablation_mode no_diff

# 移除统计特征
python run.py --model DualGAFNet --data SAHU --ablation_mode no_stat

# 移除注意力机制
python run.py --model DualGAFNet --data SAHU --ablation_mode no_attention

# 最简化模型（移除所有高级组件）
python run.py --model DualGAFNet --data SAHU --ablation_mode minimal
```

### 方法2：手动设置各个开关

```bash
# 自定义消融组合
python run.py --model DualGAFNet --data SAHU \
    --use_diff_branch False \
    --use_statistical_features True \
    --attention_type channel

# 组合消融：移除差分分支和注意力
python run.py --model DualGAFNet --data SAHU \
    --use_diff_branch False \
    --attention_type none
```

## 📊 测试结果分析

### 模型复杂度对比

| 实验配置 | 参数数量 | 模型大小(MB) | 推理时间(ms) | 复杂度减少 |
|---------|---------|-------------|-------------|-----------|
| 完整模型（基线） | 26,934,845 | 102.75 | 15.64* | - |
| 移除差分分支 | 15,742,685 | 60.05 | 4.14 | ↓41.5% |
| 移除统计特征 | 26,901,437 | 102.62 | 10.36 | ↓0.1% |
| 移除注意力机制 | 26,934,660 | 102.75 | 10.69 | ↓0.0% |
| 最简化模型 | 15,709,092 | 59.93 | 3.38 | ↓41.7% |

*注：第一次运行包含初始化时间

### 关键发现

1. **差分分支的影响最大**：
   - 移除后参数减少41.5%，推理时间大幅降低
   - 表明GAF融合是计算复杂度的主要来源

2. **统计特征影响较小**：
   - 参数数量几乎无变化
   - 主要影响多模态学习能力

3. **注意力机制开销很低**：
   - 参数增加微乎其微
   - 性价比很高的组件

## 🔧 详细配置参数

### 消融实验开关

```python
# 在run.py中的参数设置
parser.add_argument('--use_diff_branch', action='store_true', default=True,
                   help='是否使用差分分支进行GAF融合')

parser.add_argument('--ablation_mode', type=str, default='none',
                   choices=['none', 'no_diff', 'no_stat', 'no_attention', 'minimal'],
                   help='消融实验模式快捷设置')
```

### 支持的消融模式

| 模式 | 描述 | 等价设置 |
|-----|------|---------|
| `none` | 完整模型 | 所有组件启用 |
| `no_diff` | 移除差分分支 | `use_diff_branch=False` |
| `no_stat` | 移除统计特征 | `use_statistical_features=False` |
| `no_attention` | 移除注意力机制 | `attention_type=none` |
| `minimal` | 最简化模型 | 所有高级组件禁用 |

## 📈 推荐的实验流程

### 1. 基线实验
```bash
python run.py --model DualGAFNet --data SAHU --step 72 --seq_len 72 \
    --train_epochs 20 --batch_size 4 --feature_dim 32 \
    --use_statistical_features --use_diff_branch --attention_type channel \
    --des "baseline_full_model"
```

### 2. 单一组件消融
```bash
# 实验A：移除差分分支
python run.py --model DualGAFNet --data SAHU --step 72 --seq_len 72 \
    --train_epochs 20 --batch_size 4 --feature_dim 32 \
    --ablation_mode no_diff --des "ablation_no_diff"

# 实验B：移除统计特征
python run.py --model DualGAFNet --data SAHU --step 72 --seq_len 72 \
    --train_epochs 20 --batch_size 4 --feature_dim 32 \
    --ablation_mode no_stat --des "ablation_no_stat"

# 实验C：移除注意力机制
python run.py --model DualGAFNet --data SAHU --step 72 --seq_len 72 \
    --train_epochs 20 --batch_size 4 --feature_dim 32 \
    --ablation_mode no_attention --des "ablation_no_attention"
```

### 3. 组合消融实验
```bash
# 实验D：移除差分分支 + 统计特征
python run.py --model DualGAFNet --data SAHU --step 72 --seq_len 72 \
    --train_epochs 20 --batch_size 4 --feature_dim 32 \
    --use_diff_branch False --use_statistical_features False \
    --des "ablation_no_diff_no_stat"

# 实验E：最简化模型
python run.py --model DualGAFNet --data SAHU --step 72 --seq_len 72 \
    --train_epochs 20 --batch_size 4 --feature_dim 32 \
    --ablation_mode minimal --des "ablation_minimal"
```

## 📊 结果评估指标

### 1. 性能指标
- **准确率 (Accuracy)**: 整体分类性能
- **F1分数 (F1-score)**: 平衡精确率和召回率
- **精确率 (Precision)**: 正确预测的正样本比例
- **召回率 (Recall)**: 实际正样本被正确预测的比例

### 2. 效率指标
- **参数数量**: 模型复杂度
- **推理时间**: 计算效率
- **内存占用**: 资源需求
- **训练时间**: 收敛速度

### 3. 分析维度
- **组件贡献度**: (基线性能 - 消融性能) / 基线性能
- **复杂度收益**: 性能提升 / 复杂度增加
- **计算效率**: 性能 / 推理时间

## 🎯 实际应用建议

### 1. 产品部署场景
- **高性能要求**: 使用完整模型
- **资源受限**: 考虑移除差分分支（减少41%参数）
- **实时性要求**: 使用最简化模型（最快推理）

### 2. 研究验证场景
- **论文写作**: 完整的消融实验增强可信度
- **组件优化**: 识别性价比最高的组件
- **架构设计**: 指导后续模型改进方向

### 3. 调试和优化
- **性能异常**: 通过消融定位问题组件
- **过拟合问题**: 简化模型降低复杂度
- **训练不稳定**: 逐步添加组件找到平衡点

## 🔍 详细测试示例

### 运行消融实验测试
```bash
# 切换到test_env环境（如果需要）
conda activate test_env

# 运行消融实验测试脚本
python test_ablation_experiments.py

# 或使用提供的运行脚本
chmod +x scripts/run_ablation_experiments.sh
./scripts/run_ablation_experiments.sh
```

### 测试输出示例
```
🔬 消融实验状态: GAF差分分支消融
   ❌ 禁用差分分支 (use_diff_branch=False)
   
🔧 参数数量: 15,742,685 (训练: 15,742,685)
💾 模型大小: 60.05 MB
⚡ 推理时间: 4.14ms
```

## 🚨 注意事项

### 1. 数据格式要求
- **启用统计特征时**: 必须提供time_series_x数据
- **禁用统计特征时**: 可以不提供time_series_x，但数据加载器需要兼容

### 2. 参数兼容性
- 消融实验会自动调整相关参数
- 手动设置的参数优先级高于自动设置

### 3. 结果解释
- 第一次运行推理时间可能较长（包含初始化）
- 关注相对性能变化而非绝对数值
- 考虑多次运行取平均值增加可靠性

## 📚 扩展阅读

- **消融实验原理**: 通过移除组件验证其必要性
- **双路GAF网络架构**: 理解各组件的设计目的
- **HVAC异常检测**: 了解具体应用场景需求
- **模型优化策略**: 根据消融结果指导优化方向

---

## 💡 快速命令参考

```bash
# 完整模型测试
python run.py --model DualGAFNet --data SAHU --des "full_model"

# 移除差分分支
python run.py --model DualGAFNet --data SAHU --ablation_mode no_diff

# 移除统计特征  
python run.py --model DualGAFNet --data SAHU --ablation_mode no_stat

# 移除注意力机制
python run.py --model DualGAFNet --data SAHU --ablation_mode no_attention

# 最简化模型
python run.py --model DualGAFNet --data SAHU --ablation_mode minimal

# 自定义组合
python run.py --model DualGAFNet --data SAHU \
    --use_diff_branch False --attention_type none
```

通过系统性的消融实验，您可以深入理解双路GAF网络各组件的贡献，为模型优化和实际部署提供科学依据。 
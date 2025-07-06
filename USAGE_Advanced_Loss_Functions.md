# 高级损失函数使用指南

## 🎯 解决类别相似性问题

当您遇到**某个类别容易被误分类**的问题时，可以使用以下高级损失函数：

### 1. 标签平滑（Label Smoothing）- 推荐首选 ⭐⭐⭐

**适用场景**：类别间特征相似度高，模型过度自信

```bash
# 方案1：使用预设配置（推荐）
python run.py --model DualGAFNet --data DualGAF \
    --loss_preset hvac_similar

# 方案2：手动配置
python run.py --model DualGAFNet --data DualGAF \
    --loss_type label_smoothing \
    --label_smoothing 0.15
```

### 2. Focal Loss - 难分类样本聚焦 ⭐⭐

**适用场景**：存在特别难分类的样本

```bash
# 方案1：使用预设配置
python run.py --model DualGAFNet --data DualGAF \
    --loss_preset hard_samples

# 方案2：手动配置
python run.py --model DualGAFNet --data DualGAF \
    --loss_type focal \
    --focal_gamma 3.0 \
    --focal_alpha 0.25
```

### 3. 组合损失 - 综合解决方案 ⭐⭐⭐

**适用场景**：同时解决多个问题

```bash
python run.py --model DualGAFNet --data DualGAF \
    --loss_preset overconfidence_prevention
```

## 📊 类别权重配置

### ✅ 默认行为（推荐）

对于**平衡数据集**（各类别样本数量一致），系统默认**不使用类别权重**：

```bash
# 无需额外配置，类别权重默认禁用
python run.py --model DualGAFNet --data DualGAF \
    --loss_type label_smoothing
```

### 🔧 启用类别权重（仅用于不平衡数据）

如果您的数据确实不平衡，可以手动启用：

```bash
# 启用自动计算的类别权重
python run.py --model DualGAFNet --data DualGAF \
    --loss_type focal \
    --enable_class_weights

# 手动指定类别权重
python run.py --model DualGAFNet --data DualGAF \
    --loss_type focal \
    --enable_class_weights \
    --class_weights "1.0,2.0,1.5,3.0,1.0,2.5,1.8,1.2,2.2"
```

## 🚀 快速开始

### 针对相似类别问题（最常用）

```bash
python run.py --model DualGAFNet --data DualGAF \
    --loss_preset hvac_similar \
    --des "solve_similar_classes"
```

这将使用：
- 标签平滑因子：0.15
- 无类别权重（平衡数据集）
- 适合HVAC异常检测中的相似故障模式

### 对比实验

运行多种损失函数对比：

```bash
bash scripts/run_with_advanced_loss.sh
```

## 🔧 参数调优建议

### 标签平滑因子选择

| 类别相似程度 | 推荐值 | 说明 |
|------------|--------|------|
| 高度相似 | 0.15-0.25 | HVAC相似故障模式 |
| 中等相似 | 0.08-0.15 | 部分特征重叠 |
| 差异较大 | 0.05-0.08 | 特征区分度较高 |

### Focal Loss参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| gamma=2.0 | 标准聚焦 | 推荐起始值 |
| gamma=3.0-4.0 | 强烈聚焦 | 严重不平衡时 |
| alpha=0.25 | 降低易分类样本权重 | 更关注难样本 |

## 💡 最佳实践

1. **首选标签平滑**：对大多数相似类别问题都有效
2. **观察混淆矩阵**：识别具体的误分类模式
3. **渐进式调参**：从较小的平滑因子开始
4. **无需类别权重**：平衡数据集默认禁用即可
5. **结合特征改进**：可配合更好的特征提取器使用

## 📋 常见问题

**Q: 什么时候需要启用类别权重？**
A: 仅当各类别样本数量差异很大时（如某类别样本数是其他的2倍以上）

**Q: 标签平滑会降低准确率吗？**
A: 在训练集上可能略有下降，但验证集性能通常会提升（更好的泛化）

**Q: 如何选择合适的损失函数？**
A: 建议从`--loss_preset hvac_similar`开始，观察效果后再调整

**Q: 可以同时使用多种技术吗？**
A: 可以，例如标签平滑 + 更好的特征提取器 + 数据增强 
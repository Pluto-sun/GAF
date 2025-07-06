# 残差分类器优化指南

## 🎯 概述

本指南介绍了为双路GAF网络新增的残差分类器功能，这是针对HVAC异常检测等复杂分类任务的性能优化。

## 🔬 残差分类器的优势

### 📈 理论优势
1. **梯度流动优化**：残差连接缓解梯度消失问题，提高深层网络训练稳定性
2. **特征重用机制**：允许不同抽象层次的特征组合，增强决策边界学习
3. **训练收敛性**：残差结构通常具有更快的收敛速度和更稳定的训练过程
4. **表达能力增强**：特别适合相似类别的细微差异学习

### 🎯 HVAC应用优势
- **相似异常区分**：HVAC异常类别间相似性高，残差连接有助于学习细微差异
- **多模态特征融合**：残差结构更好地处理GAF图像和统计特征的融合
- **复杂决策边界**：残差连接支持更复杂的非线性决策边界学习

## 🛠️ 支持的残差分类器类型

### 1. 基础残差分类器 (`residual`)
```bash
--classifier_type residual
```
**特点**：
- 结构：输入 → 残差块1 → 残差块2 → 残差块3 → 输出
- 参数量：中等（约28M）
- 推荐场景：通用场景，平衡性能与效率

**架构细节**：
- 隐藏层：[1024, 512, 256]
- Dropout：0.1
- 批归一化：禁用
- 残差块类型：基础残差块

### 2. 瓶颈残差分类器 (`residual_bottleneck`)
```bash
--classifier_type residual_bottleneck
```
**特点**：
- 结构：降维 → 处理 → 升维的瓶颈设计
- 参数量：较大（约35M）
- 推荐场景：高维特征、计算资源充足

**架构细节**：
- 隐藏层：[2048, 1024, 512]（高维）或 [1024, 512, 256]（中维）
- Dropout：0.15
- 批归一化：启用
- 残差块类型：瓶颈残差块

### 3. 密集残差分类器 (`residual_dense`)
```bash
--classifier_type residual_dense
```
**特点**：
- 结构：更深层的密集连接残差网络
- 参数量：适中（约29M）
- 推荐场景：需要最强表达能力的复杂任务

**架构细节**：
- 隐藏层：[1024, 512, 256, 128]
- Dropout：0.2
- 批归一化：启用
- 残差块类型：密集残差块

## 📊 性能对比（测试结果）

| 分类器类型 | 参数数量 | 模型大小 | 推理时间 | 分类器占比 | 推荐场景 |
|-----------|----------|----------|----------|------------|----------|
| `simple` | 23.3M | 88.7MB | 44.1ms | 3.7% | 资源受限 |
| `mlp` | 26.9M | 102.7MB | 44.8ms | 16.8% | 传统基线 |
| `residual` | 28.0M | 106.8MB | 44.2ms | 20.0% | **平衡推荐** |
| `residual_bottleneck` | 35.2M | 134.3MB | 44.4ms | 36.4% | 高维特征 |
| `residual_dense` | 29.1M | 111.1MB | 44.4ms | 23.0% | **最强性能** |

## 🚀 使用方法

### 基本用法
```bash
# 基础残差分类器（推荐）
python run.py --model DualGAFNet --data SAHU \
    --classifier_type residual \
    --feature_dim 64 --batch_size 4

# 密集残差分类器（最强性能）
python run.py --model DualGAFNet --data SAHU \
    --classifier_type residual_dense \
    --feature_dim 64 --batch_size 4

# 瓶颈残差分类器（高维特征）
python run.py --model DualGAFNet --data SAHU \
    --classifier_type residual_bottleneck \
    --feature_dim 64 --batch_size 4
```

### 与消融实验结合
```bash
# 残差分类器 + 消融实验
python run.py --model DualGAFNet --data SAHU \
    --classifier_type residual \
    --ablation_mode no_diff \
    --des "residual_no_diff"

# 密集残差分类器 + 无注意力机制
python run.py --model DualGAFNet --data SAHU \
    --classifier_type residual_dense \
    --ablation_mode no_attention \
    --des "residual_dense_no_attention"
```

### 与优化功能结合
```bash
# 残差分类器 + 梯度累积 + RAdam
python run.py --model DualGAFNet --data SAHU \
    --classifier_type residual_dense \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized \
    --use_statistical_features \
    --train_epochs 20
```

## 🔬 技术实现细节

### 残差块结构
```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = self.shortcut(x)  # 残差连接
        out = self.main_path(x)      # 主路径
        out = out + residual         # 残差相加
        out = self.final_activation(out)  # 最终激活
        return out
```

### 自动结构适配
- **输入维度**：根据`feature_dim * num_images`自动计算
- **隐藏层设计**：根据输入维度智能选择网络深度
- **权重初始化**：使用Kaiming初始化，适合ReLU激活函数

### 批归一化策略
- `residual`：不使用BatchNorm，保持轻量级
- `residual_bottleneck`：使用BatchNorm，提高训练稳定性
- `residual_dense`：使用BatchNorm，支持更深层网络

## 📈 推荐策略

### 🏆 性能优先
```bash
--classifier_type residual_dense
```
- 最强表达能力
- 适合复杂异常模式
- 训练时间略长但效果最佳

### ⚡ 效率优先
```bash
--classifier_type residual
```
- 平衡性能与计算效率
- 适合大多数应用场景
- 推理速度快

### 🎯 高维特征
```bash
--classifier_type residual_bottleneck
```
- 专为高维特征设计
- 适合feature_dim > 128的情况
- 瓶颈结构减少计算量

### 📱 资源受限
```bash
--classifier_type simple
```
- 最轻量级选择
- 适合边缘设备部署
- 基础性能保证

## 🧪 测试验证

### 功能测试
```bash
# 运行完整测试套件
python test_residual_classifier.py
```

### 性能基准测试
```bash
# 运行性能对比测试
bash scripts/test_residual_classifiers.sh
```

## 💡 最佳实践

### 1. 分类器选择策略
- **开发阶段**：使用`residual_dense`获得最佳性能
- **生产部署**：根据资源约束选择`residual`或`simple`
- **研究实验**：使用`residual_bottleneck`进行高维特征实验

### 2. 与其他功能配合
- **梯度累积**：残差分类器与梯度累积配合使用，提高小batch训练效果
- **损失函数**：推荐使用`hvac_similar_optimized`损失预设
- **学习率调度**：残差结构通常受益于ReduceLROnPlateau调度器

### 3. 超参数调优
- **Dropout**：残差分类器通常需要较低的dropout（0.1-0.2）
- **学习率**：可以使用稍高的学习率（1e-3），残差连接提供更稳定的梯度
- **批归一化**：对于`residual_bottleneck`和`residual_dense`，BatchNorm是关键

## 🔗 相关功能

- **消融实验**：参见 `ABLATION_EXPERIMENTS_GUIDE.md`
- **梯度累积**：参见 `GRADIENT_ACCUMULATION_GUIDE.md`
- **损失函数**：参见 `RADAM_SCHEDULER_GUIDE.md`
- **完整系统**：参见 `COMPLETE_SYSTEM_SUMMARY.md`

## 📞 问题排查

### 常见问题
1. **内存不足**：尝试使用`simple`或减少`feature_dim`
2. **训练不稳定**：启用BatchNorm（使用`residual_bottleneck`或`residual_dense`）
3. **过拟合**：增加dropout或使用数据增强
4. **收敛慢**：检查学习率设置，残差结构通常支持更高学习率

### 性能优化
1. **推理加速**：使用`torch.compile()`（PyTorch 2.0+）
2. **内存优化**：使用`--use_amp`启用混合精度训练
3. **并行化**：利用`--use_multi_gpu`进行多GPU训练 
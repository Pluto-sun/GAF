# 梯度累积功能使用指南

## 概述

梯度累积（Gradient Accumulation）是一种在小batch训练时减少梯度噪声的有效技术。通过在多个小batch上累积梯度后再进行参数更新，可以模拟大batch训练的效果，提高训练稳定性。

## 适用场景

✅ **推荐使用梯度累积的情况：**
- `batch_size < 8`：小batch训练梯度噪声较大
- GPU显存限制：无法使用更大的batch_size
- 需要稳定训练：减少训练过程中的波动
- 类别相似性问题：配合标签平滑等损失函数效果更佳

❌ **不建议使用梯度累积的情况：**
- `batch_size >= 16`：已经足够大，额外的累积意义不大
- 计算资源充足：可以直接增大batch_size
- 需要快速迭代：梯度累积会稍微增加计算开销

## 功能特性

### 1. 智能配置
- **自动梯度累积**：根据batch_size自动判断是否需要累积
- **手动精确控制**：用户可以精确指定累积步数
- **兼容性强**：完全兼容现有的训练流程和损失函数

### 2. 性能优化
- **内存高效**：不会显著增加显存使用
- **计算优化**：梯度裁剪在累积完成后统一进行
- **进度显示**：实时显示累积进度和有效batch size

### 3. 与现有功能集成
- **RAdam优化器**：完美配合RAdam的稳定训练特性
- **ReduceLROnPlateau**：学习率调度器正常工作
- **高级损失函数**：兼容标签平滑、Focal Loss等所有损失函数

## 使用方法

### 方式1: 手动设置累积步数

```bash
# 推荐配置：batch_size=4, 累积步数=2, 有效batch_size=8
python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 10
```

### 方式2: 自动梯度累积

```bash
# 系统自动判断是否需要梯度累积
python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 4 \
    --enable_auto_gradient_accumulation \
    --loss_preset hvac_similar_optimized \
    --train_epochs 10
```

### 方式3: 关闭梯度累积

```bash
# 显式关闭梯度累积（默认行为）
python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 10
```

## 参数配置详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gradient_accumulation_steps` | int | 1 | 梯度累积步数，>1时启用累积 |
| `--enable_auto_gradient_accumulation` | flag | False | 自动根据batch_size启用累积 |

### 自动累积规则

| batch_size | 自动设置的累积步数 | 有效batch_size |
|------------|-------------------|----------------|
| < 4 | 4 | batch_size × 4 |
| 4-7 | 2 | batch_size × 2 |
| ≥ 8 | 1 (禁用) | batch_size |

## 最佳实践建议

### 1. 针对HVAC异常检测的推荐配置

```bash
# 最优配置：小batch + 梯度累积 + 优化损失函数 + RAdam
python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 20 \
    --patience 5
```

**理由：**
- `batch_size=4`：适合您的每epoch 1492个batch的设置
- `gradient_accumulation_steps=2`：有效batch_size=8，减少梯度噪声
- `hvac_similar_optimized`：解决类别相似性问题
- RAdam优化器：更稳定的小batch训练（系统自动使用）

### 2. 不同场景的配置策略

#### 场景A：快速验证实验
```bash
python run.py --batch_size 4 --gradient_accumulation_steps 2 --train_epochs 5
```

#### 场景B：完整训练实验
```bash
python run.py --batch_size 4 --gradient_accumulation_steps 2 --train_epochs 30 --patience 8
```

#### 场景C：对比实验
```bash
# 不使用梯度累积
python run.py --batch_size 4 --gradient_accumulation_steps 1 --des "no_accumulation"

# 使用梯度累积
python run.py --batch_size 4 --gradient_accumulation_steps 2 --des "with_accumulation"
```

### 3. 参数调优指南

#### 累积步数选择
- **batch_size=2**: 推荐`gradient_accumulation_steps=4`（有效batch_size=8）
- **batch_size=4**: 推荐`gradient_accumulation_steps=2`（有效batch_size=8）
- **batch_size=6**: 推荐`gradient_accumulation_steps=2`（有效batch_size=12）
- **batch_size≥8**: 通常不需要梯度累积

#### 有效batch_size目标
- **一般建议**: 有效batch_size=8-16
- **大模型**: 有效batch_size=16-32
- **小模型**: 有效batch_size=4-8

## 技术原理

### 实现机制
1. **损失缩放**: `loss = loss / gradient_accumulation_steps`
2. **梯度累积**: 多次调用`loss.backward()`，梯度自动累积
3. **统一更新**: 累积完成后调用`optimizer.step()`和`optimizer.zero_grad()`
4. **梯度裁剪**: 在参数更新前统一进行梯度裁剪

### 数学等价性
梯度累积在数学上等价于使用更大的batch_size：

```
# 原始大batch训练
loss = criterion(model(big_batch), labels)
loss.backward()
optimizer.step()

# 梯度累积等价实现
total_loss = 0
for mini_batch in split(big_batch, accumulation_steps):
    loss = criterion(model(mini_batch), labels) / accumulation_steps
    loss.backward()
    total_loss += loss
optimizer.step()
```

## 性能影响分析

### 计算开销
- **额外开销**: <5%，主要来自条件判断
- **内存使用**: 基本无增加
- **训练时间**: 每个epoch时间基本不变

### 训练效果
- **收敛稳定性**: 显著提升（特别是小batch场景）
- **最终性能**: 通常与大batch训练相当或更好
- **过拟合风险**: 轻微降低（相当于增大了batch_size）

## 故障排除

### 常见问题

#### Q1: 梯度累积没有生效
**检查项：**
- 确认`gradient_accumulation_steps > 1`
- 查看训练日志中的"梯度累积配置"信息
- 检查进度条是否显示"GA: x/y"

#### Q2: 训练速度明显变慢
**可能原因：**
- 累积步数设置过大
- 与其他优化选项冲突

**解决方案：**
```bash
# 减少累积步数
--gradient_accumulation_steps 2  # 而不是4或更大

# 检查其他参数
--disable_parallel False  # 确保并行优化启用
```

#### Q3: 内存使用异常增加
**原因：** 梯度累积本身不应显著增加内存
**检查：**
- 确认不是同时增大了batch_size
- 检查是否有其他内存泄漏

### 调试技巧

#### 1. 启用详细日志
```bash
python run.py --gradient_accumulation_steps 2 | tee training.log
```

#### 2. 监控GPU内存
```bash
# 训练时另开终端监控
watch -n 1 nvidia-smi
```

#### 3. 对比实验
```bash
# 不使用梯度累积
python run.py --batch_size 4 --gradient_accumulation_steps 1 --train_epochs 2 --des "baseline"

# 使用梯度累积
python run.py --batch_size 4 --gradient_accumulation_steps 2 --train_epochs 2 --des "accumulation"
```

## 测试验证

### 运行测试脚本
```bash
# 基础功能测试
python test_gradient_accumulation.py

# 完整示例测试
bash scripts/run_with_gradient_accumulation.sh
```

### 预期输出
```
🔄 梯度累积配置: 每2轮累积一次梯度
   实际batch_size: 4
   有效batch_size: 8

Epoch 1/10 [GA:2]: 100%|██████████| 1492/1492 [10:23<00:00, 2.39it/s, Loss=0.3456, Acc=85.23%, GA=2/2]
```

## 与其他功能的配合

### 1. 高级损失函数
```bash
# 梯度累积 + 标签平滑 + timm优化
python run.py \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized
```

### 2. RAdam + 学习率调度
```bash
# 完整优化配置
python run.py \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 30 \
    --patience 5
```

### 3. 数据增强
```bash
# 梯度累积 + 统计特征增强
python run.py \
    --gradient_accumulation_steps 2 \
    --use_statistical_features \
    --stat_type comprehensive
```

## 总结

梯度累积是小batch训练的有力工具，特别适合您的HVAC异常检测场景：

✨ **核心优势：**
- 减少小batch带来的训练不稳定性
- 提高模型收敛质量
- 无需额外的显存开销
- 完美集成到现有训练流程

🎯 **推荐配置：**
```bash
python run.py \
    --model DualGAFNet \
    --data DualGAF \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --loss_preset hvac_similar_optimized \
    --train_epochs 20
```

这个配置将为您提供稳定、高效的训练体验，有效解决小batch训练中的梯度噪声问题。 
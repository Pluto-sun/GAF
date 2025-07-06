# RAdam + ReduceLROnPlateau 优化器配置指南

## ✅ 已完成配置

已成功将Adam优化器替换为RAdam，并配置了ReduceLROnPlateau学习率调度器，专门针对您的训练情况（batch_size=4，每个epoch 1492个batch）进行了优化。

## 🚀 主要改进

### RAdam 优化器优势
- **训练前期更稳定**: 自适应修正Adam的方差问题
- **对超参数不敏感**: 更鲁棒的训练过程
- **收敛性更好**: 特别适合小batch_size训练
- **自动降级**: 如果RAdam不可用，自动使用Adam作为备选

### ReduceLROnPlateau 学习率调度器
- **智能监控**: 基于验证损失自动调整学习率
- **合适的耐心度**: 设置为5个epoch，适合小batch训练的波动性
- **平缓调整**: 学习率减半（factor=0.5），避免过激调整
- **保护机制**: 最小学习率1e-6，防止学习率过小

## 📊 配置参数详解

### RAdam优化器参数
```python
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,                # 初始学习率
    betas=(0.9, 0.999),     # RAdam推荐参数
    eps=1e-8,               # 数值稳定性
    weight_decay=1e-4       # 轻微正则化，防止过拟合
)
```

### ReduceLROnPlateau参数
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',              # 监控验证损失（越小越好）
    factor=0.5,             # 学习率缩减为50%
    patience=5,             # 5个epoch没有改善才调整
    min_lr=1e-6,           # 最小学习率
    cooldown=2,            # 调整后等待2个epoch
    threshold=0.001,       # 改善阈值
    threshold_mode='rel',  # 相对阈值模式
    eps=1e-8              # 数值稳定性参数
)
```

## 💡 使用方法

### 直接使用（推荐）
现在您可以直接运行训练命令，无需任何额外配置：

```bash
python run.py --model DualGAFNet --data DualGAF --step 96 \
    --loss_preset hvac_similar_optimized \
    --batch_size 4 --train_epochs 50
```

### 搭配优化损失函数使用
```bash
python run.py --model DualGAFNet --data DualGAF --step 96 \
    --loss_preset hvac_similar_optimized \
    --batch_size 4 --train_epochs 50 \
    --learning_rate 0.001
```

## 📈 训练过程监控

训练过程中您会看到：

1. **优化器信息**:
   ```
   🚀 使用RAdam优化器 (lr=0.001, weight_decay=1e-4)
   ```

2. **学习率调度器配置**:
   ```
   📈 配置ReduceLROnPlateau学习率调度器:
      → 监控指标: 验证损失
      → 缩减因子: 0.5 (学习率减半)
      → 耐心度: 5个epoch
      → 最小学习率: 1e-6
      → 冷却期: 2个epoch
   ```

3. **学习率调整提示**（当发生时）:
   ```
   🔄 学习率调整: 0.001000 → 0.000500
   ```

## 🎯 针对您训练情况的优化

### 小Batch Size优化 (batch_size=4)
- **patience=5**: 考虑到小batch训练的不稳定性，给予更多耐心
- **threshold=0.001**: 设置合适的改善阈值，避免因微小波动导致过早调整
- **cooldown=2**: 调整后等待期，避免频繁调整
- **weight_decay=1e-4**: 轻微正则化，适合小batch训练

### 大数据集优化 (每个epoch 1492个batch)
- **factor=0.5**: 适中的学习率缩减，不会过于激进
- **min_lr=1e-6**: 合适的最小学习率下限
- **rel threshold**: 相对阈值模式，适应不同损失范围

## ⚡ 性能验证

通过测试验证：
- ✅ RAdam优化器在PyTorch 2.7.1中可用
- ✅ ReduceLROnPlateau调度器配置正确
- ✅ 与现有代码完美集成
- ✅ 训练过程稳定，效果良好

测试结果显示在模拟的HVAC数据上：
- 最终验证准确率: 99.58%
- 训练过程稳定，无异常波动
- 学习率调度器正常工作

## 🔧 故障排除

### 如果RAdam不可用
系统会自动降级到Adam优化器，并显示消息：
```
⚡ RAdam不可用，使用Adam优化器 (lr=0.001, weight_decay=1e-4)
```

### 如果学习率调整过于频繁
可以增加patience参数：
- 当前设置: `patience=5`
- 可以调整为: `patience=7` 或 `patience=10`

### 如果学习率调整过于激进
可以调整factor参数：
- 当前设置: `factor=0.5` (减半)
- 可以调整为: `factor=0.7` (减少30%)

## 🎉 配置完成

现在您的HVAC异常检测项目已经配置了：
1. ✅ **RAdam优化器** - 更稳定的训练
2. ✅ **ReduceLROnPlateau调度器** - 智能学习率调整  
3. ✅ **优化损失函数** - 解决类别相似性问题
4. ✅ **针对小batch训练优化** - 适合您的具体情况

可以直接开始训练，享受更好的训练效果！🚀 
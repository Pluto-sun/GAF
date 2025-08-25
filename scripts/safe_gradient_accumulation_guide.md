# 梯度累积安全配置指南

## 🚨 问题解决总结

**内存双重释放错误的真正原因是BatchNorm动态模式切换，不是梯度累积！**

- ✅ **已修复**: BatchNorm动态模式切换 (models/DualGAFNet.py)
- ✅ **保留**: 梯度累积功能（对小batch训练很重要）
- ✅ **增强**: 安全的数据加载配置

## 📊 梯度累积配置策略

### 方案1: 自动梯度累积（推荐）
```bash
--enable_auto_gradient_accumulation  # 系统自动计算最优步数
--batch_size 2                       # 小batch避免OOM
```

### 方案2: 手动梯度累积
```bash
--gradient_accumulation_steps 4      # 手动指定步数
--batch_size 4                       # 有效batch = 4 * 4 = 16
```

### 方案3: 内存受限环境
```bash
--gradient_accumulation_steps 8      # 更多累积步数
--batch_size 2                       # 最小batch
--safe_mode                          # 启用所有安全选项
```

## 🛡️ 安全配置组合

### 基础安全配置
```bash
--safe_mode                          # 禁用多线程数据加载
--num_workers 0                      # 明确禁用多线程
--drop_last_batch                    # 避免不规则batch
```

### 环境变量配置
```bash
export CUDA_LAUNCH_BLOCKING=1                    # 同步CUDA调用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制内存碎片
export OMP_NUM_THREADS=1                         # 限制CPU线程
```

## 📈 性能优化建议

### 小数据集 (SAHU: 26信号)
```bash
--batch_size 8
--gradient_accumulation_steps 2
# 有效batch_size = 16
```

### 大数据集 (DDAHU: 120信号)
```bash
--batch_size 2
--gradient_accumulation_steps 8
# 有效batch_size = 16，但内存使用更少
```

## 🔧 故障排除

### 如果仍有内存错误
1. 降低batch_size到1
2. 增加gradient_accumulation_steps
3. 使用--safe_mode
4. 检查GPU内存使用

### 如果训练太慢
1. 增加batch_size（如果内存允许）
2. 减少gradient_accumulation_steps
3. 启用多线程（移除--safe_mode）

## ✅ 推荐的完整配置

```bash
# 对于您的DDAHU数据集
python run.py \
    --model DualGAFNet \
    --data DualGAF_DDAHU \
    --batch_size 2 \
    --enable_auto_gradient_accumulation \
    --safe_mode \
    --drop_last_batch \
    --lr_scheduler_type f1_based \
    --loss_preset hvac_hard_samples \
    [其他参数...]
```

这个配置既保证了内存安全，又保持了训练效果！ 
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import numpy as np

class F1BasedReduceLROnPlateau:
    """
    基于F1分数的学习率调度器
    
    当F1分数停滞不前时，自动降低学习率。
    这比基于损失的调度器更适合分类任务，特别是：
    1. 不平衡数据集
    2. 多分类任务
    3. 损失函数与真实性能指标不完全一致的场景
    
    Args:
        optimizer (Optimizer): 要调整的优化器
        mode (str): 'max' 因为F1分数越高越好
        factor (float): 学习率缩放因子，新学习率 = 旧学习率 * factor
        patience (int): 等待轮数，F1分数无改善时的容忍轮数
        threshold (float): 改善阈值，只有超过这个阈值才算改善
        threshold_mode (str): 'rel'(相对) 或 'abs'(绝对)
        cooldown (int): 减少学习率后的冷却期
        min_lr (float): 学习率下限
        eps (float): 学习率改变的最小值
        verbose (bool): 是否打印调整信息
    """
    
    def __init__(self, optimizer, mode='max', factor=0.5, patience=5,
                 threshold=0.01, threshold_mode='rel', cooldown=0,
                 min_lr=1e-6, eps=1e-8, verbose=True):
        
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # 表示"更差"的比较函数
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """重置内部状态"""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        """
        步进函数
        
        Args:
            metrics (float): 当前的F1分数
        """
        current = float(metrics)
        self.last_epoch += 1

        if self.best is None or self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # 冷却期间忽略bad epochs

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        """降低学习率"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'🎯 基于F1分数调整学习率: {old_lr:.6f} → {new_lr:.6f}')
                    print(f'   F1分数已{self.patience}轮无显著改善 (当前F1: {float(self.best):.4f})')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'max' and self.threshold_mode == 'rel':
            # 对于max模式，需要相对改善：current >= best * (1 + threshold)
            rel_epsilon = 1. + self.threshold
            return a >= best * rel_epsilon

        elif self.mode == 'max' and self.threshold_mode == 'abs':
            # 对于max模式，需要绝对改善：current >= best + threshold
            return a >= best + self.threshold

        else:
            raise ValueError('mode must be max for F1 score')

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'max'}:
            raise ValueError('mode must be max for F1 score')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold_mode must be one of rel, abs')

        if mode == 'max':
            self.mode_worse = -float('inf')

    def state_dict(self):
        """返回调度器状态"""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.__dict__.update(state_dict)


class CompositeScheduler:
    """
    复合调度器，同时支持基于损失和F1分数的调度
    
    可以设置权重来平衡两种指标的影响，或者设置优先级策略
    """
    
    def __init__(self, optimizer, strategy='f1_priority', 
                 loss_weight=0.3, f1_weight=0.7,
                 loss_params=None, f1_params=None):
        """
        Args:
            optimizer: 优化器
            strategy (str): 调度策略
                - 'f1_priority': 优先考虑F1分数
                - 'loss_priority': 优先考虑损失
                - 'weighted': 加权结合两个指标
                - 'strict_f1': 仅使用F1分数
                - 'strict_loss': 仅使用损失
            loss_weight (float): 损失权重（weighted策略时使用）
            f1_weight (float): F1分数权重（weighted策略时使用）
            loss_params (dict): 损失调度器参数
            f1_params (dict): F1调度器参数
        """
        self.strategy = strategy
        self.loss_weight = loss_weight
        self.f1_weight = f1_weight
        
        # 默认参数
        default_loss_params = {
            'mode': 'min', 'factor': 0.5, 'patience': 5,
            'threshold': 1e-3, 'cooldown': 2, 'min_lr': 1e-6
        }
        default_f1_params = {
            'mode': 'max', 'factor': 0.5, 'patience': 5, 
            'threshold': 0.01, 'cooldown': 2, 'min_lr': 1e-6
        }
        
        # 更新参数
        if loss_params:
            default_loss_params.update(loss_params)
        if f1_params:
            default_f1_params.update(f1_params)
        
        # 创建调度器
        if strategy in ['f1_priority', 'weighted', 'loss_priority']:
            self.loss_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **default_loss_params, verbose=False
            )
            self.f1_scheduler = F1BasedReduceLROnPlateau(
                optimizer, **default_f1_params, verbose=False
            )
        elif strategy == 'strict_f1':
            self.f1_scheduler = F1BasedReduceLROnPlateau(
                optimizer, **default_f1_params, verbose=True
            )
            self.loss_scheduler = None
        elif strategy == 'strict_loss':
            self.loss_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **default_loss_params, verbose=True
            )
            self.f1_scheduler = None
        
        self.last_loss_lr = [group['lr'] for group in optimizer.param_groups]
        self.last_f1_lr = [group['lr'] for group in optimizer.param_groups]
        
        print(f"📊 复合学习率调度器初始化:")
        print(f"   策略: {strategy}")
        if strategy == 'weighted':
            print(f"   权重: 损失={loss_weight:.1f}, F1={f1_weight:.1f}")
        print(f"   损失调度器: {'启用' if self.loss_scheduler else '禁用'}")
        print(f"   F1调度器: {'启用' if self.f1_scheduler else '禁用'}")
    
    def step(self, loss_metrics, f1_metrics):
        """
        执行调度步骤
        
        Args:
            loss_metrics (float): 验证损失
            f1_metrics (float): F1分数
        """
        if hasattr(self, 'loss_scheduler') and self.loss_scheduler:
            current_lr = [group['lr'] for group in self.loss_scheduler.optimizer.param_groups]
        else:
            current_lr = [group['lr'] for group in self.f1_scheduler.optimizer.param_groups]
        
        if self.strategy == 'strict_f1':
            if self.f1_scheduler:
                self.f1_scheduler.step(f1_metrics)
            
        elif self.strategy == 'strict_loss':
            if self.loss_scheduler:
                self.loss_scheduler.step(loss_metrics)
            
        elif self.strategy == 'f1_priority':
            # 先尝试F1调度，如果没有调整，再尝试损失调度
            old_lr = current_lr[0]
            self.f1_scheduler.step(f1_metrics)
            new_lr = self.f1_scheduler.optimizer.param_groups[0]['lr']
            
            if new_lr == old_lr:  # F1调度器没有调整
                self.loss_scheduler.step(loss_metrics)
                final_lr = self.loss_scheduler.optimizer.param_groups[0]['lr']
                if final_lr != new_lr:
                    print(f"🔄 损失调度器介入: F1无改善但损失需要调整")
            
        elif self.strategy == 'loss_priority':
            # 先尝试损失调度，如果没有调整，再尝试F1调度
            old_lr = current_lr[0]
            self.loss_scheduler.step(loss_metrics)
            new_lr = self.loss_scheduler.optimizer.param_groups[0]['lr']
            
            if new_lr == old_lr:  # 损失调度器没有调整
                self.f1_scheduler.step(f1_metrics)
                final_lr = self.f1_scheduler.optimizer.param_groups[0]['lr']
                if final_lr != new_lr:
                    print(f"🎯 F1调度器介入: 损失正常但F1需要调整")
                    
        elif self.strategy == 'weighted':
            # 加权策略：创建复合指标
            # 标准化指标（假设F1在0-1之间，损失需要归一化）
            normalized_loss = min(loss_metrics / 2.0, 1.0)  # 简单归一化
            composite_metric = self.loss_weight * (1 - normalized_loss) + self.f1_weight * f1_metrics
            
            # 使用F1调度器处理复合指标
            self.f1_scheduler.step(composite_metric)
        
        # 记录学习率变化
        new_lr = current_lr
        if hasattr(self, 'loss_scheduler') and self.loss_scheduler:
            new_lr = [group['lr'] for group in self.loss_scheduler.optimizer.param_groups]
        elif hasattr(self, 'f1_scheduler') and self.f1_scheduler:
            new_lr = [group['lr'] for group in self.f1_scheduler.optimizer.param_groups]
        
        if new_lr != current_lr:
            print(f"📈 学习率已调整至: {new_lr[0]:.6f}")


def create_lr_scheduler(optimizer, scheduler_type='f1_based', **kwargs):
    """
    学习率调度器工厂函数
    
    Args:
        optimizer: 优化器
        scheduler_type (str): 调度器类型
            - 'f1_based': 基于F1分数
            - 'loss_based': 基于损失（标准）
            - 'composite_f1_priority': 复合调度器，F1优先
            - 'composite_loss_priority': 复合调度器，损失优先
            - 'composite_weighted': 复合调度器，加权结合
        **kwargs: 其他参数
    
    Returns:
        调度器实例
    """
    if scheduler_type == 'f1_based':
        return F1BasedReduceLROnPlateau(optimizer, **kwargs)
    
    elif scheduler_type == 'loss_based':
        default_params = {
            'mode': 'min', 'factor': 0.5, 'patience': 5,
            'threshold': 1e-3, 'cooldown': 2, 'min_lr': 1e-6
        }
        default_params.update(kwargs)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **default_params)
    
    elif scheduler_type.startswith('composite_'):
        strategy = scheduler_type.replace('composite_', '')
        return CompositeScheduler(optimizer, strategy=strategy, **kwargs)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# 使用示例和测试
if __name__ == "__main__":
    print("="*80)
    print("测试基于F1分数的学习率调度器")
    print("="*80)
    
    # 创建测试模型和优化器
    import torch.nn as nn
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试不同类型的调度器
    schedulers = {
        'F1调度器': create_lr_scheduler(optimizer, 'f1_based', patience=3),
        '损失调度器': create_lr_scheduler(optimizer, 'loss_based', patience=3),
        'F1优先复合': create_lr_scheduler(optimizer, 'composite_f1_priority', 
                                        f1_params={'patience': 3}, loss_params={'patience': 3})
    }
    
    # 模拟训练过程中的指标变化
    print("\n模拟训练过程:")
    print("轮次  | F1分数 | 验证损失 | 学习率变化")
    print("-" * 50)
    
    # 模拟F1分数停滞但损失还在震荡的情况
    f1_scores = [0.65, 0.72, 0.78, 0.82, 0.83, 0.83, 0.83, 0.82, 0.83, 0.83]
    val_losses = [1.2, 1.0, 0.8, 0.6, 0.5, 0.52, 0.48, 0.51, 0.49, 0.50]
    
    for scheduler_name, scheduler in schedulers.items():
        print(f"\n🔧 测试: {scheduler_name}")
        # 重置优化器学习率
        for group in optimizer.param_groups:
            group['lr'] = 0.001
        
        for epoch, (f1, loss) in enumerate(zip(f1_scores, val_losses)):
            old_lr = optimizer.param_groups[0]['lr']
            
            if hasattr(scheduler, 'step') and len(scheduler.step.__code__.co_varnames) > 2:
                # 复合调度器
                scheduler.step(loss, f1)
            elif 'F1' in scheduler_name:
                scheduler.step(f1)
            else:
                scheduler.step(loss)
            
            new_lr = optimizer.param_groups[0]['lr']
            lr_change = "✓" if new_lr != old_lr else " "
            
            print(f"  {epoch+1:2d}  | {f1:.3f}  |  {loss:.3f}   | {old_lr:.6f} → {new_lr:.6f} {lr_change}")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("F1调度器能更好地检测到F1分数停滞，及时调整学习率")
    print("="*80) 
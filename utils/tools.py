import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        # 检查F1分数是否达到100%（val_loss接近-1.0）
        # 使用小的epsilon来处理浮点数精度问题
        eps = 1e-5
        if abs(val_loss + 1.0) < eps:  # F1分数为100%
            print(f'F1分数达到100%！立刻保存模型并早停')
            self.save_checkpoint(val_loss, model, path)
            self.early_stop = True
            return
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # 安全的模型保存机制 - 解决CUDA内存错误
        self._safe_save_model(model, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    
    def _safe_save_model(self, model, save_path):
        """
        安全的模型保存方法，处理CUDA内存问题
        """
        import gc
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试保存模型 (第{attempt + 1}次)")
                
                # 1. 同步CUDA操作
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 2. 清理GPU内存缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # 3. 强制垃圾回收
                gc.collect()
                
                # 4. 获取模型状态字典并移到CPU
                print("📦 正在提取模型参数...")
                if hasattr(model, 'module'):
                    # 处理DataParallel模型
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                # 5. 确保所有参数都在CPU上
                print("💻 正在将参数移至CPU...")
                cpu_state_dict = {}
                for key, value in state_dict.items():
                    if torch.is_tensor(value):
                        cpu_state_dict[key] = value.cpu().clone()
                    else:
                        cpu_state_dict[key] = value
                
                # 6. 再次清理内存
                del state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 7. 保存到文件
                print("💾 正在保存到文件...")
                torch.save(cpu_state_dict, save_path)
                
                # 8. 验证保存成功
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
                    print(f"✅ 模型保存成功! 文件大小: {file_size:.2f} MB")
                    
                    # 清理临时变量
                    del cpu_state_dict
                    gc.collect()
                    return True
                else:
                    raise RuntimeError("保存文件不存在")
                    
            except Exception as e:
                print(f"❌ 保存失败 (第{attempt + 1}次): {e}")
                
                # 清理可能的残留变量
                if 'state_dict' in locals():
                    del state_dict
                if 'cpu_state_dict' in locals():
                    del cpu_state_dict
                
                # 强制清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：2, 4, 8秒
                    print(f"⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"🚨 模型保存失败，已尝试 {max_retries} 次")
                    # 尝试保存到备用位置
                    backup_path = save_path.replace('.pth', '_backup.pth')
                    try:
                        print(f"🔄 尝试保存到备用位置: {backup_path}")
                        # 使用最简单的方式保存
                        torch.save(model.cpu().state_dict(), backup_path)
                        print(f"✅ 备用保存成功: {backup_path}")
                        return True
                    except Exception as backup_e:
                        print(f"🚨 备用保存也失败: {backup_e}")
                        return False
        
        return False


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
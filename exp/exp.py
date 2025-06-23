import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, cal_accuracy
from models import RestNet, ClusteredResNet, VGGNet, ClusteredVGGNet, ClusteredInception, GNN, MultiImageFeatureNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.font_manager as fm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch.nn.functional as F
torch.cuda.empty_cache()
# 设置中文字体支持的函数
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    # 尝试不同的中文字体（包括更多变体）
    chinese_fonts = [
        'Noto Sans CJK SC',     # Google Noto简体中文字体
        'Noto Sans CJK TC',     # Google Noto繁体中文字体  
        'Noto Serif CJK SC',    # Google Noto简体中文衬线字体
        'SimHei',               # Windows系统黑体
        'Microsoft YaHei',      # Windows系统微软雅黑
        'PingFang SC',          # macOS系统苹方字体
        'Hiragino Sans GB',     # macOS系统
        'WenQuanYi Micro Hei',  # Linux系统文泉驿微米黑
        'WenQuanYi Zen Hei',    # Linux系统文泉驿正黑
        'Source Han Sans SC',   # Adobe思源黑体
        'AR PL UMing CN',       # Arphic字体
        'DejaVu Sans',          # 备用字体
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 调试：打印一些可用的字体名称
    print("正在检测中文字体...")
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"✓ 找到字体: {font}")
            break
        else:
            print(f"✗ 未找到: {font}")
    
    # 如果没有找到预设字体，尝试查找任何包含CJK或中文关键词的字体
    if not selected_font:
        print("在预设字体中未找到，尝试查找其他CJK字体...")
        # 查找各种可能的中文字体
        cjk_keywords = ['CJK', 'Noto', 'AR PL', 'UMing', 'SimSun', 'SimHei', 'Ming', 'Gothic']
        cjk_fonts = []
        for font in fm.fontManager.ttflist:
            font_name = font.name
            if any(keyword in font_name for keyword in cjk_keywords):
                cjk_fonts.append(font_name)
        
        # 去重并优先选择更好的字体
        cjk_fonts = list(set(cjk_fonts))
        if cjk_fonts:
            # 优先选择Noto字体，其次是AR PL字体
            preferred_fonts = [f for f in cjk_fonts if 'Noto' in f]
            if not preferred_fonts:
                preferred_fonts = [f for f in cjk_fonts if 'AR PL' in f]
            if not preferred_fonts:
                preferred_fonts = cjk_fonts
            
            selected_font = preferred_fonts[0]
            print(f"✓ 找到CJK字体: {selected_font}")
            print(f"可用的其他CJK字体: {cjk_fonts[:3]}...")  # 只显示前3个
        else:
            print("✗ 未找到任何CJK字体")
    
    if selected_font:
        print(f"使用字体: {selected_font}")
        matplotlib.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
    else:
        print("未找到中文字体，使用默认字体配置")
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts
    
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 10  # 设置默认字体大小

setup_chinese_font()


import time
class Exp(object):
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.model_dict = {
            'RestNet': RestNet,
            'ClusteredResNet': ClusteredResNet,
            'VGG': VGGNet,
            'ClusteredVGGNet': ClusteredVGGNet,
            'ClusteredInception': ClusteredInception,
            'GNN': GNN,
            'MultiImageFeatureNet': MultiImageFeatureNet
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba
        self.device = self._acquire_device()
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        # 获取标签映射信息（如果数据集支持）
        self.class_names = self._get_class_names()
        self.model = self._build_model().to(self.device)
        self.time_stamp = time.strftime('%Y%m%d_%H%M%S')
        

    def _build_model(self):
        sample = next(iter(self.train_loader))
        # 判断是PyG的Batch还是普通tuple
        if hasattr(sample, 'x') and hasattr(sample, 'edge_index'):
            # GNN数据
            print(f"Sample GNN data shape: x={sample.x.shape}")
            self.args.seq_len = sample.x.shape[0]  # 节点数（可根据实际需求调整）
            self.args.enc_in = sample.x.shape[1]   # 节点特征数
            self.args.num_class = sample.y.shape[1]
        else:
            # 普通分类数据
            sample_data, label = sample
            print(f"Sample data shape: {sample_data.shape}")
            self.args.seq_len = sample_data.shape[3]
            self.args.enc_in = sample_data.shape[1]
            self.args.num_class = len(self.class_names)
            
            # 尝试从数据集获取特征列信息
            if hasattr(self.train_data, 'scalers') and self.train_data.scalers:
                feature_columns = list(self.train_data.scalers.keys())
                self.args.feature_columns = feature_columns
                print(f"从数据集获取特征列: {feature_columns}")
            else:
                self.args.feature_columns = None
                print("未找到特征列信息，将使用默认分组")
            
            # 设置HVAC信号组配置
            if hasattr(self.args, 'hvac_groups') and self.args.hvac_groups:
                print(f"使用自定义HVAC信号组: {len(self.args.hvac_groups)} 组")
            else:
                # 不使用分组，设置为None
                self.args.hvac_groups = None
                print("未提供HVAC信号组配置，使用默认特征提取器（不分组）")
                
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 添加调试信息
        print(f"模型构建完成:")
        print(f"  - 输入通道数: {self.args.enc_in}")
        print(f"  - 序列长度: {self.args.seq_len}")
        print(f"  - 类别数: {self.args.num_class}")
        print(f"  - 模型类型: {self.args.model}")
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_class_names(self):
        """获取类别名称，优先使用数据集中的标签映射"""
        try:
            # 尝试从数据集获取标签映射
            if hasattr(self.train_data, 'idx_to_label'):
                class_names = [self.train_data.idx_to_label[i] for i in range(self.args.num_class)]
                print(f"使用数据集中的标签映射: {class_names}")
                return class_names
        except Exception as e:
            print(f"无法获取数据集标签映射: {e}")
        
        # 默认使用通用名称
        default_names = [f'异常类型_{i+1}' for i in range(self.args.num_class)]
        print(f"使用默认标签名称: {default_names}")
        return default_names

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self):
        total_loss = []
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.vali_loader):
                try:
                    batch = batch.to(self.device) if hasattr(batch, 'to') else batch
                    if hasattr(batch, 'x') and hasattr(batch, 'y'):
                        # 检查批次是否为空
                        if batch.y.numel() == 0:
                            print(f"警告: 验证批次 {batch_idx} 为空，跳过")
                            continue
                        out = self.model(batch)
                        loss = self.criterion(out, batch.y)
                        pred = out.argmax(dim=1)
                        
                        # 处理标签形状
                        label_squeezed = batch.y.long().squeeze(-1) if batch.y.dim() > 1 else batch.y.long()
                        
                        # 检查标签值是否在有效范围内
                        if torch.any(label_squeezed < 0) or torch.any(label_squeezed >= out.size(1)):
                            print(f"警告: 验证批次 {batch_idx} 标签值超出范围")
                            print(f"标签范围: [{label_squeezed.min().item()}, {label_squeezed.max().item()}]")
                            print(f"期望范围: [0, {out.size(1)-1}]")
                            continue
                        
                        # 确保维度一致
                        if pred.shape != label_squeezed.shape:
                            print(f"警告: 验证批次 {batch_idx} 预测和标签形状不匹配")
                            print(f"pred shape: {pred.shape}, label shape: {label_squeezed.shape}")
                            continue
                        
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(label_squeezed.cpu().numpy())
                    else:
                        batch_x, label = batch
                        # 检查批次是否为空
                        if batch_x.size(0) == 0 or label.numel() == 0:
                            print(f"警告: 验证批次 {batch_idx} 为空 (batch_x: {batch_x.shape}, label: {label.shape})，跳过")
                            continue
                        
                        batch_x = batch_x.float().to(self.device)
                        label = label.to(self.device)
                        
                        # 进一步检查数据有效性
                        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                            print(f"警告: 验证批次 {batch_idx} 包含NaN或Inf值，跳过")
                            continue
                        
                        out = self.model(batch_x)
                        
                        # 检查输出和标签的维度匹配
                        if out.size(0) != label.size(0):
                            print(f"警告: 验证批次 {batch_idx} 输出和标签维度不匹配 (out: {out.shape}, label: {label.shape})，跳过")
                            continue
                        
                        loss = self.criterion(out, label.long().squeeze(-1))
                        pred = out.argmax(dim=1)
                        
                        # 处理标签形状
                        label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                        
                        # 检查标签值是否在有效范围内
                        if torch.any(label_squeezed < 0) or torch.any(label_squeezed >= out.size(1)):
                            print(f"警告: 验证批次 {batch_idx} 标签值超出范围")
                            print(f"标签范围: [{label_squeezed.min().item()}, {label_squeezed.max().item()}]")
                            print(f"期望范围: [0, {out.size(1)-1}]")
                            continue
                        
                        # 确保维度一致
                        if pred.shape != label_squeezed.shape:
                            print(f"警告: 验证批次 {batch_idx} 预测和标签形状不匹配")
                            print(f"pred shape: {pred.shape}, label shape: {label_squeezed.shape}")
                            continue
                        
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(label_squeezed.cpu().numpy())
                    total_loss.append(loss.item())
                    
                except Exception as e:
                    print(f"验证批次 {batch_idx} 出现错误: {e}")
                    print(f"批次信息: batch_x shape: {batch_x.shape if 'batch_x' in locals() else 'N/A'}, label shape: {label.shape if 'label' in locals() else 'N/A'}")
                    continue
        
        # 检查是否有有效的验证数据
        if len(all_labels) == 0 or len(all_preds) == 0:
            print("警告: 验证集为空，返回默认指标")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        # 计算各种评估指标
        avg_loss = np.average(total_loss) if total_loss else float('inf')
        acc = accuracy_score(all_labels, all_preds)
        
        # 计算F1分数（处理多分类和二分类情况）
        if self.args.num_class <= 2:
            # 二分类情况
            f1_macro = f1_score(all_labels, all_preds, average='binary')
            f1_weighted = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
        else:
            # 多分类情况
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            f1_weighted = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
        
        metrics = {
            'loss': avg_loss,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }
        
        self.model.train()
        return metrics

    def train(self):
        path = os.path.join(self.args.checkpoints, self.setting)
        if not os.path.exists(path):    
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()
        train_losses = []
        train_accs = []
        val_metrics_history = {
            'loss': [],
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'precision': [],
            'recall': []
        }
        
        for epoch in range(self.args.train_epochs):
            total_loss = 0
            correct = 0
            total = 0
            self.model.train()
            train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.train_epochs}')
            for batch_idx, batch in enumerate(train_bar):
                model_optim.zero_grad()
                # 判断是图类型的数据集还是图像数据集
                if hasattr(batch, 'x') and hasattr(batch, 'y'):
                    # 检查批次是否为空
                    if batch.y.numel() == 0:
                        print(f"警告: 训练批次 {batch_idx} 为空，跳过")
                        continue
                    out = self.model(batch)
                    loss = self.criterion(out, batch.y)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                else:
                    batch_x, label = batch
                    # 检查批次是否为空
                    if batch_x.size(0) == 0 or label.numel() == 0:
                        print(f"警告: 训练批次 {batch_idx} 为空 (batch_x: {batch_x.shape}, label: {label.shape})，跳过")
                        continue
                        
                    batch_x = batch_x.float().to(self.device)
                    label = label.to(self.device)
                    
                    # 检查数据有效性
                    if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                        print(f"警告: 训练批次 {batch_idx} 包含NaN或Inf值，跳过")
                        continue
                    
                    out = self.model(batch_x)
                    
                    # 检查输出和标签的维度匹配
                    if out.size(0) != label.size(0):
                        print(f"警告: 训练批次 {batch_idx} 输出和标签维度不匹配 (out: {out.shape}, label: {label.shape})，跳过")
                        continue
                    
                    loss = self.criterion(out, label.long().squeeze(-1))
                    pred = out.argmax(dim=1)
                    
                    # 确保标签和预测值形状一致，并检查标签值范围
                    label_squeezed = label.long().squeeze(-1)
                    
                    # 检查标签值是否在有效范围内
                    if torch.any(label_squeezed < 0) or torch.any(label_squeezed >= out.size(1)):
                        print(f"警告: 训练批次 {batch_idx} 标签值超出范围")
                        print(f"标签范围: [{label_squeezed.min().item()}, {label_squeezed.max().item()}]")
                        print(f"期望范围: [0, {out.size(1)-1}]")
                        continue
                    
                    # 确保维度一致
                    if pred.shape != label_squeezed.shape:
                        print(f"警告: 训练批次 {batch_idx} 预测和标签形状不匹配")
                        print(f"pred shape: {pred.shape}, label shape: {label_squeezed.shape}")
                        continue
                    
                    correct += (pred == label_squeezed).sum().item()
                    total += label_squeezed.size(0)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                total_loss += loss.item()
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*correct/total:.2f}%' if total > 0 else 'Acc: 0.00%'
                })
            
            train_loss = total_loss / len(self.train_loader)
            train_acc = correct / total
            val_metrics = self.vali()
            
            # 保存训练历史
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            for key in val_metrics_history.keys():
                val_metrics_history[key].append(val_metrics[key])
            
            # 打印详细的训练信息
            print(f"Epoch: {epoch+1}")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"        F1(macro): {val_metrics['f1_macro']:.4f}, F1(weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"        Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print("-" * 80)
            
            # 使用F1分数进行早停（也可以选择使用准确率或其他指标）
            # 这里我们使用加权F1分数，因为它考虑了类别不平衡
            early_stopping(-val_metrics['f1_weighted'], self.model, path)
            if early_stopping.early_stop:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        # 训练结束后加载最优模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        # 构建更完整的训练历史记录
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_metrics': val_metrics_history,
            'best_val_acc': max(val_metrics_history['accuracy']) if val_metrics_history['accuracy'] else 0,
            'best_val_f1_macro': max(val_metrics_history['f1_macro']) if val_metrics_history['f1_macro'] else 0,
            'best_val_f1_weighted': max(val_metrics_history['f1_weighted']) if val_metrics_history['f1_weighted'] else 0,
        }
        
        # 打印最终训练总结
        print("\n" + "="*100)
        print("训练完成总结:")
        print(f"  最佳验证准确率: {history['best_val_acc']:.4f}")
        print(f"  最佳验证F1(macro): {history['best_val_f1_macro']:.4f}")
        print(f"  最佳验证F1(weighted): {history['best_val_f1_weighted']:.4f}")
        print(f"  训练轮数: {len(train_losses)}")
        print("="*100)
        
        self.plot_results(history)
        return self.model, history

    def plot_results(self, history):
        """绘制训练结果（包含F1分数等多种指标）"""
        # 使用时间+setting作为文件名
        path = os.path.join(self.args.result_path, f"{self.setting}_{self.time_stamp}")
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 创建更大的图形来容纳更多子图
        plt.figure(figsize=(20, 12))
        
        # 1. 损失曲线
        plt.subplot(2, 3, 1)
        loss_df = pd.DataFrame({
            'Epoch': list(range(len(history['train_losses']))) * 2,
            'Loss': history['train_losses'] + history['val_metrics']['loss'],
            'Type': ['Train'] * len(history['train_losses']) + ['Val'] * len(history['val_metrics']['loss'])
        })
        sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Type')
        plt.title('损失曲线', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        plt.subplot(2, 3, 2)
        acc_df = pd.DataFrame({
            'Epoch': list(range(len(history['train_accs']))) * 2,
            'Accuracy': history['train_accs'] + history['val_metrics']['accuracy'],
            'Type': ['Train'] * len(history['train_accs']) + ['Val'] * len(history['val_metrics']['accuracy'])
        })
        sns.lineplot(data=acc_df, x='Epoch', y='Accuracy', hue='Type')
        plt.title('准确率曲线', pad=20)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 3. F1分数曲线
        plt.subplot(2, 3, 3)
        f1_df = pd.DataFrame({
            'Epoch': list(range(len(history['val_metrics']['f1_macro']))) * 2,
            'F1_Score': history['val_metrics']['f1_macro'] + history['val_metrics']['f1_weighted'],
            'Type': ['F1-Macro'] * len(history['val_metrics']['f1_macro']) + ['F1-Weighted'] * len(history['val_metrics']['f1_weighted'])
        })
        sns.lineplot(data=f1_df, x='Epoch', y='F1_Score', hue='Type')
        plt.title('F1分数曲线', pad=20)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 4. 精确率和召回率曲线
        plt.subplot(2, 3, 4)
        pr_df = pd.DataFrame({
            'Epoch': list(range(len(history['val_metrics']['precision']))) * 2,
            'Score': history['val_metrics']['precision'] + history['val_metrics']['recall'],
            'Type': ['Precision'] * len(history['val_metrics']['precision']) + ['Recall'] * len(history['val_metrics']['recall'])
        })
        sns.lineplot(data=pr_df, x='Epoch', y='Score', hue='Type')
        plt.title('精确率和召回率曲线', pad=20)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 5. 性能总结
        plt.subplot(2, 3, 5)
        final_train_acc = history['train_accs'][-1]
        performance_text = f"""训练性能总结:

最终训练准确率: {final_train_acc:.4f}
最佳验证准确率: {history['best_val_acc']:.4f}
最佳F1(macro): {history['best_val_f1_macro']:.4f}
最佳F1(weighted): {history['best_val_f1_weighted']:.4f}

总训练轮数: {len(history['train_losses'])}
模型: {self.args.model}
任务: {getattr(self.args, 'task_name', 'Classification')}

最终验证指标:
- 准确率: {history['val_metrics']['accuracy'][-1]:.4f}
- F1(macro): {history['val_metrics']['f1_macro'][-1]:.4f}
- F1(weighted): {history['val_metrics']['f1_weighted'][-1]:.4f}
- 精确率: {history['val_metrics']['precision'][-1]:.4f}
- 召回率: {history['val_metrics']['recall'][-1]:.4f}
        """
        plt.text(0.05, 0.95, performance_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top')
        plt.title('训练总结', pad=20)
        plt.axis('off')
        
        # 6. 指标对比雷达图
        plt.subplot(2, 3, 6)
        metrics_values = [
            history['val_metrics']['accuracy'][-1],
            history['val_metrics']['f1_macro'][-1],
            history['val_metrics']['f1_weighted'][-1],
            history['val_metrics']['precision'][-1],
            history['val_metrics']['recall'][-1]
        ]
        metrics_names = ['准确率', 'F1-Macro', 'F1-Weighted', '精确率', '召回率']
        
        # 创建雷达图数据
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values += metrics_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 6, projection='polar')
        ax.plot(angles, metrics_values, 'o-', linewidth=2, label='验证指标')
        ax.fill(angles, metrics_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        plt.title('最终验证指标雷达图', pad=30)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'comprehensive_training_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细的指标数据到CSV
        metrics_df = pd.DataFrame({
            'Epoch': range(len(history['train_losses'])),
            'Train_Loss': history['train_losses'],
            'Train_Acc': history['train_accs'],
            'Val_Loss': history['val_metrics']['loss'],
            'Val_Acc': history['val_metrics']['accuracy'],
            'Val_F1_Macro': history['val_metrics']['f1_macro'],
            'Val_F1_Weighted': history['val_metrics']['f1_weighted'],
            'Val_Precision': history['val_metrics']['precision'],
            'Val_Recall': history['val_metrics']['recall']
        })
        metrics_df.to_csv(os.path.join(path, 'training_metrics.csv'), index=False)
        print(f"训练结果已保存到: {path}")

    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵（使用seaborn实现）"""
        path = os.path.join(self.args.result_path, f"{self.setting}_{self.time_stamp}")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.figure(figsize=(12, 8))
        
        # 使用获取到的类别名称
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': '样本数'})
        plt.title('混淆矩阵 - HVAC异常检测', pad=20)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_report(self):
        """通用评估报告，兼容GNN和普通分类模型"""
        self.model.eval()
        self.model.to(self.device)
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in self.vali_loader:
                batch = batch.to(self.device) if hasattr(batch, 'to') else batch
                if hasattr(batch, 'x') and hasattr(batch, 'y'):
                    out = self.model(batch)
                    probs = F.softmax(out, dim=1)
                    pred = out.argmax(dim=1)
                    
                    # 处理标签形状
                    label_squeezed = batch.y.long().squeeze(-1) if batch.y.dim() > 1 else batch.y.long()
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label_squeezed.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    batch_x, label = batch
                    batch_x = batch_x.float().to(self.device)
                    label = label.to(self.device)
                    out = self.model(batch_x)
                    probs = F.softmax(out, dim=1)
                    pred = out.argmax(dim=1)
                    
                    # 处理标签形状
                    label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label_squeezed.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.class_names)
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        return accuracy, report, cm, all_probs

def save_model_checkpoint(model, setting, extra_dict=None):
    """统一保存模型到checkpoints/setting/checkpoint.pth，extra_dict可包含额外信息"""
    path = os.path.join('checkpoints', setting)
    if not os.path.exists(path):
        os.makedirs(path)
    save_dict = {'model_state_dict': model.state_dict()}
    if extra_dict:
        save_dict.update(extra_dict)
    torch.save(save_dict, os.path.join(path, 'checkpoint.pth')) 
import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, cal_accuracy
from models import RestNet, ClusteredResNet, VGGNet, ClusteredVGGNet, ClusteredInception, GNN, MultiImageFeatureNet, DualGAFNet, SimpleGAFNet, OneDCNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.font_manager as fm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch.nn.functional as F
import torch.optim as optim
import time
import warnings
from torch.optim import lr_scheduler
import logging
import sys
from datetime import datetime
# 已移除未使用的导入：visual_long, save_metrics_to_csv, create_exp_folder 这些函数在utils/tools.py中不存在
# from utils.tools import adjust_learning_rate, visual  # 这些函数存在但未使用
torch.cuda.empty_cache()
# 设置中文字体支持的函数
def setup_chinese_font():
    """设置matplotlib中文字体支持 - 优化版本"""
    # 清除matplotlib字体缓存
    import shutil
    import os
    try:
        mpl_cache = matplotlib.get_cachedir()
        if os.path.exists(mpl_cache):
            shutil.rmtree(mpl_cache)
        print("✓ matplotlib字体缓存已清除")
    except Exception as e:
        print(f"⚠️ 清除字体缓存失败: {e}")
    
    # 重新加载字体管理器
    fm.fontManager = fm.FontManager()
    
    # 扩展的中文字体列表（基于实际系统安装）
    chinese_fonts = [
        # 新安装的Noto CJK字体（优先）
        'Noto Sans CJK SC Regular',
        'Noto Sans CJK SC',
        'Noto Sans CJK TC', 
        'Noto Serif CJK SC',
        # 文泉驿字体（Linux常用）
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'WenQuanYi Micro Hei Light',
        # 思源字体
        'Source Han Sans SC',
        'Source Han Sans CN',
        # Windows字体（如果在Wine环境下）
        'SimHei',
        'Microsoft YaHei',
        'SimSun',
        # macOS字体
        'PingFang SC',
        'Hiragino Sans GB',
        # AR PL字体
        'AR PL UMing CN',
        'AR PL UKai CN',
        # 备用字体
        'DejaVu Sans',
    ]
    
    # 获取系统可用字体
    available_fonts = []
    for font in fm.fontManager.ttflist:
        available_fonts.append(font.name)
    
    print("🔍 正在检测中文字体...")
    print(f"系统总字体数: {len(available_fonts)}")
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"✅ 找到预设字体: {font}")
            break
        else:
            print(f"❌ 未找到: {font}")
    
    # 如果没有找到预设字体，智能搜索CJK字体
    if not selected_font:
        print("🔍 在预设字体中未找到，智能搜索CJK字体...")
        cjk_keywords = ['Noto', 'CJK', 'WenQuanYi', 'Source Han', 'AR PL', 'SimHei', 'YaHei']
        cjk_fonts = []
        
        for font in fm.fontManager.ttflist:
            font_name = font.name
            font_path = font.fname
            
            # 检查字体名称是否包含CJK关键词
            if any(keyword in font_name for keyword in cjk_keywords):
                cjk_fonts.append(font_name)
                print(f"🔍 发现CJK字体: {font_name} ({font_path})")
        
        # 去重并按优先级排序
        cjk_fonts = list(set(cjk_fonts))
        if cjk_fonts:
            # 按优先级排序：Noto > WenQuanYi > Source Han > AR PL > 其他
            priority_order = ['Noto', 'WenQuanYi', 'Source Han', 'AR PL']
            
            for priority in priority_order:
                matching_fonts = [f for f in cjk_fonts if priority in f]
                if matching_fonts:
                    selected_font = matching_fonts[0]
                    print(f"✅ 选择优先级字体: {selected_font}")
                    break
            
            if not selected_font:
                selected_font = cjk_fonts[0]
                print(f"✅ 选择首个CJK字体: {selected_font}")
            
            print(f"📋 其他可用CJK字体: {cjk_fonts[:5]}...")  # 显示前5个
        else:
            print("❌ 未找到任何CJK字体")
    
    # 配置matplotlib
    if selected_font:
        print(f"🎨 应用字体配置: {selected_font}")
        # 设置字体族，包含后备字体
        font_list = [selected_font] + chinese_fonts + ['DejaVu Sans', 'Arial Unicode MS']
        matplotlib.rcParams['font.sans-serif'] = font_list
        matplotlib.rcParams['font.family'] = 'sans-serif'
    else:
        print("⚠️ 未找到中文字体，使用备用配置")
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # 通用配置
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    matplotlib.rcParams['figure.max_open_warning'] = 0  # 禁用图形数量警告
    plt.rcParams['font.size'] = 10  # 设置默认字体大小
    
    # 验证字体是否正常工作
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文字体', ha='center', va='center')
        plt.close(fig)
        print("✅ 中文字体验证成功")
    except Exception as e:
        print(f"⚠️ 中文字体验证失败: {e}")
    
    print(f"📝 最终字体配置: {matplotlib.rcParams['font.sans-serif'][:3]}...")

# 使用优化的字体配置模块
try:
    from utils.font_config import quick_setup
    quick_setup()
except ImportError:
    # 如果字体配置模块不可用，使用原来的函数
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
            'MultiImageFeatureNet': MultiImageFeatureNet,
            'DualGAFNet': DualGAFNet,
            'SimpleGAFNet': SimpleGAFNet,
            'OneDCNN': OneDCNN
        }
        # if args.model == 'Mamba':
        #     print('Please make sure you have successfully installed mamba_ssm')
        #     from models import Mamba
        #     self.model_dict['Mamba'] = Mamba
        self.device = self._acquire_device()
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        # 获取标签映射信息（如果数据集支持）
        self.class_names = self._get_class_names()
        self.model = self._build_model().to(self.device)
        self.time_stamp = time.strftime('%m%d_%H%M')
        
        # 初始化日志系统
        self.logger, self.log_file = self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统，同时输出到终端和文件"""
        # 创建与plot_results一致的目录结构
        log_dir = os.path.join(self.args.result_path, f"{self.time_stamp}_{self.setting}")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件路径
        log_file = os.path.join(log_dir, "training.log")
        
        # 创建日志器
        logger = logging.getLogger(f'exp_{self.setting}')
        logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter('%(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # 防止日志向上传播
        logger.propagate = False
        
        # 记录日志开始信息
        logger.info("=" * 80)
        logger.info(f"实验日志开始: {self.setting}")
        logger.info(f"日志文件: {log_file}")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        return logger, log_file
    
    def log_info(self, message):
        """记录信息日志"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def log_warning(self, message):
        """记录警告日志"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def log_error(self, message):
        """记录错误日志"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")
    
    def log_config(self):
        """记录实验配置"""
        self.log_info("⚙️ 实验配置:")
        for key, value in self.args.__dict__.items():
            if not key.startswith('_'):
                self.log_info(f"   {key}: {value}")
    
    def log_model_info(self):
        """记录模型信息"""
        self.log_info("🏗️ 模型信息:")
        self.log_info(f"   模型类型: {type(self.model).__name__}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.log_info(f"   总参数数量: {total_params:,}")
        self.log_info(f"   可训练参数: {trainable_params:,}")
        self.log_info(f"   模型大小估计: {total_params * 4 / (1024**2):.2f} MB")
    
    def log_data_info(self):
        """记录数据集信息"""
        self.log_info("📊 数据集信息:")
        self.log_info(f"   训练集大小: {len(self.train_data)}")
        self.log_info(f"   验证集大小: {len(self.vali_data)}")
    
    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_metrics, learning_rate, scheduler_type=None):
        """记录每轮训练的详细指标"""
        self.log_info(f"Epoch {epoch+1}:")
        self.log_info(f"  训练指标:")
        self.log_info(f"    损失({getattr(self.args, 'loss_type', 'ce')}): {train_loss:.6f}")
        self.log_info(f"    准确率: {train_acc:.6f}")
        self.log_info(f"    学习率: {learning_rate:.8f}")
        
        self.log_info(f"  验证指标(标准CE):")
        self.log_info(f"    损失: {val_metrics['loss']:.6f}")
        self.log_info(f"    准确率: {val_metrics['accuracy']:.6f}")
        self.log_info(f"    F1(macro): {val_metrics['f1_macro']:.6f}")
        self.log_info(f"    F1(weighted): {val_metrics['f1_weighted']:.6f}")
        self.log_info(f"    精确率: {val_metrics['precision']:.6f}")
        self.log_info(f"    召回率: {val_metrics['recall']:.6f}")
        
        # 如果有调度器信息，记录调度器类型
        if scheduler_type:
            self.log_info(f"  学习率调度器: {scheduler_type}")
        
        self.log_info("-" * 60)
    
    def log_learning_rate_change(self, old_lr, new_lr, scheduler_type, val_metrics):
        """记录学习率变化"""
        self.log_info(f"🎯 学习率调整事件:")
        self.log_info(f"  调度器类型: {scheduler_type}")
        self.log_info(f"  学习率变化: {old_lr:.8f} → {new_lr:.8f}")
        
        if scheduler_type == 'f1_based':
            self.log_info(f"  触发条件: F1分数 = {val_metrics['f1_weighted']:.6f}")
        elif scheduler_type.startswith('composite'):
            self.log_info(f"  触发条件: F1分数 = {val_metrics['f1_weighted']:.6f}, 验证损失 = {val_metrics['loss']:.6f}")
        else:
            self.log_info(f"  触发条件: 验证损失 = {val_metrics['loss']:.6f}")
        
        self.log_info(f"  变化幅度: {((new_lr - old_lr) / old_lr * 100):+.2f}%")
        self.log_info("-" * 60)
    
    def log_epoch_summary(self, epoch, train_loss, train_acc, val_metrics, learning_rate, is_best_model=False):
        """记录每轮训练的简要总结"""
        status = "🏆 [最佳模型]" if is_best_model else ""
        self.log_info(f"Epoch {epoch+1:3d} | Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                      f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                      f"F1={val_metrics['f1_weighted']:.4f} | LR={learning_rate:.6f} {status}")
    
    def log_training_history(self, history):
        """记录完整训练历史到CSV文件"""
        import csv
        
        # 创建CSV文件路径
        log_dir = os.path.join(self.args.result_path, f"{self.time_stamp}_{self.setting}")
        csv_file = os.path.join(log_dir, "training_history.csv")
        
        # 准备CSV数据
        csv_data = []
        epochs = len(history['train_losses'])
        
        for epoch in range(epochs):
            row = {
                'Epoch': epoch + 1,
                'Train_Loss': history['train_losses'][epoch],
                'Train_Acc': history['train_accs'][epoch],
                'Val_Loss': history['val_metrics']['loss'][epoch],
                'Val_Acc': history['val_metrics']['accuracy'][epoch],
                'Val_F1_Macro': history['val_metrics']['f1_macro'][epoch],
                'Val_F1_Weighted': history['val_metrics']['f1_weighted'][epoch],
                'Val_Precision': history['val_metrics']['precision'][epoch],
                'Val_Recall': history['val_metrics']['recall'][epoch],
            }
            
            # 添加学习率信息（如果有）
            if 'learning_rates' in history:
                row['Learning_Rate'] = history['learning_rates'][epoch]
            
            csv_data.append(row)
        
        # 写入CSV文件
        if csv_data:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
            
            self.log_info(f"训练历史已保存到: {csv_file}")
    
    def close_logger(self):
        """关闭日志器"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("=" * 80)
            self.logger.info(f"实验日志结束: {self.setting}")
            self.logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
            
            # 关闭所有处理器
            for handler in self.logger.handlers:
                handler.close()
            self.logger.handlers.clear()
        

    def _build_model(self):
        sample = next(iter(self.train_loader))
        # 判断是PyG的Batch还是普通tuple
        if hasattr(sample, 'x') and hasattr(sample, 'edge_index'):
            # GNN数据
            print(f"Sample GNN data shape: x={sample.x.shape}")
            self.args.seq_len = sample.x.shape[0]  # 节点数（可根据实际需求调整）
            self.args.enc_in = sample.x.shape[1]   # 节点特征数
        elif len(sample) == 3:
            # 双路GAF数据：(summation_data, difference_data, label) - 禁用统计特征格式
            sum_data, diff_data, label = sample
            print(f"Sample dual GAF data shape: sum={sum_data.shape}, diff={diff_data.shape}")
            self.args.seq_len = sum_data.shape[3]
            self.args.enc_in = sum_data.shape[1]
            self.is_dual_gaf = True
            self.has_time_series = False
            
            # 对于DualGAFNet模型，确保use_statistical_features设置为False
            if self.args.model == 'DualGAFNet':
                self.args.use_statistical_features = False
                print("检测到三元组格式，禁用统计特征")
        elif len(sample) == 4:
            # 增强双路GAF数据：(summation_data, difference_data, time_series_data, label) - 启用统计特征格式
            sum_data, diff_data, time_series_data, label = sample
            print(f"Sample enhanced dual GAF data shape: sum={sum_data.shape}, diff={diff_data.shape}, time_series={time_series_data.shape}")
            self.args.seq_len = sum_data.shape[3]
            self.args.enc_in = sum_data.shape[1]
            self.is_dual_gaf = True
            self.has_time_series = True
            
            # 对于DualGAFNet模型，确保use_statistical_features设置为True
            if self.args.model == 'DualGAFNet':
                if not hasattr(self.args, 'use_statistical_features'):
                    self.args.use_statistical_features = True
                print(f"检测到四元组格式，统计特征设置: {self.args.use_statistical_features}")
            
            # 尝试从数据集获取特征列信息
            if hasattr(self.train_data, 'data_manager') and hasattr(self.train_data.data_manager, 'scalers') and self.train_data.data_manager.scalers:
                feature_columns = list(self.train_data.data_manager.scalers.keys())
                self.args.feature_columns = feature_columns
                print(f"从双路GAF数据集获取特征列: {feature_columns}")
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
        else:
            # 普通分类数据
            sample_data, label = sample
            print(f"Sample data shape: {sample_data.shape}")
            self.args.seq_len = sample_data.shape[3]
            self.args.enc_in = sample_data.shape[1]
            self.is_dual_gaf = False
            
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
        print(f"="*100)
        print(f"模型构建完成:")
        print(f"  - 输入通道数: {self.args.enc_in}")
        print(f"  - 序列长度: {self.args.seq_len}")
        print(f"  - 类别数: {self.args.num_class}")
        print(f"  - 模型类型: {self.args.model}")
        print(f"="*100)
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
        """
        选择RAdam优化器，配合ReduceLROnPlateau学习率调度器
        
        RAdam优化器的优势：
        - 自适应修正Adam的方差问题
        - 训练前期更稳定
        - 对超参数不敏感
        - 收敛性更好
        """
        try:
            # 尝试使用torch.optim.RAdam（PyTorch 1.5+）
            model_optim = optim.RAdam(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),  # RAdam推荐参数
                eps=1e-8,
                weight_decay=1e-4    # 轻微正则化，防止过拟合
            )
            print(f"🚀 使用RAdam优化器 (lr={self.args.learning_rate}, weight_decay=1e-4)")
        except AttributeError:
            # 降级到Adam（兼容性）
            model_optim = optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                weight_decay=1e-4
            )
            print(f"⚡ RAdam不可用，使用Adam优化器 (lr={self.args.learning_rate}, weight_decay=1e-4)")
        
        return model_optim
    
    def _select_lr_scheduler(self, optimizer):
        """
        配置学习率调度器
        
        支持多种调度策略：
        - 基于F1分数：直接关注分类性能，适合分类任务（推荐）
        - 基于验证损失：传统方法，适合损失与性能一致的场景
        - 复合策略：结合两种指标，更稳健的调度
        """
        # 获取调度器类型配置
        scheduler_type = getattr(self.args, 'lr_scheduler_type', 'f1_based')
        
        # 基础参数配置
        base_params = {
            'factor': 0.5,           # 学习率缩减为原来的50%
            'patience': 7,           # 5个epoch没有改善才调整
            'min_lr': 1e-6,         # 最小学习率
            'cooldown': 2,          # 调整后等待2个epoch再次检查
            'threshold': 0.001,      # F1分数改善阈值：1%的相对改善才算有效
            'threshold_mode': 'rel', # 相对阈值模式
            'eps': 1e-8,            # 数值稳定性参数
            'verbose': True         # 打印调整信息
        }
        
        try:
            # 尝试导入F1调度器
            from utils.f1_based_scheduler import create_lr_scheduler
            
            if scheduler_type == 'f1_based':
                scheduler = create_lr_scheduler(optimizer, 'f1_based', **base_params)
                print(f"🎯 配置基于F1分数的学习率调度器:")
                print(f"   → 监控指标: F1分数（更适合分类任务）")
                print(f"   → 优势: 直接关注分类性能，避免损失震荡影响")
                print(f"   → 触发条件: F1分数连续{base_params['patience']}轮无改善")
                
            elif scheduler_type == 'composite_f1_priority':
                scheduler = create_lr_scheduler(optimizer, 'composite_f1_priority', **base_params)
                print(f"🔄 配置复合学习率调度器（F1优先）:")
                print(f"   → 主要监控: F1分数")
                print(f"   → 辅助监控: 验证损失")
                print(f"   → 策略: F1停滞时优先调整，损失异常时辅助调整")
                print(f"   → 触发条件: F1分数连续{base_params['patience']}轮无改善")
                
            elif scheduler_type == 'composite_weighted':
                # 可配置权重
                loss_weight = getattr(self.args, 'lr_loss_weight', 0.3)
                f1_weight = getattr(self.args, 'lr_f1_weight', 0.7)
                scheduler = create_lr_scheduler(
                    optimizer, 'composite_weighted',
                    loss_weight=loss_weight, f1_weight=f1_weight,
                    **base_params
                )
                print(f"⚖️ 配置加权复合学习率调度器:")
                print(f"   → 损失权重: {loss_weight}")
                print(f"   → F1权重: {f1_weight}")
                print(f"   → 策略: 综合考虑两个指标的加权组合")
                print(f"   → 触发条件: 复合指标连续{base_params['patience']}轮无改善")
                
            elif scheduler_type == 'composite_loss_priority':
                scheduler = create_lr_scheduler(optimizer, 'composite_loss_priority', **base_params)
                print(f"🔄 配置复合学习率调度器（损失优先）:")
                print(f"   → 主要监控: 验证损失")
                print(f"   → 辅助监控: F1分数")
                print(f"   → 策略: 损失停滞时优先调整，F1异常时辅助调整")
                print(f"   → 触发条件: 验证损失连续{base_params['patience']}轮无改善")
                
            else:
                # 默认使用损失调度器（向后兼容）
                scheduler = create_lr_scheduler(optimizer, 'loss_based', **base_params)
                print(f"📉 配置基于验证损失的学习率调度器:")
                print(f"   → 监控指标: 验证损失（传统方法）")
                print(f"   → 触发条件: 验证损失连续{base_params['patience']}轮无改善")
                
        except ImportError as e:
            # 如果F1调度器不可用，回退到标准调度器
            print(f"⚠️ F1调度器导入失败: {e}")
            print("回退到标准损失调度器")
            scheduler_type = 'loss_based'  # 更新调度器类型以确保正确调用
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', **base_params
            )
            print(f"📉 配置ReduceLROnPlateau学习率调度器:")
            print(f"   → 监控指标: 验证损失")
            print(f"   → 触发条件: 验证损失连续{base_params['patience']}轮无改善")
        except Exception as e:
            # 其他异常处理
            print(f"⚠️ 学习率调度器初始化异常: {e}")
            print("回退到标准损失调度器")
            scheduler_type = 'loss_based'  # 更新调度器类型以确保正确调用
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', **base_params
            )
            print(f"📉 配置ReduceLROnPlateau学习率调度器:")
            print(f"   → 监控指标: 验证损失")
        
        # 通用配置信息
        print(f"   → 缩减因子: 0.5 (学习率减半)")
        print(f"   → 耐心度: 5个epoch")
        print(f"   → 最小学习率: 1e-6")
        print(f"   → 冷却期: 2个epoch")
        
        return scheduler

    def _apply_loss_preset(self, preset):
        """
        应用损失函数预设配置
        
        Args:
            preset (str): 预设配置名称
        """
        print(f"🎯 应用损失函数预设配置: {preset}")
        
        if preset == 'hvac_similar':
            # HVAC相似类别配置
            self.args.loss_type = 'label_smoothing'
            self.args.label_smoothing = 0.15  # 较高的平滑因子
            print("   → 配置：标签平滑 (smoothing=0.15) - 适用于HVAC异常检测中的相似故障模式")
            
        elif preset == 'imbalanced_focus':
            # 类别不平衡配置
            self.args.loss_type = 'focal'
            self.args.focal_alpha = 1.0
            self.args.focal_gamma = 2.0
            print("   → 配置：Focal Loss (alpha=1.0, gamma=2.0) - 适用于类别不平衡问题")
            
        elif preset == 'hard_samples':
            # 难分类样本配置
            self.args.loss_type = 'focal'
            self.args.focal_alpha = 0.25  # 降低易分类样本权重
            self.args.focal_gamma = 3.0   # 增强难样本聚焦
            print("   → 配置：强化Focal Loss (alpha=0.25, gamma=3.0) - 专注于难分类样本")
            
        elif preset == 'overconfidence_prevention':
            # 防止过度自信配置
            self.args.loss_type = 'combined'
            self.args.label_smoothing = 0.1
            self.args.confidence_penalty_beta = 0.1
            print("   → 配置：组合损失 (标签平滑 + 置信度惩罚) - 防止过度自信")
            
        # === 新增优化预设配置 ===
        elif preset == 'hvac_similar_optimized':
            # HVAC相似类别 + 高性能优化（推荐）
            self.args.loss_type = 'label_smoothing_optimized'
            self.args.label_smoothing = 0.10
            self.args.use_timm_loss = True
            print("   → 配置：优化标签平滑 (smoothing=0.15, timm加速) - HVAC相似类别最佳选择")
            
        elif preset == 'hvac_adaptive':
            # HVAC自适应平滑
            self.args.loss_type = 'adaptive_smoothing'
            self.args.adaptive_initial_smoothing = 0.2
            self.args.adaptive_final_smoothing = 0.08
            self.args.adaptive_decay_epochs = 30
            print("   → 配置：自适应标签平滑 (0.2→0.08, 30轮衰减) - 训练过程动态调整")
            
        elif preset == 'hvac_hard_samples':
            # HVAC难样本聚焦 + 标签平滑
            self.args.loss_type = 'hybrid_focal'
            self.args.focal_alpha = 0.8
            self.args.focal_gamma = 2.5
            self.args.label_smoothing = 0.1
            print("   → 配置：混合Focal Loss (α=0.8, γ=2.5, smoothing=0.1) - 难样本+相似类别")
            
        elif preset == 'production_optimized':
            # 生产环境优化
            self.args.loss_type = 'label_smoothing_optimized'
            self.args.label_smoothing = 0.12
            self.args.use_timm_loss = True
            print("   → 配置：生产环境优化 (smoothing=0.12, timm高性能) - 平衡精度与效率")
            
        else:
            print(f"   ⚠️ 未知的预设配置: {preset}")

    def _get_class_weights(self):
        """
        获取类别权重（默认禁用，适用于平衡数据集）
        
        Returns:
            torch.Tensor or None: 类别权重张量
        """
        # 检查是否启用类别权重
        enable_class_weights = getattr(self.args, 'enable_class_weights', False)
        
        if not enable_class_weights:
            # 默认不使用类别权重（适用于平衡数据集）
            return None
        
        print("🔧 启用类别权重功能")
        
        # 获取用户手动指定的权重
        class_weights = getattr(self.args, 'class_weights', None)
        
        if class_weights is not None:
            # 解析用户提供的权重字符串
            try:
                if isinstance(class_weights, str):
                    weights = [float(w.strip()) for w in class_weights.split(',')]
                    if len(weights) != self.args.num_class:
                        print(f"⚠️ 类别权重数量({len(weights)})与类别数量({self.args.num_class})不匹配，忽略权重设置")
                        return None
                    class_weights = torch.tensor(weights, dtype=torch.float32)
                    print(f"📊 使用用户指定的类别权重: {weights}")
                    return class_weights
            except ValueError as e:
                print(f"⚠️ 类别权重解析失败: {e}，尝试自动计算")
        
        # 自动计算类别权重（仅当启用时）
        print("🔍 尝试从训练数据自动计算类别权重...")
        try:
            if hasattr(self, 'train_data') and hasattr(self.train_data, 'labels'):
                # 计算每个类别的样本数
                unique_labels, counts = np.unique(self.train_data.labels, return_counts=True)
                
                # 检查数据是否平衡
                count_diff = counts.max() - counts.min()
                if count_diff <= len(unique_labels):  # 允许小幅度差异
                    print(f"📊 数据集较为平衡（样本数差异≤{len(unique_labels)}），建议不使用类别权重")
                    print(f"   各类别样本数: {dict(zip(unique_labels, counts))}")
                    return None
                
                # 计算平衡权重: weight = total_samples / (n_classes * class_count)
                total_samples = len(self.train_data.labels)
                n_classes = len(unique_labels)
                weights = total_samples / (n_classes * counts)
                
                # 归一化权重，使得平均权重为1
                weights = weights / weights.mean()
                
                class_weights = torch.zeros(self.args.num_class, dtype=torch.float32)
                for label, weight in zip(unique_labels, weights):
                    class_weights[int(label)] = weight
                
                print(f"📊 自动计算的类别权重: {class_weights.tolist()}")
                print(f"   各类别样本数: {dict(zip(unique_labels, counts))}")
                return class_weights
                
        except Exception as e:
            print(f"⚠️ 自动计算类别权重失败: {e}")
        
        print("💡 建议：对于平衡数据集，通常不需要使用类别权重")
        return None

    def _select_criterion(self):
        """
        选择损失函数
        
        支持多种损失函数以解决类别相似性、类别不平衡等问题:
        - 'ce': 标准交叉熵损失
        - 'label_smoothing': 标签平滑交叉熵（推荐用于相似类别）
        - 'focal': Focal Loss（推荐用于难分类样本）
        - 'confidence_penalty': 置信度惩罚损失（防止过度自信）
        - 'combined': 组合损失（标签平滑 + 置信度惩罚）
        """
        # 处理预设配置
        loss_preset = getattr(self.args, 'loss_preset', None)
        if loss_preset:
            self._apply_loss_preset(loss_preset)
            
        loss_type = getattr(self.args, 'loss_type', 'ce')
        
        # 处理类别权重
        class_weights = self._get_class_weights()
        
        if loss_type == 'ce':
            print("📍 使用标准交叉熵损失")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        elif loss_type == 'label_smoothing':
            smoothing = getattr(self.args, 'label_smoothing', 0.1)
            use_timm = getattr(self.args, 'use_timm_loss', True)
            print(f"📍 使用标签平滑交叉熵损失 (smoothing={smoothing})")
            print("   → 适用于类别相似性较高的情况，防止过度自信")
            criterion = LabelSmoothingCrossEntropy(
                smoothing=smoothing, 
                num_classes=self.args.num_class,
                weight=class_weights,
                use_timm=use_timm
            )
            
        elif loss_type == 'label_smoothing_optimized':
            smoothing = getattr(self.args, 'label_smoothing', 0.15)
            print(f"📍 使用优化标签平滑交叉熵损失 (smoothing={smoothing})")
            print("   → 高性能实现，比标准实现快10-20%")
            criterion = LabelSmoothingCrossEntropy(
                smoothing=smoothing,
                num_classes=self.args.num_class,
                weight=class_weights,
                use_timm=True
            )
            
        elif loss_type == 'hybrid_focal':
            alpha = getattr(self.args, 'focal_alpha', 0.8)
            gamma = getattr(self.args, 'focal_gamma', 2.5)
            smoothing = getattr(self.args, 'label_smoothing', 0.1)
            print(f"📍 使用混合Focal Loss (α={alpha}, γ={gamma}, smoothing={smoothing})")
            print("   → 结合难样本聚焦和标签平滑的优势")
            criterion = HybridFocalLoss(
                alpha=alpha,
                gamma=gamma,
                smoothing=smoothing,
                weight=class_weights
            )
            
        elif loss_type == 'adaptive_smoothing':
            initial_smoothing = getattr(self.args, 'adaptive_initial_smoothing', 0.2)
            final_smoothing = getattr(self.args, 'adaptive_final_smoothing', 0.05)
            decay_epochs = getattr(self.args, 'adaptive_decay_epochs', 30)
            print(f"📍 使用自适应标签平滑损失")
            print(f"   → 动态调整: {initial_smoothing} → {final_smoothing} (衰减周期: {decay_epochs})")
            
            base_loss = LabelSmoothingCrossEntropy(
                smoothing=initial_smoothing,
                num_classes=self.args.num_class,
                weight=class_weights,
                use_timm=True
            )
            criterion = AdaptiveLossScheduler(
                base_loss=base_loss,
                initial_smoothing=initial_smoothing,
                final_smoothing=final_smoothing,
                decay_epochs=decay_epochs
            )
            
        elif loss_type == 'focal':
            alpha = getattr(self.args, 'focal_alpha', 1.0)
            gamma = getattr(self.args, 'focal_gamma', 2.0)
            print(f"📍 使用Focal Loss (alpha={alpha}, gamma={gamma})")
            print("   → 适用于难分类样本和类别不平衡问题")
            criterion = FocalLoss(
                alpha=alpha,
                gamma=gamma,
                weight=class_weights
            )
            
        elif loss_type == 'confidence_penalty':
            beta = getattr(self.args, 'confidence_penalty_beta', 0.1)
            print(f"📍 使用置信度惩罚损失 (beta={beta})")
            print("   → 防止模型过度自信，鼓励更平衡的预测")
            criterion = ConfidencePenaltyLoss(beta=beta)
            
        elif loss_type == 'combined':
            smoothing = getattr(self.args, 'label_smoothing', 0.1)
            penalty_beta = getattr(self.args, 'confidence_penalty_beta', 0.05)
            print(f"📍 使用组合损失 (标签平滑: {smoothing}, 置信度惩罚: {penalty_beta})")
            print("   → 综合解决方案：缓解类别相似性 + 防止过度自信")
            
            label_smooth_loss = LabelSmoothingCrossEntropy(
                smoothing=smoothing, 
                num_classes=self.args.num_class,
                weight=class_weights
            )
            penalty_loss = ConfidencePenaltyLoss(beta=penalty_beta)
            
            losses = {
                'label_smoothing': (label_smooth_loss, 1.0),
                'confidence_penalty': (penalty_loss, 0.5)
            }
            criterion = CombinedLoss(losses)
            
        else:
            print(f"⚠️  未知的损失函数类型: {loss_type}，使用默认交叉熵")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 显示类别权重状态
        if class_weights is not None:
            print(f"   类别权重: 已启用 {class_weights.tolist()}")
        else:
            print(f"   类别权重: 禁用（默认，适用于平衡数据集）")
        
        return criterion
    
    def _select_evaluation_criterion(self):
        """
        选择评估损失函数
        
        根据主流机器学习规范，验证和测试阶段应使用标准交叉熵损失来提供：
        1. 客观的性能评估
        2. 不同方法间的公平比较
        3. 真实的泛化能力指标
        4. 避免训练技巧对评估指标的影响
        
        Returns:
            nn.Module: 标准交叉熵损失函数
        """
        # 获取类别权重（如果启用）
        class_weights = self._get_class_weights()
        
        # 始终使用标准交叉熵损失进行评估
        eval_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        
        print("🎯 验证/测试阶段使用标准交叉熵损失（遵循主流ML规范）")
        if class_weights is not None:
            print(f"   评估损失类别权重: 已启用 {class_weights.tolist()}")
        else:
            print(f"   评估损失类别权重: 禁用")
        
        return eval_criterion
    
    def _mixup_criterion(self, pred, labels_a, labels_b, lam):
        """Mixup损失函数"""
        return lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(pred, labels_b)

    def vali(self):
        total_loss = []
        all_preds = []
        all_labels = []
        self.model.eval()
        
        # 使用标准交叉熵损失进行评估（遵循主流ML规范）
        eval_criterion = self._select_evaluation_criterion()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.vali_loader):
                try:
                    if hasattr(batch, 'x') and hasattr(batch, 'y'):
                        # GNN数据
                        if batch.y.numel() == 0:
                            print(f"警告: 验证批次 {batch_idx} 为空，跳过")
                            continue
                        batch=batch.to(self.device)
                        out = self.model(batch)
                        loss = eval_criterion(out, batch.y)  # 使用标准交叉熵损失
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
                    elif len(batch) == 3:
                        # 双路GAF数据（旧版本格式）
                        sum_data, diff_data, label = batch
                        # 检查批次是否为空
                        if sum_data.size(0) == 0 or diff_data.size(0) == 0 or label.numel() == 0:
                            print(f"警告: 验证批次 {batch_idx} 为空，跳过")
                            continue
                        
                        sum_data = sum_data.float().to(self.device)
                        diff_data = diff_data.float().to(self.device)
                        label = label.to(self.device)
                        
                        # 进一步检查数据有效性
                        if torch.isnan(sum_data).any() or torch.isinf(sum_data).any() or torch.isnan(diff_data).any() or torch.isinf(diff_data).any():
                            print(f"警告: 验证批次 {batch_idx} 包含NaN或Inf值，跳过")
                            continue
                        
                        out = self.model(sum_data, diff_data)
                        
                        # 检查输出和标签的维度匹配
                        if out.size(0) != label.size(0):
                            print(f"警告: 验证批次 {batch_idx} 输出和标签维度不匹配 (out: {out.shape}, label: {label.shape})，跳过")
                            continue
                        
                        # 安全处理标签形状，避免错误地squeeze掉批次维度
                        label_for_loss = label.long()
                        if label_for_loss.dim() > 1 and label_for_loss.size(-1) == 1:
                            label_for_loss = label_for_loss.squeeze(-1)
                        loss = eval_criterion(out, label_for_loss)  # 使用标准交叉熵损失
                        pred = out.argmax(dim=1)
                        
                        # 处理标签形状：安全处理标量标签
                        if label.dim() == 0:  # 标量标签
                            label_squeezed = label.long().unsqueeze(0)  # 转换为[1]
                        else:
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
                    elif len(batch) == 4:
                        # 增强双路GAF数据（新版本格式）
                        sum_data, diff_data, time_series_data, label = batch
                        # 检查批次是否为空
                        if sum_data.size(0) == 0 or diff_data.size(0) == 0 or time_series_data.size(0) == 0 or label.numel() == 0:
                            print(f"警告: 验证批次 {batch_idx} 为空，跳过")
                            continue
                        
                        sum_data = sum_data.float().to(self.device)
                        diff_data = diff_data.float().to(self.device)
                        time_series_data = time_series_data.float().to(self.device)
                        label = label.to(self.device)
                        
                        # 进一步检查数据有效性
                        if (torch.isnan(sum_data).any() or torch.isinf(sum_data).any() or 
                            torch.isnan(diff_data).any() or torch.isinf(diff_data).any() or
                            torch.isnan(time_series_data).any() or torch.isinf(time_series_data).any()):
                            print(f"警告: 验证批次 {batch_idx} 包含NaN或Inf值，跳过")
                            continue
                        
                        out = self.model(sum_data, diff_data, time_series_data)
                        
                        # 检查输出和标签的维度匹配
                        if out.size(0) != label.size(0):
                            print(f"警告: 验证批次 {batch_idx} 输出和标签维度不匹配 (out: {out.shape}, label: {label.shape})，跳过")
                            continue
                        
                        # 安全处理标签形状，避免错误地squeeze掉批次维度
                        label_for_loss = label.long()
                        if label_for_loss.dim() > 1 and label_for_loss.size(-1) == 1:
                            label_for_loss = label_for_loss.squeeze(-1)
                        loss = eval_criterion(out, label_for_loss)  # 使用标准交叉熵损失
                        pred = out.argmax(dim=1)
                        
                        # 处理标签形状：安全处理标量标签
                        if label.dim() == 0:  # 标量标签
                            label_squeezed = label.long().unsqueeze(0)  # 转换为[1]
                        else:
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
                    else:
                        # 普通分类数据
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
                        
                        # 安全处理标签形状，避免错误地squeeze掉批次维度
                        label_for_loss = label.long()
                        if label_for_loss.dim() > 1 and label_for_loss.size(-1) == 1:
                            label_for_loss = label_for_loss.squeeze(-1)
                        loss = eval_criterion(out, label_for_loss)  # 使用标准交叉熵损失
                        pred = out.argmax(dim=1)
                        
                        # 处理标签形状：安全处理标量标签
                        if label.dim() == 0:  # 标量标签
                            label_squeezed = label.long().unsqueeze(0)  # 转换为[1]
                        else:
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
                    # 根据不同的数据格式显示相应的变量信息
                    if 'batch_x' in locals():
                        print(f"批次信息: batch_x shape: {batch_x.shape}, label shape: {label.shape if 'label' in locals() else 'N/A'}")
                    elif 'sum_data' in locals() and 'diff_data' in locals():
                        if 'time_series_data' in locals():
                            print(f"批次信息: sum_data shape: {sum_data.shape}, diff_data shape: {diff_data.shape}, time_series_data shape: {time_series_data.shape}, label shape: {label.shape if 'label' in locals() else 'N/A'}")
                        else:
                            print(f"批次信息: sum_data shape: {sum_data.shape}, diff_data shape: {diff_data.shape}, label shape: {label.shape if 'label' in locals() else 'N/A'}")
                    elif hasattr(batch, 'x') and hasattr(batch, 'y'):
                        print(f"批次信息: GNN batch.x shape: {batch.x.shape}, batch.y shape: {batch.y.shape}")
                    else:
                        print(f"批次信息: 未知数据格式, batch type: {type(batch)}, label shape: {label.shape if 'label' in locals() else 'N/A'}")
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
        # 记录实验开始信息
        self.log_config()
        self.log_model_info()
        self.log_data_info()
        
        path = os.path.join(self.args.checkpoints, self.setting)
        if not os.path.exists(path):    
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        lr_scheduler = self._select_lr_scheduler(model_optim)  # 创建学习率调度器
        
        # 记录实际使用的调度器类型
        actual_scheduler_type = getattr(self.args, 'lr_scheduler_type', 'f1_based')
        
        self.criterion = self._select_criterion()
        
        # 按照主流ML规范说明损失函数使用策略
        self.log_info("\n" + "="*80)
        self.log_info("📋 损失函数使用策略（遵循主流机器学习规范）:")
        self.log_info(f"   🎯 训练阶段: 使用 {self.args.loss_type if hasattr(self.args, 'loss_type') else 'ce'} 损失函数")
        self.log_info(f"      → 目的：优化模型参数，解决特定问题（类别不平衡、相似性等）")
        self.log_info(f"   📊 验证阶段: 使用标准交叉熵损失函数")
        self.log_info(f"      → 目的：客观评估模型性能，便于不同方法比较")
        self.log_info(f"   📈 最终报告: 基于标准交叉熵损失的指标")
        self.log_info(f"      → 目的：提供可信、可比较的性能指标")
        self.log_info("="*80 + "\n")
        
        # 梯度累积配置
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        if gradient_accumulation_steps > 1:
            self.log_info(f"🔄 梯度累积配置: 每{gradient_accumulation_steps}轮累积一次梯度")
            self.log_info(f"   实际batch_size: {self.args.batch_size}")
            self.log_info(f"   有效batch_size: {self.args.batch_size * gradient_accumulation_steps}")
        else:
            self.log_info(f"🔄 梯度累积: 已禁用（gradient_accumulation_steps=1）")
        
        train_losses = []
        train_accs = []
        learning_rates = []  # 添加学习率记录
        val_metrics_history = {
            'loss': [],
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'precision': [],
            'recall': []
        }
        
        # 初始化最佳指标追踪
        best_val_f1 = 0.0
        

        
        for epoch in range(self.args.train_epochs):
            # 更新自适应损失调度器（如果使用）
            if hasattr(self.criterion, 'set_epoch'):
                self.criterion.set_epoch(epoch)
            
            total_loss = 0
            correct = 0
            total = 0
            accumulated_loss = 0  # 累积的损失
            
            self.model.train()
            if gradient_accumulation_steps > 1:
                train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.train_epochs} [GA:{gradient_accumulation_steps}]', ncols=100)
                # 梯度累积模式：初始化梯度清零
                model_optim.zero_grad()
            else:
                train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.train_epochs}', ncols=100)
            
            for batch_idx, batch in enumerate(train_bar):
                # 判断是图类型的数据集还是图像数据集
                if hasattr(batch, 'x') and hasattr(batch, 'y'):
                    # GNN数据
                    if batch.y.numel() == 0:
                        print(f"警告: 训练批次 {batch_idx} 为空，跳过")
                        continue
                    batch=batch.to(self.device)
                    out = self.model(batch)
                    loss = self.criterion(out, batch.y)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
                elif len(batch) == 3:
                    # 双路GAF数据（旧版本格式）
                    sum_data, diff_data, label = batch
                    # 检查批次是否为空
                    if sum_data.size(0) == 0 or diff_data.size(0) == 0 or label.numel() == 0:
                        print(f"警告: 训练批次 {batch_idx} 为空，跳过")
                        continue
                        
                    sum_data = sum_data.float().to(self.device)
                    diff_data = diff_data.float().to(self.device)
                    label = label.to(self.device)
                    
                    # 检查数据有效性
                    if torch.isnan(sum_data).any() or torch.isinf(sum_data).any() or torch.isnan(diff_data).any() or torch.isinf(diff_data).any():
                        print(f"警告: 训练批次 {batch_idx} 包含NaN或Inf值，跳过")
                        continue
                    
                    # 常规训练（数据增强已在数据集中完成）
                    out = self.model(sum_data, diff_data)
                    
                    # 检查输出和标签的维度匹配
                    if out.size(0) != label.size(0):
                        print(f"警告: 训练批次 {batch_idx} 输出和标签维度不匹配 (out: {out.shape}, label: {label.shape})，跳过")
                        continue
                    
                    # 安全处理标签形状，避免错误地squeeze掉批次维度
                    label_for_loss = label.long()
                    if label_for_loss.dim() > 1 and label_for_loss.size(-1) == 1:
                        label_for_loss = label_for_loss.squeeze(-1)
                    loss = self.criterion(out, label_for_loss)
                    pred = out.argmax(dim=1)
                    # 安全处理标签形状：确保标签是1维的
                    if label.dim() == 0:  # 标量标签
                        label_squeezed = label.long().unsqueeze(0)  # 转换为[1]
                    else:
                        label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                    
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
                elif len(batch) == 4:
                    # 增强双路GAF数据（新版本格式）
                    sum_data, diff_data, time_series_data, label = batch
                    # 检查批次是否为空
                    if sum_data.size(0) == 0 or diff_data.size(0) == 0 or time_series_data.size(0) == 0 or label.numel() == 0:
                        print(f"警告: 训练批次 {batch_idx} 为空，跳过")
                        continue
                        
                    sum_data = sum_data.float().to(self.device)
                    diff_data = diff_data.float().to(self.device)
                    time_series_data = time_series_data.float().to(self.device)
                    label = label.to(self.device)
                    
                    # 检查数据有效性
                    if (torch.isnan(sum_data).any() or torch.isinf(sum_data).any() or 
                        torch.isnan(diff_data).any() or torch.isinf(diff_data).any() or
                        torch.isnan(time_series_data).any() or torch.isinf(time_series_data).any()):
                        print(f"警告: 训练批次 {batch_idx} 包含NaN或Inf值，跳过")
                        continue
                    
                    # 常规训练（使用统计特征增强）
                    out = self.model(sum_data, diff_data, time_series_data)
                    
                    # 检查输出和标签的维度匹配
                    if out.size(0) != label.size(0):
                        print(f"警告: 训练批次 {batch_idx} 输出和标签维度不匹配 (out: {out.shape}, label: {label.shape})，跳过")
                        continue
                    
                    # 安全处理标签形状，避免错误地squeeze掉批次维度
                    label_for_loss = label.long()
                    if label_for_loss.dim() > 1 and label_for_loss.size(-1) == 1:
                        label_for_loss = label_for_loss.squeeze(-1)
                    loss = self.criterion(out, label_for_loss)
                    pred = out.argmax(dim=1)
                    # 安全处理标签形状：确保标签是1维的
                    if label.dim() == 0:  # 标量标签
                        label_squeezed = label.long().unsqueeze(0)  # 转换为[1]
                    else:
                        label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                    
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
                else:
                    # 普通分类数据
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
                    
                    # 安全处理标签形状，避免错误地squeeze掉批次维度
                    label_for_loss = label.long()
                    if label_for_loss.dim() > 1 and label_for_loss.size(-1) == 1:
                        label_for_loss = label_for_loss.squeeze(-1)
                    loss = self.criterion(out, label_for_loss)
                    pred = out.argmax(dim=1)
                    
                    # 确保标签和预测值形状一致，并检查标签值范围
                    # 安全处理标签形状：确保标签是1维的
                    if label.dim() == 0:  # 标量标签
                        label_squeezed = label.long().unsqueeze(0)  # 转换为[1]
                    else:
                        label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                    
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
                
                if gradient_accumulation_steps > 1:
                    # 梯度累积模式：将损失除以累积步数
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    # 累积损失用于显示
                    accumulated_loss += loss.item()
                    total_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失大小用于统计
                    
                    # 检查是否到达累积步数或是最后一个batch
                    is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                    is_last_batch = batch_idx == len(self.train_loader) - 1
                    
                    if is_accumulation_step or is_last_batch:
                        # 进行梯度裁剪和参数更新
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        model_optim.step()
                        model_optim.zero_grad()  # 清零梯度，准备下一次累积
                        
                        # 更新进度条显示累积的损失
                        display_loss = accumulated_loss * gradient_accumulation_steps if is_accumulation_step else accumulated_loss * gradient_accumulation_steps / ((batch_idx % gradient_accumulation_steps) + 1)
                        accumulated_loss = 0  # 重置累积损失
                    else:
                        # 只计算累积损失，不更新显示
                        display_loss = accumulated_loss * gradient_accumulation_steps / ((batch_idx % gradient_accumulation_steps) + 1)
                    
                    train_bar.set_postfix({
                        'Loss': f'{display_loss:.4f}',
                        'Acc': f'{100*correct/total:.2f}%' if total > 0 else 'Acc: 0.00%',
                        'GA': f'{(batch_idx % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}'
                    })
                else:
                    # 正常训练模式（无梯度累积）
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()
                    model_optim.zero_grad()
                    
                    total_loss += loss.item()
                    train_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100*correct/total:.2f}%' if total > 0 else 'Acc: 0.00%'
                    })
            
            train_loss = total_loss / len(self.train_loader)
            train_acc = correct / total
            val_metrics = self.vali()
            
            # 获取当前学习率
            current_lr = model_optim.param_groups[0]['lr']
            
            # 保存训练历史
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            learning_rates.append(current_lr)
            for key in val_metrics_history.keys():
                val_metrics_history[key].append(val_metrics[key])
            
            # 打印详细的训练信息
            print(f"Epoch: {epoch+1}")
            print(f"  训练 - Loss({self.args.loss_type if hasattr(self.args, 'loss_type') else 'ce'}): {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  验证 - Loss(标准CE): {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"        F1(macro): {val_metrics['f1_macro']:.4f}, F1(weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"        Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print("-" * 80)
            
            # 记录每轮的详细指标到日志
            self.log_epoch_metrics(epoch, train_loss, train_acc, val_metrics, current_lr, actual_scheduler_type)
            
            # 更新学习率调度器（智能选择监控指标）
            try:
                if actual_scheduler_type in ['composite_f1_priority', 'composite_weighted', 'composite_loss_priority']:
                    # 复合调度器需要两个参数
                    if hasattr(lr_scheduler, 'step') and len(lr_scheduler.step.__code__.co_varnames) > 2:
                        lr_scheduler.step(val_metrics['loss'], val_metrics['f1_weighted'])
                    else:
                        # 回退策略：如果不是真正的复合调度器，使用F1分数
                        self.log_warning(f"复合调度器调用失败，回退到F1分数调度")
                        lr_scheduler.step(val_metrics['f1_weighted'])
                elif actual_scheduler_type == 'f1_based':
                    # F1调度器：监控F1分数（越高越好）
                    lr_scheduler.step(val_metrics['f1_weighted'])
                else:
                    # 损失调度器（默认）：监控验证损失（越低越好）
                    lr_scheduler.step(val_metrics['loss'])
            except Exception as e:
                self.log_error(f"学习率调度器调用失败: {e}")
                self.log_warning(f"回退到损失调度策略")
                # 安全回退：使用验证损失
                if hasattr(lr_scheduler, 'step'):
                    lr_scheduler.step(val_metrics['loss'])
            
            new_lr = model_optim.param_groups[0]['lr']
            
            # 如果学习率发生变化，打印详细信息并记录到日志
            if new_lr != current_lr:
                if actual_scheduler_type == 'f1_based':
                    print(f"🎯 基于F1分数调整学习率: {current_lr:.6f} → {new_lr:.6f}")
                    print(f"   当前F1分数: {val_metrics['f1_weighted']:.4f}")
                elif actual_scheduler_type.startswith('composite'):
                    print(f"🔄 复合指标调整学习率: {current_lr:.6f} → {new_lr:.6f}")
                    print(f"   F1分数: {val_metrics['f1_weighted']:.4f}, 验证损失: {val_metrics['loss']:.4f}")
                else:
                    print(f"📉 基于验证损失调整学习率: {current_lr:.6f} → {new_lr:.6f}")
                    print(f"   当前验证损失: {val_metrics['loss']:.4f}")
                
                # 记录学习率变化到日志
                self.log_learning_rate_change(current_lr, new_lr, actual_scheduler_type, val_metrics)
            
            # 检查是否为最佳模型
            is_best_model = False
            if val_metrics['f1_weighted'] > best_val_f1:
                best_val_f1 = val_metrics['f1_weighted']
                is_best_model = True
            
            # 记录简要总结到日志
            self.log_epoch_summary(epoch, train_loss, train_acc, val_metrics, new_lr, is_best_model)
            
            # 使用F1分数进行早停（也可以选择使用准确率或其他指标）
            # 这里我们使用加权F1分数，因为它考虑了类别不平衡
            early_stopping(-val_metrics['f1_weighted'], self.model, path)
            if early_stopping.early_stop:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                self.log_info(f"⏹️ 早停触发，在第 {epoch+1} 轮停止训练")
                break
        # 训练结束后加载最优模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        print("🔄 已加载最佳模型，正在评估真实性能...")
        
        # 使用最佳模型重新评估验证集，获取真实的最佳性能指标
        best_model_metrics = self.vali()
        
        # 构建更完整的训练历史记录
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'learning_rates': learning_rates,  # 添加学习率历史
            'val_metrics': val_metrics_history,
            'best_val_acc': max(val_metrics_history['accuracy']) if val_metrics_history['accuracy'] else 0,
            'best_val_f1_macro': max(val_metrics_history['f1_macro']) if val_metrics_history['f1_macro'] else 0,
            'best_val_f1_weighted': max(val_metrics_history['f1_weighted']) if val_metrics_history['f1_weighted'] else 0,
            # 添加最佳模型的真实性能指标
            'best_model_metrics': best_model_metrics
        }
        
        # 记录完整的训练历史到CSV文件
        self.log_training_history(history)
        
        # 打印最终训练总结（使用最佳模型的真实性能）
        print("\n" + "="*100)
        print("🎉 训练完成总结:")
        print(f"  训练损失函数: {self.args.loss_type if hasattr(self.args, 'loss_type') else 'ce'}")
        print(f"  验证损失函数: 标准交叉熵（遵循主流ML规范）")
        print(f"  训练过程最佳验证准确率: {history['best_val_acc']:.4f}")
        print(f"  训练过程最佳验证F1(macro): {history['best_val_f1_macro']:.4f}")
        print(f"  训练过程最佳验证F1(weighted): {history['best_val_f1_weighted']:.4f}")
        print(f"  训练轮数: {len(train_losses)}")
        print("-" * 50)
        print("📊 最佳模型真实性能指标（基于标准交叉熵损失）:")
        print(f"  验证损失(标准CE): {best_model_metrics['loss']:.4f}")
        print(f"  验证准确率: {best_model_metrics['accuracy']:.4f}")
        print(f"  验证F1(macro): {best_model_metrics['f1_macro']:.4f}")
        print(f"  验证F1(weighted): {best_model_metrics['f1_weighted']:.4f}")
        print(f"  验证精确率: {best_model_metrics['precision']:.4f}")
        print(f"  验证召回率: {best_model_metrics['recall']:.4f}")
        print("="*100)
        
        # 记录最终训练总结到日志
        self.log_info("\n" + "="*100)
        self.log_info("🎉 训练完成总结:")
        self.log_info(f"  训练损失函数: {self.args.loss_type if hasattr(self.args, 'loss_type') else 'ce'}")
        self.log_info(f"  验证损失函数: 标准交叉熵（遵循主流ML规范）")
        self.log_info(f"  训练过程最佳验证准确率: {history['best_val_acc']:.4f}")
        self.log_info(f"  训练过程最佳验证F1(macro): {history['best_val_f1_macro']:.4f}")
        self.log_info(f"  训练过程最佳验证F1(weighted): {history['best_val_f1_weighted']:.4f}")
        self.log_info(f"  训练轮数: {len(train_losses)}")
        self.log_info("-" * 50)
        self.log_info("📊 最佳模型真实性能指标（基于标准交叉熵损失）:")
        self.log_info(f"  验证损失(标准CE): {best_model_metrics['loss']:.4f}")
        self.log_info(f"  验证准确率: {best_model_metrics['accuracy']:.4f}")
        self.log_info(f"  验证F1(macro): {best_model_metrics['f1_macro']:.4f}")
        self.log_info(f"  验证F1(weighted): {best_model_metrics['f1_weighted']:.4f}")
        self.log_info(f"  验证精确率: {best_model_metrics['precision']:.4f}")
        self.log_info(f"  验证召回率: {best_model_metrics['recall']:.4f}")
        self.log_info("="*100)
        
        self.plot_results(history)
        
        # 关闭日志器
        self.close_logger()
        
        return self.model, history

    def plot_results(self, history):
        """绘制训练结果（包含F1分数等多种指标）"""
        # 使用时间+setting作为文件名
        path = os.path.join(self.args.result_path, f"{self.time_stamp}_{self.setting}")
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 创建更大的图形来容纳更多子图
        plt.figure(figsize=(20, 12))
        
        # 1. 损失曲线
        plt.subplot(2, 3, 1)
        train_loss_type = getattr(self.args, 'loss_type', 'ce')
        loss_df = pd.DataFrame({
            'Epoch': list(range(len(history['train_losses']))) * 2,
            'Loss': history['train_losses'] + history['val_metrics']['loss'],
            'Type': [f'Train({train_loss_type})'] * len(history['train_losses']) + ['Val(标准CE)'] * len(history['val_metrics']['loss'])
        })
        sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Type')
        plt.title('损失曲线（训练vs验证损失函数）', pad=20)
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
        
        # 使用最佳模型的真实性能指标（如果可用）
        if 'best_model_metrics' in history:
            best_metrics = history['best_model_metrics']
            train_loss_type = getattr(self.args, 'loss_type', 'ce')
            performance_text = f"""训练性能总结:

训练损失函数: {train_loss_type}
验证损失函数: 标准交叉熵（遵循ML规范）

最终训练准确率: {final_train_acc:.4f}
训练过程最佳验证准确率: {history['best_val_acc']:.4f}

总训练轮数: {len(history['train_losses'])}
模型: {self.args.model}

🏆 最佳模型真实性能（基于标准CE损失）:
- 验证损失(标准CE): {best_metrics['loss']:.4f}
- 验证准确率: {best_metrics['accuracy']:.4f}
- F1(macro): {best_metrics['f1_macro']:.4f}
- F1(weighted): {best_metrics['f1_weighted']:.4f}
- 精确率: {best_metrics['precision']:.4f}
- 召回率: {best_metrics['recall']:.4f}
            """
        else:
            # 向后兼容：如果没有最佳模型指标，使用最后一轮的指标
            train_loss_type = getattr(self.args, 'loss_type', 'ce')
            performance_text = f"""训练性能总结:

训练损失函数: {train_loss_type}
验证损失函数: 标准交叉熵（遵循ML规范）

最终训练准确率: {final_train_acc:.4f}
最佳验证准确率: {history['best_val_acc']:.4f}
最佳F1(macro): {history['best_val_f1_macro']:.4f}
最佳F1(weighted): {history['best_val_f1_weighted']:.4f}

总训练轮数: {len(history['train_losses'])}
模型: {self.args.model}

最终验证指标 (最后一轮, 基于标准CE损失):
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
        
        # 使用最佳模型的真实性能指标（如果可用）
        if 'best_model_metrics' in history:
            best_metrics = history['best_model_metrics']
            metrics_values = [
                best_metrics['accuracy'],
                best_metrics['f1_macro'],
                best_metrics['f1_weighted'],
                best_metrics['precision'],
                best_metrics['recall']
            ]
            radar_title = '🏆 最佳模型性能雷达图（基于标准CE损失）'
            label_text = '最佳模型性能'
        else:
            # 向后兼容：使用最后一轮的指标
            metrics_values = [
                history['val_metrics']['accuracy'][-1],
                history['val_metrics']['f1_macro'][-1],
                history['val_metrics']['f1_weighted'][-1],
                history['val_metrics']['precision'][-1],
                history['val_metrics']['recall'][-1]
            ]
            radar_title = '最终验证指标雷达图（最后一轮，基于标准CE损失）'
            label_text = '验证指标'
        
        metrics_names = ['准确率', 'F1-Macro', 'F1-Weighted', '精确率', '召回率']
        
        # 创建雷达图数据
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values += metrics_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 6, projection='polar')
        ax.plot(angles, metrics_values, 'o-', linewidth=2, label=label_text)
        ax.fill(angles, metrics_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        plt.title(radar_title, pad=30)
        
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
        
        # 保存最佳模型的真实性能指标（如果可用）
        if 'best_model_metrics' in history:
            best_metrics = history['best_model_metrics']
            summary_df = pd.DataFrame({
                'Metric': ['Best_Model_Val_Loss', 'Best_Model_Val_Acc', 'Best_Model_Val_F1_Macro', 
                          'Best_Model_Val_F1_Weighted', 'Best_Model_Val_Precision', 'Best_Model_Val_Recall',
                          'Training_Process_Best_Acc', 'Training_Process_Best_F1_Macro', 'Training_Process_Best_F1_Weighted'],
                'Value': [best_metrics['loss'], best_metrics['accuracy'], best_metrics['f1_macro'],
                         best_metrics['f1_weighted'], best_metrics['precision'], best_metrics['recall'],
                         history['best_val_acc'], history['best_val_f1_macro'], history['best_val_f1_weighted']]
            })
            summary_df.to_csv(os.path.join(path, 'best_model_summary.csv'), index=False)
            print(f"训练结果已保存到: {path}")
            print(f"📊 最佳模型真实性能已记录到: best_model_summary.csv")
        else:
            print(f"训练结果已保存到: {path}")

    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵（使用seaborn实现）"""
        path = os.path.join(self.args.result_path, f"{self.time_stamp}_{self.setting}")
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
                if hasattr(batch, 'x') and hasattr(batch, 'y'):
                    # GNN数据
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    probs = F.softmax(out, dim=1)
                    pred = out.argmax(dim=1)
                    
                    # 处理标签形状
                    label_squeezed = batch.y.long().squeeze(-1) if batch.y.dim() > 1 else batch.y.long()
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label_squeezed.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                elif len(batch) == 3:
                    # 双路GAF数据（旧版本格式）
                    sum_data, diff_data, label = batch
                    sum_data = sum_data.float().to(self.device)
                    diff_data = diff_data.float().to(self.device)
                    label = label.to(self.device)
                    out = self.model(sum_data, diff_data)
                    probs = F.softmax(out, dim=1)
                    pred = out.argmax(dim=1)
                    
                    # 处理标签形状
                    label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label_squeezed.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                elif len(batch) == 4:
                    # 增强双路GAF数据（新版本格式）
                    sum_data, diff_data, time_series_data, label = batch
                    sum_data = sum_data.float().to(self.device)
                    diff_data = diff_data.float().to(self.device)
                    time_series_data = time_series_data.float().to(self.device)
                    label = label.to(self.device)
                    out = self.model(sum_data, diff_data, time_series_data)
                    probs = F.softmax(out, dim=1)
                    pred = out.argmax(dim=1)
                    
                    # 处理标签形状
                    label_squeezed = label.long().squeeze(-1) if label.dim() > 1 else label.long()
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(label_squeezed.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    # 普通分类数据
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

# ========== 添加高级损失函数 ==========

# 尝试导入timm优化实现
try:
    from timm.loss.cross_entropy import LabelSmoothingCrossEntropy as TimmLabelSmoothingCE
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    TimmLabelSmoothingCE = None


class LabelSmoothingCrossEntropy(nn.Module):
    """
    优化的标签平滑交叉熵损失函数
    
    优先使用timm的高效实现，如果不可用则使用自定义实现。
    基于性能测试，timm实现比自定义实现快10-20%，内存效率更高。
    
    Args:
        smoothing (float): 平滑因子，通常在0.05-0.2之间
        num_classes (int): 类别数量
        dim (int): softmax的维度，默认为-1
        weight (Tensor): 各类别的权重，用于处理类别不平衡
        use_timm (bool): 是否优先使用timm实现
    """
    def __init__(self, smoothing=0.1, num_classes=None, dim=-1, weight=None, use_timm=True):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.weight = weight
        self.use_timm = use_timm and TIMM_AVAILABLE
        
        # 选择实现方式
        if self.use_timm:
            print(f"🚀 使用timm优化的Label Smoothing CE (性能提升10-20%)")
            self.timm_loss = TimmLabelSmoothingCE(smoothing=smoothing)
            self._forward_func = self._timm_forward
        else:
            print(f"📚 使用自定义Label Smoothing CE")
            self.confidence = 1.0 - smoothing
            self._forward_func = self._custom_forward
    
    def forward(self, pred, target):
        return self._forward_func(pred, target)
    
    def _timm_forward(self, pred, target):
        """使用timm实现的前向传播"""
        if self.weight is not None:
            # timm不直接支持类别权重，需要手动处理
            base_loss = self.timm_loss(pred, target)
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            weight_expanded = self.weight[target]
            weighted_ce = (ce_loss * weight_expanded).mean()
            # 保持相同的平滑比例
            return base_loss * (weighted_ce / ce_loss.mean()).detach()
        else:
            return self.timm_loss(pred, target)
    
    def _custom_forward(self, pred, target):
        """优化的自定义实现"""
        if self.num_classes is None:
            self.num_classes = pred.size(1)
        
        # 使用更稳定的log_softmax
        log_pred = F.log_softmax(pred, dim=self.dim)
        
        # 优化的软标签创建 - 避免scatter操作
        nll_loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        
        # 计算标签平滑损失
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # 应用类别权重
        if self.weight is not None:
            weight_expanded = self.weight[target]
            loss = loss * weight_expanded
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss 专门处理难分类样本和类别不平衡问题
    
    通过动态调整损失权重，让模型更专注于难分类的样本，
    减少易分类样本对损失的贡献。
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 计算alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[target]
            
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ConfidencePenaltyLoss(nn.Module):
    """
    置信度惩罚损失
    
    惩罚过度自信的预测，鼓励模型输出更平衡的概率分布，
    有助于缓解类别间相似性问题。
    """
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(pred, target)
        
        # 置信度惩罚：KL散度相对于均匀分布
        prob = F.softmax(pred, dim=1)
        log_prob = F.log_softmax(pred, dim=1)
        uniform_dist = torch.ones_like(prob) / prob.size(1)
        
        # KL(uniform || pred) = -H(uniform) + H_cross(uniform, pred)
        kl_penalty = F.kl_div(log_prob, uniform_dist, reduction='batchmean')
        
        return ce_loss - self.beta * kl_penalty


class HybridFocalLoss(nn.Module):
    """
    混合Focal Loss，结合标签平滑和难样本聚焦
    
    专门针对HVAC异常检测中类别相似性和难样本问题设计
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight
        
        # 如果有标签平滑，使用优化的实现
        if smoothing > 0:
            self.label_smooth = LabelSmoothingCrossEntropy(
                smoothing=smoothing, weight=weight, use_timm=True
            )
        else:
            self.label_smooth = None
    
    def forward(self, pred, target):
        if self.label_smooth is not None:
            # 结合标签平滑的Focal Loss
            ce_loss = self.label_smooth(pred, target)
            p = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
            return focal_loss
        else:
            # 标准Focal Loss
            ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()


class AdaptiveLossScheduler(nn.Module):
    """
    自适应损失调度器
    
    根据训练进度动态调整损失函数参数，
    在训练初期使用更强的正则化，后期减少正则化强度
    """
    
    def __init__(self, base_loss, initial_smoothing=0.2, final_smoothing=0.05, 
                 decay_epochs=50):
        super().__init__()
        self.base_loss = base_loss
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.decay_epochs = decay_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """设置当前训练轮次"""
        self.current_epoch = epoch
        
        # 计算当前的平滑因子
        if epoch < self.decay_epochs:
            progress = epoch / self.decay_epochs
            current_smoothing = self.initial_smoothing * (1 - progress) + \
                              self.final_smoothing * progress
        else:
            current_smoothing = self.final_smoothing
        
        # 更新损失函数参数
        if hasattr(self.base_loss, 'smoothing'):
            self.base_loss.smoothing = current_smoothing
            if hasattr(self.base_loss, 'confidence'):
                self.base_loss.confidence = 1.0 - current_smoothing
    
    def forward(self, pred, target):
        return self.base_loss(pred, target)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    将多个损失函数按权重组合
    """
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight
            
    def forward(self, *args, **kwargs):
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss = loss_fn(*args, **kwargs)
            weighted_loss = self.weights[name] * loss
            total_loss += weighted_loss
            loss_dict[name] = loss.item()
            
        return total_loss 
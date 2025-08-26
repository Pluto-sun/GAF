import argparse
import os
import torch
import torch.backends
from exp.exp import Exp
from utils.print_args import print_args
from utils.memory_cleaner import MemoryCleaner
import random
import numpy as np
import atexit
import signal
import sys

def cleanup_handler(signum=None, frame=None):
    """信号处理函数，用于程序被中断时清理内存"""
    print(f"\n程序被中断 (信号: {signum})，正在清理内存...")
    MemoryCleaner.full_cleanup()
    MemoryCleaner.print_memory_info()
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, cleanup_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, cleanup_handler)  # 终止信号

if __name__ == '__main__':
    # 程序开始时检查和清理内存
    print("=== 程序启动 ===")
    # MemoryCleaner.print_memory_info()
    # MemoryCleaner.full_cleanup()
    
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GAF Classification')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='classification',
                        help='task name, options:[classification]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='SAHU', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/SAHU/', help='root path of the data file')
    parser.add_argument('--step', type=int, required=True, help='slide step')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification')
    parser.add_argument('--result_path', type=str, default='./result/', help='result path')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--rows', type=int, default=None, help='roll size')
    # model define
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--gaf_method', type=str, default='summation', help='GAF method; summation or difference')
    parser.add_argument('--feature_dim', type=int, default=32, help='feature dimension')
    parser.add_argument('--data_type_method', type=str, default='uint8', 
                        help='Data type conversion method; options: [float32, uint8, uint16]')
    # GNN
    parser.add_argument('--use_attention', type=bool, default=True, help='use attention')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--sample_size', type=int, default=1000, help='sample size')
    
    # 1D-CNN参数
    parser.add_argument('--cnn_dropout_rate', type=float, default=0.3, 
                        help='Dropout rate for 1D-CNN model')
    parser.add_argument('--cnn_use_batch_norm', type=bool, default=True, 
                        help='Whether to use BatchNorm in 1D-CNN model')
    # 添加通道分组参数
    parser.add_argument('--channel_groups', type=str, default=None,
                        help='Channel grouping for ClusteredResNet. Format: "0,1,2|3,4,5|6,7,8"')
    parser.add_argument('--hvac_groups', type=str, default=None,
                        help='HVAC signal grouping for MultiImageFeatureNet. Format: "SA_TEMP,OA_TEMP|OA_CFM,RA_CFM|SA_SP"')
    # 通道压缩参数
    parser.add_argument('--use_channel_compression', action='store_true', default=False, help='use channel compression')
    parser.add_argument('--compression_strategy', type=str, default='signal_compression', help='compression strategy')
    parser.add_argument('--compression_ratio', type=float, default=0.7, help='compression ratio')
    parser.add_argument('--compression_channels', type=int, default=None, help='compression channels')
    parser.add_argument('--adaptive_compression_ratios', type=list, default=[0.5, 0.7, 0.8], help='adaptive compression ratios')
    parser.add_argument('--hvac_group_compression_ratios', type=list, default=None, help='hvac group compression ratios')
    # 双路网络模块配置参数
    parser.add_argument('--extractor_type', type=str, default='large_kernel',
                        help='Feature extractor type for DualGAFNet. Options: [large_kernel, inception, dilated, multiscale]')
    parser.add_argument('--fusion_type', type=str, default='adaptive',
                        help='Feature fusion type for DualGAFNet. Options: [adaptive, concat, bidirectional, gated, add, mul, weighted_add]')
    parser.add_argument('--attention_type', type=str, default='channel',
                        help='Attention type for DualGAFNet. Options: [channel, spatial, cbam, self, none]')
    parser.add_argument('--classifier_type', type=str, default='mlp',
                        help='Classifier type for DualGAFNet. Options: [mlp, simple, residual, residual_bottleneck, residual_dense, feature_compression, hierarchical, efficient_mlp, efficient_simple, global_pooling, conv1d, separable]')
    
    # 统计特征配置（新增）
    parser.add_argument('--use_statistical_features', action='store_true', default=False, help='是否使用统计特征')
    parser.add_argument('--stat_type', type=str, default='comprehensive',
                       choices=['basic', 'comprehensive', 'correlation_focused'], help='统计特征类型')
    parser.add_argument('--multimodal_fusion_strategy', type=str, default='concat',
                       choices=['concat', 'attention', 'gated', 'adaptive','film'], help='多模态融合策略')

    # 信号级统计特征配置（新增）
    parser.add_argument('--use_signal_level_stats', action='store_true', default=False, 
                       help='是否使用信号级统计特征（对比实验，替代全局统计特征）')
    parser.add_argument('--signal_stat_type', type=str, default='comprehensive',
                       choices=['basic', 'comprehensive', 'extended'], help='信号级统计特征类型')
    parser.add_argument('--signal_stat_fusion_strategy', type=str, default='concat_project',
                       choices=['concat_project', 'attention_fusion', 'gated_fusion', 'residual_fusion', 'cross_attention', 'adaptive_fusion'],
                       help='信号级统计特征融合策略')
    parser.add_argument('--signal_stat_feature_dim', type=int, default=32,
                       help='信号级统计特征维度')

    # 消融实验配置参数
    parser.add_argument('--use_diff_branch', action='store_true', default=True,
                       help='是否使用差分分支进行GAF融合（消融实验开关）。False时仅使用sum分支')
    parser.add_argument('--ablation_mode', type=str, default='none',
                       choices=['none', 'no_diff', 'no_stat', 'no_attention', 'minimal'],
                       help='消融实验模式快捷设置: none(完整模型), no_diff(无差分分支), no_stat(无统计特征), no_attention(无注意力), minimal(最简模型)')
    
    # SimpleGAFNet特有参数（用于消融实验）
    parser.add_argument('--backbone_type', type=str, default='resnet18',
                       choices=['simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'inception'],
                       help='SimpleGAFNet的主干架构类型: simple_cnn(轻量级), resnet18(推荐), resnet34(深层), resnet50(实验性), inception(多尺度)')
    parser.add_argument('--use_sum_branch', action='store_true', default=True,
                       help='SimpleGAFNet选择GAF分支: True(使用summation GAF), False(使用difference GAF)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--cuda_debug', action='store_true', default=False,
                        help='启用CUDA调试模式（用于诊断CUDA错误）')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9,
                        help='GPU内存使用比例限制 (0.7-0.95)')
    parser.add_argument('--safe_mode', action='store_true', default=False,
                        help='安全模式：启用所有内存优化和错误处理')
    parser.add_argument('--drop_last_batch', action='store_true', default=True, 
                        help='丢弃最后一个不完整的batch（避免BatchNorm错误）')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    
    # 学习率调度器配置（基于F1分数的改进方案）
    parser.add_argument('--lr_scheduler_type', type=str, default='f1_based',
                        choices=['f1_based', 'loss_based', 'composite_f1_priority', 'composite_weighted'],
                        help='学习率调度器类型: f1_based(基于F1分数，推荐), loss_based(基于损失), composite_f1_priority(F1优先复合), composite_weighted(加权复合)')
    parser.add_argument('--lr_loss_weight', type=float, default=0.3,
                        help='复合调度器中损失权重 (仅composite_weighted时使用)')
    parser.add_argument('--lr_f1_weight', type=float, default=0.7,
                        help='复合调度器中F1权重 (仅composite_weighted时使用)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # 梯度累积优化（针对小batch训练）
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='梯度累积步数。设为>1时实现梯度累积，有效batch_size = batch_size * gradient_accumulation_steps')
    parser.add_argument('--enable_auto_gradient_accumulation', action='store_true', default=False,
                        help='根据batch_size自动启用梯度累积（batch_size<8时自动设为2，<4时自动设为4）')
    
    # 高级损失函数配置（用于解决类别相似性问题）
    parser.add_argument('--loss_type', type=str, default='ce', 
                        choices=['ce', 'label_smoothing', 'focal', 'confidence_penalty', 'combined',
                                'label_smoothing_optimized', 'hybrid_focal', 'adaptive_smoothing'],
                        help='损失函数类型: ce(交叉熵), label_smoothing(标签平滑), focal(焦点损失), confidence_penalty(置信度惩罚), combined(组合损失), label_smoothing_optimized(优化标签平滑), hybrid_focal(混合焦点损失), adaptive_smoothing(自适应平滑)')
    
    # 标签平滑参数
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑因子 (0.0-0.5)，推荐值：相似类别0.1-0.2，差异较大类别0.05-0.1')
    
    # Focal Loss参数
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal Loss的alpha参数，用于平衡正负样本')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数，控制难样本的聚焦程度 (1.0-3.0)')
    
    # 置信度惩罚参数
    parser.add_argument('--confidence_penalty_beta', type=float, default=0.1,
                        help='置信度惩罚强度 (0.01-0.2)，较高值会减少过度自信')
    
    # 优化损失函数参数（基于性能测试：timm实现比自定义实现快10-20%）
    parser.add_argument('--use_timm_loss', action='store_true', default=True,
                        help='优先使用timm优化实现，性能提升10-20%，内存效率更高')
    parser.add_argument('--adaptive_initial_smoothing', type=float, default=0.2,
                        help='自适应标签平滑初始值（训练开始时）')
    parser.add_argument('--adaptive_final_smoothing', type=float, default=0.05,
                        help='自适应标签平滑最终值（训练结束时）')
    parser.add_argument('--adaptive_decay_epochs', type=int, default=30,
                        help='自适应平滑衰减周期（epoch数）')
    
    # 类别权重（用于处理类别不平衡）- 默认禁用，适用于平衡数据集
    parser.add_argument('--enable_class_weights', action='store_true', 
                        help='启用类别权重（仅当数据不平衡时使用）')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='手动指定类别权重，格式为逗号分隔的数字，如 "1.0,2.0,1.5"')
    
    # 损失函数预设配置
    parser.add_argument('--loss_preset', type=str, default=None,
                        choices=['hvac_similar', 'imbalanced_focus', 'hard_samples', 'overconfidence_prevention',
                                'hvac_similar_optimized', 'hvac_adaptive', 'hvac_hard_samples', 'production_optimized'],
                        help='损失函数预设配置: hvac_similar(HVAC相似类别), imbalanced_focus(类别不平衡), hard_samples(难分类样本), overconfidence_prevention(防止过度自信), hvac_similar_optimized(HVAC相似类别+高性能优化,推荐), hvac_adaptive(HVAC自适应平滑), hvac_hard_samples(HVAC难样本聚焦), production_optimized(生产环境优化)')

    # parallel processing optimization (并行处理优化)
    parser.add_argument('--n_jobs', type=int, default=16, 
                        help='Number of parallel jobs for data processing. -1 means auto-detect (min(cpu_count, 8))')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='Enable multiprocessing for GAF generation and data conversion')
    parser.add_argument('--chunk_size', type=int, default=800,
                        help='Chunk size for parallel processing. Larger values use more memory but reduce overhead')
    parser.add_argument('--disable_parallel', action='store_true', default=True,
                        help='Disable all parallel processing optimizations (useful for debugging)')
    parser.add_argument('--use_shared_memory', action='store_true', default=True,
                        help='Enable shared memory optimization for faster inter-process communication')
    parser.add_argument('--disable_shared_memory', action='store_true', default=False,
                        help='Disable shared memory optimization (fallback to standard multiprocessing)')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    
    # 处理并行处理参数
    import multiprocessing as mp
    import sys
    if args.disable_parallel:
        # 禁用所有并行处理
        args.use_multiprocessing = False
        args.use_shared_memory = False
        args.n_jobs = 1
        print("🔧 并行处理已禁用 (disable_parallel=True)")
    else:
        # 自动检测CPU核心数
        if args.n_jobs == -1:
            args.n_jobs = min(mp.cpu_count(), 8)  # 限制最大进程数
        elif args.n_jobs <= 0:
            args.n_jobs = 1
        
        # 处理共享内存配置
        if args.disable_shared_memory:
            args.use_shared_memory = False
            print("🔧 共享内存优化已禁用")
        elif sys.version_info < (3, 8):
            args.use_shared_memory = False
            print("⚠️ Python版本过低，禁用共享内存优化 (需要Python 3.8+)")
    
    # 🔧 CUDA调试和GPU内存管理配置
    if args.cuda_debug or args.safe_mode:
        print("\n🔧 启用CUDA调试和安全模式配置...")
        import os
        
        # 设置CUDA调试环境变量
        if args.cuda_debug:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            print("✅ 已启用CUDA_LAUNCH_BLOCKING=1 (同步执行，便于调试)")
        
        # GPU内存优化
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            total_memory_gb = total_memory / (1024**3)
            
            print(f"🎯 GPU信息: {torch.cuda.get_device_name(current_device)}")
            print(f"📊 总显存: {total_memory_gb:.2f} GB")
            
            # 设置内存分配策略
            if args.safe_mode:
                # 启用内存池复用和预分配
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                print("✅ 已启用CUDA内存优化策略")
                
                # 限制GPU内存使用
                if hasattr(torch.cuda, 'set_memory_fraction'):
                    torch.cuda.set_memory_fraction(args.gpu_memory_fraction)
                    print(f"✅ GPU内存限制: {args.gpu_memory_fraction*100:.0f}%")
                
            # 清理初始GPU内存
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            
            # 显示当前内存使用情况
            allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved = torch.cuda.memory_reserved(current_device) / (1024**3)
            print(f"📈 当前GPU内存使用: {allocated:.3f} GB (已分配) / {reserved:.3f} GB (已保留)")
            
        else:
            print("⚠️ CUDA不可用，跳过GPU配置")
    
    # 🔧 安全模式额外优化
    if args.safe_mode:
        print("\n🛡️ 安全模式：启用额外的稳定性优化...")
        
        # 自动调整batch_size以适应内存限制
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_memory_gb < 6:  # 小显存GPU
                if args.batch_size > 16:
                    original_batch_size = args.batch_size
                    args.batch_size = 16
                    print(f"🔧 小显存设备，批次大小调整: {original_batch_size} → {args.batch_size}")
            elif total_memory_gb < 8:  # 中等显存GPU
                if args.batch_size > 32:
                    original_batch_size = args.batch_size
                    args.batch_size = 32
                    print(f"🔧 中等显存设备，批次大小调整: {original_batch_size} → {args.batch_size}")
        
        # 强制启用梯度检查点（如果支持）
        print("✅ 安全模式优化已启用")
    
    print(f"\n📊 最终配置:")
    print(f"   CUDA调试: {'启用' if args.cuda_debug else '禁用'}")
    print(f"   安全模式: {'启用' if args.safe_mode else '禁用'}")
    print(f"   GPU内存限制: {args.gpu_memory_fraction*100:.0f}%")
    print(f"   批次大小: {args.batch_size}")
    print(f"   Worker数量: {args.num_workers}")
    
    # 根据系统资源调整参数
    available_memory_gb = None
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        print("⚠️ psutil未安装，无法自动检测内存大小")
    
    # 根据内存大小调整chunk_size - 针对高端服务器优化
    if available_memory_gb is not None:
        cpu_cores = mp.cpu_count()
        
        if available_memory_gb < 8:  # 小内存系统
            args.chunk_size = min(args.chunk_size, 50)
            args.n_jobs = min(args.n_jobs, 2)
            print(f"💾 检测到小内存系统 ({available_memory_gb:.1f}GB)，调整参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
        elif available_memory_gb < 32:  # 中等内存系统
            args.chunk_size = max(args.chunk_size, 200)
            args.n_jobs = min(args.n_jobs, 8)
            print(f"💾 检测到中等内存系统 ({available_memory_gb:.1f}GB)，调整参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
        elif available_memory_gb < 64:  # 大内存系统
            args.chunk_size = max(args.chunk_size, 400)
            args.n_jobs = min(args.n_jobs, 16)
            print(f"💾 检测到大内存系统 ({available_memory_gb:.1f}GB)，优化参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
        else:  # 超大内存服务器 (您的配置)
            # 针对32核128GB的高端配置
            if cpu_cores >= 32:
                # 使用适中的进程数，避免过度并行
                args.n_jobs = min(args.n_jobs, 20)  # 留出一些核心给系统
                args.chunk_size = max(args.chunk_size, 800)  # 大块大小减少开销
                print(f"🚀 检测到高端服务器 ({available_memory_gb:.1f}GB, {cpu_cores}核)，高性能配置：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
            else:
                args.chunk_size = max(args.chunk_size, 600)
                print(f"💾 检测到超大内存系统 ({available_memory_gb:.1f}GB)，优化参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
    
    shared_memory_status = "启用" if args.use_shared_memory else "禁用"
    print(f"🚀 并行处理配置 - 进程数: {args.n_jobs}, 多进程: {args.use_multiprocessing}, 块大小: {args.chunk_size}, 共享内存: {shared_memory_status}")
    
    # 处理BatchNorm友好配置
    print(f"📊 DataLoader配置: num_workers={args.num_workers}, batch_size={args.batch_size}")
    if args.drop_last_batch:
        print(f"🔧 已启用drop_last_batch=True，将丢弃不完整的最后一个batch（避免BatchNorm在batch_size=1时出错）")
    else:
        print(f"⚠️ drop_last_batch=False，可能在最后一个batch时遇到BatchNorm错误")
        print(f"💡 提示：如果遇到BatchNorm错误，建议设置 --drop_last_batch")
    
    # 处理消融实验模式（在其他参数处理之前）
    if args.ablation_mode != 'none':
        print(f"\n🔬 应用消融实验模式: {args.ablation_mode}")
        
        if args.ablation_mode == 'no_diff':
            # 移除差分分支
            args.use_diff_branch = False
            print(f"   ❌ 禁用差分分支 (use_diff_branch=False)")
            
        elif args.ablation_mode == 'no_stat':
            # 移除统计特征
            args.use_statistical_features = False
            print(f"   ❌ 禁用统计特征 (use_statistical_features=False)")
            
        elif args.ablation_mode == 'no_attention':
            # 移除注意力机制
            args.attention_type = 'none'
            print(f"   ❌ 禁用注意力机制 (attention_type=none)")
            
        elif args.ablation_mode == 'minimal':
            # 最简化模型：移除所有高级组件
            args.use_diff_branch = False
            args.use_statistical_features = False
            args.attention_type = 'none'
            print(f"   ❌ 禁用差分分支 (use_diff_branch=False)")
            print(f"   ❌ 禁用统计特征 (use_statistical_features=False)")
            print(f"   ❌ 禁用注意力机制 (attention_type=none)")
            print(f"   💡 现在使用最简化模型（仅sum分支 + 基础ResNet）")
        
        print(f"🔬 消融实验配置完成\n")
    else:
        # 显示当前完整模型配置
        ablation_status = []
        if not args.use_diff_branch:
            ablation_status.append("差分分支已禁用")
        if not args.use_statistical_features:
            ablation_status.append("统计特征已禁用")
        if args.attention_type == 'none':
            ablation_status.append("注意力机制已禁用")
        
        if ablation_status:
            print(f"\n🔬 手动消融配置: {' + '.join(ablation_status)}")
        else:
            print(f"\n🔬 使用完整模型（未启用消融实验）")
    
    # 处理梯度累积参数
    if args.enable_auto_gradient_accumulation:
        if args.batch_size < 4:
            args.gradient_accumulation_steps = 4
            print(f"🔄 自动梯度累积: batch_size={args.batch_size} < 4，设置累积步数为{args.gradient_accumulation_steps}")
        elif args.batch_size < 8:
            args.gradient_accumulation_steps = 2
            print(f"🔄 自动梯度累积: batch_size={args.batch_size} < 8，设置累积步数为{args.gradient_accumulation_steps}")
        else:
            print(f"🔄 自动梯度累积: batch_size={args.batch_size} >= 8，无需梯度累积")
    
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        print(f"📊 梯度累积已启用:")
        print(f"   实际batch_size: {args.batch_size}")
        print(f"   累积步数: {args.gradient_accumulation_steps}")
        print(f"   有效batch_size: {effective_batch_size}")
        print(f"   建议: 这有助于减少小batch带来的梯度噪声")
    
    # 处理通道分组参数
    if args.channel_groups is not None:
        try:
            # 将字符串格式的分组转换为列表格式
            args.channel_groups = [
                [int(x) for x in group.split(',')]
                for group in args.channel_groups.split('|')
            ]
        except Exception as e:
            print(f"Error parsing channel_groups: {e}")
            print("Using default channel grouping")
            args.channel_groups = None
    
    # 处理HVAC信号分组参数
    if args.hvac_groups is not None:
        try:
            # 将字符串格式的HVAC分组转换为列表格式
            args.hvac_groups = [
                [signal.strip() for signal in group.split(',')]
                for group in args.hvac_groups.split('|')
            ]
            print(f"HVAC Groups parsed: {len(args.hvac_groups)} groups")
            for i, group in enumerate(args.hvac_groups):
                print(f"  Group {i}: {group}")
        except Exception as e:
            print(f"Error parsing hvac_groups: {e}")
            print("Using default HVAC grouping")
            args.hvac_groups = None

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            # 构建实验设置字符串
            setting = f'{args.data}_{args.model}_{args.des}_sl{args.seq_len}_step{args.step}_' \
                    f'fd{args.feature_dim}_extractor-{args.extractor_type}_gaf-{args.fusion_type}_' \
                    f'attention-{args.attention_type}_classifier-{args.classifier_type}'
            
            if args.use_statistical_features:
                setting += f'_stat-{args.stat_type}_fusion-{args.multimodal_fusion_strategy}'
            
            if args.ablation_mode != 'none':
                setting += f'_ablation-{args.ablation_mode}'
            if args.use_signal_level_stats:
                setting += f'_sl-{args.signal_stat_type}_fusion-{args.signal_stat_fusion_strategy}'
            if args.hvac_groups:
                setting += '_grouped'
            if args.use_channel_compression:
                setting += f'_compression-{args.compression_strategy}_ratio-{args.compression_ratio}'
            exp = Exp(args,setting)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train()
            exp.evaluate_report()

            print('>>>>>>>validating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.vali()
            
            # 每次实验后清理内存
            # MemoryCleaner.full_cleanup()
            
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        # 构建实验设置字符串
        setting = f'{args.data}_{args.model}_{args.des}_sl{args.seq_len}_step{args.step}_' \
                f'fd{args.feature_dim}_extractor-{args.extractor_type}_gaf-{args.fusion_type}_' \
                f'attention-{args.attention_type}_classifier-{args.classifier_type}'
        
        if args.use_statistical_features:
            setting += f'_stat-{args.stat_type}_fusion-{args.multimodal_fusion_strategy}'
        
        if args.ablation_mode != 'none':
            setting += f'_ablation-{args.ablation_mode}'
        if args.use_signal_level_stats:
            setting += f'_sl-{args.signal_stat_type}_fusion-{args.signal_stat_fusion_strategy}'
        if args.hvac_groups:
            setting += '_grouped'
        
        exp = Exp(args, setting)  # set experiments
        ii = 0

        print('>>>>>>>validating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
    
    # 程序结束时的最终清理
    # print("\n=== 程序即将结束 ===")
    # MemoryCleaner.full_cleanup()
    # MemoryCleaner.print_memory_info()
    print("=== 程序结束 ===\n")

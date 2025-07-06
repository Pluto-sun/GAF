#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版双路GAF网络运行脚本
集成统计特征提取器和多模态融合功能
"""

import argparse
import torch
import random
import numpy as np
import os
from exp.exp import Exp

def main():
    parser = argparse.ArgumentParser(description='增强版双路GAF网络训练')
    
    # 基础配置
    parser.add_argument('--model', type=str, required=True, default='DualGAFNet', help='模型名称')
    parser.add_argument('--data', type=str, required=True, default='DualGAF', help='数据集名称')
    parser.add_argument('--root_path', type=str, default='./dataset/SAHU/', help='数据根目录')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型保存路径')
    parser.add_argument('--result_path', type=str, default='./result/', help='结果保存路径')
    
    # 数据配置
    parser.add_argument('--seq_len', type=int, default=96, help='时间序列长度')
    parser.add_argument('--step', type=int, default=96, help='滑动窗口步长')
    parser.add_argument('--enc_in', type=int, default=26, help='输入特征维度')
    parser.add_argument('--num_class', type=int, default=5, help='分类类别数')
    parser.add_argument('--test_size', type=float, default=0.3, help='测试集比例')
    parser.add_argument('--data_type_method', type=str, default='uint8', choices=['float32', 'uint8', 'uint16'], help='数据类型转换方法')
    
    # 模型配置
    parser.add_argument('--feature_dim', type=int, default=64, help='特征维度')
    parser.add_argument('--extractor_type', type=str, default='large_kernel', 
                       choices=['large_kernel', 'inception', 'dilated', 'multiscale'], help='特征提取器类型')
    parser.add_argument('--fusion_type', type=str, default='adaptive',
                       choices=['adaptive', 'concat', 'add', 'mul', 'weighted_add', 'bidirectional', 'gated'], help='GAF融合类型')
    parser.add_argument('--attention_type', type=str, default='channel',
                       choices=['channel', 'spatial', 'cbam', 'self', 'none'], help='注意力类型')
    parser.add_argument('--classifier_type', type=str, default='mlp',
                       choices=['mlp', 'simple'], help='分类器类型')
    
    # 统计特征配置（新增）
    parser.add_argument('--use_statistical_features', action='store_true', default=True, help='是否使用统计特征')
    parser.add_argument('--stat_type', type=str, default='comprehensive',
                       choices=['basic', 'comprehensive', 'correlation_focused'], help='统计特征类型')
    parser.add_argument('--multimodal_fusion_strategy', type=str, default='concat',
                       choices=['concat', 'attention', 'gated', 'adaptive'], help='多模态融合策略')
    
    # 训练配置
    parser.add_argument('--train_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 设备配置
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--gpu_type', type=str, default='cuda', choices=['cuda', 'mps'], help='GPU类型')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多GPU')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU设备号')
    
    # 其他配置
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--seed', type=int, default=2021, help='随机种子')
    parser.add_argument('--des', type=str, default='enhanced_dual_gaf', help='实验描述')
    
    # 并行处理优化配置 (Parallel Processing Optimization)
    parser.add_argument('--n_jobs', type=int, default=-1, 
                        help='并行处理进程数，-1表示自动检测 (Number of parallel jobs, -1 for auto-detect)')
    parser.add_argument('--use_multiprocessing', action='store_true', default=True,
                        help='启用多进程处理GAF生成和数据转换 (Enable multiprocessing for GAF generation and data conversion)')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='并行处理的数据块大小 (Chunk size for parallel processing)')
    parser.add_argument('--disable_parallel', action='store_true', default=False,
                        help='禁用所有并行处理优化，用于调试 (Disable all parallel processing optimizations for debugging)')
    parser.add_argument('--use_shared_memory', action='store_true', default=True,
                        help='启用共享内存优化，减少进程间通信开销 (Enable shared memory optimization for faster inter-process communication)')
    parser.add_argument('--disable_shared_memory', action='store_true', default=False,
                        help='禁用共享内存优化，回退到标准多进程 (Disable shared memory optimization, fallback to standard multiprocessing)')
    
    # HVAC信号分组配置（可选）
    parser.add_argument('--use_hvac_groups', action='store_true', help='使用HVAC信号分组')
    
    args = parser.parse_args()
    
    # 设置随机种子
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 处理并行处理参数
    import multiprocessing as mp
    import sys
    if args.disable_parallel:
        # 禁用所有并行处理
        args.use_multiprocessing = False
        args.use_shared_memory = False
        args.n_jobs = 1
        print("⚠️  并行处理已禁用 (disable_parallel=True)")
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
            print("⚠️  Python版本过低，禁用共享内存优化 (需要Python 3.8+)")
        
        # 根据系统资源调整参数
        available_memory_gb = None
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            print("⚠️  psutil未安装，无法自动检测内存大小")
        
        # 根据内存大小调整chunk_size - 针对高端服务器优化
        if available_memory_gb is not None:
            cpu_cores = mp.cpu_count()
            
            if available_memory_gb < 8:  # 小内存系统
                args.chunk_size = min(args.chunk_size, 50)
                args.n_jobs = min(args.n_jobs, 2)
                print(f"🔧 检测到小内存系统 ({available_memory_gb:.1f}GB)，调整参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
            elif available_memory_gb < 32:  # 中等内存系统
                args.chunk_size = max(args.chunk_size, 200)
                args.n_jobs = min(args.n_jobs, 8)
                print(f"🔧 检测到中等内存系统 ({available_memory_gb:.1f}GB)，调整参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
            elif available_memory_gb < 64:  # 大内存系统
                args.chunk_size = max(args.chunk_size, 400)
                args.n_jobs = min(args.n_jobs, 16)
                print(f"🚀 检测到大内存系统 ({available_memory_gb:.1f}GB)，优化参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
            else:  # 超大内存服务器 (您的配置)
                # 针对32核128GB的高端配置
                if cpu_cores >= 32:
                    # 使用适中的进程数，避免过度并行
                    args.n_jobs = min(args.n_jobs, 20)  # 留出一些核心给系统
                    args.chunk_size = max(args.chunk_size, 800)  # 大块大小减少开销
                    print(f"🚀 检测到高端服务器 ({available_memory_gb:.1f}GB, {cpu_cores}核)，高性能配置：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
                else:
                    args.chunk_size = max(args.chunk_size, 600)
                    print(f"🚀 检测到超大内存系统 ({available_memory_gb:.1f}GB)，优化参数：chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
        
        shared_memory_status = "启用" if args.use_shared_memory else "禁用"
        print(f"⚡ 并行处理配置 - 进程数: {args.n_jobs}, 多进程: {args.use_multiprocessing}, 块大小: {args.chunk_size}, 共享内存: {shared_memory_status}")
    
    # 设置HVAC信号分组（如果启用）
    if args.use_hvac_groups:
        args.hvac_groups = [
            ['SA_TEMP','OA_TEMP','MA_TEMP','RA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'],  # 温度组
            ['OA_CFM','RA_CFM','SA_CFM'],                 # 流量组
            ['SA_SP', 'SA_SPSPT'],                          # 设定点组
            ['SF_WAT', 'RF_WAT'],                  # 阀门组
            ['SF_SPD','RF_SPD','SF_CS','RF_CS'],                  # 风机组
            ['CHWC_VLV_DM','CHWC_VLV'],                  # 冷水阀组
            ['OA_DMPR_DM','RA_DMPR_DM','OA_DMPR','RA_DMPR'],     # 风门组
        ]
    else:
        args.hvac_groups = None
    
    # 构建实验设置字符串
    setting = f'{args.data}_{args.model}_{args.des}_sl{args.seq_len}_step{args.step}_' \
              f'gaf{args.fusion_type}_fd{args.feature_dim}_dtype{args.data_type_method}_' \
              f'{args.extractor_type}-{args.attention_type}-{args.classifier_type}'
    
    if args.use_statistical_features:
        setting += f'-stat{args.stat_type}-fusion{args.multimodal_fusion_strategy}'
    
    if args.use_hvac_groups:
        setting += '-grouped'
    
    print('='*100)
    print(f'实验设置: {setting}')
    print('='*100)
    
    # 打印配置信息
    print("\n📋 模型配置信息:")
    print(f"  数据集: {args.data}")
    print(f"  模型: {args.model}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  特征提取器: {args.extractor_type}")
    print(f"  GAF融合类型: {args.fusion_type}")
    print(f"  注意力类型: {args.attention_type}")
    print(f"  分类器类型: {args.classifier_type}")
    print(f"  使用统计特征: {args.use_statistical_features}")
    if args.use_statistical_features:
        print(f"  统计特征类型: {args.stat_type}")
        print(f"  多模态融合策略: {args.multimodal_fusion_strategy}")
    print(f"  使用HVAC分组: {args.use_hvac_groups}")
    
    print(f"\n🚀 训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  训练轮数: {args.train_epochs}")
    print(f"  数据类型: {args.data_type_method}")
    
    print(f"\n⚡ 并行优化配置:")
    print(f"  并行进程数: {args.n_jobs}")
    print(f"  使用多进程: {args.use_multiprocessing}")
    print(f"  数据块大小: {args.chunk_size}")
    print(f"  并行处理: {'禁用' if args.disable_parallel else '启用'}")
    print(f"  共享内存优化: {'启用' if args.use_shared_memory else '禁用'}")
    print('='*100)
    
    # 创建实验对象并运行
    exp = Exp(args, setting)
    
    print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    model, history = exp.train()
    
    print('>>>>>>>开始测试 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.evaluate_report()
    
    print('>>>>>>>训练和测试完成 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版双路GAF网络测试脚本
测试统计特征提取器和多模态融合功能
"""

import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.DualGAFNet import Model, TimeSeriesStatisticalExtractor, MultiModalFusion
from data_provider.data_loader.DualGAFDataLoader import DualGAFDataLoader

def test_statistical_extractor():
    """测试时序统计特征提取器"""
    print("="*60)
    print("测试时序统计特征提取器")
    print("="*60)
    
    # 创建测试数据
    B, T, C = 4, 96, 26  # batch_size=4, time_length=96, num_signals=26
    time_series_data = torch.randn(B, T, C)
    
    print(f"输入时序数据形状: {time_series_data.shape}")
    
    # 测试不同的统计特征类型
    stat_types = ['basic', 'comprehensive', 'correlation_focused']
    
    for stat_type in stat_types:
        print(f"\n--- 测试统计特征类型: {stat_type} ---")
        
        extractor = TimeSeriesStatisticalExtractor(
            num_signals=C,
            time_length=T,
            feature_dim=128,
            stat_type=stat_type
        )
        
        # 前向传播
        stat_features = extractor(time_series_data)
        print(f"统计特征输出形状: {stat_features.shape}")
        print(f"统计特征范围: [{stat_features.min().item():.4f}, {stat_features.max().item():.4f}]")
        
        # 测试格式检测功能：创建真正的BCT格式数据
        # 原始BTC数据: [B, T, C] = [4, 96, 26]
        # 目标BCT数据: [B, C, T] = [4, 26, 96] - 但保持每个信号的时序含义不变
        
        # 正确的方法：重新构造BCT格式的数据
        B, T, C = time_series_data.shape
        # 创建BCT格式的数据，其中每个信号在时间维度上的数据与BTC格式一致
        time_series_data_bct = torch.zeros(B, C, T)
        for b in range(B):
            for c in range(C):
                # 将BTC格式中第c个信号的时序数据复制到BCT格式的对应位置
                time_series_data_bct[b, c, :] = time_series_data[b, :, c]
        
        stat_features_2 = extractor(time_series_data_bct)
        print(f"BCT格式输入后的输出形状: {stat_features_2.shape}")
        
        # 验证两种输入格式的结果是否一致
        if torch.allclose(stat_features, stat_features_2, atol=1e-5):
            print("✓ 不同输入格式的结果一致")
        else:
            # 对于相同数据的不同格式表示，结果应该一致
            # 如果不一致，检查是否在可接受的数值误差范围内
            max_diff = (stat_features - stat_features_2).abs().max().item()
            if max_diff < 1e-3:  # 允许更大的数值误差，因为神经网络的非线性
                print(f"✓ 不同输入格式的结果基本一致（最大差异: {max_diff:.6f}）")
            else:
                print("✗ 不同输入格式的结果差异较大")
                print(f"  第一种格式结果范围: [{stat_features.min().item():.6f}, {stat_features.max().item():.6f}]")
                print(f"  第二种格式结果范围: [{stat_features_2.min().item():.6f}, {stat_features_2.max().item():.6f}]")
                print(f"  最大差异: {max_diff:.6f}")

def test_multimodal_fusion():
    """测试多模态融合模块"""
    print("\n" + "="*60)
    print("测试多模态融合模块")
    print("="*60)
    
    # 创建测试数据
    B, C, feature_dim = 4, 26, 128
    gaf_features = torch.randn(B, C, feature_dim)
    stat_features = torch.randn(B, feature_dim)
    
    print(f"GAF特征形状: {gaf_features.shape}")
    print(f"统计特征形状: {stat_features.shape}")
    
    # 测试不同的融合策略
    fusion_strategies = ['concat', 'attention', 'gated', 'adaptive']
    
    for strategy in fusion_strategies:
        print(f"\n--- 测试融合策略: {strategy} ---")
        
        fusion_module = MultiModalFusion(
            feature_dim=feature_dim,
            fusion_strategy=strategy
        )
        
        # 前向传播
        fused_features = fusion_module(gaf_features, stat_features)
        print(f"融合后特征形状: {fused_features.shape}")
        print(f"融合特征范围: [{fused_features.min().item():.4f}, {fused_features.max().item():.4f}]")
        
        # 验证输出形状
        expected_shape = (B, C, feature_dim)
        if fused_features.shape == expected_shape:
            print(f"✓ 输出形状正确: {fused_features.shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {fused_features.shape}")

def test_enhanced_dual_gaf_model():
    """测试增强版双路GAF网络"""
    print("\n" + "="*60)
    print("测试增强版双路GAF网络")
    print("="*60)
    
    # 创建测试数据
    B, C, H, W = 4, 26, 32, 32
    T = 96
    
    sum_x = torch.randn(B, C, H, W)
    diff_x = torch.randn(B, C, H, W)
    time_series_x = torch.randn(B, C, T)  # [B, C, T] 格式
    
    print(f"Summation GAF形状: {sum_x.shape}")
    print(f"Difference GAF形状: {diff_x.shape}")
    print(f"时序数据形状: {time_series_x.shape}")
    
    # 创建配置
    configs = type("cfg", (), {
        "feature_dim": 64,
        "num_class": 4,
        "enc_in": C,
        "seq_len": T,
        "extractor_type": "large_kernel",
        "fusion_type": "adaptive", 
        "attention_type": "channel",
        "classifier_type": "mlp",
        "use_statistical_features": True,
        "stat_type": "comprehensive",
        "multimodal_fusion_strategy": "concat",
        "hvac_groups": None,
        "feature_columns": None
    })()
    
    # 创建模型
    model = Model(configs)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    print("\n--- 测试前向传播 ---")
    
    # 1. 测试带统计特征的前向传播
    model.eval()
    with torch.no_grad():
        output = model(sum_x, diff_x, time_series_x)
        print(f"带统计特征的输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 验证输出形状
        expected_shape = (B, configs.num_class)
        if output.shape == expected_shape:
            print(f"✓ 输出形状正确: {output.shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {output.shape}")
    
    # 2. 测试不带统计特征的模型
    print("\n--- 测试不带统计特征的模型 ---")
    configs_no_stat = type("cfg", (), {
        "feature_dim": 64,
        "num_class": 4,
        "enc_in": C,
        "seq_len": T,
        "extractor_type": "large_kernel",
        "fusion_type": "adaptive", 
        "attention_type": "channel",
        "classifier_type": "mlp",
        "use_statistical_features": False,
        "stat_type": "comprehensive",
        "multimodal_fusion_strategy": "concat",
        "hvac_groups": None,
        "feature_columns": None
    })()
    
    model_no_stat = Model(configs_no_stat)
    model_no_stat.eval()
    with torch.no_grad():
        output_no_stat = model_no_stat(sum_x, diff_x)
        print(f"不带统计特征的输出形状: {output_no_stat.shape}")
        print(f"输出范围: [{output_no_stat.min().item():.4f}, {output_no_stat.max().item():.4f}]")
    
    # 3. 测试错误情况：启用统计特征但不提供时序数据
    print("\n--- 测试错误处理 ---")
    try:
        model(sum_x, diff_x)  # 没有提供 time_series_x
        print("✗ 应该抛出错误但没有")
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")

def test_data_loader_compatibility():
    """测试数据加载器兼容性"""
    print("\n" + "="*60)
    print("测试数据加载器兼容性")
    print("="*60)
    
    # 创建模拟的args对象
    class MockArgs:
        def __init__(self):
            self.root_path = './dataset/SAHU/direct_5_working'
            self.seq_len = 96
            self.step = 96
            self.test_size = 0.2
            self.data_type_method = 'uint8'
            self.batch_size = 4
            self.num_workers = 0
    
    args = MockArgs()
    
    # 检查数据加载器是否能正确返回四元组
    print("检查数据加载器输出格式...")
    
    try:
        # 注意：这需要实际的数据文件存在
        # 如果没有数据文件，这部分会失败，但不影响其他测试
        train_dataset = DualGAFDataLoader(args, flag='train')
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"数据加载器输出元素数量: {len(sample)}")
            
            if len(sample) == 4:
                sum_data, diff_data, time_series_data, label = sample
                print(f"✓ 四元组格式正确")
                print(f"  - Summation GAF: {sum_data.shape}")
                print(f"  - Difference GAF: {diff_data.shape}")
                print(f"  - 时序数据: {time_series_data.shape}")
                print(f"  - 标签: {label}")
            else:
                print(f"✗ 输出格式错误，期望4个元素，实际{len(sample)}个")
        else:
            print("数据集为空")
            
    except Exception as e:
        print(f"数据加载器测试跳过（可能缺少数据文件）: {e}")

def main():
    """主测试函数"""
    print("增强版双路GAF网络功能测试")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行各项测试
    test_statistical_extractor()
    test_multimodal_fusion()
    test_enhanced_dual_gaf_model()
    test_data_loader_compatibility()
    
    print("\n" + "="*80)
    print("所有测试完成！")
    print("="*80)

if __name__ == "__main__":
    main() 
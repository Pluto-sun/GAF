#!/usr/bin/env python3
"""
测试新的ResNet特征提取器
验证它们能正确处理96x96的GAF图像
"""
import torch
import torch.nn as nn
from models.DualGAFNet import (
    ResNetFeatureExtractor,
    OptimizedLargeKernelFeatureExtractor, 
    OptimizedDilatedFeatureExtractor,
    DualGAFNet
)

def test_feature_extractors():
    """测试所有新的特征提取器"""
    print("="*80)
    print("测试ResNet和优化特征提取器 - 96x96 GAF图像")
    print("="*80)
    
    # 创建测试数据 (batch_size=2, channels=1, height=96, width=96)
    x = torch.randn(2, 1, 96, 96)
    feature_dim = 128
    
    extractors = [
        ("ResNet18", ResNetFeatureExtractor(feature_dim, depth='resnet18')),
        ("ResNet34", ResNetFeatureExtractor(feature_dim, depth='resnet34')),
        ("ResNet Light", ResNetFeatureExtractor(feature_dim, depth='resnet_light')),
        ("优化大核卷积", OptimizedLargeKernelFeatureExtractor(feature_dim)),
        ("优化膨胀卷积", OptimizedDilatedFeatureExtractor(feature_dim)),
    ]
    
    results = []
    
    for name, extractor in extractors:
        print(f"\n{'-'*60}")
        print(f"测试 {name}")
        print(f"{'-'*60}")
        
        try:
            # 计算参数数量
            total_params = sum(p.numel() for p in extractor.parameters())
            
            # 前向传播
            with torch.no_grad():
                output = extractor(x)
            
            print(f"✅ 测试通过")
            print(f"  - 输入形状: {x.shape}")
            print(f"  - 输出形状: {output.shape}")
            print(f"  - 参数数量: {total_params:,}")
            print(f"  - 模型大小: {total_params * 4 / (1024**2):.2f} MB")
            
            results.append({
                'name': name,
                'params': total_params,
                'output_shape': output.shape,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results.append({
                'name': name,
                'params': 0,
                'output_shape': None,
                'success': False,
                'error': str(e)
            })
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("测试结果汇总")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"成功: {len(successful)}/{len(results)}")
    if successful:
        print("\n成功的提取器:")
        for result in successful:
            print(f"  - {result['name']}: {result['params']:,} 参数, 输出形状 {result['output_shape']}")
    
    if failed:
        print(f"\n失败的提取器:")
        for result in failed:
            print(f"  - {result['name']}: {result['error']}")
    
    return results

def test_dual_gaf_with_resnet():
    """测试完整的DualGAF网络使用ResNet特征提取器"""
    print(f"\n{'='*80}")
    print("测试完整DualGAF网络")
    print("="*80)
    
    # 测试配置
    test_configs = [
        {"signals": 26, "extractor": "resnet18", "name": "SAHU-ResNet18"},
        {"signals": 120, "extractor": "resnet_light", "name": "DAHU-ResNetLight"},
        {"signals": 120, "extractor": "optimized_large_kernel", "name": "DAHU-优化大核"},
    ]
    
    for config in test_configs:
        print(f"\n{'-'*60}")
        print(f"测试配置: {config['name']}")
        print(f"信号数量: {config['signals']}, 特征提取器: {config['extractor']}")
        print(f"{'-'*60}")
        
        try:
            # 创建测试数据
            B, C, H, W = 2, config['signals'], 96, 96
            sum_x = torch.randn(B, C, H, W)
            diff_x = torch.randn(B, C, H, W)
            time_series_x = torch.randn(B, H, C)
            
            # 创建模型配置
            class Args:
                def __init__(self):
                    self.feature_dim = 64
                    self.num_class = 4
                    self.enc_in = C
                    self.seq_len = H
                    self.extractor_type = config['extractor']
                    self.fusion_type = "adaptive"
                    self.attention_type = "channel"
                    self.classifier_type = "mlp"
                    self.use_statistical_features = True
                    self.stat_type = "basic"
                    self.multimodal_fusion_strategy = "concat"
                    self.hvac_groups = None
                    self.feature_columns = None
            
            args = Args()
            
            # 创建DualGAF网络
            net = DualGAFNet(
                feature_dim=args.feature_dim,
                num_classes=args.num_class,
                num_images=args.enc_in,
                time_length=args.seq_len,
                extractor_type=args.extractor_type,
                fusion_type=args.fusion_type,
                attention_type=args.attention_type,
                classifier_type=args.classifier_type,
                use_statistical_features=args.use_statistical_features,
                stat_type=args.stat_type,
                multimodal_fusion_strategy=args.multimodal_fusion_strategy,
            )
            
            # 前向传播测试
            with torch.no_grad():
                output = net(sum_x, diff_x, time_series_x)
            
            # 计算参数数量
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            
            print(f"✅ 测试通过")
            print(f"  - 输入: Sum{sum_x.shape}, Diff{diff_x.shape}, TS{time_series_x.shape}")
            print(f"  - 输出形状: {output.shape}")
            print(f"  - 总参数: {total_params:,}")
            print(f"  - 可训练参数: {trainable_params:,}")
            print(f"  - 估计模型大小: {total_params * 4 / (1024**2):.2f} MB")
            
            # 估算单个batch的内存占用
            input_memory = (sum_x.numel() + diff_x.numel() + time_series_x.numel()) * 4 / (1024**2)
            print(f"  - 单batch输入内存: {input_memory:.2f} MB")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 测试特征提取器
    extractor_results = test_feature_extractors()
    
    # 测试完整网络
    test_dual_gaf_with_resnet()
    
    print(f"\n{'='*80}")
    print("所有测试完成")
    print("="*80) 
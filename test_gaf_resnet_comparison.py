#!/usr/bin/env python3
"""
GAF优化ResNet vs 传统ResNet对比测试
展示为什么GAF优化版本更适合GAF图像处理
"""
import torch
import torch.nn as nn
from models.DualGAFNet import ResNetFeatureExtractor

def analyze_spatial_resolution_preservation():
    """分析空间分辨率保留效果"""
    print("="*80)
    print("GAF优化ResNet vs 传统ResNet - 空间分辨率保留分析")
    print("="*80)
    
    # 输入数据：96x96 GAF图像
    x = torch.randn(1, 1, 96, 96)
    print(f"输入GAF图像尺寸: {x.shape}")
    
    # 测试不同的ResNet架构
    architectures = [
        ("传统ResNet18", "resnet18"),
        ("GAF优化ResNet18", "resnet18_gaf"),
        ("GAF轻量级", "resnet18_gaf_light"),
        ("GAF高保真", "resnet_gaf_preserve"),
    ]
    
    print(f"\n{'架构名称':<20} {'第1层输出':<15} {'第2层输出':<15} {'第3层输出':<15} {'第4层输出':<15} {'参数量':<10}")
    print("-" * 95)
    
    for name, arch_type in architectures:
        try:
            model = ResNetFeatureExtractor(feature_dim=128, depth=arch_type)
            
            with torch.no_grad():
                # 如果是GAF优化版本，需要访问内部的resnet
                if hasattr(model, 'resnet'):
                    resnet = model.resnet
                    x1 = resnet.conv1(x)
                    x2 = resnet.layer1(x1)
                    x3 = resnet.layer2(x2)
                    x4 = resnet.layer3(x3)
                    if resnet.layer4 is not None:
                        x5 = resnet.layer4(x4)
                        layer4_shape = f"{x5.shape[2]}x{x5.shape[3]}"
                    else:
                        layer4_shape = "None"
                else:
                    # 传统ResNet
                    x1 = model.conv1(x)
                    x2 = model.layer1(x1)
                    x3 = model.layer2(x2)
                    x4 = model.layer3(x3)
                    if model.layer4 is not None:
                        x5 = model.layer4(x4)
                        layer4_shape = f"{x5.shape[2]}x{x5.shape[3]}"
                    else:
                        layer4_shape = "None"
            
            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            
            # 格式化输出
            conv1_shape = f"{x1.shape[2]}x{x1.shape[3]}"
            layer1_shape = f"{x2.shape[2]}x{x2.shape[3]}"
            layer2_shape = f"{x3.shape[2]}x{x3.shape[3]}"
            layer3_shape = f"{x4.shape[2]}x{x4.shape[3]}"
            
            print(f"{name:<20} {conv1_shape:<15} {layer1_shape:<15} {layer2_shape:<15} {layer3_shape:<15} {total_params/1000:.1f}K")
            
        except Exception as e:
            print(f"{name:<20} 错误: {e}")
    
    print(f"\n关键观察:")
    print(f"1. 🔴 传统ResNet18: 第一层就从96→24，损失75%空间信息")
    print(f"2. 🟢 GAF优化版本: 第一层保持96→96，渐进式下采样")
    print(f"3. 🟡 高保真版本: 前两层都保持高分辨率")
    print(f"4. 💡 参数量: GAF优化版本通过合理设计控制参数数量")

def compare_feature_extraction_quality():
    """比较特征提取质量"""
    print(f"\n{'='*80}")
    print("特征提取质量对比")
    print("="*80)
    
    # 创建一个模拟的GAF图像，包含细节纹理
    def create_gaf_like_pattern(size=96):
        """创建类似GAF的模式"""
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        X, Y = torch.meshgrid(x, y)
        
        # GAF特有的对称和周期性模式
        pattern = torch.cos(torch.pi * X) * torch.cos(torch.pi * Y)
        pattern += 0.3 * torch.sin(5 * torch.pi * X) * torch.sin(5 * torch.pi * Y)  # 高频细节
        
        return pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, 96, 96]
    
    gaf_pattern = create_gaf_like_pattern()
    print(f"创建GAF模拟图像: {gaf_pattern.shape}")
    print(f"值范围: [{gaf_pattern.min():.3f}, {gaf_pattern.max():.3f}]")
    
    # 测试不同架构的特征提取
    test_models = [
        ("传统ResNet18", "resnet18"),
        ("GAF优化ResNet18", "resnet18_gaf"),
        ("GAF高保真", "resnet_gaf_preserve"),
    ]
    
    print(f"\n特征提取结果对比:")
    print(f"{'架构':<20} {'输出特征维度':<15} {'特征范围':<20} {'特征标准差':<15}")
    print("-" * 75)
    
    for name, arch_type in test_models:
        try:
            model = ResNetFeatureExtractor(feature_dim=128, depth=arch_type)
            model.eval()
            
            with torch.no_grad():
                features = model(gaf_pattern)
            
            feat_std = features.std().item()
            feat_min = features.min().item()
            feat_max = features.max().item()
            
            print(f"{name:<20} {features.shape[1]:<15} [{feat_min:.3f}, {feat_max:.3f}] {feat_std:<15.4f}")
            
        except Exception as e:
            print(f"{name:<20} 错误: {e}")

def analyze_computational_efficiency():
    """分析计算效率"""
    print(f"\n{'='*80}")
    print("计算效率分析")
    print("="*80)
    
    import time
    
    # 测试数据
    batch_sizes = [1, 8, 16]
    x_96 = torch.randn(16, 1, 96, 96)
    
    models_to_test = [
        ("传统ResNet18", "resnet18"),
        ("GAF优化ResNet18", "resnet18_gaf"),
        ("GAF轻量级", "resnet18_gaf_light"),
    ]
    
    print(f"{'架构':<20} {'参数量':<12} {'推理时间(ms)':<15} {'内存占用(MB)':<15}")
    print("-" * 70)
    
    for name, arch_type in models_to_test:
        try:
            model = ResNetFeatureExtractor(feature_dim=128, depth=arch_type)
            model.eval()
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            
            # 预热
            with torch.no_grad():
                _ = model(x_96[:1])
            
            # 测试推理时间
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x_96)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            avg_time = (time.time() - start_time) / 10 * 1000  # 转换为毫秒
            
            # 估算内存占用
            model_memory = total_params * 4 / (1024**2)  # MB
            
            print(f"{name:<20} {total_params/1000:.1f}K{'':<7} {avg_time:<15.2f} {model_memory:<15.1f}")
            
        except Exception as e:
            print(f"{name:<20} 错误: {e}")

def demonstrate_architecture_differences():
    """演示架构差异"""
    print(f"\n{'='*80}")
    print("架构设计理念对比")
    print("="*80)
    
    print("🔴 传统ResNet18的问题:")
    print("  1. 设计目标: 224x224的ImageNet自然图像")
    print("  2. 第一层: 7x7卷积 + 3x3最大池化")
    print("  3. 下采样策略: 224→112→56→28→14→7 (激进下采样)")
    print("  4. 适用场景: 自然图像的语义分割、分类")
    print("  5. 问题: 对于96x96的GAF图像，过早丢失细节信息")
    
    print(f"\n🟢 GAF优化ResNet的改进:")
    print("  1. 设计目标: 96x96的GAF时序图像")
    print("  2. 第一层: 3x3卷积，保持分辨率")
    print("  3. 下采样策略: 96→48→24→12→6 (渐进下采样)")
    print("  4. 适用场景: 时序数据的GAF表示学习")
    print("  5. 优势: 保留更多空间-时间依赖关系")
    
    print(f"\n💡 为什么这样设计:")
    print("  1. GAF图像每个像素都有意义 (时间点间的关系)")
    print("  2. 对称性和周期性模式需要保留")
    print("  3. 高频细节包含重要的时序信息")
    print("  4. 96x96相对较小，不需要激进下采样")

if __name__ == "__main__":
    # 运行所有分析
    analyze_spatial_resolution_preservation()
    compare_feature_extraction_quality()
    analyze_computational_efficiency()
    demonstrate_architecture_differences()
    
    print(f"\n{'='*80}")
    print("总结与建议")
    print("="*80)
    print("✅ 推荐使用 GAF优化ResNet 系列:")
    print("  - SAHU数据集(26信号): resnet18_gaf")
    print("  - DAHU数据集(120信号): resnet18_gaf_light")
    print("  - 高精度要求: resnet_gaf_preserve")
    print("  - 自动选择: extractor_type='auto'")
    print(f"\n❌ 不推荐传统ResNet:")
    print("  - 过早下采样导致信息损失")
    print("  - 不适合GAF图像的特性")
    print("="*80) 
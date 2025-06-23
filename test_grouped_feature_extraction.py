#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证分组特征提取功能

使用示例:
python test_grouped_feature_extraction.py
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_grouped_feature_extraction():
    """测试分组特征提取功能"""
    print("=" * 80)
    print("测试HVAC分组特征提取网络")
    print("=" * 80)
    
    try:
        from models.MultiImageFeatureNet import Model
        
        # 设置测试参数
        B, C, H, W = 2, 10, 32, 32
        x = torch.randn(B, C, H, W)
        
        # 模拟HVAC特征列名
        feature_columns = [
            'SA_TEMP', 'OA_TEMP', 'MA_TEMP',     # 温度组
            'OA_CFM', 'RA_CFM',                  # 流量组
            'SA_SP', 'SA_SPSPT',                 # 设定点组
            'CHW_VLV_POS', 'HW_VLV_POS',        # 阀门组
            'UNKNOWN_SIGNAL'                     # 未知信号（测试默认组）
        ]
        
        # 创建配置
        configs = type("Config", (), {
            "feature_dim": 32, 
            "num_class": 5, 
            "enc_in": C,
            "feature_columns": feature_columns,
            "hvac_groups": [
                ['SA_TEMP', 'OA_TEMP', 'MA_TEMP', 'RA_TEMP'],  # 温度组
                ['OA_CFM', 'RA_CFM', 'SA_CFM'],                 # 流量组
                ['SA_SP', 'SA_SPSPT'],                          # 设定点组
                ['CHW_VLV_POS', 'HW_VLV_POS'],                  # 阀门组
            ]
        })()
        
        print(f"1. 测试参数:")
        print(f"   输入维度: {x.shape}")
        print(f"   特征列数: {len(feature_columns)}")
        print(f"   信号组数: {len(configs.hvac_groups)}")
        print(f"   类别数: {configs.num_class}")
        
        print(f"\n2. 特征列名:")
        for i, col in enumerate(feature_columns):
            print(f"   通道{i}: {col}")
        
        print(f"\n3. 信号分组:")
        for i, group in enumerate(configs.hvac_groups):
            print(f"   组{i}: {group}")
        
        # 创建模型
        print(f"\n4. 创建模型...")
        model = Model(configs)
        
        print(f"\n5. 通道分组映射:")
        for channel_idx, group_idx in model.model.channel_to_group.items():
            col_name = feature_columns[channel_idx] if channel_idx < len(feature_columns) else "Unknown"
            group_name = f"group_{group_idx}" if group_idx >= 0 else "default"
            print(f"   通道{channel_idx} ({col_name}) -> {group_name}")
        
        print(f"\n6. 特征提取器:")
        for name, extractor in model.model.feature_extractors.items():
            print(f"   {name}: {type(extractor).__name__}")
        
        # 测试forward过程
        print(f"\n7. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"   输出维度: {output.shape}")
        print(f"   输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 测试梯度传播
        print(f"\n8. 测试梯度传播...")
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        gradients_exist = any(p.grad is not None for p in model.parameters())
        print(f"   梯度是否存在: {'是' if gradients_exist else '否'}")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n9. 模型统计:")
        print(f"   总参数数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   特征提取器数量: {len(model.model.feature_extractors)}")
        
        print(f"\n✓ 分组特征提取功能测试通过！")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性（不提供分组信息时的默认行为）"""
    print("\n" + "=" * 80)
    print("测试向后兼容性")
    print("=" * 80)
    
    try:
        from models.MultiImageFeatureNet import Model
        
        # 不提供分组信息的配置
        configs = type("Config", (), {
            "num_class": 4, 
            "enc_in": 10,
        })()
        
        print(f"1. 创建没有分组信息的模型...")
        model = Model(configs)
        
        print(f"2. 模型默认行为:")
        print(f"   使用默认HVAC分组: {len(model.model.hvac_groups)} 组")
        print(f"   特征提取器数量: {len(model.model.feature_extractors)}")
        
        # 测试forward
        x = torch.randn(2, 10, 32, 32)
        output = model(x)
        print(f"   输出维度: {output.shape}")
        
        print(f"\n✓ 向后兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 向后兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试分组特征提取功能...")
    
    success1 = test_grouped_feature_extraction()
    success2 = test_backward_compatibility()
    
    if success1 and success2:
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！分组特征提取功能正常工作。")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 部分测试失败，请检查代码。")
        print("=" * 80) 
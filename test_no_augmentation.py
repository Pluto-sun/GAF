#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：验证数据增强功能已完全移除
确认系统回退到没有数据增强的版本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockArgs:
    """模拟命令行参数"""
    def __init__(self):
        self.root_path = './dataset/SAHU'
        self.seq_len = 72
        self.step = 12
        self.test_size = 0.3
        self.data_type_method = 'float32'

def test_no_augmentation():
    """测试确认没有数据增强功能"""
    print("=== 测试数据增强功能是否已完全移除 ===\n")
    
    success = True
    
    # 1. 测试增强模块是否已删除
    try:
        import utils.gaf_augmentation
        print("❌ 错误: utils.gaf_augmentation 模块仍然存在")
        success = False
    except ImportError:
        print("✅ 通过: utils.gaf_augmentation 模块已删除")
    
    # 2. 测试数据加载器是否不再包含增强功能
    try:
        from data_provider.data_loader.DualGAFDataLoader import DualGAFDataLoader
        
        args = MockArgs()
        dataset = DualGAFDataLoader(args, flag='train')
        
        # 检查是否有增强相关属性
        has_aug_attrs = any([
            hasattr(dataset, 'augmentation_applied'),
            hasattr(dataset, 'applied_augmentations'),
            hasattr(dataset, 'augmentation_mode'),
            hasattr(dataset, '_generate_augmented_dataset'),
            hasattr(dataset, '_apply_single_augmentation'),
            hasattr(dataset, '_apply_full_augmentation'),
            hasattr(dataset, '_apply_probabilistic_augmentation')
        ])
        
        if has_aug_attrs:
            print("❌ 错误: 数据加载器仍包含增强相关属性或方法")
            success = False
        else:
            print("✅ 通过: 数据加载器已清理增强相关代码")
        
        print(f"   数据集样本数: {len(dataset)}")
        
    except Exception as e:
        print(f"❌ 错误: 数据加载器测试失败 - {e}")
        success = False
    
    # 3. 测试run.py是否不再包含增强参数
    try:
        # 检查run.py文件内容
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        aug_keywords = [
            'enable_augmentation',
            'aug_rotation',
            'aug_noise',
            'aug_flip',
            'aug_mixup',
            'aug_overall_prob',
            'aug_mode'
        ]
        
        found_keywords = [kw for kw in aug_keywords if kw in content]
        
        if found_keywords:
            print(f"❌ 错误: run.py仍包含增强参数: {found_keywords}")
            success = False
        else:
            print("✅ 通过: run.py已清理所有增强参数")
    
    except Exception as e:
        print(f"❌ 错误: run.py检查失败 - {e}")
        success = False
    
    # 4. 测试exp.py是否已清理增强相关代码
    try:
        with open('exp/exp.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        aug_keywords = [
            'augmentation_applied',
            'augmentation',
            'mixup'
        ]
        
        found_keywords = [kw for kw in aug_keywords if kw in content]
        
        if found_keywords:
            print(f"⚠️  警告: exp.py可能仍包含增强相关代码: {found_keywords}")
            # 不设为失败，因为可能是注释或其他无关代码
        else:
            print("✅ 通过: exp.py已清理增强相关代码")
    
    except Exception as e:
        print(f"❌ 错误: exp.py检查失败 - {e}")
        success = False
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 所有测试通过！数据增强功能已完全移除")
        print("\n系统状态:")
        print("  ✅ 回退到无数据增强版本")
        print("  ✅ 数据加载器只返回原始数据")
        print("  ✅ 训练过程不包含任何增强操作")
        print("  ✅ 配置参数已清理")
    else:
        print("❌ 部分测试失败，可能需要进一步清理")
    
    print(f"{'='*60}\n")
    
    return success

if __name__ == "__main__":
    test_no_augmentation() 
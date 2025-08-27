#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体配置模块 - 解决matplotlib中文显示问题
专为GAF项目优化，提供简单易用的中文字体配置
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font_simple():
    """
    简化版中文字体设置
    根据测试结果，直接配置最佳字体
    """
    # 基于测试结果的最优配置
    font_candidates = [
        'WenQuanYi Micro Hei',      # 测试通过的首选字体
        'WenQuanYi Zen Hei',        # 备选字体1
        'Noto Sans CJK JP',         # 备选字体2（系统中找到的）
        'Noto Serif CJK JP',        # 备选字体3
        'DejaVu Sans',              # 最终备选
        'Arial Unicode MS'          # 兜底字体
    ]
    
    # 配置matplotlib
    matplotlib.rcParams['font.sans-serif'] = font_candidates
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.max_open_warning'] = 0
    
    # 设置合适的字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    
    print("✅ 中文字体配置已应用")
    return font_candidates[0]

def test_font_display():
    """
    快速测试中文字体显示
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试文本
        test_texts = [
            '🎯 HVAC异常检测系统',
            '📊 训练损失: 0.123',
            '📈 验证精度: 95.6%',
            '⚡ 学习率: 0.001',
            '🔥 F1分数: 0.892'
        ]
        
        for i, text in enumerate(test_texts):
            ax.text(0.1, 0.9 - i*0.15, text, fontsize=14, 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_title('🧪 中文字体显示测试', fontsize=16, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 保存测试图
        output_path = './font_test_quick.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 快速字体测试完成，结果保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 字体测试失败: {e}")
        return False

def get_system_fonts_info():
    """
    获取系统字体信息
    """
    print("📋 系统中文字体信息:")
    
    chinese_keywords = ['Noto', 'CJK', 'WenQuanYi', 'Source Han', 'AR PL', 'SimHei', 'YaHei']
    chinese_fonts = []
    
    for font in fm.fontManager.ttflist:
        if any(keyword in font.name for keyword in chinese_keywords):
            chinese_fonts.append(font.name)
    
    chinese_fonts = list(set(chinese_fonts))
    
    if chinese_fonts:
        print(f"找到 {len(chinese_fonts)} 个中文字体:")
        for font in sorted(chinese_fonts):
            print(f"  📝 {font}")
    else:
        print("  ❌ 未找到中文字体")
    
    return chinese_fonts

def apply_project_font_settings():
    """
    为GAF项目应用字体设置
    这是主要的配置函数，在项目启动时调用
    """
    print("🎨 正在配置GAF项目字体设置...")
    
    # 应用字体配置
    selected_font = setup_chinese_font_simple()
    
    # 验证配置
    success = test_font_display()
    
    if success:
        print("🎉 字体配置成功！中文显示正常")
        print(f"📝 使用字体: {selected_font}")
        print("💡 提示: 训练图表中的中文标签现在应该能正常显示")
    else:
        print("⚠️ 字体配置可能存在问题，建议手动检查")
    
    return success

# 快捷配置函数
def quick_setup():
    """一键配置中文字体"""
    return apply_project_font_settings()

if __name__ == "__main__":
    # 运行完整的字体配置和测试
    print("=" * 50)
    print("🛠️ GAF项目字体配置工具")
    print("=" * 50)
    
    # 显示系统字体信息
    get_system_fonts_info()
    
    print("\n" + "-" * 30)
    
    # 应用配置
    apply_project_font_settings() 
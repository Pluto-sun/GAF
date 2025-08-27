#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类错误样本分析工具
用于分析和可视化模型评估过程中保存的错误分类样本的时序数据

使用方法:
python utils/analyze_misclassified_samples.py --result_dir ./result/0821_2024_DualGAF_DDAHU_DualGAFNet_基准版_sl96_step96_fd128_extractor-dilated_gaf-adaptive_attention-channel_classifier-mlp_stat-basic_fusion-concat
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MisclassifiedSampleAnalyzer:
    """分类错误样本分析器"""
    
    def __init__(self, result_dir):
        """
        初始化分析器
        
        Args:
            result_dir (str): 结果目录路径，包含misclassified_samples子目录
        """
        self.result_dir = Path(result_dir)
        self.misclassified_dir = self.result_dir / 'misclassified_samples'
        
        if not self.misclassified_dir.exists():
            raise ValueError(f"错误样本目录不存在: {self.misclassified_dir}")
        
        # 加载错误样本概要信息
        self.summary_file = self.misclassified_dir / 'misclassified_samples_summary.csv'
        self.error_stats_file = self.misclassified_dir / 'error_type_statistics.csv'
        
        if self.summary_file.exists():
            self.summary_df = pd.read_csv(self.summary_file)
            print(f"✅ 加载错误样本概要: {len(self.summary_df)} 个错误样本")
        else:
            self.summary_df = None
            print("⚠️ 未找到错误样本概要文件")
        
        if self.error_stats_file.exists():
            self.error_stats_df = pd.read_csv(self.error_stats_file)
            print(f"✅ 加载错误类型统计: {len(self.error_stats_df)} 种错误类型")
        else:
            self.error_stats_df = None
            print("⚠️ 未找到错误类型统计文件")
    
    def show_summary(self):
        """显示错误样本总体概况"""
        if self.summary_df is None:
            print("❌ 无法显示概况：缺少概要数据")
            return
        
        print("\n" + "="*80)
        print("📊 分类错误样本分析概况")
        print("="*80)
        
        # 基本统计
        total_errors = len(self.summary_df)
        has_time_series = self.summary_df['Has_Time_Series'].sum()
        
        print(f"总错误样本数: {total_errors}")
        print(f"包含时序数据的错误样本: {has_time_series}")
        print(f"时序数据覆盖率: {has_time_series/total_errors*100:.1f}%")
        
        # 数据类型分布
        print(f"\n📋 数据类型分布:")
        data_type_counts = self.summary_df['Data_Type'].value_counts()
        for data_type, count in data_type_counts.items():
            print(f"  {data_type}: {count} 个样本 ({count/total_errors*100:.1f}%)")
        
        # 错误类型分布
        if self.error_stats_df is not None:
            print(f"\n🎯 主要错误类型 (前10名):")
            top_errors = self.error_stats_df.head(10)
            for _, row in top_errors.iterrows():
                print(f"  {row['True_Label_Name']} → {row['Predicted_Label_Name']}: {row['Count']} 个样本")
        
        # 真实标签分布
        print(f"\n📈 真实标签错误分布:")
        true_label_errors = self.summary_df['True_Label_Name'].value_counts()
        for label, count in true_label_errors.items():
            print(f"  {label}: {count} 个错误样本")
        
        # 预测标签分布
        print(f"\n🎲 预测标签错误分布:")
        pred_label_errors = self.summary_df['Predicted_Label_Name'].value_counts()
        for label, count in pred_label_errors.items():
            print(f"  {label}: {count} 个错误预测")
    
    def plot_error_distribution(self, save_plot=True):
        """绘制错误分布图"""
        if self.summary_df is None or self.error_stats_df is None:
            print("❌ 无法绘制分布图：缺少统计数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 错误类型热力图
        pivot_table = self.error_stats_df.pivot(
            index='True_Label_Name', 
            columns='Predicted_Label_Name', 
            values='Count'
        ).fillna(0)
        
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Reds', ax=axes[0,0])
        axes[0,0].set_title('错误分类热力图')
        axes[0,0].set_xlabel('预测标签')
        axes[0,0].set_ylabel('真实标签')
        
        # 2. 真实标签错误分布
        true_label_errors = self.summary_df['True_Label_Name'].value_counts()
        true_label_errors.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('各类别的错误样本数量')
        axes[0,1].set_xlabel('真实标签')
        axes[0,1].set_ylabel('错误样本数')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 预测置信度分布
        if 'Prob_' in self.summary_df.columns[0] or any('Prob_' in col for col in self.summary_df.columns):
            prob_columns = [col for col in self.summary_df.columns if col.startswith('Prob_')]
            if prob_columns:
                # 计算最高预测概率
                prob_values = self.summary_df[prob_columns].values
                max_probs = np.max(prob_values, axis=1)
                
                axes[1,0].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
                axes[1,0].set_title('错误预测的最高置信度分布')
                axes[1,0].set_xlabel('最高预测概率')
                axes[1,0].set_ylabel('样本数量')
                axes[1,0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='0.5阈值')
                axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, '无预测概率数据', ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('预测概率分布（无数据）')
        
        # 4. 数据类型分布饼图
        data_type_counts = self.summary_df['Data_Type'].value_counts()
        axes[1,1].pie(data_type_counts.values, labels=data_type_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('数据类型分布')
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.result_dir / 'error_distribution_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 错误分布图已保存: {plot_file}")
        
        plt.show()
    
    def analyze_time_series_sample(self, sample_index=None, true_label=None, pred_label=None):
        """分析特定错误样本的时序数据"""
        if self.summary_df is None:
            print("❌ 无法分析样本：缺少概要数据")
            return
        
        # 查找样本
        if sample_index is not None:
            sample_row = self.summary_df[self.summary_df['Sample_Index'] == sample_index]
        elif true_label is not None and pred_label is not None:
            sample_row = self.summary_df[
                (self.summary_df['True_Label_Name'] == true_label) & 
                (self.summary_df['Predicted_Label_Name'] == pred_label)
            ]
        else:
            print("❌ 请指定sample_index或(true_label, pred_label)")
            return
        
        if sample_row.empty:
            print("❌ 未找到匹配的错误样本")
            return
        
        sample_row = sample_row.iloc[0]  # 取第一个匹配的样本
        
        if not sample_row['Has_Time_Series']:
            print(f"❌ 样本 {sample_row['Sample_Index']} 没有时序数据")
            return
        
        # 加载时序数据
        time_series_file = self.misclassified_dir / f"sample_{sample_row['Sample_Index']}_true_{sample_row['True_Label']}_pred_{sample_row['Predicted_Label']}.csv"
        
        if not time_series_file.exists():
            print(f"❌ 时序数据文件不存在: {time_series_file}")
            return
        
        # 读取时序数据（跳过注释行）
        time_series_df = pd.read_csv(time_series_file, comment='#', index_col=0)
        
        print(f"\n🔍 样本 {sample_row['Sample_Index']} 时序数据分析")
        print(f"真实标签: {sample_row['True_Label_Name']} ({sample_row['True_Label']})")
        print(f"预测标签: {sample_row['Predicted_Label_Name']} ({sample_row['Predicted_Label']})")
        print(f"数据类型: {sample_row['Data_Type']}")
        print(f"时序数据形状: {time_series_df.shape}")
        
        # 绘制时序数据
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 所有信号的时序图
        if time_series_df.shape[1] <= 10:  # 如果信号数量不多，全部显示
            for col in time_series_df.columns:
                axes[0].plot(time_series_df.index, time_series_df[col], label=col, alpha=0.7)
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:  # 如果信号太多，只显示前几个
            for i, col in enumerate(time_series_df.columns[:6]):
                axes[0].plot(time_series_df.index, time_series_df[col], label=col, alpha=0.7)
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].set_title(f'时序数据 (显示前6个信号，共{time_series_df.shape[1]}个)')
        
        axes[0].set_title(f'错误样本 {sample_row["Sample_Index"]} - {sample_row["True_Label_Name"]} → {sample_row["Predicted_Label_Name"]}')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('信号值')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 统计特征热力图
        if time_series_df.shape[1] > 1:
            # 计算各信号的统计特征
            stats = pd.DataFrame({
                'Mean': time_series_df.mean(),
                'Std': time_series_df.std(),
                'Min': time_series_df.min(),
                'Max': time_series_df.max(),
                'Range': time_series_df.max() - time_series_df.min()
            })
            
            # 标准化以便可视化
            stats_normalized = (stats - stats.min()) / (stats.max() - stats.min())
            
            sns.heatmap(stats_normalized.T, annot=True, fmt='.2f', cmap='viridis', ax=axes[1])
            axes[1].set_title('信号统计特征 (标准化)')
            axes[1].set_xlabel('信号')
            axes[1].set_ylabel('统计特征')
        else:
            axes[1].text(0.5, 0.5, '单一信号，无法绘制统计特征热力图', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        # 保存单个样本分析图
        sample_plot_file = self.result_dir / f'sample_{sample_row["Sample_Index"]}_analysis.png'
        plt.savefig(sample_plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 样本分析图已保存: {sample_plot_file}")
        
        plt.show()
        
        return time_series_df
    
    def get_available_samples(self):
        """获取可用的错误样本列表"""
        if self.summary_df is None:
            print("❌ 无概要数据")
            return None
        
        # 只返回有时序数据的样本
        samples_with_ts = self.summary_df[self.summary_df['Has_Time_Series'] == True]
        
        print(f"\n📋 可用的错误样本 (共 {len(samples_with_ts)} 个):")
        print("样本索引 | 真实标签 → 预测标签 | 数据类型")
        print("-" * 60)
        
        for _, row in samples_with_ts.head(20).iterrows():  # 只显示前20个
            print(f"{row['Sample_Index']:8d} | {row['True_Label_Name']:12s} → {row['Predicted_Label_Name']:12s} | {row['Data_Type']}")
        
        if len(samples_with_ts) > 20:
            print(f"... 还有 {len(samples_with_ts) - 20} 个样本")
        
        return samples_with_ts
    
    def compare_error_patterns(self, error_type1, error_type2, max_samples=5):
        """比较不同错误类型的模式"""
        if self.summary_df is None:
            print("❌ 无概要数据")
            return
        
        # 获取两种错误类型的样本
        type1_samples = self.summary_df[
            (self.summary_df['True_Label_Name'] == error_type1[0]) & 
            (self.summary_df['Predicted_Label_Name'] == error_type1[1]) &
            (self.summary_df['Has_Time_Series'] == True)
        ]
        
        type2_samples = self.summary_df[
            (self.summary_df['True_Label_Name'] == error_type2[0]) & 
            (self.summary_df['Predicted_Label_Name'] == error_type2[1]) &
            (self.summary_df['Has_Time_Series'] == True)
        ]
        
        if type1_samples.empty or type2_samples.empty:
            print(f"❌ 错误类型样本不足: {error_type1} ({len(type1_samples)} 个), {error_type2} ({len(type2_samples)} 个)")
            return
        
        print(f"\n🔍 比较错误模式:")
        print(f"类型1: {error_type1[0]} → {error_type1[1]} ({len(type1_samples)} 个样本)")
        print(f"类型2: {error_type2[0]} → {error_type2[1]} ({len(type2_samples)} 个样本)")
        
        # 随机选择样本进行比较
        selected_type1 = type1_samples.sample(min(max_samples, len(type1_samples)))
        selected_type2 = type2_samples.sample(min(max_samples, len(type2_samples)))
        
        fig, axes = plt.subplots(2, max_samples, figsize=(max_samples*4, 8))
        if max_samples == 1:
            axes = axes.reshape(-1, 1)
        
        # 绘制错误类型1的样本
        for i, (_, sample) in enumerate(selected_type1.iterrows()):
            if i >= max_samples:
                break
            
            time_series_file = self.misclassified_dir / f"sample_{sample['Sample_Index']}_true_{sample['True_Label']}_pred_{sample['Predicted_Label']}.csv"
            if time_series_file.exists():
                time_series_df = pd.read_csv(time_series_file, comment='#', index_col=0)
                
                # 只绘制前几个信号
                for j, col in enumerate(time_series_df.columns[:3]):
                    axes[0, i].plot(time_series_df.index, time_series_df[col], alpha=0.7, label=col)
                
                axes[0, i].set_title(f'样本{sample["Sample_Index"]}\n{error_type1[0]}→{error_type1[1]}')
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].legend()
        
        # 绘制错误类型2的样本
        for i, (_, sample) in enumerate(selected_type2.iterrows()):
            if i >= max_samples:
                break
            
            time_series_file = self.misclassified_dir / f"sample_{sample['Sample_Index']}_true_{sample['True_Label']}_pred_{sample['Predicted_Label']}.csv"
            if time_series_file.exists():
                time_series_df = pd.read_csv(time_series_file, comment='#', index_col=0)
                
                # 只绘制前几个信号
                for j, col in enumerate(time_series_df.columns[:3]):
                    axes[1, i].plot(time_series_df.index, time_series_df[col], alpha=0.7, label=col)
                
                axes[1, i].set_title(f'样本{sample["Sample_Index"]}\n{error_type2[0]}→{error_type2[1]}')
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].legend()
        
        plt.tight_layout()
        
        # 保存比较图
        compare_plot_file = self.result_dir / f'error_pattern_comparison_{error_type1[0]}_{error_type1[1]}_vs_{error_type2[0]}_{error_type2[1]}.png'
        plt.savefig(compare_plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 错误模式比较图已保存: {compare_plot_file}")
        
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析分类错误样本的时序数据')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='结果目录路径')
    parser.add_argument('--action', type=str, default='summary',
                       choices=['summary', 'plot', 'analyze_sample', 'list_samples', 'compare'],
                       help='分析动作')
    parser.add_argument('--sample_index', type=int, default=None,
                       help='要分析的样本索引')
    parser.add_argument('--true_label', type=str, default=None,
                       help='真实标签名称')
    parser.add_argument('--pred_label', type=str, default=None,
                       help='预测标签名称')
    
    args = parser.parse_args()
    
    try:
        analyzer = MisclassifiedSampleAnalyzer(args.result_dir)
        
        if args.action == 'summary':
            analyzer.show_summary()
        
        elif args.action == 'plot':
            analyzer.plot_error_distribution()
        
        elif args.action == 'analyze_sample':
            analyzer.analyze_time_series_sample(args.sample_index, args.true_label, args.pred_label)
        
        elif args.action == 'list_samples':
            analyzer.get_available_samples()
        
        elif args.action == 'compare':
            # 示例：比较最常见的两种错误类型
            if analyzer.error_stats_df is not None and len(analyzer.error_stats_df) >= 2:
                error1 = (analyzer.error_stats_df.iloc[0]['True_Label_Name'], 
                         analyzer.error_stats_df.iloc[0]['Predicted_Label_Name'])
                error2 = (analyzer.error_stats_df.iloc[1]['True_Label_Name'], 
                         analyzer.error_stats_df.iloc[1]['Predicted_Label_Name'])
                analyzer.compare_error_patterns(error1, error2)
            else:
                print("❌ 错误类型不足，无法进行比较")
    
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

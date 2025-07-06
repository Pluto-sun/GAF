#!/usr/bin/env python3
"""
快速块大小优化测试脚本
用于验证块大小优化功能的快速测试版本（样本数较少）

测试样本数: 1000 (相比完整版的5000)
功能与完整版相同，但测试规模较小，用于快速验证
"""

import os
import sys
import time
import numpy as np
import multiprocessing as mp
import psutil
import gc

# 添加项目路径
sys.path.append('.')

# 导入完整版测试器
from test_chunk_size_optimization import ChunkSizeOptimizer


class QuickChunkSizeOptimizer(ChunkSizeOptimizer):
    """快速块大小优化测试器（继承完整版，只改变样本数和测试配置）"""
    
    def __init__(self, test_samples=1000):
        """使用较少的样本数进行快速测试"""
        print("🚀 快速块大小优化测试器")
        print("=" * 40)
        
        # 调用父类初始化，但使用较少的样本
        super().__init__(test_samples)
        
        # 简化测试配置，减少测试时间
        self.chunk_sizes = [50, 100, 200, 500]  # 减少测试的块大小数量
        self.n_jobs_list = [4, 8]  # 只测试两种进程数配置
        
        print("📝 快速测试配置:")
        print(f"   样本数: {self.test_samples}")
        print(f"   测试块大小: {self.chunk_sizes}")
        print(f"   测试进程数: {self.n_jobs_list}")
        print()
    
    def test_gaf_chunk_sizes_quick(self):
        """快速GAF块大小测试（减少测试组合）"""
        print("=" * 40)
        print("🧮 快速GAF块大小测试")
        print("=" * 40)
        
        baseline_time = self.baseline_results['gaf_summation_time']
        
        # 只测试一种进程数配置以节省时间
        n_jobs = min(8, self.cpu_count)
        print(f"\n--- 测试进程数: {n_jobs} ---")
        
        for chunk_size in self.chunk_sizes:
            if chunk_size > self.test_samples // 2:
                continue
                
            print(f"\n🔄 测试块大小: {chunk_size}")
            
            # 测试标准多进程
            print(f"   📊 测试标准多进程...")
            start_time = time.time()
            result_standard = self._test_standard_multiprocess_gaf(
                self.test_data, "summation", n_jobs, chunk_size
            )
            standard_time = time.time() - start_time
            speedup_standard = baseline_time / standard_time if standard_time > 0 else 0
            
            # 测试共享内存模式
            print(f"   🚀 测试共享内存...")
            start_time = time.time()
            result_shared = self._test_shared_memory_gaf(
                self.test_data, "summation", n_jobs, chunk_size
            )
            shared_time = time.time() - start_time
            speedup_shared = baseline_time / shared_time if shared_time > 0 else 0
            
            # 验证结果一致性 (快速采样检查)
            consistency_check = self._fast_consistency_check(result_standard, result_shared)
            
            # 记录结果
            result_entry = {
                'test_type': 'gaf_generation',
                'n_jobs': n_jobs,
                'chunk_size': chunk_size,
                'standard_time': standard_time,
                'shared_time': shared_time,
                'baseline_time': baseline_time,
                'standard_speedup': speedup_standard,
                'shared_speedup': speedup_shared,
                'consistency_check': consistency_check,
                'samples_per_chunk': chunk_size,
                'total_chunks': max(1, self.test_samples // chunk_size)
            }
            self.gaf_results.append(result_entry)
            
            print(f"   📊 标准多进程: {standard_time:.2f}s (加速: {speedup_standard:.2f}x)")
            print(f"   🚀 共享内存: {shared_time:.2f}s (加速: {speedup_shared:.2f}x)")
            print(f"   ✅ 结果一致性: {consistency_check}")
            
            # 清理内存
            del result_standard, result_shared
            gc.collect()
        
        return self.gaf_results
    
    def test_conversion_chunk_sizes_quick(self):
        """快速数据转换块大小测试"""
        print("=" * 40)
        print("🔄 快速数据转换块大小测试")
        print("=" * 40)
        
        # 生成GAF数据用于转换测试
        print("🔄 生成GAF数据...")
        gaf_data = self._single_thread_gaf(self.test_data, "summation")
        print(f"📊 GAF数据大小: {gaf_data.nbytes / 1024**3:.2f}GB")
        
        baseline_time = self.baseline_results['conversion_uint8_time']
        
        # 只测试一种线程数配置
        n_jobs = min(8, self.cpu_count)
        print(f"\n--- 测试线程数: {n_jobs} ---")
        
        for chunk_size in self.chunk_sizes:
            if chunk_size > self.test_samples // 2:
                continue
                
            print(f"\n🔄 测试块大小: {chunk_size}")
            
            # 测试标准多线程转换
            print(f"   📊 测试标准多线程转换...")
            start_time = time.time()
            result_standard = self._test_standard_multithread_conversion(
                gaf_data, np.uint8, 255, n_jobs, chunk_size
            )
            standard_time = time.time() - start_time
            speedup_standard = baseline_time / standard_time if standard_time > 0 else 0
            
            # 测试共享内存转换
            print(f"   🚀 测试共享内存转换...")
            start_time = time.time()
            result_shared = self._test_shared_memory_conversion(
                gaf_data, np.uint8, 255, n_jobs, chunk_size
            )
            shared_time = time.time() - start_time
            speedup_shared = baseline_time / shared_time if shared_time > 0 else 0
            
            # 验证结果一致性 (快速采样检查)
            consistency_check = self._fast_consistency_check(result_standard, result_shared)
            
            # 记录结果
            result_entry = {
                'test_type': 'data_conversion',
                'n_jobs': n_jobs,
                'chunk_size': chunk_size,
                'standard_time': standard_time,
                'shared_time': shared_time,
                'baseline_time': baseline_time,
                'standard_speedup': speedup_standard,
                'shared_speedup': speedup_shared,
                'consistency_check': consistency_check,
                'samples_per_chunk': chunk_size,
                'total_chunks': max(1, self.test_samples // chunk_size)
            }
            self.conversion_results.append(result_entry)
            
            print(f"   📊 标准多线程: {standard_time:.2f}s (加速: {speedup_standard:.2f}x)")
            print(f"   🚀 共享内存: {shared_time:.2f}s (加速: {speedup_shared:.2f}x)")
            print(f"   ✅ 结果一致性: {consistency_check}")
            
            # 清理内存
            del result_standard, result_shared
            gc.collect()
        
        # 清理GAF数据
        del gaf_data
        gc.collect()
        
        return self.conversion_results
    
    def run_quick_test(self):
        """运行快速测试"""
        print("🚀 开始快速块大小优化测试")
        print(f"📊 测试样本数: {self.test_samples}")
        print()
        
        try:
            # 1. 单线程基准测试
            print("1️⃣ 基准测试...")
            self.test_single_thread_baseline()
            
            # 2. 快速GAF块大小测试
            print("2️⃣ GAF块大小测试...")
            self.test_gaf_chunk_sizes_quick()
            
            # 3. 快速转换块大小测试
            print("3️⃣ 数据转换块大小测试...")
            self.test_conversion_chunk_sizes_quick()
            
            # 4. 结果分析
            print("4️⃣ 结果分析...")
            self.analyze_results()
            
            # 5. 保存结果
            print("5️⃣ 保存结果...")
            self.save_results("quick_test_results")
            
            print("\n✅ 快速测试完成!")
            print("📈 查看quick_test_results/目录获取详细结果")
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("🔬 快速块大小优化测试")
    print("=" * 50)
    print("这是一个验证性的快速测试，使用较少的样本数")
    print("如需完整测试，请运行 test_chunk_size_optimization.py")
    print()
    
    # 检查系统要求
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"💻 系统信息:")
    print(f"   CPU核心数: {cpu_count}")
    print(f"   总内存: {memory_gb:.1f}GB")
    print()
    
    if cpu_count < 4:
        print("⚠️  警告: 系统CPU核心数较少，可能影响测试效果")
    
    if memory_gb < 8:
        print("⚠️  警告: 系统内存较少，建议降低测试样本数")
    
    # 创建快速测试器并运行
    optimizer = QuickChunkSizeOptimizer(test_samples=1000)
    optimizer.run_quick_test()


if __name__ == "__main__":
    main() 
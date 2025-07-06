#!/usr/bin/env python3
"""
并行优化性能测试脚本

测试DualGAFDataLoader的并行处理性能改进
比较单进程vs多进程的处理时间和资源使用情况
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np
import psutil
from contextlib import contextmanager

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_loader.DualGAFDataLoader import DualGAFDataManager


class PerformanceMonitor:
    """性能监控类"""
    def __init__(self):
        self.stats = {}
        
    def __enter__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024**3  # GB
        self.start_cpu_percent = self.process.cpu_percent()
        
        print(f"开始监控 - 内存: {self.start_memory:.2f}GB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024**3  # GB
        end_cpu_percent = self.process.cpu_percent()
        
        elapsed_time = end_time - self.start_time
        memory_diff = end_memory - self.start_memory
        
        print(f"性能统计:")
        print(f"  执行时间: {elapsed_time:.2f}秒")
        print(f"  内存变化: {memory_diff:+.2f}GB")
        print(f"  峰值内存: {end_memory:.2f}GB")
        print(f"  CPU使用率: {end_cpu_percent:.1f}%")
        
        self.stats = {
            'time': elapsed_time,
            'memory_diff': memory_diff,
            'peak_memory': end_memory,
            'cpu_percent': end_cpu_percent
        }


def monitor_performance():
    """返回性能监控器实例"""
    return PerformanceMonitor()


class TestArgs:
    """测试用参数配置"""
    def __init__(self, use_parallel=True, n_jobs=None, chunk_size=100):
        self.root_path = "./dataset/SAHU/direct_5_working"
        self.seq_len = 96
        self.step = 96
        self.test_size = 0.2
        self.data_type_method = 'uint8'
        
        # 并行配置
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), 8)
            
        self.n_jobs = n_jobs if use_parallel else 1
        self.use_multiprocessing = use_parallel
        self.chunk_size = chunk_size
        
        print(f"测试配置 - 并行: {use_parallel}, 进程数: {self.n_jobs}, 块大小: {chunk_size}")


def create_synthetic_test_data():
    """创建合成测试数据，用于在没有真实数据时进行测试"""
    print("创建合成测试数据...")
    
    # 模拟文件数据结构
    n_samples = 4000
    n_features = 26
    seq_len = 96
    
    # 生成时序数据
    data = []
    for i in range(n_samples):
        # 生成带有趋势和噪声的时序数据
        trend = np.linspace(0, 1, seq_len)
        noise = np.random.normal(0, 0.1, (seq_len, n_features))
        seasonal = np.sin(2 * np.pi * np.arange(seq_len) / 24)[:, np.newaxis]
        
        sample = trend[:, np.newaxis] + seasonal + noise
        data.append(sample)
    
    return np.array(data)


def test_gaf_generation_performance():
    """测试GAF生成的性能"""
    print("\n" + "="*60)
    print("GAF生成性能测试")
    print("="*60)
    
    # 创建测试数据
    test_data = create_synthetic_test_data()
    print(f"测试数据形状: {test_data.shape}")
    
    results = {}
    
    # 测试单进程
    print(f"\n--- 单进程测试 ---")
    args_single = TestArgs(use_parallel=False)
    manager_single = DualGAFDataManager.__new__(DualGAFDataManager, args_single)
    manager_single.args = args_single
    manager_single.n_jobs = 16
    manager_single.use_multiprocessing = False
    manager_single.chunk_size = 100
    
    with monitor_performance() as monitor:
        result_single = manager_single._generate_gaf_matrix_single(test_data, "summation", False)
    
    results['single_process'] = monitor.stats
    print(f"结果形状: {result_single.shape}")
    
    # 测试多进程
    print(f"\n--- 多进程测试 ---")
    args_multi = TestArgs(use_parallel=True, n_jobs=16, chunk_size=100)
    manager_multi = DualGAFDataManager.__new__(DualGAFDataManager, args_multi)
    manager_multi.args = args_multi
    manager_multi.n_jobs = 16
    manager_multi.use_multiprocessing = True
    manager_multi.chunk_size = 100
    
    with monitor_performance() as monitor:
        result_multi = manager_multi.generate_gaf_matrix_parallel(test_data, "summation", False)
    
    results['multi_process'] = monitor.stats
    print(f"结果形状: {result_multi.shape}")
    
    # 验证结果一致性
    print(f"\n--- 结果验证 ---")
    if np.allclose(result_single, result_multi, rtol=1e-5):
        print("✓ 单进程和多进程结果一致")
    else:
        print("✗ 结果不一致，需要检查实现")
        diff = np.abs(result_single - result_multi).mean()
        print(f"平均差异: {diff}")
    
    return results


def test_data_conversion_performance():
    """测试数据类型转换的性能"""
    print("\n" + "="*60)
    print("数据类型转换性能测试")
    print("="*60)
    
    # 创建GAF数据 (模拟)
    n_samples, n_channels, height, width = 2000, 26, 72, 72
    test_gaf_data = np.random.uniform(-1, 1, (n_samples, n_channels, height, width)).astype(np.float32)
    print(f"测试GAF数据形状: {test_gaf_data.shape}")
    print(f"数据大小: {test_gaf_data.nbytes / 1024**3:.2f}GB")
    
    results = {}
    
    # 测试单进程
    print(f"\n--- 单进程转换测试 ---")
    args_single = TestArgs(use_parallel=False)
    manager_single = DualGAFDataManager.__new__(DualGAFDataManager, args_single)
    manager_single.args = args_single
    manager_single.data_type_method = 'uint8'
    manager_single.n_jobs = 1
    manager_single.use_multiprocessing = False
    manager_single.chunk_size = 100
    
    with monitor_performance() as monitor:
        result_single = manager_single._gaf_to_int(test_gaf_data, dtype=np.uint8)
    
    results['single_conversion'] = monitor.stats
    print(f"转换后数据类型: {result_single.dtype}")
    print(f"转换后大小: {result_single.nbytes / 1024**3:.2f}GB")
    
    # 测试多进程
    print(f"\n--- 多进程转换测试 ---")
    args_multi = TestArgs(use_parallel=True, n_jobs=16, chunk_size=50)
    manager_multi = DualGAFDataManager.__new__(DualGAFDataManager, args_multi)
    manager_multi.args = args_multi
    manager_multi.data_type_method = 'uint8'
    manager_multi.n_jobs = 16
    manager_multi.use_multiprocessing = True
    manager_multi.chunk_size = 100
    
    with monitor_performance() as monitor:
        result_multi = manager_multi._gaf_to_int_parallel(test_gaf_data, dtype=np.uint8)
    
    results['multi_conversion'] = monitor.stats
    print(f"转换后数据类型: {result_multi.dtype}")
    print(f"转换后大小: {result_multi.nbytes / 1024**3:.2f}GB")
    
    # 验证结果一致性
    print(f"\n--- 结果验证 ---")
    if np.array_equal(result_single, result_multi):
        print("✓ 单进程和多进程转换结果一致")
    else:
        print("✗ 转换结果不一致")
        diff_count = np.sum(result_single != result_multi)
        print(f"不同元素数量: {diff_count} / {result_single.size}")
    
    return results


def print_performance_summary(gaf_results, conversion_results):
    """打印性能总结"""
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)
    
    print("\n📊 GAF生成性能对比:")
    single_time = gaf_results['single_process']['time']
    multi_time = gaf_results['multi_process']['time']
    speedup = single_time / multi_time if multi_time > 0 else 0
    
    print(f"  单进程时间: {single_time:.2f}s")
    print(f"  多进程时间: {multi_time:.2f}s")
    print(f"  加速比: {speedup:.2f}x")
    
    if speedup > 1.5:
        print("  ✓ 显著性能提升")
    elif speedup > 1.1:
        print("  ✓ 适度性能提升")
    else:
        print("  ⚠️  性能提升有限")
    
    print("\n📊 数据转换性能对比:")
    single_conv_time = conversion_results['single_conversion']['time']
    multi_conv_time = conversion_results['multi_conversion']['time']
    conv_speedup = single_conv_time / multi_conv_time if multi_conv_time > 0 else 0
    
    print(f"  单进程时间: {single_conv_time:.2f}s")
    print(f"  多进程时间: {multi_conv_time:.2f}s")
    print(f"  加速比: {conv_speedup:.2f}x")
    
    if conv_speedup > 1.5:
        print("  ✓ 显著性能提升")
    elif conv_speedup > 1.1:
        print("  ✓ 适度性能提升")
    else:
        print("  ⚠️  性能提升有限")
    
    # 系统信息
    print(f"\n🖥️  系统信息:")
    print(f"  CPU核心数: {mp.cpu_count()}")
    print(f"  可用内存: {psutil.virtual_memory().available / 1024**3:.2f}GB")
    print(f"  总内存: {psutil.virtual_memory().total / 1024**3:.2f}GB")
    
    # 优化建议
    print(f"\n💡 优化建议:")
    if speedup < 1.5:
        print("  - 考虑增加chunk_size以减少进程通信开销")
        print("  - 检查系统资源是否充足")
        print("  - 对于小数据集，单进程可能更高效")
    
    if conv_speedup < 1.2:
        print("  - 数据转换可能受I/O限制，考虑使用SSD")
        print("  - 适当调整进程数和块大小")


def main():
    """主测试函数"""
    print("DualGAFDataLoader 并行优化性能测试")
    print("="*60)
    print(f"Python版本: {sys.version}")
    print(f"CPU核心数: {mp.cpu_count()}")
    print(f"可用内存: {psutil.virtual_memory().available / 1024**3:.2f}GB")
    
    try:
        # 测试GAF生成性能
        gaf_results = test_gaf_generation_performance()
        
        # 测试数据转换性能
        conversion_results = test_data_conversion_performance()
        
        # 打印总结
        print_performance_summary(gaf_results, conversion_results)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n✅ 性能测试完成")
    return 0


if __name__ == "__main__":
    exit(main()) 
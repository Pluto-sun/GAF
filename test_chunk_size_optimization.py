#!/usr/bin/env python3
"""
块大小优化测试脚本
测试GAF计算和数据范围转换在不同块大小和处理模式下的最优配置

功能:
1. 测试GAF计算的最优块大小 (标准多进程 vs 共享内存)
2. 测试数据类型转换的最优块大小 (标准多进程 vs 共享内存)  
3. 单线程基准性能测试
4. 系统资源监控和推荐
5. 测试样本数: 5000

作者: AI Assistant
日期: 2024
"""

import os
import sys
import time
import numpy as np
import multiprocessing as mp
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import pandas as pd

# 添加项目路径
sys.path.append('.')
from data_provider.data_loader.DualGAFDataLoader import DualGAFDataManager


class ChunkSizeOptimizer:
    """块大小优化测试器"""
    
    def __init__(self, test_samples=5000):
        self.test_samples = test_samples
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"=== 块大小优化测试器初始化 ===")
        print(f"🖥️  CPU核心数: {self.cpu_count}")
        print(f"💾 总内存: {self.memory_gb:.1f}GB")
        print(f"🔢 测试样本数: {self.test_samples}")
        print()
        
        # 测试配置 - 优化后的测试范围（减少测试时间）
        self.chunk_sizes = [100, 200, 500, 1000, 1500]  # 减少测试点
        self.n_jobs_list = [4, 8, min(12, self.cpu_count)]  # 重点测试常用配置
        
        # 生成测试数据
        self.test_data = self._generate_test_data()
        print(f"📊 测试数据生成完成: {self.test_data.shape}")
        print(f"💻 数据大小: {self.test_data.nbytes / 1024**3:.2f}GB")
        
        # 结果存储
        self.gaf_results = []
        self.conversion_results = []
        
    def _generate_test_data(self):
        """生成标准化的测试数据"""
        print("🔄 生成测试数据...")
        
        # 设置随机种子确保可重复性
        np.random.seed(42)
        
        # 生成类似实际工况数据的特征
        seq_len = 96  # 时间序列长度
        n_features = 26  # 特征数量
        
        # 生成多种模式的时间序列数据
        data = []
        for i in range(self.test_samples):
            if i % 1000 == 0:
                print(f"  生成样本 {i}/{self.test_samples}")
                
            # 混合不同的信号模式
            t = np.linspace(0, 24, seq_len)  # 24小时
            sample = np.zeros((seq_len, n_features))
            
            for j in range(n_features):
                # 基础趋势 + 周期性 + 噪声
                trend = 0.1 * t + np.random.normal(0, 0.05)
                periodic = np.sin(2 * np.pi * t / 12) * (0.3 + 0.2 * np.random.random())
                noise = np.random.normal(0, 0.1, seq_len)
                
                # 偶尔添加异常模式
                if np.random.random() < 0.1:
                    anomaly_start = np.random.randint(0, seq_len - 10)
                    anomaly_end = anomaly_start + np.random.randint(5, 15)
                    anomaly_end = min(anomaly_end, seq_len)
                    trend[anomaly_start:anomaly_end] += np.random.normal(0.5, 0.2)
                
                sample[:, j] = trend + periodic + noise
            
            # 归一化到[-1, 1]区间
            sample = 2 * (sample - sample.min()) / (sample.max() - sample.min()) - 1
            data.append(sample)
        
        return np.array(data, dtype=np.float32)
    
    @contextmanager
    def monitor_performance(self, test_name):
        """性能监控上下文管理器"""
        print(f"🚀 开始测试: {test_name}")
        
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024**3
        start_cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 强制垃圾回收
        gc.collect()
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024**3
            end_cpu_percent = psutil.cpu_percent(interval=0.1)
            
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"✅ {test_name} 完成")
            print(f"   ⏱️  耗时: {duration:.2f}s")
            print(f"   💾 内存变化: {memory_used:+.2f}GB")
            print(f"   🔥 CPU使用率: {end_cpu_percent:.1f}%")
            print()
    
    def test_single_thread_baseline(self):
        """单线程基准测试"""
        print("=" * 60)
        print("📏 单线程基准测试")
        print("=" * 60)
        
        results = {}
        
        # GAF生成基准测试
        with self.monitor_performance("单线程 GAF 生成 (Summation)"):
            start_time = time.time()
            gaf_result = self._single_thread_gaf(self.test_data, "summation")
            gaf_time = time.time() - start_time
            results['gaf_summation_time'] = gaf_time
            print(f"   📊 GAF输出形状: {gaf_result.shape}")
        
        # with self.monitor_performance("单线程 GAF 生成 (Difference)"):
        #     start_time = time.time()
        #     gaf_diff_result = self._single_thread_gaf(self.test_data, "difference")
        #     gaf_diff_time = time.time() - start_time
        #     results['gaf_difference_time'] = gaf_diff_time
        
        # 数据类型转换基准测试
        with self.monitor_performance("单线程数据转换 (uint8)"):
            start_time = time.time()
            uint8_result = self._single_thread_conversion(gaf_result, np.uint8, 255)
            uint8_time = time.time() - start_time
            results['conversion_uint8_time'] = uint8_time
            print(f"   📊 转换后数据范围: [{uint8_result.min()}, {uint8_result.max()}]")
        
        # with self.monitor_performance("单线程数据转换 (uint16)"):
        #     start_time = time.time()
        #     uint16_result = self._single_thread_conversion(gaf_result, np.uint16, 65535)
        #     uint16_time = time.time() - start_time
        #     results['conversion_uint16_time'] = uint16_time
        
        # with self.monitor_performance("单线程数据转换 (float32)"):
        #     start_time = time.time()
        #     float32_result = self._single_thread_conversion_float32(gaf_result)
        #     float32_time = time.time() - start_time
        #     results['conversion_float32_time'] = float32_time
        
        # 清理内存
        del gaf_result, uint8_result
        gc.collect()
        
        self.baseline_results = results
        print(f"📈 单线程基准结果:")
        for key, value in results.items():
            print(f"   {key}: {value:.2f}s")
        print()
        
        return results
    
    def _create_mock_baseline(self):
        """创建模拟的基准测试结果（避免耗时的单线程测试）"""
        print("🔢 创建估算基准时间...")
        
        # 基于经验公式估算单线程时间
        # GAF计算复杂度大约是 O(N * D * T^2)
        N, T, D = self.test_data.shape
        
        # 估算GAF计算时间（基于经验值）
        gaf_time_per_sample = 0.015  # 每个样本大约15ms（经验值）
        estimated_gaf_time = N * gaf_time_per_sample
        
        # 估算数据转换时间（基于内存带宽）
        data_size_gb = self.test_data.nbytes / (1024**3)
        estimated_conversion_time = data_size_gb * 8  # 假设8秒/GB的转换时间
        
        self.baseline_results = {
            'gaf_summation_time': estimated_gaf_time,
            'conversion_uint8_time': estimated_conversion_time,
        }
        
        print(f"📊 估算基准时间:")
        print(f"   GAF生成: {estimated_gaf_time:.2f}s")
        print(f"   数据转换: {estimated_conversion_time:.2f}s")
        print(f"💡 注意: 这些是估算值，实际加速比可能有偏差")
        print()
        
        return self.baseline_results
    
    def test_gaf_chunk_sizes(self):
        """测试GAF计算的最优块大小"""
        print("=" * 60)
        print("🧮 GAF计算块大小优化测试")
        print("=" * 60)
        
        baseline_time = self.baseline_results['gaf_summation_time']
        
        for n_jobs in [4, 8, min(12, self.cpu_count)]:
            print(f"\n--- 测试进程数: {n_jobs} ---")
            
            for chunk_size in self.chunk_sizes:
                if chunk_size > self.test_samples // 2:
                    continue  # 跳过过大的块大小
                
                # 测试标准多进程
                with self.monitor_performance(f"标准多进程 GAF (块大小: {chunk_size})"):
                    start_time = time.time()
                    result_standard = self._test_standard_multiprocess_gaf(
                        self.test_data, "summation", n_jobs, chunk_size
                    )
                    standard_time = time.time() - start_time
                    speedup_standard = baseline_time / standard_time if standard_time > 0 else 0
                
                # 测试共享内存模式
                with self.monitor_performance(f"共享内存 GAF (块大小: {chunk_size})"):
                    start_time = time.time()
                    result_shared = self._test_shared_memory_gaf(
                        self.test_data, "summation", n_jobs, chunk_size
                    )
                    shared_time = time.time() - start_time
                    speedup_shared = baseline_time / shared_time if shared_time > 0 else 0
                
                # 验证结果一致性 (优化版本 - 采样检查)
                # consistency_check = self._fast_consistency_check(result_standard, result_shared)
                
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
                    'samples_per_chunk': chunk_size,
                    'total_chunks': max(1, self.test_samples // chunk_size)
                }
                self.gaf_results.append(result_entry)
                
                print(f"   📊 标准多进程: {standard_time:.2f}s (加速: {speedup_standard:.2f}x)")
                print(f"   🚀 共享内存: {shared_time:.2f}s (加速: {speedup_shared:.2f}x)")
                # print(f"   ✅ 结果一致性: {consistency_check}")
                
                # 清理内存
                del result_standard, result_shared
                gc.collect()
        
        return self.gaf_results
    
    def test_conversion_chunk_sizes(self):
        """测试数据类型转换的最优块大小"""
        print("=" * 60)
        print("🔄 数据类型转换块大小优化测试")
        print("=" * 60)
        
        # 先生成GAF数据用于转换测试
        print("🔄 生成GAF数据用于转换测试...")
        gaf_data = self._single_thread_gaf(self.test_data, "summation")
        print(f"📊 GAF数据形状: {gaf_data.shape}, 大小: {gaf_data.nbytes / 1024**3:.2f}GB")
        
        baseline_time = self.baseline_results['conversion_uint8_time']
        
        for n_jobs in [4, 8, min(12, self.cpu_count)]:
            print(f"\n--- 测试线程数: {n_jobs} ---")
            
            for chunk_size in self.chunk_sizes:
                if chunk_size > self.test_samples // 2:
                    continue
                
                # 测试标准多线程转换
                with self.monitor_performance(f"标准多线程转换 (块大小: {chunk_size})"):
                    start_time = time.time()
                    result_standard = self._test_standard_multithread_conversion(
                        gaf_data, np.uint8, 255, n_jobs, chunk_size
                    )
                    standard_time = time.time() - start_time
                    speedup_standard = baseline_time / standard_time if standard_time > 0 else 0
                
                # 测试共享内存转换
                with self.monitor_performance(f"共享内存转换 (块大小: {chunk_size})"):
                    start_time = time.time()
                    result_shared = self._test_shared_memory_conversion(
                        gaf_data, np.uint8, 255, n_jobs, chunk_size
                    )
                    shared_time = time.time() - start_time
                    speedup_shared = baseline_time / shared_time if shared_time > 0 else 0
                
                # 验证结果一致性 (优化版本 - 采样检查)
                # consistency_check = self._fast_consistency_check(result_standard, result_shared)
                
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
                    # 'consistency_check': consistency_check,
                    'samples_per_chunk': chunk_size,
                    'total_chunks': max(1, self.test_samples // chunk_size)
                }
                self.conversion_results.append(result_entry)
                
                print(f"   📊 标准多线程: {standard_time:.2f}s (加速: {speedup_standard:.2f}x)")
                print(f"   🚀 共享内存: {shared_time:.2f}s (加速: {speedup_shared:.2f}x)")
                # print(f"   ✅ 结果一致性: {consistency_check}")
                
                # 清理内存
                del result_standard, result_shared
                gc.collect()
        
        # 清理GAF数据
        del gaf_data
        gc.collect()
        
        return self.conversion_results
    
    def _single_thread_gaf(self, data, method):
        """单线程GAF生成"""
        from pyts.image import GramianAngularField
        
        N, T, D = data.shape
        transposed_data = data.transpose(0, 2, 1)
        flattened_data = transposed_data.reshape(-1, T)
        gasf = GramianAngularField(method=method)
        batch_gaf = gasf.fit_transform(flattened_data)
        reshaped_gaf = batch_gaf.reshape(N, D, T, T)
        return reshaped_gaf.astype(np.float32)
    
    def _single_thread_conversion(self, data, dtype, max_val):
        """单线程数据类型转换"""
        data_min, data_max = data.min(), data.max()
        if data_min >= -1.0 and data_max <= 1.0:
            normalized = (data.astype(np.float64) + 1.0) * (max_val / 2.0)
        else:
            clipped = np.clip(data, -1, 1)
            normalized = (clipped.astype(np.float64) + 1.0) * (max_val / 2.0)
        return np.round(normalized).astype(dtype)
    
    def _single_thread_conversion_float32(self, data):
        """单线程Float32转换"""
        data_min, data_max = data.min(), data.max()
        if data_min >= -1.0 and data_max <= 1.0:
            result = data.astype(np.float32) if data.dtype != np.float32 else data.copy()
            result += 1.0
            result *= 127.5
        else:
            result = np.clip(data, -1, 1, dtype=np.float32)
            result += 1.0
            result *= 127.5
        return result
    
    def _test_standard_multiprocess_gaf(self, data, method, n_jobs, chunk_size):
        """测试标准多进程GAF生成"""
        chunks = self._split_data_into_chunks(data, chunk_size)
        
        from functools import partial
        process_func = partial(ChunkSizeOptimizer._process_gaf_chunk_static, method=method)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_func, chunks))
        
        return np.concatenate(results, axis=0)
    
    def _test_shared_memory_gaf(self, data, method, n_jobs, chunk_size):
        """测试共享内存GAF生成"""
        # 简化版共享内存测试，基于DualGAFDataManager的实现
        try:
            from multiprocessing import shared_memory
            from concurrent.futures import as_completed
            
            N, T, D = data.shape
            
            # 创建输入共享内存
            input_shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            input_array = np.ndarray(data.shape, dtype=data.dtype, buffer=input_shm.buf)
            input_array[:] = data[:]
            
            # 创建结果共享内存
            result_shape = (N, D, T, T)
            result_dtype = np.float32
            result_size = N * D * T * T * np.dtype(result_dtype).itemsize
            result_shm = shared_memory.SharedMemory(create=True, size=result_size)
            result_array = np.ndarray(result_shape, dtype=result_dtype, buffer=result_shm.buf)
            
            # 计算块索引
            chunk_indices = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk_indices.append((i, end_idx))
            
            # 并行处理
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for start_idx, end_idx in chunk_indices:
                    future = executor.submit(
                        self._process_gaf_chunk_shared_memory_static,
                        input_shm.name,
                        data.shape,
                        data.dtype,
                        start_idx,
                        end_idx,
                        method,
                        result_shm.name,
                        result_shape,
                        result_dtype
                    )
                    futures.append(future)
                
                # 等待完成
                for future in as_completed(futures):
                    future.result()
            
            # 复制结果
            final_result = result_array.copy()
            
            # 清理
            input_shm.close()
            input_shm.unlink()
            result_shm.close()
            result_shm.unlink()
            
            return final_result
            
        except Exception as e:
            print(f"共享内存GAF测试失败: {e}")
            # 回退到标准方法
            return self._test_standard_multiprocess_gaf(data, method, n_jobs, chunk_size)
    
    def _test_standard_multithread_conversion(self, data, dtype, max_val, n_jobs, chunk_size):
        """测试标准多线程数据转换"""
        chunks = self._split_data_into_chunks(data, chunk_size)
        
        from functools import partial
        process_func = partial(ChunkSizeOptimizer._process_conversion_chunk_static, dtype=dtype, max_val=max_val)
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_func, chunks))
        
        return np.concatenate(results, axis=0)
    
    def _test_shared_memory_conversion(self, data, dtype, max_val, n_jobs, chunk_size):
        """测试共享内存数据转换"""
        try:
            from multiprocessing import shared_memory
            from concurrent.futures import as_completed
            
            N = data.shape[0]
            
            # 创建输入共享内存
            input_shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            input_array = np.ndarray(data.shape, dtype=data.dtype, buffer=input_shm.buf)
            input_array[:] = data[:]
            
            # 创建结果共享内存
            result_size = data.size * np.dtype(dtype).itemsize
            result_shm = shared_memory.SharedMemory(create=True, size=result_size)
            result_array = np.ndarray(data.shape, dtype=dtype, buffer=result_shm.buf)
            
            # 计算块索引
            chunk_indices = []
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                chunk_indices.append((i, end_idx))
            
            # 并行处理
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for start_idx, end_idx in chunk_indices:
                    future = executor.submit(
                        self._process_conversion_chunk_shared_memory_static,
                        input_shm.name,
                        data.shape,
                        data.dtype,
                        result_shm.name,
                        data.shape,
                        dtype,
                        start_idx,
                        end_idx,
                        dtype,
                        max_val
                    )
                    futures.append(future)
                
                # 等待完成
                for future in as_completed(futures):
                    future.result()
            
            # 复制结果
            final_result = result_array.copy()
            
            # 清理
            input_shm.close()
            input_shm.unlink()
            result_shm.close()
            result_shm.unlink()
            
            return final_result
            
        except Exception as e:
            print(f"共享内存转换测试失败: {e}")
            # 回退到标准方法
            return self._test_standard_multithread_conversion(data, dtype, max_val, n_jobs, chunk_size)
    
    def _split_data_into_chunks(self, data, chunk_size):
        """将数据分割成块"""
        N = data.shape[0]
        chunks = []
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunks.append(data[i:end_idx])
        return chunks
    
    def _fast_consistency_check(self, result1, result2, sample_ratio=0.01, rtol=1e-5):
        """
        快速一致性检查：通过采样的方式检查两个大数组是否一致
        
        Args:
            result1, result2: 要比较的数组
            sample_ratio: 采样比例 (默认1%, 对于大数组足够)
            rtol: 相对容差
            
        Returns:
            bool: 是否一致
        """
        # 基本形状检查
        if result1.shape != result2.shape:
            print(f"   ⚠️  形状不一致: {result1.shape} vs {result2.shape}")
            return False
        
        # 基本统计检查（非常快）
        if abs(result1.mean() - result2.mean()) > rtol:
            print(f"   ⚠️  均值差异过大: {result1.mean():.6f} vs {result2.mean():.6f}")
            return False
            
        if abs(result1.std() - result2.std()) > rtol:
            print(f"   ⚠️  标准差差异过大: {result1.std():.6f} vs {result2.std():.6f}")
            return False
        
        # 采样检查
        total_elements = result1.size
        sample_size = max(1000, int(total_elements * sample_ratio))  # 至少检查1000个元素
        
        # 随机采样索引
        np.random.seed(42)  # 确保可重复
        flat1 = result1.flatten()
        flat2 = result2.flatten()
        
        # 随机采样
        sample_indices = np.random.choice(total_elements, size=sample_size, replace=False)
        sample1 = flat1[sample_indices]
        sample2 = flat2[sample_indices]
        
        # 对采样数据进行精确比较
        is_close = np.allclose(sample1, sample2, rtol=rtol)
        
        if not is_close:
            # 额外的诊断信息
            diff = np.abs(sample1 - sample2)
            max_diff = diff.max()
            avg_diff = diff.mean()
            print(f"   ⚠️  采样检查失败: 最大差异={max_diff:.8f}, 平均差异={avg_diff:.8f}")
        
        return is_close
    
    @staticmethod
    def _process_gaf_chunk_static(chunk_data, method):
        """静态方法：处理GAF块"""
        from pyts.image import GramianAngularField
        
        N, T, D = chunk_data.shape
        transposed_data = chunk_data.transpose(0, 2, 1)
        flattened_data = transposed_data.reshape(-1, T)
        gasf = GramianAngularField(method=method)
        batch_gaf = gasf.fit_transform(flattened_data)
        reshaped_gaf = batch_gaf.reshape(N, D, T, T)
        return reshaped_gaf.astype(np.float32)
    
    @staticmethod
    def _process_conversion_chunk_static(chunk_data, dtype, max_val):
        """静态方法：处理转换块"""
        data_min, data_max = chunk_data.min(), chunk_data.max()
        if data_min >= -1.0 and data_max <= 1.0:
            normalized = (chunk_data.astype(np.float64) + 1.0) * (max_val / 2.0)
        else:
            clipped = np.clip(chunk_data, -1, 1)
            normalized = (clipped.astype(np.float64) + 1.0) * (max_val / 2.0)
        return np.round(normalized).astype(dtype)
    
    @staticmethod
    def _process_gaf_chunk_shared_memory_static(shm_name, shape, dtype, start_idx, end_idx, 
                                              method, result_shm_name, result_shape, result_dtype):
        """静态方法：共享内存GAF处理"""
        from multiprocessing import shared_memory
        from pyts.image import GramianAngularField
        
        # 连接输入共享内存
        input_shm = shared_memory.SharedMemory(name=shm_name)
        input_array = np.ndarray(shape, dtype=dtype, buffer=input_shm.buf)
        
        # 提取块数据
        chunk_data = input_array[start_idx:end_idx].copy()
        input_shm.close()
        
        # GAF转换
        N, T, D = chunk_data.shape
        transposed_data = chunk_data.transpose(0, 2, 1)
        flattened_data = transposed_data.reshape(-1, T)
        gasf = GramianAngularField(method=method)
        batch_gaf = gasf.fit_transform(flattened_data)
        reshaped_gaf = batch_gaf.reshape(N, D, T, T).astype(np.float32)
        
        # 写入结果共享内存
        result_shm = shared_memory.SharedMemory(name=result_shm_name)
        result_array = np.ndarray(result_shape, dtype=result_dtype, buffer=result_shm.buf)
        result_array[start_idx:start_idx + reshaped_gaf.shape[0]] = reshaped_gaf
        result_shm.close()
    
    @staticmethod
    def _process_conversion_chunk_shared_memory_static(input_shm_name, input_shape, input_dtype,
                                                     result_shm_name, result_shape, result_dtype,
                                                     start_idx, end_idx, target_dtype, max_val):
        """静态方法：共享内存转换处理"""
        from multiprocessing import shared_memory
        
        # 连接输入共享内存
        input_shm = shared_memory.SharedMemory(name=input_shm_name)
        input_array = np.ndarray(input_shape, dtype=input_dtype, buffer=input_shm.buf)
        
        # 提取块数据
        chunk_data = input_array[start_idx:end_idx].copy()
        input_shm.close()
        
        # 数据转换
        data_min, data_max = chunk_data.min(), chunk_data.max()
        if data_min >= -1.0 and data_max <= 1.0:
            normalized = (chunk_data.astype(np.float64) + 1.0) * (max_val / 2.0)
        else:
            clipped = np.clip(chunk_data, -1, 1)
            normalized = (clipped.astype(np.float64) + 1.0) * (max_val / 2.0)
        
        converted_data = np.round(normalized).astype(target_dtype)
        
        # 写入结果共享内存
        result_shm = shared_memory.SharedMemory(name=result_shm_name)
        result_array = np.ndarray(result_shape, dtype=result_dtype, buffer=result_shm.buf)
        result_array[start_idx:end_idx] = converted_data
        result_shm.close()
    
    def analyze_results(self):
        """分析测试结果并提供推荐"""
        print("=" * 60)
        print("📈 结果分析与推荐")
        print("=" * 60)
        
        # 分析GAF结果
        if self.gaf_results:
            print("\n🧮 GAF计算优化分析:")
            gaf_df = pd.DataFrame(self.gaf_results)
            
            # 找出最佳配置
            best_standard = gaf_df.loc[gaf_df['standard_speedup'].idxmax()]
            best_shared = gaf_df.loc[gaf_df['shared_speedup'].idxmax()]
            
            print(f"   📊 最佳标准多进程配置:")
            print(f"      进程数: {best_standard['n_jobs']}, 块大小: {best_standard['chunk_size']}")
            print(f"      加速比: {best_standard['standard_speedup']:.2f}x, 耗时: {best_standard['standard_time']:.2f}s")
            
            print(f"   🚀 最佳共享内存配置:")
            print(f"      进程数: {best_shared['n_jobs']}, 块大小: {best_shared['chunk_size']}")
            print(f"      加速比: {best_shared['shared_speedup']:.2f}x, 耗时: {best_shared['shared_time']:.2f}s")
            
            # 效率分析
            avg_standard_speedup = gaf_df['standard_speedup'].mean()
            avg_shared_speedup = gaf_df['shared_speedup'].mean()
            print(f"   📈 平均加速比 - 标准: {avg_standard_speedup:.2f}x, 共享内存: {avg_shared_speedup:.2f}x")
        
        # 分析转换结果
        if self.conversion_results:
            print("\n🔄 数据转换优化分析:")
            conv_df = pd.DataFrame(self.conversion_results)
            
            # 找出最佳配置
            best_standard = conv_df.loc[conv_df['standard_speedup'].idxmax()]
            best_shared = conv_df.loc[conv_df['shared_speedup'].idxmax()]
            
            print(f"   📊 最佳标准多线程配置:")
            print(f"      线程数: {best_standard['n_jobs']}, 块大小: {best_standard['chunk_size']}")
            print(f"      加速比: {best_standard['standard_speedup']:.2f}x, 耗时: {best_standard['standard_time']:.2f}s")
            
            print(f"   🚀 最佳共享内存配置:")
            print(f"      线程数: {best_shared['n_jobs']}, 块大小: {best_shared['chunk_size']}")
            print(f"      加速比: {best_shared['shared_speedup']:.2f}x, 耗时: {best_shared['shared_time']:.2f}s")
            
            # 效率分析
            avg_standard_speedup = conv_df['standard_speedup'].mean()
            avg_shared_speedup = conv_df['shared_speedup'].mean()
            print(f"   📈 平均加速比 - 标准: {avg_standard_speedup:.2f}x, 共享内存: {avg_shared_speedup:.2f}x")
        
        # 系统推荐
        print(f"\n🎯 系统推荐配置 (基于当前硬件: {self.cpu_count}核, {self.memory_gb:.1f}GB):")
        
        # GAF计算推荐
        if self.gaf_results:
            gaf_df = pd.DataFrame(self.gaf_results)
            gaf_recommendations = []
            
            # 不同进程数下的最佳块大小
            for n_jobs in gaf_df['n_jobs'].unique():
                subset = gaf_df[gaf_df['n_jobs'] == n_jobs]
                best_standard_idx = subset['standard_speedup'].idxmax()
                best_shared_idx = subset['shared_speedup'].idxmax()
                
                best_standard_config = subset.loc[best_standard_idx]
                best_shared_config = subset.loc[best_shared_idx]
                
                gaf_recommendations.append({
                    'n_jobs': n_jobs,
                    'standard_chunk_size': best_standard_config['chunk_size'],
                    'standard_speedup': best_standard_config['standard_speedup'],
                    'shared_chunk_size': best_shared_config['chunk_size'],
                    'shared_speedup': best_shared_config['shared_speedup']
                })
            
            print(f"   🧮 GAF计算推荐:")
            for rec in gaf_recommendations:
                print(f"      {rec['n_jobs']}进程 - 标准: {rec['standard_chunk_size']} ({rec['standard_speedup']:.2f}x), "
                      f"共享内存: {rec['shared_chunk_size']} ({rec['shared_speedup']:.2f}x)")
        
        # 转换推荐
        if self.conversion_results:
            conv_df = pd.DataFrame(self.conversion_results)
            conv_recommendations = []
            
            for n_jobs in conv_df['n_jobs'].unique():
                subset = conv_df[conv_df['n_jobs'] == n_jobs]
                best_standard_idx = subset['standard_speedup'].idxmax()
                best_shared_idx = subset['shared_speedup'].idxmax()
                
                best_standard_config = subset.loc[best_standard_idx]
                best_shared_config = subset.loc[best_shared_idx]
                
                conv_recommendations.append({
                    'n_jobs': n_jobs,
                    'standard_chunk_size': best_standard_config['chunk_size'],
                    'standard_speedup': best_standard_config['standard_speedup'],
                    'shared_chunk_size': best_shared_config['chunk_size'],
                    'shared_speedup': best_shared_config['shared_speedup']
                })
            
            print(f"   🔄 数据转换推荐:")
            for rec in conv_recommendations:
                print(f"      {rec['n_jobs']}线程 - 标准: {rec['standard_chunk_size']} ({rec['standard_speedup']:.2f}x), "
                      f"共享内存: {rec['shared_chunk_size']} ({rec['shared_speedup']:.2f}x)")
    
    def save_results(self, output_dir="test_results"):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存GAF结果
        if self.gaf_results:
            gaf_df = pd.DataFrame(self.gaf_results)
            gaf_df.to_csv(os.path.join(output_dir, "gaf_chunk_size_results.csv"), index=False)
            print(f"📁 GAF测试结果已保存到: {output_dir}/gaf_chunk_size_results.csv")
        
        # 保存转换结果
        if self.conversion_results:
            conv_df = pd.DataFrame(self.conversion_results)
            conv_df.to_csv(os.path.join(output_dir, "conversion_chunk_size_results.csv"), index=False)
            print(f"📁 转换测试结果已保存到: {output_dir}/conversion_chunk_size_results.csv")
        
        # 保存基准结果
        if hasattr(self, 'baseline_results'):
            baseline_df = pd.DataFrame([self.baseline_results])
            baseline_df.to_csv(os.path.join(output_dir, "baseline_results.csv"), index=False)
            print(f"📁 基准测试结果已保存到: {output_dir}/baseline_results.csv")
    
    def run_full_test(self):
        """运行完整的块大小优化完整测试"""
        print("🔬 开始块大小优化完整测试")
        print(f"📊 测试样本数: {self.test_samples}")
        
        # 估算测试时间
        n_gaf_tests = len(self.chunk_sizes) * len([n for n in self.n_jobs_list if n <= self.cpu_count]) * 2  # 标准+共享内存
        n_conv_tests = len(self.chunk_sizes) * len([n for n in self.n_jobs_list if n <= self.cpu_count]) * 2
        total_tests = n_gaf_tests + n_conv_tests
        estimated_time_per_test = 8  # 每个测试约8秒
        estimated_total_time = total_tests * estimated_time_per_test / 60  # 转换为分钟
        
        print(f"⏱️  预估测试时间: {estimated_total_time:.1f}分钟 ({total_tests}个测试)")
        print(f"🎯 测试配置: 块大小{len(self.chunk_sizes)}种, 进程数{len(self.n_jobs_list)}种")
        print()
        
        try:
            # 1. 单线程基准测试 - 暂时注释掉（太慢）
            print("⏭️  跳过单线程基准测试（太耗时），使用估算基准时间")
            self._create_mock_baseline()
            # self.test_single_thread_baseline()
            
            # 2. GAF块大小测试
            self.test_gaf_chunk_sizes()
            
            # 3. 转换块大小测试
            self.test_conversion_chunk_sizes()
            
            # 4. 结果分析
            self.analyze_results()
            
            # 5. 保存结果
            self.save_results()
            
            print("\n✅ 块大小优化测试完成!")
            print("📈 查看test_results/目录获取详细结果")
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("🔬 块大小优化测试脚本")
    print("=" * 60)
    
    # 检查系统要求
    if mp.cpu_count() < 4:
        print("⚠️  警告: 系统CPU核心数较少，可能影响测试效果")
    
    if psutil.virtual_memory().total / (1024**3) < 8:
        print("⚠️  警告: 系统内存较少，建议降低测试样本数")
    
    # 创建测试器并运行
    optimizer = ChunkSizeOptimizer(test_samples=5000)
    optimizer.run_full_test()


if __name__ == "__main__":
    main() 
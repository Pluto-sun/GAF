import torch
import gc
import os
import psutil
import subprocess
import sys

class MemoryCleaner:
    """内存清理工具类"""
    
    @staticmethod
    def clear_gpu_memory():
        """清理GPU内存"""
        if torch.cuda.is_available():
            print("正在清理GPU内存...")
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            # 同步所有CUDA操作
            torch.cuda.synchronize()
            print("GPU内存清理完成")
        else:
            print("未检测到CUDA设备")
    
    @staticmethod
    def clear_system_memory():
        """清理系统内存"""
        print("正在清理系统内存...")
        # 强制垃圾回收
        gc.collect()
        print("系统内存清理完成")
    
    @staticmethod
    def get_memory_info():
        """获取内存使用信息"""
        info = {}
        
        # GPU内存信息
        if torch.cuda.is_available():
            info['gpu'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
                'max_reserved': torch.cuda.max_memory_reserved() / 1024**3
            }
        
        # 系统内存信息
        memory = psutil.virtual_memory()
        info['system'] = {
            'total': memory.total / 1024**3,
            'available': memory.available / 1024**3,
            'used': memory.used / 1024**3,
            'percent': memory.percent
        }
        
        return info
    
    @staticmethod
    def print_memory_info():
        """打印内存使用信息"""
        info = MemoryCleaner.get_memory_info()
        
        print("\n=== 内存使用情况 ===")
        
        if 'gpu' in info:
            print("GPU内存:")
            print(f"  已分配: {info['gpu']['allocated']:.2f} GB")
            print(f"  已缓存: {info['gpu']['reserved']:.2f} GB")
            print(f"  最大分配: {info['gpu']['max_allocated']:.2f} GB")
            print(f"  最大缓存: {info['gpu']['max_reserved']:.2f} GB")
        
        print("系统内存:")
        print(f"  总内存: {info['system']['total']:.2f} GB")
        print(f"  已使用: {info['system']['used']:.2f} GB")
        print(f"  可用内存: {info['system']['available']:.2f} GB")
        print(f"  使用率: {info['system']['percent']:.1f}%")
        print("==================\n")
    
    @staticmethod
    def full_cleanup():
        """完整清理"""
        print("\n开始完整内存清理...")
        MemoryCleaner.clear_system_memory()
        MemoryCleaner.clear_gpu_memory()
        print("完整内存清理完成\n")
    
    @staticmethod
    def reset_gpu():
        """重置GPU状态（高级清理）"""
        if torch.cuda.is_available():
            print("正在重置GPU状态...")
            try:
                # 重置内存统计信息
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # 清空缓存
                torch.cuda.empty_cache()
                
                # 同步所有CUDA操作
                torch.cuda.synchronize()
                
                print("GPU状态重置完成")
            except Exception as e:
                print(f"GPU状态重置失败: {e}")
        else:
            print("未检测到CUDA设备")

def cleanup_on_exit():
    """程序退出时的清理函数"""
    print("\n程序即将退出，正在清理内存...")
    MemoryCleaner.full_cleanup()
    MemoryCleaner.print_memory_info()

# 注册退出时的清理函数
import atexit
atexit.register(cleanup_on_exit)

# 使用示例
if __name__ == "__main__":
    # 打印当前内存状态
    MemoryCleaner.print_memory_info()
    
    # 执行清理
    MemoryCleaner.full_cleanup()
    
    # 打印清理后的内存状态
    MemoryCleaner.print_memory_info() 
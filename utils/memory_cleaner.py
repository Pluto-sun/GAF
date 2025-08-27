import torch
import gc
import os
import psutil
import subprocess
import sys
import platform

class MemoryCleaner:
    """内存清理工具类"""
    
    @staticmethod
    def is_wsl():
        """检测是否在WSL环境中运行"""
        try:
            # 检查 /proc/version 文件
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    version = f.read().lower()
                    if 'microsoft' in version or 'wsl' in version:
                        return True
            
            # 检查环境变量
            if 'WSL_DISTRO_NAME' in os.environ or 'WSLENV' in os.environ:
                return True
                
            return False
        except:
            return False
    
    @staticmethod
    def is_cuda_stable():
        """检查CUDA状态是否稳定"""
        if not torch.cuda.is_available():
            return False
        
        try:
            # 尝试基本的CUDA操作
            device = torch.cuda.current_device()
            torch.cuda.get_device_properties(device)
            return True
        except:
            return False
    
    @staticmethod
    def clear_gpu_memory():
        """清理GPU内存"""
        if not torch.cuda.is_available():
            print("未检测到CUDA设备")
            return
        
        print("正在清理GPU内存...")
        
        try:
            # 检查CUDA是否稳定
            if not MemoryCleaner.is_cuda_stable():
                print("CUDA状态不稳定，跳过GPU内存清理")
                return
            
            # WSL环境下使用更温和的清理方式
            if MemoryCleaner.is_wsl():
                print("检测到WSL环境，使用安全清理模式")
                # 在WSL中，退出时不执行激进的GPU操作
                try:
                    # 仅尝试基本的同步操作
                    torch.cuda.synchronize()
                    print("GPU同步完成")
                except Exception as e:
                    print(f"GPU同步失败（WSL环境下这是正常的）: {e}")
                return
            
            # 非WSL环境的标准清理
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            # 同步所有CUDA操作
            torch.cuda.synchronize()
            print("GPU内存清理完成")
            
        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"GPU内存清理失败（CUDA错误）: {e}")
                if MemoryCleaner.is_wsl():
                    print("这在WSL环境中是已知问题，不影响程序功能")
                else:
                    print("建议检查CUDA驱动和环境配置")
            else:
                print(f"GPU内存清理失败: {e}")
        except Exception as e:
            print(f"GPU内存清理出现意外错误: {e}")
    
    @staticmethod
    def clear_system_memory():
        """清理系统内存"""
        print("正在清理系统内存...")
        try:
            # 强制垃圾回收
            gc.collect()
            print("系统内存清理完成")
        except Exception as e:
            print(f"系统内存清理失败: {e}")
    
    @staticmethod
    def get_memory_info():
        """获取内存使用信息"""
        info = {}
        
        # GPU内存信息
        if torch.cuda.is_available():
            try:
                info['gpu'] = {
                    'allocated': torch.cuda.memory_allocated() / 1024**3,
                    'reserved': torch.cuda.memory_reserved() / 1024**3,
                    'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
                    'max_reserved': torch.cuda.max_memory_reserved() / 1024**3
                }
            except Exception as e:
                print(f"获取GPU内存信息失败: {e}")
                info['gpu'] = {'error': str(e)}
        
        # 系统内存信息
        try:
            memory = psutil.virtual_memory()
            info['system'] = {
                'total': memory.total / 1024**3,
                'available': memory.available / 1024**3,
                'used': memory.used / 1024**3,
                'percent': memory.percent
            }
        except Exception as e:
            print(f"获取系统内存信息失败: {e}")
            info['system'] = {'error': str(e)}
        
        return info
    
    @staticmethod
    def print_memory_info():
        """打印内存使用信息"""
        info = MemoryCleaner.get_memory_info()
        
        print("\n=== 内存使用情况 ===")
        
        if 'gpu' in info:
            if 'error' in info['gpu']:
                print(f"GPU内存: 信息获取失败 - {info['gpu']['error']}")
            else:
                print("GPU内存:")
                print(f"  已分配: {info['gpu']['allocated']:.2f} GB")
                print(f"  已缓存: {info['gpu']['reserved']:.2f} GB")
                print(f"  最大分配: {info['gpu']['max_allocated']:.2f} GB")
                print(f"  最大缓存: {info['gpu']['max_reserved']:.2f} GB")
        
        if 'system' in info:
            if 'error' in info['system']:
                print(f"系统内存: 信息获取失败 - {info['system']['error']}")
            else:
                print("系统内存:")
                print(f"  总内存: {info['system']['total']:.2f} GB")
                print(f"  已使用: {info['system']['used']:.2f} GB")
                print(f"  可用内存: {info['system']['available']:.2f} GB")
                print(f"  使用率: {info['system']['percent']:.1f}%")
        
        # 显示环境信息
        print(f"运行环境: {'WSL' if MemoryCleaner.is_wsl() else platform.system()}")
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
        if not torch.cuda.is_available():
            print("未检测到CUDA设备")
            return
        
        print("正在重置GPU状态...")
        try:
            # 检查CUDA是否稳定
            if not MemoryCleaner.is_cuda_stable():
                print("CUDA状态不稳定，跳过GPU重置")
                return
            
            # WSL环境下跳过重置操作
            if MemoryCleaner.is_wsl():
                print("WSL环境下跳过GPU重置操作")
                return
            
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
            if MemoryCleaner.is_wsl():
                print("这在WSL环境中是已知问题")

def cleanup_on_exit():
    """程序退出时的清理函数"""
    try:
        print("\n程序即将退出，正在清理内存...")
        
        # 在WSL环境下使用更安全的清理方式
        if MemoryCleaner.is_wsl():
            print("检测到WSL环境，使用安全退出模式")
            # 只清理系统内存，避免CUDA清理引起的问题
            MemoryCleaner.clear_system_memory()
            print("WSL环境下的安全清理完成")
        else:
            # 非WSL环境执行完整清理
            MemoryCleaner.full_cleanup()
        
        # 尝试打印内存信息（如果失败也不要影响程序退出）
        try:
            MemoryCleaner.print_memory_info()
        except:
            print("内存信息获取失败，但不影响程序退出")
            
    except Exception as e:
        print(f"退出清理过程中出现错误: {e}")
        print("程序将继续退出...")

# 注册退出时的清理函数
import atexit
atexit.register(cleanup_on_exit)

# 使用示例
if __name__ == "__main__":
    # 显示环境信息
    print(f"运行环境: {'WSL' if MemoryCleaner.is_wsl() else platform.system()}")
    
    # 打印当前内存状态
    MemoryCleaner.print_memory_info()
    
    # 执行清理
    MemoryCleaner.full_cleanup()
    
    # 打印清理后的内存状态
    MemoryCleaner.print_memory_info() 
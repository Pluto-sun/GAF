#!/usr/bin/env python3
"""
独立的GPU和系统内存清理脚本
可以在训练程序结束后运行，或者作为独立工具使用
"""

import torch
import gc
import os
import sys
import subprocess
import time

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        print("🔧 正在清理GPU内存...")
        
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU设备")
        
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            print(f"  清理GPU {i}...")
            
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 同步操作
            torch.cuda.synchronize()
            
            # 重置统计信息
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
            
        print("✅ GPU内存清理完成")
        return True
    else:
        print("❌ 未检测到CUDA设备")
        return False

def clear_system_memory():
    """清理系统内存"""
    print("🔧 正在清理系统内存...")
    
    # Python垃圾回收
    collected = gc.collect()
    print(f"  垃圾回收清理了 {collected} 个对象")
    
    # 尝试释放Python内存池
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print("  释放了libc内存池")
    except:
        pass
    
    print("✅ 系统内存清理完成")

def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        info = {
            'device': i,
            'name': torch.cuda.get_device_name(i),
            'allocated': torch.cuda.memory_allocated(i) / 1024**3,
            'reserved': torch.cuda.memory_reserved(i) / 1024**3,
            'total': torch.cuda.get_device_properties(i).total_memory / 1024**3
        }
        gpu_info.append(info)
    
    return gpu_info

def get_system_memory_info():
    """获取系统内存信息"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024**3,
            'available': memory.available / 1024**3,
            'used': memory.used / 1024**3,
            'percent': memory.percent
        }
    except ImportError:
        print("警告: psutil未安装，无法显示系统内存信息")
        return None

def print_memory_status():
    """打印内存状态"""
    print("\n" + "="*60)
    print("📊 内存状态报告")
    print("="*60)
    
    # GPU信息
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\n🖥️  GPU内存状态:")
        for info in gpu_info:
            print(f"  GPU {info['device']} ({info['name']}):")
            print(f"    已分配: {info['allocated']:.2f} GB")
            print(f"    已缓存: {info['reserved']:.2f} GB")
            print(f"    总容量: {info['total']:.2f} GB")
            print(f"    使用率: {(info['allocated']/info['total']*100):.1f}%")
    
    # 系统内存信息
    sys_memory = get_system_memory_info()
    if sys_memory:
        print(f"\n💻 系统内存状态:")
        print(f"  总内存: {sys_memory['total']:.2f} GB")
        print(f"  已使用: {sys_memory['used']:.2f} GB")
        print(f"  可用内存: {sys_memory['available']:.2f} GB")
        print(f"  使用率: {sys_memory['percent']:.1f}%")
    
    print("="*60 + "\n")

def kill_python_processes():
    """杀死其他Python进程（危险操作，慎用）"""
    print("⚠️  警告: 正在查找Python进程...")
    
    try:
        result = subprocess.run(['pgrep', '-f', 'python'], 
                              capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        current_pid = str(os.getpid())
        
        other_pids = [pid for pid in pids if pid and pid != current_pid]
        
        if other_pids:
            print(f"找到 {len(other_pids)} 个其他Python进程")
            response = input("是否要杀死这些进程? (y/N): ")
            if response.lower() == 'y':
                for pid in other_pids:
                    try:
                        os.kill(int(pid), 9)
                        print(f"  杀死进程 {pid}")
                    except:
                        print(f"  无法杀死进程 {pid}")
        else:
            print("未找到其他Python进程")
    except Exception as e:
        print(f"操作失败: {e}")

def force_gpu_reset():
    """强制重置GPU（需要nvidia-smi）"""
    print("🔄 尝试强制重置GPU...")
    
    try:
        # 重置GPU
        result = subprocess.run(['nvidia-smi', '--gpu-reset'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU重置成功")
        else:
            print(f"❌ GPU重置失败: {result.stderr}")
    except FileNotFoundError:
        print("❌ nvidia-smi未找到，无法重置GPU")
    except Exception as e:
        print(f"❌ GPU重置失败: {e}")

def main():
    print("🧹 GPU和系统内存清理工具")
    print("="*40)
    
    # 显示清理前状态
    print("\n📋 清理前状态:")
    print_memory_status()
    
    # 执行清理
    print("🚀 开始清理...")
    
    # 清理系统内存
    clear_system_memory()
    
    # 清理GPU内存
    gpu_available = clear_gpu_memory()
    
    # 等待一下让清理生效
    time.sleep(2)
    
    # 显示清理后状态
    print("\n📋 清理后状态:")
    print_memory_status()
    
    # 交互式选项
    while True:
        print("\n🔧 额外选项:")
        print("1. 再次清理")
        print("2. 强制重置GPU (需要nvidia-smi)")
        print("3. 查看进程并清理 (危险)")
        print("4. 退出")
        
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            clear_system_memory()
            clear_gpu_memory()
            print_memory_status()
        elif choice == '2':
            force_gpu_reset()
        elif choice == '3':
            kill_python_processes()
        elif choice == '4':
            break
        else:
            print("无效选择")
    
    print("\n✨ 清理完成，程序退出")

if __name__ == "__main__":
    main() 
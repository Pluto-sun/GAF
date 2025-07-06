#!/bin/bash

# GPU和系统内存清理脚本
# 使用方法: ./cleanup_system.sh [选项]
# 选项: --gpu-only, --memory-only, --force

echo "🧹 系统内存清理工具"
echo "===================="

# 检查是否有root权限
check_root() {
    if [[ $EUID -eq 0 ]]; then
        echo "✅ 检测到root权限，可以执行系统级清理"
        return 0
    else
        echo "⚠️  建议使用sudo运行以获得更好的清理效果"
        return 1
    fi
}

# 清理GPU内存
cleanup_gpu() {
    echo "🔧 清理GPU内存..."
    
    # 检查nvidia-smi是否可用
    if command -v nvidia-smi &> /dev/null; then
        echo "  使用nvidia-smi清理GPU进程..."
        
        # 显示GPU状态
        echo "  当前GPU状态:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
        
        # 杀死占用GPU的进程（可选）
        if [[ "$1" == "--force" ]]; then
            echo "  强制清理GPU进程..."
            nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | while read pid; do
                if [[ -n "$pid" ]]; then
                    echo "    杀死GPU进程: $pid"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        fi
        
        # 重置GPU
        nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true
        
        echo "✅ GPU清理完成"
    else
        echo "❌ nvidia-smi未找到，跳过GPU清理"
    fi
}

# 清理系统内存
cleanup_memory() {
    echo "🔧 清理系统内存..."
    
    # 显示清理前的内存状态
    echo "  清理前内存状态:"
    free -h
    
    # 清理页面缓存
    echo "  清理页面缓存..."
    sync
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'
    
    # 清理目录项和inode缓存
    echo "  清理目录项和inode缓存..."
    echo 2 > /proc/sys/vm/drop_caches 2>/dev/null || sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'
    
    # 清理所有缓存
    echo "  清理所有缓存..."
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    
    # 清理交换分区
    if [[ "$1" == "--force" ]]; then
        echo "  清理交换分区..."
        swapoff -a 2>/dev/null && swapon -a 2>/dev/null || true
    fi
    
    # 显示清理后的内存状态
    echo "  清理后内存状态:"
    free -h
    
    echo "✅ 系统内存清理完成"
}

# 清理Python进程
cleanup_python() {
    echo "🔧 清理Python进程..."
    
    # 找到所有Python进程
    python_pids=$(pgrep -f python | grep -v $$)
    
    if [[ -n "$python_pids" ]]; then
        echo "  找到Python进程:"
        ps -p $python_pids -o pid,ppid,cmd --no-headers || true
        
        if [[ "$1" == "--force" ]]; then
            echo "  强制清理Python进程..."
            echo "$python_pids" | xargs kill -9 2>/dev/null || true
        else
            echo "  (使用 --force 选项来强制清理这些进程)"
        fi
    else
        echo "  未找到Python进程"
    fi
}

# 显示系统状态
show_status() {
    echo "📊 系统状态报告"
    echo "==============="
    
    # 内存状态
    echo "💻 内存状态:"
    free -h
    
    # GPU状态
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "🖥️  GPU状态:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi
    
    # 磁盘状态
    echo ""
    echo "💾 磁盘状态:"
    df -h / | tail -1
    
    # 进程数量
    echo ""
    echo "🔢 进程统计:"
    echo "  总进程数: $(ps aux | wc -l)"
    echo "  Python进程数: $(pgrep -f python | wc -l)"
    
    echo ""
}

# 主函数
main() {
    # 检查参数
    gpu_only=false
    memory_only=false
    force=false
    
    for arg in "$@"; do
        case $arg in
            --gpu-only)
                gpu_only=true
                ;;
            --memory-only)
                memory_only=true
                ;;
            --force)
                force=true
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --gpu-only    仅清理GPU"
                echo "  --memory-only 仅清理系统内存"
                echo "  --force       强制清理，包括杀死进程"
                echo "  --help        显示帮助"
                exit 0
                ;;
        esac
    done
    
    # 显示初始状态
    show_status
    
    # 检查权限
    check_root
    
    echo ""
    echo "🚀 开始清理..."
    echo ""
    
    # 执行清理
    if [[ "$memory_only" == false ]]; then
        cleanup_gpu "$@"
        echo ""
    fi
    
    if [[ "$gpu_only" == false ]]; then
        cleanup_memory "$@"
        echo ""
        cleanup_python "$@"
        echo ""
    fi
    
    # 等待系统稳定
    echo "⏳ 等待系统稳定..."
    sleep 3
    
    # 显示最终状态
    echo "📋 清理完成，最终状态:"
    show_status
    
    echo "✨ 清理完成！"
}

# 运行主函数
main "$@" 
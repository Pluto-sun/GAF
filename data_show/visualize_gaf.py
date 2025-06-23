import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
import matplotlib.pyplot as plt
import numpy as np
from data_provider.data_loader import ClassificationSegLoader, ClassificationDayWindowLoader
import argparse
import inspect

print(sys.path)

def visualize_gaf_samples(args, save_dir, flag='train', num_per_class=3):
    # 加载数据
    dataset = ClassificationSegLoader(args, flag)
    # 获取原始特征名
    feature_names = list(dataset.scalers.keys())
    # 获取数据和标签
    if flag == 'train':
        data = dataset.train
        labels = dataset.train_labels
    else:
        data = dataset.val
        labels = dataset.val_labels

    data = np.array(data)
    labels = np.array(labels)
    num_classes = len(set(labels))
    np.random.seed(42)

    # 按类别分组索引
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    for class_idx, indices in class_indices.items():
        if len(indices) < num_per_class:
            print(f"类别{class_idx}样本不足{num_per_class}个，仅可视化{len(indices)}个")
        selected = np.random.choice(indices, min(num_per_class, len(indices)), replace=False)
        for num, idx in enumerate(selected):
            sample = data[idx]  # [T, T, 30]
            plt.figure(figsize=(18, 20))
            for ch in range(sample.shape[0]):
                plt.subplot(5, 6, ch + 1)
                plt.imshow(sample[ch, :, :], cmap='gray', vmin=0, vmax=255)
                plt.title(feature_names[ch], fontsize=8)
                plt.axis('off')
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'class_{class_idx}_num{num+1}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"已保存: {save_path}")

# 用法示例（假设你有args对象，且save_dir为保存目录）

args = argparse.Namespace(
    root_path='./dataset/SAHU/direct_5_working',   # 数据集路径，根据你的实际路径修改
    seq_len=96,                   # 窗口长度
    step=96,                      # 滑动步长
    batch_size=32,                # 批量大小
    num_workers=4,                # 数据加载线程数
    data='SAHU',                  # 数据集类型
    gaf_method='difference',       # GAF方法，可选'summation'或'difference'
    test_size=0.3,                # 测试集比例
    val_ratio=0.3,                # 验证集比例（如果用到RawFeatureWindowLoader）
    data_type_method='uint8',
)
visualize_gaf_samples(args, save_dir='./gaf_vis/direct_5_diff/', flag='train', num_per_class=4)
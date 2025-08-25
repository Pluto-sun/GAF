import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_provider.data_loader.DualGAFDataLoader import DualGAFDataLoader

parser = argparse.ArgumentParser(description='可视化DDAHU Summation GAF样本')
parser.add_argument('--root_path', type=str, required=True, help='数据根目录')
parser.add_argument('--seq_len', type=int, default=24, help='窗口长度')
parser.add_argument('--step', type=int, default=1, help='滑动步长')
parser.add_argument('--test_size', type=float, default=0.2, help='验证集比例')
parser.add_argument('--num_samples', type=int, default=3, help='采集样本数')
parser.add_argument('--output_dir', type=str, default='gaf_samples', help='输出目录')
parser.add_argument('--format', type=str, default='svg', choices=['pdf', 'svg'], help='保存格式')
parser.add_argument('--rows', type=int, default=None, help='roll size')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 构造args对象
class Args:
    pass
loader_args = Args()
loader_args.root_path = args.root_path
loader_args.seq_len = args.seq_len
loader_args.step = args.step
loader_args.test_size = args.test_size
loader_args.data_type_method = 'uint8'  # 保证渲染精度
loader_args.use_statistical_features = False
loader_args.rows = args.rows

# 加载数据集
print('加载DualGAFDataLoader...')
dataset = DualGAFDataLoader(loader_args, flag='train')

# 获取信号名
feature_names = list(dataset.data_manager.scalers.keys())
print(f'信号数量: {len(feature_names)}')

# 采集样本
for sample_idx in range(min(args.num_samples, len(dataset))):
    summation_gaf, _, label = dataset[sample_idx]
    # summation_gaf: [num_signals, seq_len, seq_len]
    summation_gaf = summation_gaf.numpy()
    for sig_idx, sig_name in enumerate(feature_names):
        gaf_img = summation_gaf[sig_idx]
        plt.figure(figsize=(16, 16))
        plt.imshow(gaf_img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        plt.axis('off')
        fname = f'{sig_name}_sample{sample_idx}.{args.format}'
        fpath = os.path.join(args.output_dir, fname)
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'保存: {fpath}')
print('全部样本渲染完成。') 
from data_provider.data_loader import ClassificationSegLoader
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import random
data_dict = {
    'SAHU':ClassificationSegLoader,
    'DDAHU':ClassificationSegLoader
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    drop_last = False
    data_set = Data(
        args = args,
        root_path=args.root_path,
        win_size=args.seq_len,
        step=args.step,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

# data_set = ClassificationSegLoader(
#     args = None,
#     root_path='./dataset/SAHU',
#     win_size=64,
#     step=64,
#     flag='train'
# )
# data_loader = DataLoader(
#     data_set,
#     batch_size=64,
#     shuffle=True,
#     num_workers=10,
#     drop_last=True)

# # 配置参数
# NUM_SAMPLES_TO_DISPLAY = 16  # 总展示样本数（4x4网格）
# TARGET_CHANNEL = 11  # 指定展示的通道索引（可修改为任意有效通道）

# # 收集所有样本（从数据加载器中获取足够多的样本）
# all_samples = []
# all_labels = []
# for images, labels in data_loader:
#     # 转换为numpy格式（假设images形状为 [batch_size, time_step, time_step, channels]）
#     batch_samples = images.numpy()  # 整个批次的样本
#     batch_labels = labels.numpy()   # 整个批次的标签
#     all_samples.extend(batch_samples)
#     all_labels.extend(batch_labels)
    
#     # 提前终止：当收集到足够样本时停止（避免内存溢出）
#     if len(all_samples) >= NUM_SAMPLES_TO_DISPLAY:
#         break

# # 随机抽取16个样本（避免重复）
# if len(all_samples) < NUM_SAMPLES_TO_DISPLAY:
#     raise ValueError(f"数据加载器中样本总数不足{NUM_SAMPLES_TO_DISPLAY}个")
# random_indices = random.sample(range(len(all_samples)), NUM_SAMPLES_TO_DISPLAY)
# selected_samples = [all_samples[i] for i in random_indices]
# selected_labels = [all_labels[i] for i in random_indices]

# # 创建4x4网格子图
# plt.figure(figsize=(12, 12))  # 总尺寸（每个子图约3x3英寸，4x4排列）
# for idx, (sample, label) in enumerate(zip(selected_samples, selected_labels)):
#     # 处理通道：提取指定通道（假设通道在最后一维，形状为 [H, W, C]）
#     if sample.ndim == 3:
#         display_image = sample[:, :, TARGET_CHANNEL]  # 提取指定通道
#     elif sample.ndim == 2:
#         display_image = sample  # 单通道直接使用
#     else:
#         raise ValueError("图像维度需为2D（单通道）或3D（H, W, C）")
    
#     if display_image.dtype != np.uint8:
#         display_image = display_image.astype(np.uint8)  # 直接转换，不乘以255
    
#     # 子图布局：4行4列，索引从1开始
#     plt.subplot(4, 4, idx + 1)
#     plt.imshow(display_image, cmap='gray', vmin=0, vmax=255)
#     plt.title(f"Label: {label}", fontsize=8)  # 子图标题显示标签
#     plt.axis('off')  # 关闭坐标轴

# # 整体标题和保存
# plt.suptitle(f"GAF Images - Channel {TARGET_CHANNEL}", fontsize=14, y=0.95)  # 顶部总标题
# plt.tight_layout(pad=3)  # 调整子图间距，避免重叠
# save_path = os.path.join("./dataset/SAHU", f"channel_{TARGET_CHANNEL}_random_16_samples.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"成功保存4x4网格图，包含{NUM_SAMPLES_TO_DISPLAY}个随机样本（通道{TARGET_CHANNEL}）")
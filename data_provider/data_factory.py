from data_provider.data_loader import (
    ClassificationSegLoader,
    ClassificationDayWindowLoader,
    DualGAFDataLoader,
    HVACGraphDataset,
    RawFeatureWindowLoader,
    DualGAFDataLoaderDDAHU,
    DualGAFDataLoaderFCU,
)
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import random

data_dict = {
    "SAHU": ClassificationSegLoader,
    "DDAHU": ClassificationSegLoader,
    "SAHU_day": ClassificationDayWindowLoader,
    "Graph": HVACGraphDataset,
    "DualGAF": DualGAFDataLoader,
    "DualGAF_DDAHU": DualGAFDataLoaderDDAHU,
    "DualGAF_FCU": DualGAFDataLoaderFCU,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == "test" or flag == "TEST") else True
    batch_size = args.batch_size
    
    # 使用args中的drop_last_batch参数，训练时启用，测试时禁用
    drop_last = getattr(args, 'drop_last_batch', True) and flag == "train"
    
    # 安全模式：禁用多线程数据加载以避免内存竞争和双重释放错误
    safe_mode = getattr(args, 'safe_mode', False)
    num_workers = 0 if safe_mode else getattr(args, 'num_workers', 4)
    
    if safe_mode:
        print(f"🛡️ 安全模式启用: num_workers=0, 禁用多线程数据加载")
    
    data_set = Data(
        args=args,
        flag=flag,
    )
    print(flag, len(data_set))
    if isinstance(data_set, HVACGraphDataset):
        data_loader = GeoDataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last,
        )
    return data_set, data_loader


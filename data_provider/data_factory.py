from data_provider.data_loader import (
    ClassificationSegLoader,
    ClassificationDayWindowLoader,
    DualGAFDataLoader,
    HVACGraphDataset,
    RawFeatureWindowLoader,
    DualGAFDataLoaderDDAHU,
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
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == "test" or flag == "TEST") else True
    drop_last = False
    batch_size = args.batch_size

    drop_last = False
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
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
    return data_set, data_loader


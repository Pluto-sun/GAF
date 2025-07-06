from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import pickle
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import hashlib  # 添加哈希库
import os
import torch
from torch_geometric.data import Data
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class RawFeatureWindowLoader(Dataset):
    def __init__(self, args, flag):
        """
        读取原始csv，按文件标签映射，滑动窗口切分，归一化，输出原始特征窗口和标签。
        flag: 'train' 或 'val'，用于切分数据集
        """
        self.args = args
        self.root_path = args.root_path
        self.win_size = args.seq_len
        self.step = args.step
        self.flag = flag
        self.val_ratio = args.val_ratio
        self.seed = args.seed

        # 使用有意义的字符串标签而不是数字
        self.file_label_map = {
            "AHU_annual_resampled_direct_5T_working.csv": "正常状态",
            "coi_stuck_025_resampled_direct_5T_working.csv": "冷却盘管卡死异常",
            "damper_stuck_025_annual_resampled_direct_5T_working.csv": "风阀卡死异常",
            "coi_leakage_050_annual_resampled_direct_5T_working.csv": "冷却盘管泄漏异常",
            "oa_bias_-4_annual_resampled_direct_5T_working.csv": "新风偏置异常",
        }
        
        # 创建标签映射：字符串标签 -> 数字索引
        unique_labels = sorted(set(self.file_label_map.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.scalers = {}
        self.feature_columns = None
        self.labeled_windows = None
        self.labeled_labels = None
        self._load_or_process()
        self._split_data()

    def _load_or_process(self):
        all_segments = []
        all_labels = []
        feature_columns = None
        for file_name, label in self.file_label_map.items():
            file_path = os.path.join(self.root_path, file_name)
            df = pd.read_csv(file_path)
            df = df[df["is_working"] == 1]
            # 去除无关列
            exclude_columns = [
                "Datetime",
                "is_working",
                "ts",
                "date",
                "hour",
                "time_diff",
                "segment_id",
            ]
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            features = df[feature_columns]
            all_segments.append(features.values)
            # 转换字符串标签为数字索引
            numeric_label = self.label_to_idx[label]
            all_labels.append(np.full(len(features), numeric_label))
        X = np.concatenate(all_segments, axis=0)  # [总样本数, 通道数]
        y = np.concatenate(all_labels, axis=0)
        # 归一化
        self.scalers = {}
        for i, col in enumerate(X.T):
            self.scalers[i] = MinMaxScaler(feature_range=(-1, 1))
            self.scalers[i].fit(col.reshape(-1, 1))
            X[:, i] = self.scalers[i].transform(X[:, i].reshape(-1, 1)).flatten()
        # 打乱并划分
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        train_split = int(len(X) * (1 - self.val_ratio))
        if self.flag == "train":
            self.X = X[:train_split]
            self.y = y[:train_split]
        else:
            self.X = X[train_split:]
            self.y = y[train_split:]
        self.labeled_windows = X
        self.labeled_labels = y
        self.feature_columns = list(range(X.shape[1]))
        # 持久化
        if self.persist_path:
            with open(self.persist_path, "wb") as f:
                pickle.dump(
                    {
                        "labeled_windows": self.labeled_windows,
                        "labeled_labels": self.labeled_labels,
                        "scalers": self.scalers,
                        "feature_columns": self.feature_columns,
                    },
                    f,
                )

    def _split_data(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.labeled_windows))
        X = self.labeled_windows[indices]
        y = self.labeled_labels[indices]
        train_split = int(len(X) * (1 - self.val_ratio))
        if self.flag == "train":
            self.X = X[:train_split]
            self.y = y[:train_split]
        else:
            self.X = X[train_split:]
            self.y = y[train_split:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.float32(self.X[idx]), np.float32(self.y[idx])

    def get_train_val(self):
        """
        返回划分好的原始窗口特征和标签，便于传统ML基线。
        """
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.labeled_windows))
        X = self.labeled_windows[indices]
        y = self.labeled_labels[indices]
        train_split = int(len(X) * (1 - self.val_ratio))
        X_train = X[:train_split]
        y_train = y[:train_split]
        X_val = X[train_split:]
        y_val = y[train_split:]
        return (X_train, y_train), (X_val, y_val)

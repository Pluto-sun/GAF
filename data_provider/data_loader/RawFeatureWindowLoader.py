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
    def __init__(self, args, flag, max_samples=None):
        """
        读取原始csv，按文件标签映射，滑动窗口切分，归一化，输出原始特征窗口和标签。
        flag: 'train' 或 'val'，用于切分数据集
        max_samples: 限制总样本数，None为全量
        """
        self.args = args
        self.root_path = args.root_path
        self.win_size = args.win_size
        self.step = args.step
        self.flag = flag
        self.val_ratio = args.val_ratio
        self.seed = 42
        self.max_samples = max_samples

        # 使用有意义的字符串标签而不是数字
        # self.file_label_map = {
        #     "AHU_annual_resampled_direct_5min_working.csv": "AHU_annual",
        #     "coi_bias_-4_annual_resampled_direct_5min_working.csv": "coi_bias_-4",
        #     "coi_bias_4_annual_direct_5min_working.csv": "coi_bias_4",
        #     "oa_bias_4_annual_direct_5min_working.csv": "oa_bias_4", 
        #     "coi_leakage_010_annual_direct_5min_working.csv": "coi_leakage_010",
        #     "coi_stuck_075_annual_resampled_direct_5min_working.csv": "coi_stuck_075",
        #     "coi_stuck_025_annual_direct_5min_working.csv": "coi_stuck_025",
        #     "damper_stuck_075_annual_resampled_direct_5min_working.csv": "damper_stuck_075",
        #     "damper_stuck_025_annual_direct_5min_working.csv": "damper_stuck_025",
        # }
        self.file_label_map = {
            "DualDuct_DMPRStuck_Cold_0_direct_5min_working.csv": "DMPRStuck_Cold_0",
            "DualDuct_DMPRStuck_Cold_50_direct_5min_working.csv": "DMPRStuck_Cold_50",
            "DualDuct_DMPRStuck_Cold_100_direct_5min_working.csv": "DMPRStuck_Cold_100",
            "DualDuct_DMPRStuck_Hot_0_direct_5min_working.csv": "DMPRStuck_Hot_0",
            "DualDuct_DMPRStuck_Hot_50_direct_5min_working.csv": "DMPRStuck_Hot_50",
            "DualDuct_DMPRStuck_Hot_100_direct_5min_working.csv": "DMPRStuck_Hot_100",
            "DualDuct_DMPRStuck_OA_0_direct_5min_working.csv": "DMPRStuck_OA_0",
            "DualDuct_DMPRStuck_OA_45_direct_5min_working.csv": "DMPRStuck_OA_45",
            "DualDuct_DMPRStuck_OA_100_direct_5min_working.csv": "DMPRStuck_OA_100",
            "DualDuct_Fouling_Cooling_Airside_Moderate_direct_5min_working.csv": "Fouling_Cooling_Airside_Moderate",
            "DualDuct_Fouling_Cooling_Airside_Severe_direct_5min_working.csv": "Fouling_Cooling_Airside_Severe",
            "DualDuct_Fouling_Cooling_Waterside_Moderate_direct_5min_working.csv": "Fouling_Cooling_Waterside_Moderate",
            "DualDuct_Fouling_Cooling_Waterside_Severe_direct_5min_working.csv": "Fouling_Cooling_Waterside_Severe",
            "DualDuct_Fouling_Heating_Airside_Moderate_direct_5min_working.csv": "Fouling_Heating_Airside_Moderate",
            "DualDuct_Fouling_Heating_Airside_Severe_direct_5min_working.csv": "Fouling_Heating_Airside_Severe",
            "DualDuct_Fouling_Heating_Waterside_Moderate_direct_5min_working.csv": "Fouling_Heating_Waterside_Moderate",
            "DualDuct_Fouling_Heating_Waterside_Severe_direct_5min_working.csv": "Fouling_Heating_Waterside_Severe",
            "DualDuct_VLVStuck_Cooling_0__direct_5min_working.csv": "VLVStuck_Cooling_0",
            "DualDuct_VLVStuck_Cooling_50__direct_5min_working.csv": "VLVStuck_Cooling_50",
            "DualDuct_VLVStuck_Cooling_100__direct_5min_working.csv": "VLVStuck_Cooling_100",
            "DualDuct_VLVStuck_Heating_0__direct_5min_working.csv": "VLVStuck_Heating_0",
            "DualDuct_VLVStuck_Heating_50__direct_5min_working.csv": "VLVStuck_Heating_50",
            "DualDuct_VLVStuck_Heating_100__direct_5min_working.csv": "VLVStuck_Heating_100",
            "DualDuct_SensorBias_CSA_+4C_direct_5min_working.csv": "SensorBias_CSA_+4C",
            "DualDuct_SensorBias_CSP_+4inwg_direct_5min_working.csv": "SensorBias_CSP_+4inwg",
            "DualDuct_SensorBias_HSA_+4C_direct_5min_working.csv": "SensorBias_HSA_+4C",
            "DualDuct_SensorBias_HSP_+4inwg_direct_5min_working.csv": "SensorBias_HSP_+4inwg",
            "DualDuct_HeatSeqUnstable_direct_5min_working.csv": "HeatSeqUnstable",
            "DualDuct_CoolSeqUnstable_direct_5min_working.csv": "CoolSeqUnstable",
        }
        # 创建标签映射：字符串标签 -> 数字索引
        unique_labels = sorted(set(self.file_label_map.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.scalers = {}
        self.feature_columns = None
        self.labeled_windows = np.empty((0,))
        self.labeled_labels = np.empty((0,))
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
            all_segments.append(np.array(features))
            # 转换字符串标签为数字索引
            numeric_label = self.label_to_idx[label]
            all_labels.append(np.full(len(features), numeric_label))
        X = np.concatenate(all_segments, axis=0)  # [总样本数, 通道数]
        y = np.concatenate(all_labels, axis=0)
        # 限制样本数
        if self.max_samples is not None and len(X) > self.max_samples:
            np.random.seed(self.seed)
            indices = np.random.permutation(len(X))[:self.max_samples]
            X = X[indices]
            y = y[indices]
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
        # if self.persist_path:
        #     with open(self.persist_path, "wb") as f:
        #         pickle.dump(
        #             {
        #                 "labeled_windows": self.labeled_windows,
        #                 "labeled_labels": self.labeled_labels,
        #                 "scalers": self.scalers,
        #                 "feature_columns": self.feature_columns,
        #             },
        #             f,
        #         )

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

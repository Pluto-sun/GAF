import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
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

class ClassificationDayWindowLoader(Dataset):
    def __init__(self, args, flag):
        """
        按天分组后滑窗切分的分类数据集加载器
        Args:
            args: 命令行参数
            root_path: 数据根目录
            win_size: 窗口大小
            step: 滑动步长
            flag: 数据集类型，可选值为'train', 'val'
            file_label_map: 文件名和标签的映射，格式为{文件名: 标签}
            persist_path: 持久化保存路径
        """
        self.flag = flag
        self.step = args.step
        self.win_size = args.seq_len
        self.gaf_method = (
            args.gaf_method if hasattr(args, "gaf_method") else "summation"
        )
        self.root_path = args.root_path

        self.file_label_map = {
            # "AHU_annual_resampled_direct_15T.csv": 0,
            "coi_stuck_025_annual_resampled_direct_15T.csv": 0,
            "damper_stuck_025_annual_resampled_direct_15T.csv": 1,
            "coi_leakage_050_annual_resampled_direct_15T.csv": 2,
            "oa_bias_-4_annual_resampled_direct_15T.csv": 3,
        }
        file_keys = sorted(self.file_label_map.keys())
        file_str = "|".join(file_keys).encode()
        file_hash = hashlib.md5(file_str).hexdigest()
        self._auto_persist_path = os.path.join(
            self.root_path,
            f"classifier_day_win{self.win_size}_step{self.step}_files{len(self.file_label_map)}_{file_hash}_gaf{self.gaf_method}.pkl",
        )
        self.persist_path = self._auto_persist_path
        if os.path.exists(self.persist_path):
            print(f"检测到已存在的持久化文件: {self.persist_path}")
            self.load_persisted_data(self.persist_path)
            return

        def load_and_segment_by_day(path):
            exclude_columns = [
                "Datetime",
                "is_working",
                "ts",
                "date",
                "hour",
                "time_diff",
                "segment_id",
                "SA_TEMPSPT",
                "SF_SPD",
                "SA_SPSPT",
                "OA_CFM",
            ]
            df = pd.read_csv(path)
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df["date"] = df["Datetime"].dt.date
            # 按天分组
            day_groups = df.groupby("date")
            segments = []
            for _, group in day_groups:
                # 只保留特征列
                feature_columns = [
                    col for col in group.columns if col not in exclude_columns
                ]
                group_features = group[feature_columns].values
                # 滑窗切分
                if len(group_features) >= self.win_size:
                    for i in range(
                        0, len(group_features) - self.win_size + 1, self.step
                    ):
                        segments.append(group_features[i : i + self.win_size])
            return segments, feature_columns

        def generate_gaf_matrix(
            data: np.ndarray, method: str = "summation", normalize: bool = False
        ) -> np.ndarray:
            if data.ndim != 3:
                raise ValueError(f"输入数据必须为3维，当前维度数：{data.ndim}")
            N, T, D = data.shape
            valid_methods = {"summation", "difference"}
            if method not in valid_methods:
                raise ValueError(
                    f"method必须为{sorted(valid_methods)}之一，当前输入：{method}"
                )
            transposed_data = data.transpose(0, 2, 1)
            flattened_data = transposed_data.reshape(-1, T)
            gasf = GramianAngularField(method=method)
            batch_gaf = gasf.fit_transform(flattened_data)
            reshaped_gaf = batch_gaf.reshape(N, D, T, T)
            target_gaf = reshaped_gaf.transpose(0, 2, 3, 1)
            return target_gaf.astype(np.float32)

        def gaf_to_float32(data: np.ndarray) -> np.ndarray:
            clipped_data = np.clip(data, -1, 1)
            mapped_data = (clipped_data + 1) / 2 * 255
            float_data = mapped_data.astype(np.float32)
            return float_data

        all_segments = []
        all_labels = []
        feature_columns = None
        print("\n=== 开始加载数据文件（按天分组） ===")
        for i, (file_name, label) in enumerate(self.file_label_map.items()):
            file_path = os.path.join(self.root_path, file_name)
            print(f"\n处理文件 {i+1}/{len(self.file_label_map)}: {file_path}")
            print(f"标签值: {label}")
            segments, cols = load_and_segment_by_day(file_path)
            if not segments:
                print(f"警告: 文件 {file_name} 未包含有效数据段")
                continue
            print(f"成功加载 {len(segments)} 个窗口")
            print(f"特征列数量: {len(cols)}")
            if feature_columns is None:
                feature_columns = cols
            elif set(feature_columns) != set(cols):
                print(f"警告: 文件 {file_name} 的特征列与之前不匹配")
                print(f"当前特征列: {set(cols)}")
                print(f"之前特征列: {set(feature_columns)}")
            for segment in segments:
                all_segments.append(segment)
                all_labels.append(label)
        print(f"\n=== 数据加载完成 ===")
        print(f"总窗口数: {len(all_segments)}")
        print(f"总标签数: {len(all_labels)}")
        print("\n=== 开始通道级别归一化 ===")
        print(f"特征数量: {len(feature_columns)}")
        self.scalers = {}
        for i, col in enumerate(feature_columns):
            print(f"\n处理特征 {i+1}/{len(feature_columns)}: {col}")
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
            feature_data = np.concatenate(
                [seg[:, i].reshape(-1, 1) for seg in all_segments], axis=0
            )

            self.scalers[col].fit(feature_data)
            print(f"特征 {col} 归一化完成")
        print("\n=== 应用归一化到所有窗口 ===")
        for seg_idx in range(len(all_segments)):
            if seg_idx % 100 == 0:
                print(f"处理窗口 {seg_idx+1}/{len(all_segments)}")
            for i, col in enumerate(feature_columns):
                all_segments[seg_idx][:, i] = (
                    self.scalers[col]
                    .transform(all_segments[seg_idx][:, i].reshape(-1, 1))
                    .flatten()
                )
        print("\n=== 转换为numpy数组 ===")
        labeled_windows = np.array(all_segments)
        labeled_labels = np.array(all_labels)
        print(f"窗口数据形状: {labeled_windows.shape}")
        print(f"标签数据形状: {labeled_labels.shape}")
        if len(labeled_windows) == 0:
            raise ValueError("未能生成任何有效的窗口")
        print("\n=== 打乱数据 ===")
        np.random.seed(42)
        indices = np.random.permutation(len(labeled_windows))
        labeled_windows = labeled_windows[indices]
        labeled_labels = labeled_labels[indices]
        print("\n=== 开始GAF转换 ===")
        print(f"输入数据形状: {labeled_windows.shape}")
        gaf_data = generate_gaf_matrix(labeled_windows, self.gaf_method, False)
        print(f"GAF转换后数据形状: {gaf_data.shape}")
        print("\n=== 开始数据范围转换 ===")
        gaf_data = gaf_to_float32(gaf_data)
        print(f"数据范围: [{gaf_data.min():.2f}, {gaf_data.max():.2f}]")
        gaf_data = gaf_data.transpose(0, 3, 1, 2)
        train_split = int(len(gaf_data) * 0.7)
        self.train = gaf_data[:train_split]
        self.train_labels = labeled_labels[:train_split]
        self.val = gaf_data[train_split:]
        self.val_labels = labeled_labels[train_split:]
        print("\n=== 数据集划分完成 ===")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")
        print("\n=== 保存预处理数据 ===")
        self.persist_data(self.persist_path, labeled_windows, labeled_labels)
        print(f"已自动保存预处理数据到: {self.persist_path}")

    def load_persisted_data(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        required_keys = ["train", "val", "train_labels", "val_labels", "scalers"]
        if not all(key in data for key in required_keys):
            raise ValueError("持久化文件数据格式不完整，可能版本不兼容")
        self.train = data["train"]
        self.val = data["val"]
        self.train_labels = data["train_labels"]
        self.val_labels = data["val_labels"]
        self.scalers = data["scalers"]
        print(f"从 {path} 加载数据完成")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")

    def persist_data(self, path, labeled_windows, labeled_labels):
        data = {
            "train": self.train,
            "val": self.val,
            "train_labels": self.train_labels,
            "val_labels": self.val_labels,
            "scalers": self.scalers,
            "win_size": self.win_size,
            "step": self.step,
            "file_map": self.file_label_map,
            "gaf_method": self.gaf_method,
            "labeled_windows": labeled_windows,
            "labeled_labels": labeled_labels,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"数据持久化保存到 {path} 完成")

    def __len__(self):
        return {"train": len(self.train), "val": len(self.val)}[self.flag]

    def __getitem__(self, index):
        if self.flag == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.val[index]), np.float32(self.val_labels[index])


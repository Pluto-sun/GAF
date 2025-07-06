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

warnings.filterwarnings("ignore")

class HVACGraphDataset(Dataset):
    """HVAC异常检测图数据集 - 基于变量关联的静态图方法"""

    def __init__(self, args, flag, correlation_threshold=0.3, random_state=42):
        self.flag = flag
        self.args = args
        self.root_path = args.root_path
        self.correlation_threshold = correlation_threshold
        self.sample_size = args.sample_size
        self.random_state = random_state
        self.data_list = []
        self.labels = []
        self.feature_names = None
        self.global_adj_matrix = None
        self.num_classes = None
        self.scaler = StandardScaler()
        self.test_size = args.test_size
        self.normalize_on = "normal"
        self._load_and_process_data()
        self._split_indices(self.test_size)

    def _load_and_process_data(self):
        print("正在加载数据文件...")
        csv_files = [f for f in os.listdir(self.root_path) if f.endswith(".csv")]
        csv_files.sort()
        self.num_classes = len(csv_files)
        print(f"找到 {self.num_classes} 个CSV文件")
        normal_file = csv_files[0]
        normal_file_path = os.path.join(self.root_path, normal_file)
        print(f"用于图结构构建的正常数据文件: {normal_file}")
        try:
            df_normal = pd.read_csv(normal_file_path)
            if "is_working" in df_normal.columns:
                df_normal = df_normal[df_normal["is_working"] == 1]
            exclude_columns = [
                "Datetime",
                "is_working",
                "ts",
                "date",
                "hour",
                "time_diff",
                "segment_id",
            ]
            df_normal = df_normal.drop(columns=exclude_columns, errors="ignore")
            df_normal = df_normal.fillna(df_normal.mean())
            self.feature_names = df_normal.columns.tolist()
            print(f"正常数据特征列: {self.feature_names}")
            self._compute_global_adjacency(df_normal.values)
        except Exception as e:
            print(f"处理正常数据文件 {normal_file} 时出错: {e}")
            raise e
        all_data = []
        all_labels = []
        for i, file in enumerate(csv_files):
            file_path = os.path.join(self.root_path, file)
            print(f"处理文件: {file}")
            try:
                df = pd.read_csv(file_path)
                if "is_working" in df.columns:
                    df = df[df["is_working"] == 1]
                df = df.drop(columns=exclude_columns, errors="ignore")
                df = df[self.feature_names]
                df = df.fillna(df.mean())
                if self.sample_size is not None and len(df) > self.sample_size:
                    sampled_df = df.sample(n=self.sample_size, random_state=42)
                else:
                    sampled_df = df
                all_data.append(sampled_df.values)
                all_labels.extend([i] * len(sampled_df))
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
                continue
        combined_data = np.vstack(all_data)
        combined_labels = np.array(all_labels)
        print(f"总数据形状: {combined_data.shape}")
        print(f"类别分布: {np.bincount(combined_labels)}")
        print("正在进行数据标准化...")

        # 新增：根据设置选择归一化数据
        if self.normalize_on == "normal":
            print("使用正常数据进行归一化")
            norm_data = df_normal.values
        else:
            print("使用所有数据进行归一化")
            norm_data = combined_data

        combined_data = self.scaler.fit(norm_data).transform(combined_data)
        print("正在创建图数据...")
        self._create_graph_data(combined_data, combined_labels)
        print(f"创建了 {len(self.data_list)} 个图样本")
        print(f"self.data_list形式: {self.data_list[0]}")

    def _compute_global_adjacency(self, data):
        corr_matrix = np.corrcoef(data.T)
        corr_matrix = np.nan_to_num(corr_matrix)
        self.global_adj_matrix = (
            np.abs(corr_matrix) > self.correlation_threshold
        ).astype(float)
        np.fill_diagonal(self.global_adj_matrix, 0)
        if np.sum(self.global_adj_matrix) == 0:
            print("警告：相关性阈值过高，创建全连接图")
            self.global_adj_matrix = np.ones_like(corr_matrix) - np.eye(
                len(corr_matrix)
            )
        print(
            f"邻接矩阵密度: {np.sum(self.global_adj_matrix) / (self.global_adj_matrix.shape[0] ** 2 - self.global_adj_matrix.shape[0]):.3f}"
        )
        self.correlation_matrix = corr_matrix

    def _create_graph_data(self, data, labels):
        num_features = data.shape[1]
        edge_index = []
        edge_attr = []
        for i in range(num_features):
            for j in range(num_features):
                if self.global_adj_matrix[i, j] > 0:
                    edge_index.append([i, j])
                    edge_attr.append(self.correlation_matrix[i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        for i in range(len(data)):
            x = torch.tensor(data[i].reshape(-1, 1), dtype=torch.float)
            y = torch.tensor(labels[i], dtype=torch.long)
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            self.data_list.append(graph_data)
            self.labels.append(labels[i])

    def _split_indices(self, test_size):
        indices = np.arange(len(self.data_list))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.labels,
        )
        if self.flag == "train":
            self.indices = train_idx
        else:
            self.indices = val_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data_list[real_idx]

    def visualize_graph_structure(self, save_path="graph_structure.png"):
        """可视化图结构"""
        plt.figure(figsize=(15, 12))

        # 绘制相关性矩阵
        plt.subplot(2, 2, 1)
        sns.heatmap(
            self.correlation_matrix,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Feature Correlation Matrix", pad=20)

        # 绘制邻接矩阵
        plt.subplot(2, 2, 2)
        sns.heatmap(
            self.global_adj_matrix, cmap="Blues", cbar_kws={"label": "Connection"}
        )
        plt.title("Graph Adjacency Matrix", pad=20)

        # 度分布
        plt.subplot(2, 2, 3)
        degrees = np.sum(self.global_adj_matrix, axis=1)
        sns.histplot(degrees, bins=20, kde=True)
        plt.xlabel("Node Degree")
        plt.ylabel("Frequency")
        plt.title("Node Degree Distribution", pad=20)

        # 图统计信息
        plt.subplot(2, 2, 4)
        num_nodes = self.global_adj_matrix.shape[0]
        num_edges = np.sum(self.global_adj_matrix) // 2  # 无向图
        density = num_edges / (num_nodes * (num_nodes - 1) / 2)

        stats_text = f"""
        Graph Statistics:
        - Nodes: {num_nodes}
        - Edges: {num_edges}
        - Density: {density:.3f}
        - Avg Degree: {np.mean(degrees):.2f}
        """
        plt.text(
            0.1,
            0.5,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="center",
        )
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
import pickle
from pyts.image import GramianAngularField
import hashlib

warnings.filterwarnings("ignore")


class ClassificationSegDataManager:
    """
    分类数据管理器
    负责数据的加载、处理和持久化，只处理一次数据
    """
    _instances = {}  # 类级别的实例缓存
    
    def __new__(cls, args):
        # 基于关键参数生成唯一标识
        key = (
            args.root_path,
            args.seq_len,
            args.step,
            args.test_size,
            getattr(args, 'gaf_method', 'summation'),
            getattr(args, 'data_type_method', 'float32')
        )
        
        # 如果已存在相同配置的实例，直接返回
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance._initialized = False
        return cls._instances[key]
    
    def __init__(self, args):
        # 避免重复初始化
        if self._initialized:
            return
            
        self.args = args
        self.root_path = args.root_path
        self.step = args.step
        self.win_size = args.seq_len
        self.gaf_method = getattr(args, "gaf_method", "summation")
        self.test_size = args.test_size
        
        # 数据类型转换方法选择
        self.data_type_method = getattr(args, 'data_type_method', 'float32')
        print(f"使用数据类型转换方法: {self.data_type_method}")

        # 文件标签映射
        self.file_label_map = {
            "coi_bias_-4_annual_resampled_direct_5min_working.csv": "coi_bias_-4",
            "coi_leakage_050_annual_resampled_direct_5min_working.csv": "coi_leakage_050", 
            "coi_stuck_075_annual_resampled_direct_5min_working.csv": "coi_stuck_075",
            "damper_stuck_075_annual_resampled_direct_5min_working.csv": "damper_stuck_075",
            "oa_bias_-4_annual_resampled_direct_5min_working.csv": "oa_bias_-4",
        }
        
        # 创建标签映射
        unique_labels = sorted(set(self.file_label_map.values()))
        self.args.num_class = len(unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"标签映射: {self.label_to_idx}")
        
        # 生成持久化文件路径
        file_keys = sorted(self.file_label_map.keys())
        file_str = "|".join(file_keys).encode()
        file_hash = hashlib.md5(file_str).hexdigest()
        self.persist_path = os.path.join(
            self.root_path,
            f"classifier_data_win{self.win_size}_step{self.step}_files{len(self.file_label_map)}_{file_hash}_gaf{self.gaf_method}_dtype{self.data_type_method}.pkl",
        )

        # 初始化数据存储
        self.train = None
        self.train_labels = None
        self.val = None
        self.val_labels = None
        self.scalers = None
        
        # 加载或处理数据
        if os.path.exists(self.persist_path):
            print(f"检测到已存在的持久化文件: {self.persist_path}")
            self.load_persisted_data(self.persist_path)
        else:
            self._process_data()
        
        self._initialized = True
        print(f"分类数据管理器初始化完成")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")

    def _process_data(self):
        """处理数据的主要逻辑"""
        def load_and_segment(path, rows=None):
            exclude_columns = [
                "Datetime", "is_working", "ts", "date", "hour", "time_diff", "segment_id",
            ]

            if rows:
                df = pd.read_csv(path, nrows=rows)
            else:
                df = pd.read_csv(path)
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            # 生成时间序列特征
            df["ts"] = df["Datetime"].astype("int64") // 10**9
            df["date"] = df["Datetime"].dt.date
            df["hour"] = df["Datetime"].dt.hour

            # 使用is_working列过滤工作时间
            df = df[df["is_working"] == 1]

            # 分段逻辑：连续工作时间段识别
            df = df.sort_values("ts").reset_index(drop=True)
            df["time_diff"] = df["ts"].diff() > 3600 * 2  # 两小时不连续视为新时段
            df["segment_id"] = df["time_diff"].cumsum()

            # 获取特征列
            feature_columns = [
                col for col in df.columns
                if col not in [
                    "Datetime", "ts", "date", "hour", "time_diff", "segment_id", "is_working",
                    "SA_TEMPSPT", "SYS_CTL", "RF_SPD_DM", "SF_SPD_DM",
                ]
            ]

            # 提取所有工作时段数据
            segments = []
            for seg_id, group in df.groupby("segment_id"):
                if len(group) >= self.win_size:
                    segment_data = group[feature_columns].values
                    segments.append(segment_data)

            return segments, feature_columns

        def create_segment_windows(segments):
            all_windows = []
            for seg in segments:
                if len(seg) == self.win_size:
                    all_windows.append(seg)
                else:
                    for i in range(0, len(seg) - self.win_size + 1, self.step):
                        all_windows.append(seg[i : i + self.win_size])
            return np.array(all_windows) if len(all_windows) > 0 else np.array([])

        def generate_gaf_matrix(data: np.ndarray, method: str = "summation") -> np.ndarray:
            """GAF矩阵生成函数"""
            if data.ndim != 3:
                raise ValueError(f"输入数据必须为3维，当前维度数：{data.ndim}")

            N, T, D = data.shape
            valid_methods = {"summation", "difference"}
            if method not in valid_methods:
                raise ValueError(f"method必须为{sorted(valid_methods)}之一")

            transposed_data = data.transpose(0, 2, 1)
            flattened_data = transposed_data.reshape(-1, T)
            gasf = GramianAngularField(method=method)
            batch_gaf = gasf.fit_transform(flattened_data)
            reshaped_gaf = batch_gaf.reshape(N, D, T, T)
            return reshaped_gaf.astype(np.float32)

        def convert_gaf_data_type(data: np.ndarray) -> np.ndarray:
            """根据配置转换GAF数据类型"""
            print(f"处理GAF数据转换，数据形状: {data.shape}, 内存占用: {data.nbytes / 1024**3:.2f} GB")
            
            if self.data_type_method == 'float32':
                return self._gaf_to_float32(data)
            elif self.data_type_method == 'uint8':
                return self._gaf_to_int(data, dtype=np.uint8)
            elif self.data_type_method == 'uint16':
                return self._gaf_to_int(data, dtype=np.uint16)
            else:
                raise ValueError(f"不支持的数据类型方法: {self.data_type_method}")

        # 收集所有文件的数据和标签
        all_segments = []
        all_labels = []
        feature_columns = None

        print("\n=== 开始加载数据文件 ===")
        for i, (file_name, label) in enumerate(self.file_label_map.items()):
            file_path = os.path.join(self.root_path, file_name)
            print(f"\n处理文件 {i+1}/{len(self.file_label_map)}: {file_path}")
            print(f"标签值: {label}")

            segments, cols = load_and_segment(file_path, None)

            if not segments:
                print(f"警告: 文件 {file_name} 未包含有效数据段")
                continue

            print(f"成功加载 {len(segments)} 个数据段")
            print(f"特征列数量: {len(cols)}")

            if feature_columns is None:
                feature_columns = cols
            elif set(feature_columns) != set(cols):
                print(f"警告: 文件 {file_name} 的特征列与之前不匹配")

            # 为每个段添加对应标签
            numeric_label = self.label_to_idx[label]
            for segment in segments:
                all_segments.append(segment)
                all_labels.append(numeric_label)

        print(f"\n=== 数据加载完成 ===")
        print(f"总数据段数: {len(all_segments)}")

        # 通道级别归一化
        print("\n=== 开始通道级别归一化 ===")
        self.scalers = {}
        for i, col in enumerate(feature_columns):
            print(f"\n处理特征 {i+1}/{len(feature_columns)}: {col}")
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
            feature_data = np.concatenate(
                [seg[:, i].reshape(-1, 1) for seg in all_segments], axis=0
            )
            self.scalers[col].fit(feature_data)

        # 应用归一化
        print("\n=== 应用归一化到所有数据段 ===")
        for seg_idx in range(len(all_segments)):
            if seg_idx % 100 == 0:
                print(f"处理数据段 {seg_idx+1}/{len(all_segments)}")
            for i, col in enumerate(feature_columns):
                all_segments[seg_idx][:, i] = (
                    self.scalers[col]
                    .transform(all_segments[seg_idx][:, i].reshape(-1, 1))
                    .flatten()
                )

        # 创建窗口
        print("\n=== 开始创建时间窗口 ===")
        labeled_windows = []
        labeled_labels = []

        for seg_idx, (seg, label) in enumerate(zip(all_segments, all_labels)):
            if seg_idx % 100 == 0:
                print(f"处理数据段 {seg_idx+1}/{len(all_segments)}")
            windows = create_segment_windows([seg])
            if len(windows) > 0:
                for window in windows:
                    labeled_windows.append(window)
                    labeled_labels.append(label)

        print(f"\n=== 窗口创建完成 ===")
        print(f"生成的窗口数量: {len(labeled_windows)}")

        # 转换为numpy数组
        labeled_windows = np.array(labeled_windows)
        labeled_labels = np.array(labeled_labels)

        if len(labeled_windows) == 0:
            raise ValueError("未能生成任何有效的时间窗口")

        # 打乱数据
        print("\n=== 打乱数据 ===")
        np.random.seed(42)
        indices = np.random.permutation(len(labeled_windows))
        labeled_windows = labeled_windows[indices]
        labeled_labels = labeled_labels[indices]

        # GAF转换
        print("\n=== 开始GAF转换 ===")
        print(f"输入数据形状: {labeled_windows.shape}")
        gaf_data = generate_gaf_matrix(labeled_windows, self.gaf_method)
        print(f"GAF转换后数据形状: {gaf_data.shape}")

        # 数据类型转换
        print("\n=== 开始数据范围转换 ===")
        gaf_data = convert_gaf_data_type(gaf_data)

        print(f"转换后数据范围: [{gaf_data.min()}, {gaf_data.max()}]")
        print(f"转换后数据类型: {gaf_data.dtype}")

        # 计算划分点
        train_split = int(len(gaf_data) * (1 - self.test_size))
        
        # 划分数据集
        self.train = gaf_data[:train_split]
        self.train_labels = labeled_labels[:train_split]
        self.val = gaf_data[train_split:]
        self.val_labels = labeled_labels[train_split:]

        print("\n=== 数据集划分完成 ===")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")

        # 保存预处理数据
        print("\n=== 保存预处理数据 ===")
        self.persist_data(self.persist_path, labeled_windows, labeled_labels)
        print(f"已自动保存预处理数据到: {self.persist_path}")

    def _gaf_to_float32(self, data: np.ndarray) -> np.ndarray:
        """Float32转换方法"""
        batch_size = 1000
        if data.shape[0] <= batch_size:
            return self._process_batch_float32(data)
        else:
            print(f"使用分批处理，批大小: {batch_size}")
            results = []
            for i in range(0, data.shape[0], batch_size):
                end_idx = min(i + batch_size, data.shape[0])
                batch = data[i:end_idx]
                if i % (batch_size * 5) == 0:
                    print(f"处理进度: {i}/{data.shape[0]} ({100*i/data.shape[0]:.1f}%)")
                batch_result = self._process_batch_float32(batch)
                results.append(batch_result)
            return np.concatenate(results, axis=0)

    def _process_batch_float32(self, batch_data: np.ndarray) -> np.ndarray:
        """Float32批处理"""
        data_min, data_max = batch_data.min(), batch_data.max()
        if data_min >= -1.0 and data_max <= 1.0:
            if batch_data.dtype != np.float32:
                result = batch_data.astype(np.float32)
            else:
                result = batch_data.copy()
            result += 1.0
            result *= 127.5
        else:
            result = np.clip(batch_data, -1, 1, dtype=np.float32)
            result += 1.0
            result *= 127.5
        return result

    def _gaf_to_int(self, data: np.ndarray, dtype=np.uint8) -> np.ndarray:
        """整数转换方法"""
        if dtype == np.uint8:
            max_val = 255
        elif dtype == np.uint16:
            max_val = 65535
        else:
            raise ValueError(f"不支持的数据类型: {dtype}")
        
        batch_size = 1000
        if data.shape[0] <= batch_size:
            return self._process_batch_int(data, dtype, max_val)
        else:
            print(f"使用分批处理，批大小: {batch_size}")
            results = []
            for i in range(0, data.shape[0], batch_size):
                end_idx = min(i + batch_size, data.shape[0])
                batch = data[i:end_idx]
                if i % (batch_size * 5) == 0:
                    print(f"处理进度: {i}/{data.shape[0]} ({100*i/data.shape[0]:.1f}%)")
                batch_result = self._process_batch_int(batch, dtype, max_val)
                results.append(batch_result)
            return np.concatenate(results, axis=0)

    def _process_batch_int(self, batch_data: np.ndarray, dtype, max_val: int) -> np.ndarray:
        """整数批处理"""
        data_min, data_max = batch_data.min(), batch_data.max()
        if data_min >= -1.0 and data_max <= 1.0:
            normalized = (batch_data.astype(np.float64) + 1.0) * (max_val / 2.0)
        else:
            clipped = np.clip(batch_data, -1, 1)
            normalized = (clipped.astype(np.float64) + 1.0) * (max_val / 2.0)
        result = np.round(normalized).astype(dtype)
        return result

    def load_persisted_data(self, path):
        """从文件加载预处理好的数据"""
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
        
        # 加载标签映射
        if "label_to_idx" in data and "idx_to_label" in data:
            self.label_to_idx = data["label_to_idx"]
            self.idx_to_label = data["idx_to_label"]
        else:
            print("警告：持久化文件中没有标签映射，正在重新生成...")
            unique_labels = sorted(set(self.file_label_map.values()))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 加载数据类型方法
        if "data_type_method" in data:
            saved_method = data["data_type_method"]
            if saved_method != self.data_type_method:
                print(f"警告：持久化文件的数据类型方法({saved_method})与当前设置({self.data_type_method})不匹配")
                self.data_type_method = saved_method
        
        print(f"从 {path} 加载数据完成")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")

    def persist_data(self, path, labeled_windows, labeled_labels):
        """持久化保存预处理好的数据"""
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
            "data_type_method": self.data_type_method,
            "labeled_windows": labeled_windows,
            "labeled_labels": labeled_labels,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"数据持久化保存到 {path} 完成")


class ClassificationSegLoader(Dataset):
    """
    分类数据集视图
    轻量级的Dataset包装器，不重复处理数据
    """
    
    def __init__(self, args, flag):
        """
        Args:
            args: 命令行参数
            flag: 数据集类型，'train' 或 'val'
        """
        self.flag = flag
        self.data_manager = ClassificationSegDataManager(args)
        
        # 根据flag选择对应的数据
        if flag == "train":
            self.data = self.data_manager.train
            self.labels = self.data_manager.train_labels
        elif flag == "val":
            self.data = self.data_manager.val
            self.labels = self.data_manager.val_labels
        else:
            raise ValueError(f"不支持的flag值: {flag}，应为'train'或'val'")
        
        print(f"创建{flag}数据集视图，包含{len(self)}个样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """返回data, label两个值"""
        data = self.data[index]
        label = self.labels[index]
        
        # 如果使用整数类型存储，需要转换回浮点数进行训练
        if self.data_manager.data_type_method in ['uint8', 'uint16']:
            data = data.astype(np.float32)
        
        return data.astype(np.float32), np.float32(label)
    
    @property
    def label_to_idx(self):
        """获取标签映射"""
        return self.data_manager.label_to_idx
    
    @property
    def idx_to_label(self):
        """获取索引到标签的映射"""
        return self.data_manager.idx_to_label


"""
使用示例：

# 1. 设置参数
class Args:
    def __init__(self):
        self.root_path = './dataset/SAHU'
        self.seq_len = 72
        self.step = 12
        self.test_size = 0.3
        self.gaf_method = 'summation'  # 或 'difference'
        self.data_type_method = 'uint8'  # 或 'float32', 'uint16'
        self.batch_size = 32
        self.num_workers = 4

args = Args()

# 2. 使用data_factory创建数据集和数据加载器
from data_provider.data_factory import data_provider

# 创建训练集（第一次调用时会处理数据）
train_dataset, train_loader = data_provider(args, flag='train')
print(f"训练集样本数: {len(train_dataset)}")

# 创建验证集（复用已处理的数据，无冗余）
val_dataset, val_loader = data_provider(args, flag='val')
print(f"验证集样本数: {len(val_dataset)}")

# 3. 在训练循环中使用
for epoch in range(num_epochs):
    for data, labels in train_loader:
        # data: [batch_size, channels, height, width] - GAF图像
        # labels: [batch_size] - 分类标签
        
        # 网络处理
        outputs = model(data)
        
        # 计算损失和反向传播
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 4. 访问标签映射
print("标签映射:", train_dataset.label_to_idx)
print("索引到标签:", train_dataset.idx_to_label)

特点：
- 只处理一次数据：使用单例模式，相同配置的数据管理器只创建一次
- GAF转换：支持summation和difference两种GAF方法
- 内存优化：支持uint8/uint16数据类型，显著减少内存占用
- 持久化缓存：自动保存和加载预处理结果
- 轻量级视图：train和val数据集只是数据管理器的轻量级包装
- 时间段分割：基于工作时间和时间连续性进行智能分段
- 滑动窗口：支持可配置的窗口大小和步长
""" 
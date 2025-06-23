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


class ClassificationSegLoader(Dataset):
    def __init__(self, args, flag):
        """
        初始化分类数据集加载器

        Args:
            args: 命令行参数
            root_path: 数据根目录
            win_size: 窗口大小
            file_label_map: 文件名和标签的映射，格式为{文件名: 标签}，第一个为正常数据
            flag: 数据集类型，可选值为'train', 'val', 'test'
            persist_path: 持久化保存路径，如果提供则保存或加载处理好的数据
        """
        self.flag = flag
        self.args = args
        self.root_path = args.root_path
        self.step = args.step
        self.win_size = args.seq_len
        self.gaf_method = (
            args.gaf_method if hasattr(args, "gaf_method") else "summation"
        )  # 新增：从args获取GAF方法
        self.test_size = args.test_size
        
        # 新增：数据类型转换方法选择
        self.data_type_method = getattr(args, 'data_type_method', 'float32')  # 'float32', 'uint8', 'uint16'
        print(f"使用数据类型转换方法: {self.data_type_method}")

        # 使用有意义的字符串标签而不是数字
        self.file_label_map = {
            "coi_bias_-4_annual_resampled_direct_5min_working.csv": "coi_bias_-4",
            "coi_leakage_050_annual_resampled_direct_5min_working.csv": "coi_leakage_050", 
            "coi_stuck_075_annual_resampled_direct_5min_working.csv": "coi_stuck_075",
            "damper_stuck_075_annual_resampled_direct_5min_working.csv": "damper_stuck_075",
            "oa_bias_-4_annual_resampled_direct_5min_working.csv": "oa_bias_-4",
        }
        
        # 创建标签映射：字符串标签 -> 数字索引
        unique_labels = sorted(set(self.file_label_map.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"标签映射: {self.label_to_idx}")
        # 生成持久化文件名标识
        file_keys = sorted(self.file_label_map.keys())
        file_str = "|".join(file_keys).encode()
        file_hash = hashlib.md5(file_str).hexdigest()  # 使用MD5哈希
        self._auto_persist_path = os.path.join(
            self.root_path,
            f"classifier_data_win{self.win_size}_step{self.step}_files{len(self.file_label_map)}_{file_hash}_gaf{self.gaf_method}_dtype{self.data_type_method}.pkl",  # 修改：在文件名中包含数据类型方法
        )

        # 优先使用用户指定路径，否则使用自动生成路径
        self.persist_path = self._auto_persist_path
        # 如果持久化文件存在，直接加载
        if os.path.exists(self.persist_path):
            print(f"检测到已存在的持久化文件: {self.persist_path}")
            self.load_persisted_data(self.persist_path)
            return

        # 增强型数据加载器
        def load_and_segment(path, rows=None, skip_normalize=True):
            # 定义要排除的列
            exclude_columns = [
                "Datetime",
                "is_working",
                "ts",
                "date",
                "hour",
                "time_diff",
                "segment_id",
            ]

            if rows:
                df = pd.read_csv(
                    path,
                    # usecols=lambda c: c not in exclude_columns,
                    nrows=rows,
                )
            else:
                df = pd.read_csv(
                    path,
                    # usecols=lambda c: c not in exclude_columns
                )
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            # 生成时间序列特征
            df["ts"] = df["Datetime"].astype("int64") // 10**9
            df["date"] = df["Datetime"].dt.date
            df["hour"] = df["Datetime"].dt.hour

            # 使用is_working列过滤工作时间
            df = df[df["is_working"] == 1]

            # 分段逻辑：连续工作时间段识别
            df = df.sort_values("ts").reset_index(drop=True)

            # 计算is_working的变化点或者时间间隔过大的点
            df["time_diff"] = df["ts"].diff() > 3600 * 2  # 两小时不连续视为新时段

            # 生成时间段ID
            df["segment_id"] = df["time_diff"].cumsum()

            # 获取特征列（非时间相关列）
            feature_columns = [
                col
                for col in df.columns
                if col
                not in [
                    "Datetime",
                    "ts",
                    "date",
                    "hour",
                    "time_diff",
                    "segment_id",
                    "is_working",
                    "SA_TEMPSPT",
                    "SYS_CTL",
                    "RF_SPD_DM",
                    "SF_SPD_DM",
                ]
            ]

            # 提取所有工作时段数据
            segments = []
            for seg_id, group in df.groupby("segment_id"):
                # 确保最小长度满足窗口要求
                if len(group) >= self.win_size:
                    # 只保留特征列数据
                    segment_data = group[feature_columns].values
                    segments.append(segment_data)

            return segments, feature_columns

        # 分段窗口生成器
        def create_segment_windows(segments):
            all_windows = []
            for seg in segments:
                # 单段内滑动窗口处理
                if len(seg) == self.win_size:
                    all_windows.append(seg)
                else:
                    # 单段内滑动窗口处理
                    for i in range(0, len(seg) - self.win_size + 1, self.step):
                        all_windows.append(seg[i : i + self.win_size])
            return np.array(all_windows) if len(all_windows) > 0 else np.array([])

        def generate_gaf_matrix(
            data: np.ndarray, method: str = "summation", normalize: bool = False
        ) -> np.ndarray:
            """
            将多维时间序列转换为Gramian角场(GAF)矩阵

            参数:
            - data: 输入数据（形状[N, T, D]，N=样本数，T=时间步，D=维度数）
            - method: GAF方法，可选"summation"（和）或"difference"（差），默认"summation"
            - normalize: 是否使用pyts内置归一化（若数据已在[-1, 1]，设为False）

            返回:
            - gaf_data: GAF矩阵（形状[N, T, T, D]，数据类型np.float32）
            """
            # 1. 输入维度检查
            if data.ndim != 3:
                raise ValueError(
                    f"输入数据必须为3维，当前维度数：{data.ndim}，正确形状应为[N, T, D]"
                )

            N, T, D = data.shape  # 提取维度尺寸
            valid_methods = {"summation", "difference"}
            if method not in valid_methods:
                raise ValueError(
                    f"method必须为{sorted(valid_methods)}之一，当前输入：{method}"
                )

            # 2. 调整维度顺序为[N, D, T]（pyts要求时间步在最后一维）
            transposed_data = data.transpose(0, 2, 1)  # 形状[N, D, T]

            # 3. 展开为单维度时间序列批量输入格式[N*D, T]
            flattened_data = transposed_data.reshape(-1, T)  # 形状[N*D, T]

            # 4. 初始化GAF并生成矩阵
            gasf = GramianAngularField(method=method)
            batch_gaf = gasf.fit_transform(flattened_data)  # 形状[N*D, T, T]

            # 5. 重组维度为[N, D, T, T]
            reshaped_gaf = batch_gaf.reshape(N, D, T, T)

            # # 6. 调整维度顺序为目标形状[N, T, T, D]
            # target_gaf = reshaped_gaf.transpose(0, 2, 3, 1)

            # 7. 转换为深度学习友好的float32类型
            return reshaped_gaf.astype(np.float32)

        def gaf_to_float32(data: np.ndarray) -> np.ndarray:
            """
            将四维数组中的每个二维矩阵（值范围[-1, 1]）映射到[0, 255]并转换为浮点数（保留小数精度）
            针对大规模数据集优化版本

            参数:
            - data: 输入数组（形状[N, T, T, D]，每个元素值需在[-1, 1]范围内）

            返回:
            - float_data: 转换后的浮点数数组（形状相同，数据类型为np.float32，值范围[0.0, 255.0]）
            """
            print(f"处理GAF数据转换，数据形状: {data.shape}, 内存占用: {data.nbytes / 1024**3:.2f} GB")
            
            # 针对大规模数据的分批处理优化
            batch_size = 1000  # 每批处理1000个样本，减少内存峰值
            
            if data.shape[0] <= batch_size:
                # 小数据集，直接处理
                return _process_batch_optimized(data)
            else:
                # 大数据集，分批处理
                print(f"使用分批处理，批大小: {batch_size}")
                results = []
                
                for i in range(0, data.shape[0], batch_size):
                    end_idx = min(i + batch_size, data.shape[0])
                    batch = data[i:end_idx]
                    
                    if i % (batch_size * 5) == 0:  # 每5个批次打印一次进度
                        print(f"处理进度: {i}/{data.shape[0]} ({100*i/data.shape[0]:.1f}%)")
                    
                    batch_result = _process_batch_optimized(batch)
                    results.append(batch_result)
                
                print("合并分批结果...")
                return np.concatenate(results, axis=0)
        
        def _process_batch_optimized(batch_data: np.ndarray) -> np.ndarray:
            """
            优化的批处理函数
            """
            # 1. 使用原地操作减少内存分配
            # 检查数据范围，避免不必要的clip
            data_min, data_max = batch_data.min(), batch_data.max()
            
            if data_min >= -1.0 and data_max <= 1.0:
                # 数据在范围内，直接转换（原地操作）
                if batch_data.dtype != np.float32:
                    result = batch_data.astype(np.float32)
                else:
                    result = batch_data.copy()
                result += 1.0
                result *= 127.5
            else:
                # 需要裁剪
                result = np.clip(batch_data, -1, 1, dtype=np.float32)
                result += 1.0
                result *= 127.5
            
            return result

        def gaf_to_int(data: np.ndarray, dtype=np.uint8) -> np.ndarray:
            """
            将四维数组中的每个二维矩阵（值范围[-1, 1]）映射到[0, 255]并转换为整数类型
            针对大规模数据集优化版本，使用整数存储节省内存

            参数:
            - data: 输入数组（形状[N, T, T, D]，每个元素值需在[-1, 1]范围内）
            - dtype: 目标整数类型，可选 np.uint8 (0-255) 或 np.uint16 (0-65535)

            返回:
            - int_data: 转换后的整数数组（形状相同，数据类型为指定的整数类型）
            """
            print(f"处理GAF数据转换为整数，数据形状: {data.shape}, 内存占用: {data.nbytes / 1024**3:.2f} GB")
            print(f"目标数据类型: {dtype.__name__}")
            
            # 确定量化范围
            if dtype == np.uint8:
                max_val = 255
                print("使用8位整数存储 (0-255)")
            elif dtype == np.uint16:
                max_val = 65535
                print("使用16位整数存储 (0-65535)")
            else:
                raise ValueError(f"不支持的数据类型: {dtype}，请使用 np.uint8 或 np.uint16")
            
            # 针对大规模数据的分批处理优化
            batch_size = 1000  # 每批处理1000个样本
            
            if data.shape[0] <= batch_size:
                # 小数据集，直接处理
                return _process_batch_int_optimized(data, dtype, max_val)
            else:
                # 大数据集，分批处理
                print(f"使用分批处理，批大小: {batch_size}")
                results = []
                
                for i in range(0, data.shape[0], batch_size):
                    end_idx = min(i + batch_size, data.shape[0])
                    batch = data[i:end_idx]
                    
                    if i % (batch_size * 5) == 0:  # 每5个批次打印一次进度
                        print(f"处理进度: {i}/{data.shape[0]} ({100*i/data.shape[0]:.1f}%)")
                    
                    batch_result = _process_batch_int_optimized(batch, dtype, max_val)
                    results.append(batch_result)
                
                print("合并分批结果...")
                return np.concatenate(results, axis=0)
        
        def _process_batch_int_optimized(batch_data: np.ndarray, dtype, max_val: int) -> np.ndarray:
            """
            优化的整数批处理函数
            """
            # 检查数据范围
            data_min, data_max = batch_data.min(), batch_data.max()
            
            # 将[-1, 1]映射到[0, max_val]
            if data_min >= -1.0 and data_max <= 1.0:
                # 数据在范围内，直接转换
                # 使用更高精度的中间计算避免舍入误差
                normalized = (batch_data.astype(np.float64) + 1.0) * (max_val / 2.0)
            else:
                # 需要裁剪
                clipped = np.clip(batch_data, -1, 1)
                normalized = (clipped.astype(np.float64) + 1.0) * (max_val / 2.0)
            
            # 四舍五入并转换为目标整数类型
            result = np.round(normalized).astype(dtype)
            
            return result

        # 收集所有文件的数据和标签
        all_segments = []
        all_labels = []
        feature_columns = None

        # 读取所有数据文件
        print("\n=== 开始加载数据文件 ===")
        for i, (file_name, label) in enumerate(self.file_label_map.items()):
            file_path = os.path.join(self.root_path, file_name)
            print(f"\n处理文件 {i+1}/{len(self.file_label_map)}: {file_path}")
            print(f"标签值: {label}")

            segments, cols = load_and_segment(file_path, None, True)

            if not segments:
                print(f"警告: 文件 {file_name} 未包含有效数据段")
                continue

            print(f"成功加载 {len(segments)} 个数据段")
            print(f"特征列数量: {len(cols)}")

            if feature_columns is None:
                feature_columns = cols
            elif set(feature_columns) != set(cols):
                print(f"警告: 文件 {file_name} 的特征列与之前不匹配")
                print(f"当前特征列: {set(cols)}")
                print(f"之前特征列: {set(feature_columns)}")

            # 为每个段添加对应标签（转换为数字索引）
            numeric_label = self.label_to_idx[label]
            for segment in segments:
                all_segments.append(segment)
                all_labels.append(numeric_label)

        print(f"\n=== 数据加载完成 ===")
        print(f"总数据段数: {len(all_segments)}")
        print(f"总标签数: {len(all_labels)}")

        # 对所有特征进行通道级别的归一化
        print("\n=== 开始通道级别归一化 ===")
        print(f"特征数量: {len(feature_columns)}")

        # 创建归一化器
        self.scalers = {}
        for i, col in enumerate(feature_columns):
            print(f"\n处理特征 {i+1}/{len(feature_columns)}: {col}")
            self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
            # 收集该特征的所有数据点
            feature_data = np.concatenate(
                [seg[:, i].reshape(-1, 1) for seg in all_segments], axis=0
            )
            print(f"特征数据形状: {feature_data.shape}")
            # 拟合归一化器
            self.scalers[col].fit(feature_data)
            print(f"特征 {col} 归一化完成")

        # 应用归一化
        print("\n=== 应用归一化到所有数据段 ===")
        for seg_idx in range(len(all_segments)):
            if seg_idx % 100 == 0:  # 每处理100个段打印一次进度
                print(f"处理数据段 {seg_idx+1}/{len(all_segments)}")
            for i, col in enumerate(feature_columns):
                all_segments[seg_idx][:, i] = (
                    self.scalers[col]
                    .transform(all_segments[seg_idx][:, i].reshape(-1, 1))
                    .flatten()
                )

        print("\n=== 开始创建时间窗口 ===")
        # 创建窗口
        labeled_windows = []
        labeled_labels = []

        for seg_idx, (seg, label) in enumerate(zip(all_segments, all_labels)):
            if seg_idx % 100 == 0:  # 每处理100个段打印一次进度
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

        print(f"窗口数据形状: {labeled_windows.shape}")
        print(f"标签数据形状: {labeled_labels.shape}")

        # 确保数据非空
        if len(labeled_windows) == 0:
            raise ValueError("未能生成任何有效的时间窗口")

        # 打乱数据
        print("\n=== 打乱数据 ===")
        np.random.seed(42)  # 固定随机种子保证可复现
        indices = np.random.permutation(len(labeled_windows))
        labeled_windows = labeled_windows[indices]
        labeled_labels = labeled_labels[indices]

        print("\n=== 开始GAF转换 ===")
        print(f"输入数据形状: {labeled_windows.shape}")
        gaf_data = generate_gaf_matrix(labeled_windows, self.gaf_method, False)
        print(f"GAF转换后数据形状: {gaf_data.shape}")

        print("\n=== 开始数据范围转换 ===")
        print(f"转换前内存使用: {gaf_data.nbytes / 1024**3:.2f} GB")
        print(f"转换前数据类型: {gaf_data.dtype}")
        
        # 对于大数据集（您的数据约11.3GB），启用内存优化模式
        if gaf_data.nbytes > 8 * 1024**3:  # 超过8GB
            print("检测到大数据集，启用内存优化模式")
        
        # 根据配置选择转换方法
        if self.data_type_method == 'float32':
            print("使用Float32转换方法")
            gaf_data = gaf_to_float32(gaf_data)
        elif self.data_type_method == 'uint8':
            print("使用UInt8转换方法 (内存节省75%)")
            gaf_data = gaf_to_int(gaf_data, dtype=np.uint8)
        elif self.data_type_method == 'uint16':
            print("使用UInt16转换方法 (内存节省50%)")
            gaf_data = gaf_to_int(gaf_data, dtype=np.uint16)
        else:
            raise ValueError(f"不支持的数据类型方法: {self.data_type_method}")
        
        print(f"转换后数据范围: [{gaf_data.min()}, {gaf_data.max()}]")
        print(f"转换后数据类型: {gaf_data.dtype}")
        print(f"转换后内存使用: {gaf_data.nbytes / 1024**3:.2f} GB")
        
        # 计算内存节省比例
        if self.data_type_method != 'float32':
            original_size = gaf_data.size * 4  # float32 大小
            current_size = gaf_data.nbytes
            memory_saving = (original_size - current_size) / original_size * 100
            print(f"相比Float32节省内存: {memory_saving:.1f}%")

        # 计算划分点
        train_split = int(len(gaf_data) * (1 - self.test_size))
        # 划分数据集
        self.train = gaf_data[:train_split]
        self.train_labels = labeled_labels[:train_split]
        self.val = gaf_data[train_split:]
        self.val_labels = labeled_labels[train_split:]
        # 输出数据集信息
        print("\n=== 数据集划分完成 ===")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")
        # 数据处理完成后自动保存
        print("\n=== 保存预处理数据 ===")
        self.persist_data(self.persist_path, labeled_windows, labeled_labels)
        print(f"已自动保存预处理数据到: {self.persist_path}")

    def load_persisted_data(self, path):
        """从文件加载预处理好的数据"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # 校验数据格式完整性
        required_keys = ["train", "val", "train_labels", "val_labels", "scalers"]
        if not all(key in data for key in required_keys):
            raise ValueError("持久化文件数据格式不完整，可能版本不兼容")
        self.train = data["train"]
        self.val = data["val"]
        self.train_labels = data["train_labels"]
        self.val_labels = data["val_labels"]
        self.scalers = data["scalers"]
        
        # 加载标签映射（如果存在）
        if "label_to_idx" in data and "idx_to_label" in data:
            self.label_to_idx = data["label_to_idx"]
            self.idx_to_label = data["idx_to_label"]
        else:
            # 兼容旧版本数据，重新生成映射
            print("警告：持久化文件中没有标签映射，正在重新生成...")
            unique_labels = sorted(set(self.file_label_map.values()))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 加载数据类型方法（如果存在）
        if "data_type_method" in data:
            saved_method = data["data_type_method"]
            if saved_method != self.data_type_method:
                print(f"警告：持久化文件的数据类型方法({saved_method})与当前设置({self.data_type_method})不匹配")
                print("将使用持久化文件中的数据类型方法")
                self.data_type_method = saved_method
        else:
            print("警告：持久化文件中没有数据类型方法信息，假设为float32")
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
            "file_map": self.file_label_map,  # 保存文件映射用于版本校验
            "gaf_method": self.gaf_method,  # 保存GAF方法
            "data_type_method": self.data_type_method,  # 保存数据类型方法
            "labeled_windows": labeled_windows,  # 新增：保存原始窗口
            "labeled_labels": labeled_labels,  # 新增：保存原始标签
            "label_to_idx": self.label_to_idx,  # 保存标签映射
            "idx_to_label": self.idx_to_label,  # 保存索引到标签的映射
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"数据持久化保存到 {path} 完成")

    def __len__(self):
        return {"train": len(self.train), "val": len(self.val)}[self.flag]

    def __getitem__(self, index):
        if self.flag == "train":
            data = self.train[index]
            label = self.train_labels[index]
        else:  # val
            data = self.val[index]
            label = self.val_labels[index]
        
        # 如果使用整数类型存储，需要转换回浮点数进行训练
        if self.data_type_method in ['uint8', 'uint16']:
            # 将整数数据转换回[0, 255]范围的浮点数
            data = data.astype(np.float32)
        
        return data.astype(np.float32), np.float32(label)


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

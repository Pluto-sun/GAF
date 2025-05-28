import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import  MinMaxScaler
import warnings
import pickle
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import hashlib  # 添加哈希库
warnings.filterwarnings('ignore')


class ClassificationSegLoader(Dataset):
    def __init__(self, args, root_path, win_size,step,flag, file_label_map=None, persist_path=None):
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
        self.step = step
        self.win_size = win_size
        self.gaf_method = args.gaf_method if hasattr(args, 'gaf_method') else 'summation'  # 新增：从args获取GAF方法

        # 如果没有提供文件标签映射，则使用默认映射
        # if file_label_map is None:
        #     file_label_map = {
        #         "DualDuct_FaultFree_resampled_direct_5T_working.csv": 0,  # 正常数据
        #         "DualDuct_DMPRStuck_OA_100_resampled_direct_5T_working.csv": 1,   # 异常数据
        #         "DualDuct_Fouling_Cooling_Airside_Severe_resampled_direct_5T_working.csv": 2
        #     }
        if file_label_map is None:
            file_label_map = {
                "AHU_annual_resampled_direct_5T_working.csv": 0,  # 正常数据
                "coi_stuck_025_resampled_direct_5T_working.csv": 1,   # 异常数据
                "damper_stuck_025_annual_resampled_direct_5T_working.csv": 2
            }
        self.file_label_map=file_label_map    
        # 生成持久化文件名标识
        file_keys = sorted(file_label_map.keys()) if file_label_map else []
        file_str = '|'.join(file_keys).encode()
        file_hash = hashlib.md5(file_str).hexdigest()  # 使用MD5哈希
        self._auto_persist_path = os.path.join(
            root_path,
            f"classifier_data_win{win_size}_step{step}_files{len(file_keys)}_{file_hash}_gaf{self.gaf_method}.pkl"  # 修改：在文件名中包含GAF方法
        )
        
        # 优先使用用户指定路径，否则使用自动生成路径
        self.persist_path = persist_path or self._auto_persist_path
        # 如果持久化文件存在，直接加载
        if os.path.exists(self.persist_path):
            print(f"检测到已存在的持久化文件: {self.persist_path}")
            self.load_persisted_data(self.persist_path)
            return



            
            
        # 增强型数据加载器
        def load_and_segment(path, rows=None, skip_normalize=True):
            # 定义要排除的列
            # exclude_columns=[
            # #     管道静压
            #     "SA_SPSPT",
            #     "SA_SP",
            #     # 系统控制信号
            #     "SYS_CTL",
            #     # 一些阀门的实际信号，只保留设定信号
            #     "CHWC_VLV",#冷却盘管
            #     "RA_DMPR",#回风阀门
            #     "OA_DMPR",#外部阀门
            #     #功率信号
            #     "SF_WAT",
            #     "RF_WAT",
            #     #风扇的实际转速，只保留控制信号
            #     "RF_SPD",
            #     "SF_SPD",
            #     #温度设定值
            #     "SA_TEMPSPT",
            #     # 排除之前定义的列
            #     'SYS_CTL', 'SF_SPD_DM', 'SF_SPD', 'SA_TEMPSPT', 'RF_SPD_DM', 'OA_CFM'
            # ]
            exclude_columns=[]
            
            # 移除重复项
            exclude_columns = list(set(exclude_columns))
            
            if rows:
                df = pd.read_csv(
                    path,
                    usecols=lambda c: c not in exclude_columns,
                    nrows=rows
                )
            else:
                df = pd.read_csv(
                    path,
                    usecols=lambda c: c not in exclude_columns
                )
            df['Datetime'] = pd.to_datetime(df['Datetime'])

            # 生成时间序列特征
            df['ts'] = df['Datetime'].astype('int64') // 10 ** 9
            df['date'] = df['Datetime'].dt.date
            df['hour'] = df['Datetime'].dt.hour

            # 使用is_working列过滤工作时间
            df = df[df['is_working'] == 1]

            # 分段逻辑：连续工作时间段识别
            df = df.sort_values('ts').reset_index(drop=True)
            
            # 计算is_working的变化点或者时间间隔过大的点
            df['time_diff'] = df['ts'].diff() > 3600 * 2  # 两小时不连续视为新时段

            # 生成时间段ID
            df['segment_id'] = df['time_diff'].cumsum()

            # 获取特征列（非时间相关列）
            feature_columns = [col for col in df.columns if col not in ['Datetime', 'ts', 'date', 'hour', 'time_diff', 'segment_id', 'is_working']]
            
            # 提取所有工作时段数据
            segments = []
            for seg_id, group in df.groupby('segment_id'):
                # 确保最小长度满足窗口要求
                if len(group) >= win_size:
                    # 只保留特征列数据
                    segment_data = group[feature_columns].values
                    segments.append(segment_data)
            
            return segments, feature_columns
        
        # 分段窗口生成器
        def create_segment_windows(segments):
            all_windows = []
            for seg in segments:
                # 单段内滑动窗口处理
                if len(seg) == win_size:
                    all_windows.append(seg)
                else:
                    # 单段内滑动窗口处理
                    for i in range(0, len(seg) - win_size + 1, self.step):
                        all_windows.append(seg[i:i + win_size])
            return np.array(all_windows) if len(all_windows) > 0 else np.array([])
        
        def generate_gaf_matrix(
            data: np.ndarray,
            method: str = "summation",
            normalize: bool = False
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
                raise ValueError(f"输入数据必须为3维，当前维度数：{data.ndim}，正确形状应为[N, T, D]")
            
            N, T, D = data.shape  # 提取维度尺寸
            valid_methods = {"summation", "difference"}
            if method not in valid_methods:
                raise ValueError(f"method必须为{sorted(valid_methods)}之一，当前输入：{method}")
            
            # 2. 调整维度顺序为[N, D, T]（pyts要求时间步在最后一维）
            transposed_data = data.transpose(0, 2, 1)  # 形状[N, D, T]
            
            # 3. 展开为单维度时间序列批量输入格式[N*D, T]
            flattened_data = transposed_data.reshape(-1, T)  # 形状[N*D, T]
            
            # 4. 初始化GAF并生成矩阵
            gasf = GramianAngularField(method=method)
            batch_gaf = gasf.fit_transform(flattened_data)  # 形状[N*D, T, T]
            
            # 5. 重组维度为[N, D, T, T]
            reshaped_gaf = batch_gaf.reshape(N, D, T, T)
            
            # 6. 调整维度顺序为目标形状[N, T, T, D]
            target_gaf = reshaped_gaf.transpose(0, 2, 3, 1)
            
            # 7. 转换为深度学习友好的float32类型
            return target_gaf.astype(np.float32)
        def gaf_to_float32(data: np.ndarray) -> np.ndarray:
            """
            将四维数组中的每个二维矩阵（值范围[-1, 1]）映射到[0, 255]并转换为浮点数（保留小数精度）

            参数:
            - data: 输入数组（形状[N, T, T, D]，每个元素值需在[-1, 1]范围内）

            返回:
            - float_data: 转换后的浮点数数组（形状相同，数据类型为np.float32，值范围[0.0, 255.0]）
            """
            # 1. 防御性剪枝（确保所有值在理论范围内，处理可能的浮点误差）
            clipped_data = np.clip(data, -1, 1)  # 剪枝到[-1, 1]

            # 2. 固定范围映射到[0, 255]（浮点运算保留精度）
            mapped_data = (clipped_data + 1) / 2 * 255  # 浮点结果，范围[0.0, 255.0]

            # 3. 四舍五入取整（可选：根据深度学习需求决定是否保留小数）
            #    若需要保留原始浮点值，可注释掉这一步
            # rounded_data = np.round(mapped_data)  # 转换为整数（0-255）

            # 4. 转换为深度学习常用的float32类型（避免整数类型精度损失）
            float_data = mapped_data.astype(np.float32)

            return float_data
        # 收集所有文件的数据和标签
        all_segments = []
        all_labels = []
        feature_columns = None
        
        # 读取所有数据文件
        print("\n=== 开始加载数据文件 ===")
        for i, (file_name, label) in enumerate(file_label_map.items()):
            file_path = os.path.join(root_path, file_name)
            print(f"\n处理文件 {i+1}/{len(file_label_map)}: {file_path}")
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
            
            # 为每个段添加对应标签
            for segment in segments:
                all_segments.append(segment)
                all_labels.append(label)
        
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
            feature_data = np.concatenate([seg[:, i].reshape(-1, 1) for seg in all_segments], axis=0)
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
                all_segments[seg_idx][:, i] = self.scalers[col].transform(
                    all_segments[seg_idx][:, i].reshape(-1, 1)).flatten()
        
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
        gaf_data = gaf_to_float32(gaf_data)
        print(f"数据范围: [{gaf_data.min():.2f}, {gaf_data.max():.2f}]")
        
        # 计算划分点
        train_split = int(len(gaf_data) * 0.6)
        val_split = train_split + int(len(gaf_data) * 0.2)
        
        # 划分数据集
        self.train = gaf_data[:train_split]
        self.train_labels = labeled_labels[:train_split]
        
        self.val = gaf_data[train_split:val_split]
        self.val_labels = labeled_labels[train_split:val_split]
        
        self.test = gaf_data[val_split:]
        self.test_labels = labeled_labels[val_split:]
        
        # 输出数据集信息
        print("\n=== 数据集划分完成 ===")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")
        print(f"测试集: {len(self.test)} 样本")
        
        # 数据处理完成后自动保存
        print("\n=== 保存预处理数据 ===")
        self.persist_data(self.persist_path)
        print(f"已自动保存预处理数据到: {self.persist_path}")

    def load_persisted_data(self, path):
        """从文件加载预处理好的数据"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # 校验数据格式完整性
        required_keys = ['train', 'val', 'test', 'train_labels', 'val_labels', 'test_labels', 'scalers']
        if not all(key in data for key in required_keys):
            raise ValueError("持久化文件数据格式不完整，可能版本不兼容")    
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']
        self.train_labels = data['train_labels']
        self.val_labels = data['val_labels']
        self.test_labels = data['test_labels']
        self.scalers = data['scalers']
        
        print(f"从 {path} 加载数据完成")
        print(f"训练集: {len(self.train)} 样本")
        print(f"验证集: {len(self.val)} 样本")
        print(f"测试集: {len(self.test)} 样本")
    
    def persist_data(self, path):
        """持久化保存预处理好的数据"""
        data = {
            'train': self.train,
            'val': self.val,
            'test': self.test,
            'train_labels': self.train_labels,
            'val_labels': self.val_labels,
            'test_labels': self.test_labels,
            'scalers': self.scalers,
            'win_size': self.win_size,
            'step': self.step,
            'file_map': self.file_label_map,  # 保存文件映射用于版本校验
            'gaf_method': self.gaf_method  # 保存GAF方法
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"数据持久化保存到 {path} 完成")
    
    def __len__(self):
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test)
        }[self.flag]

    def __getitem__(self, index):
        if self.flag == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        elif self.flag == "val":
            return np.float32(self.val[index]), np.float32(self.val_labels[index])
        else:  # test
            return np.float32(self.test[index]), np.float32(self.test_labels[index])

    


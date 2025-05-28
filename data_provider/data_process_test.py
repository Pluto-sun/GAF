import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
flag='train'
step=64
win_size=64
root_path='./dataset/SAHU'
# 创建保存图像的文件夹
save_dir = './data_provider/figures/feature_distributions'
os.makedirs(save_dir, exist_ok=True)

# 增强型数据加载器
def load_and_segment(path, rows=None, scalers=None, fit_scalers=False, skip_normalize=False):
    # 定义要排除的列
    exclude_columns=[
    #     管道静压
        "SA_SPSPT",
        "SA_SP",
        # 系统控制信号
        "SYS_CTL",
        # 一些阀门的实际信号，只保留设定信号
        "CHWC_VLV",#冷却盘管
        "RA_DMPR",#回风阀门
        "OA_DMPR",#外部阀门
        #功率信号
        "SF_WAT",
        "RF_WAT",
        #风扇的实际转速，只保留控制信号
        "RF_SPD",
        "SF_SPD",
        #温度设定值
        "SA_TEMPSPT",
        # 排除之前定义的列
        'SYS_CTL', 'SF_SPD_DM', 'SF_SPD', 'SA_TEMPSPT', 'RF_SPD_DM', 'OA_CFM'
    ]
    
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
    
    # 如果要跳过归一化处理，直接返回原始数据
    if skip_normalize:
        return segments, None
    
    # 所有数据读取完成后再进行归一化
    if len(segments) > 0:
        # 如果没有提供归一化器，创建新的归一化器
        if scalers is None:
            scalers = {}
            for i, col in enumerate(feature_columns):
                scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                # 收集该特征的所有数据点
                feature_data = np.concatenate([seg[:, i].reshape(-1, 1) for seg in segments], axis=0)
                # 拟合归一化器
                scalers[col].fit(feature_data)
        
        # 对每个时段的每个特征应用归一化
        for seg_idx in range(len(segments)):
            for i, col in enumerate(feature_columns):
                if fit_scalers:
                    # 对训练集，如果需要重新拟合归一化器
                    feature_data = segments[seg_idx][:, i].reshape(-1, 1)
                    # scalers[col].fit(feature_data)
                    segments[seg_idx][:, i] = scalers[col].transform(feature_data).flatten()
                else:
                    # 对验证集和测试集，只使用transform不进行fit
                    segments[seg_idx][:, i] = scalers[col].transform(segments[seg_idx][:, i].reshape(-1, 1)).flatten()
    
    return segments, scalers

# 分段窗口生成器
def create_segment_windows(segments):
    all_windows = []
    for seg in segments:
        # 单段内滑动窗口处理
        if len(seg) == win_size:
            all_windows.append(seg)
        else:
            # 单段内滑动窗口处理
            
            for i in range(0, len(seg) - win_size + 1, step):
                all_windows.append(seg[i:i + win_size])
    return np.array(all_windows)

# 直接读取原始数据（不分段，不归一化）用于绘图
def load_raw_data(path, rows=1000):
    # 定义要排除的列
    exclude_columns=[
    #     管道静压
        "SA_SPSPT",
        "SA_SP",
        # 系统控制信号
        "SYS_CTL",
        # 一些阀门的实际信号，只保留设定信号
        "CHWC_VLV",#冷却盘管
        "RA_DMPR",#回风阀门
        "OA_DMPR",#外部阀门
        #功率信号
        "SF_WAT",
        "RF_WAT",
        #风扇的实际转速，只保留控制信号
        "RF_SPD",
        "SF_SPD",
        #温度设定值
        "SA_TEMPSPT",
        # 排除之前定义的列
        'SYS_CTL', 'SF_SPD_DM', 'SF_SPD', 'SA_TEMPSPT', 'RF_SPD_DM', 'OA_CFM'
    ]
    
    # 移除重复项
    exclude_columns = list(set(exclude_columns))
    
    # 读取数据
    df = pd.read_csv(
        path,
        usecols=lambda c: c not in exclude_columns,
        nrows=rows
    )
    
    # 不过滤工作时间，保留所有数据
    
    # 获取特征列和工作状态列
    feature_columns = [col for col in df.columns if col not in ['Datetime', 'is_working']]
    
    # 返回特征数据、列名和工作状态
    return df[feature_columns].values, feature_columns, df['is_working'].values

# 获取并绘制原始数据
raw_data, feature_names, working_status = load_raw_data(os.path.join(root_path, "AHU_annual_resampled_mean_5T_working.csv"))
if raw_data.shape[0] > 0:
    plt.figure(figsize=(12, 36))
    for i in range(raw_data.shape[1]):
        plt.subplot(raw_data.shape[1], 1, i+1)
        
        # 取前1000个数据点
        data_len = min(1000, len(raw_data))
        x = np.arange(data_len)
        y = raw_data[:data_len, i]
        work_mask = working_status[:data_len] == 1
        non_work_mask = working_status[:data_len] == 0
        
        # 创建工作状态的掩码数组
        work_y = np.copy(y)
        non_work_y = np.copy(y)
        
        # 对于工作时间的线，在非工作时间点处设置为NaN
        work_y[non_work_mask] = np.nan
        # 对于非工作时间的线，在工作时间点处设置为NaN
        non_work_y[work_mask] = np.nan
        
        # 绘制两条线
        plt.plot(x, work_y, 'b-', linewidth=1.5, label='工作时间')
        plt.plot(x, non_work_y, 'r-', linewidth=1.5, label='非工作时间')
        
        plt.title(f'{feature_names[i]}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'raw_feature_distribution.png'), dpi=300)
    plt.close()  # 关闭图形，释放内存

# 主处理流程
# 先处理训练数据并获取归一化器
normal_segments_train, scalers = load_and_segment(
    os.path.join(root_path, "AHU_annual_resampled_mean_5T_working.csv"), 
    1000, 
    None, 
    fit_scalers=True
)

# 使用训练数据的归一化器处理所有数据
normal_segments, _ = load_and_segment(
    os.path.join(root_path, "AHU_annual_resampled_mean_5T_working.csv"), 
    1000, 
    scalers, 
    fit_scalers=False
)
print(f'normal_segments的长度为{len(normal_segments)}')
print(f'normal_segments的形状为{normal_segments[0].shape}')

# 选择第一个segments
first_segment = normal_segments[0]

# 读取特征列名称
_, feature_names, _ = load_raw_data(os.path.join(root_path, "AHU_annual_resampled_mean_5T_working.csv"))

# 绘制归一化后的第一个segments的数据分布
plt.figure(figsize=(12, 36))
for i in range(first_segment.shape[1]):
    plt.subplot(first_segment.shape[1], 1, i+1)
    plt.plot(first_segment[:, i])
    plt.title(f'{feature_names[i]}')
plt.tight_layout()
# 保存图像到指定文件夹
plt.savefig(os.path.join(save_dir, 'normalized_feature_distribution.png'), dpi=300)
plt.close()  # 关闭图形，释放内存

# all_normal_windows = create_segment_windows(normal_segments)

# # 打乱所有窗口数据
# np.random.seed(42)  # 固定随机种子保证可复现
# shuffled_indices = np.random.permutation(len(all_normal_windows))
# shuffled_windows = [all_normal_windows[i] for i in shuffled_indices]

# # 计算划分点
# train_split = int(len(shuffled_windows) * 0.6)
# val_split = train_split + int(len(shuffled_windows) * 0.2)

# # 根据划分点划分数据集
# train = shuffled_windows[:train_split]
# train = np.array(train)
# val = shuffled_windows[train_split:val_split]
# val = np.array(val)
# test_normal = shuffled_windows[val_split:]

# # 处理异常数据 - 使用训练数据的归一化器
# fault_segments, _ = load_and_segment(
#     os.path.join(root_path, "coi_stuck_025_resampled_mean_5T.csv"),
#     None, 
#     scalers,
#     fit_scalers=False
# )
# fault_windows = create_segment_windows(fault_segments)

# # 创建平衡测试集
# np.random.shuffle(fault_windows)
# test_fault = fault_windows[:len(test_normal)]

# # 合并测试数据
# print(f"正常数据长度{len(test_normal)}")
# test = np.concatenate([test_normal, test_fault], axis=0)
# test_labels = np.concatenate([
#     np.zeros((len(test_normal), win_size)),
#     np.ones((len(test_fault), win_size))
# ])

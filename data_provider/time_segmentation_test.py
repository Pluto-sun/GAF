import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 创建保存图像的文件夹
save_dir = './data_provider/figures/time_segmentation'
os.makedirs(save_dir, exist_ok=True)

# 设置参数
win_size = 64
root_path = './dataset/SAHU'
data_limit = 10000  # 读取前10000条数据

# 时间判断函数优化
def is_working_time(ts):
    dt = datetime.fromtimestamp(ts)
    weekday = dt.weekday()
    hour = dt.hour

    if weekday < 5:  # 周一到周五
        return 6 <= hour < 22
    elif weekday == 5:  # 周六
        return 6 <= hour < 18
    return False

# 定义要排除的列
exclude_columns = [
    # 管道静压
    "SA_SPSPT",
    "SA_SP",
    # 系统控制信号
    "SYS_CTL",
    # 一些阀门的实际信号，只保留设定信号
    "CHWC_VLV",  # 冷却盘管
    "RA_DMPR",  # 回风阀门
    "OA_DMPR",  # 外部阀门
    # 功率信号
    "SF_WAT",
    "RF_WAT",
    # 风扇的实际转速，只保留控制信号
    "RF_SPD",
    "RF_SPD_DM",
    "SF_SPD",
    "SF_SPD_DM",
    # 温度设定值
    "SA_TEMPSPT",
    # 风量
    "OA_CFM"
]

# 读取原始数据
df = pd.read_csv(
    os.path.join(root_path, "AHU_annual_resampled_mean_5T.csv"),
    usecols=lambda c: c not in exclude_columns,
    nrows=data_limit
)

# 转换时间格式
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['ts'] = df['Datetime'].astype('int64') // 10 ** 9

# 1. 绘制原始数据时间序列图
plt.figure(figsize=(15, 10))
plt.suptitle('原始数据时间序列图', fontsize=16)

# 选择几个重要指标进行绘制
key_metrics = ['SA_TEMP', 'RA_TEMP', 'MA_TEMP', 'OA_TEMP']
colors = ['blue', 'green', 'red', 'purple']

for i, metric in enumerate(key_metrics):
    if metric in df.columns:
        plt.subplot(len(key_metrics), 1, i+1)
        plt.plot(df['Datetime'], df[metric], color=colors[i], label=metric)
        plt.ylabel(metric)
        plt.title(f'{metric} 原始时间序列')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        # 优化x轴日期显示
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(save_dir, 'original_time_series.png'), dpi=300)
plt.close()

# 2. 应用工作时间过滤
# 添加工作日标记列
df['is_working'] = df['ts'].apply(is_working_time)
df_working = df[df['is_working']]

# 绘制工作时间过滤后的数据
plt.figure(figsize=(15, 10))
plt.suptitle('工作时间过滤后的数据', fontsize=16)

for i, metric in enumerate(key_metrics):
    if metric in df.columns:
        plt.subplot(len(key_metrics), 1, i+1)
        plt.plot(df_working['Datetime'], df_working[metric], color=colors[i], label=f'{metric} (工作时间)')
        plt.ylabel(metric)
        plt.title(f'{metric} 工作时间序列')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(save_dir, 'working_hours_filtered.png'), dpi=300)
plt.close()

# 3. 应用时间段切分逻辑
df_working = df_working.sort_values('ts').reset_index(drop=True)
df_working['time_diff'] = df_working['ts'].diff() > 3600 * 2  # 两小时不连续视为新时段
df_working['segment_id'] = df_working['time_diff'].cumsum()

# 4. 绘制分段结果可视化
plt.figure(figsize=(15, 10))
plt.suptitle('按工作时间分段结果', fontsize=16)

# 获取前5个段落(或全部如果少于5个)
segment_ids = df_working['segment_id'].unique()
segment_count = min(5, len(segment_ids))

for i, metric in enumerate(key_metrics[:2]):  # 只用两个指标以免图太复杂
    if metric in df.columns:
        plt.subplot(2, 1, i+1)
        
        for j in range(segment_count):
            segment_data = df_working[df_working['segment_id'] == segment_ids[j]]
            plt.plot(segment_data['Datetime'], segment_data[metric], 
                    label=f'段落 {j+1}', alpha=0.8)
        
        plt.ylabel(metric)
        plt.title(f'{metric} 分段后时间序列')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(save_dir, 'time_segments_visualization.png'), dpi=300)
plt.close()

# 5. 分段统计信息
segment_stats = df_working.groupby('segment_id').agg({
    'Datetime': ['min', 'max', 'count'],
})

segment_stats.columns = ['开始时间', '结束时间', '数据点数']
segment_stats['持续时间(小时)'] = (segment_stats['结束时间'] - segment_stats['开始时间']).dt.total_seconds() / 3600

# 保存分段统计信息到CSV
segment_stats.to_csv(os.path.join(save_dir, 'segment_statistics.csv'))

# 打印统计信息
print(f"总数据点: {len(df)}")
print(f"工作时间数据点: {len(df_working)}")
print(f"分段总数: {len(segment_ids)}")
print(f"分段详情已保存至: {os.path.join(save_dir, 'segment_statistics.csv')}")

# 6. 选择一个较长的段落，绘制其中几个通道的数据
longest_segment_id = segment_stats['数据点数'].idxmax()
longest_segment = df_working[df_working['segment_id'] == longest_segment_id]

# 绘制最长段落的多通道数据
plt.figure(figsize=(15, 12))
plt.suptitle(f'最长工作时间段落数据 (段落ID: {longest_segment_id})', fontsize=16)

feature_columns = [col for col in df.columns if col not in ['Datetime', 'ts', 'date', 'hour', 'time_diff', 'segment_id', 'is_working']]
selected_features = feature_columns[:min(8, len(feature_columns))]  # 最多显示8个特征

for i, feature in enumerate(selected_features):
    plt.subplot(len(selected_features), 1, i+1)
    plt.plot(longest_segment['Datetime'], longest_segment[feature], label=feature)
    plt.ylabel(feature)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 在每个子图上标注时间跨度
    time_span = (longest_segment['Datetime'].max() - longest_segment['Datetime'].min()).total_seconds() / 3600
    plt.title(f'{feature} (时间跨度: {time_span:.2f}小时)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(save_dir, 'longest_segment_features.png'), dpi=300)
plt.close()

print(f"所有图像已保存至: {save_dir}") 
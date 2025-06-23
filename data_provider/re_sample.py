import os
import pandas as pd
import re

def batch_resample_hvac_data(file_paths, output_path, interval='5T'):
    """
    批量重新采样HVAC数据。

    参数：
    - file_paths (list): 输入的CSV文件路径列表。
    - interval (str): 重新采样的时间间隔（例如 '5T' 表示每5分钟，'10T' 表示每10分钟）。

    输出：
    - 为每个文件保存两个重新采样结果的CSV文件：
        1. 直接采样（如整点00、05等）。
        2. 均值采样（计算每个时间段的均值）。
    """
    # 兼容采样间隔格式，将如'5T'自动转换为'5min'
    interval = re.sub(r'(\d+)T$', r'\1min', interval)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file_name in os.listdir(file_paths):
        if file_name.endswith('.csv') and 'resampled' not in file_name:
            file_path = os.path.join(file_paths, file_name)
            data = pd.read_csv(file_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
            resampled_direct = data.resample(interval).asfreq()
            resampled_mean = data.resample(interval).mean()
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            direct_file = os.path.join(output_path, f"{base_name}_resampled_direct_{interval}.csv")
            mean_file = os.path.join(output_path, f"{base_name}_resampled_mean_{interval}.csv")
            resampled_direct.to_csv(direct_file)
            resampled_mean.to_csv(mean_file)
            print(f"文件 {file_path} 重新采样完成，文件已保存：\n{direct_file}\n{mean_file}")

# 示例调用
batch_resample_hvac_data("./dataset/SAHU/origin", "./dataset/SAHU/resampled_5T", interval='5T')
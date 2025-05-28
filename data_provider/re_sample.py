import pandas as pd
def resample_hvac_data(file_path, output_prefix, interval='5T'):
    """
    重新采样HVAC数据。

    参数：
    - file_path (str): 输入的CSV文件路径。
    - output_prefix (str): 输出文件名前缀。
    - interval (str): 重新采样的时间间隔（例如 '5T' 表示每5分钟，'10T' 表示每10分钟）。

    输出：
    - 保存两个重新采样结果的CSV文件：
        1. 直接采样（如整点00、05等）。
        2. 均值采样（计算每个时间段的均值）。
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 确保时间戳列为datetime类型
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # 将时间戳设置为索引
    data.set_index('Datetime', inplace=True)

    # 方式1：直接采样
    resampled_direct = data.resample(interval).asfreq()

    # 方式2：均值处理
    resampled_mean = data.resample(interval).mean()

    # 保存结果到CSV文件
    direct_file = f"{output_prefix}_resampled_direct_{interval}.csv"
    mean_file = f"{output_prefix}_resampled_mean_{interval}.csv"

    resampled_direct.to_csv(direct_file)
    resampled_mean.to_csv(mean_file)

    print(f"重新采样完成，文件已保存：\n{direct_file}\n{mean_file}")
# 示例调用
resample_hvac_data('./dataset/DDAHU/DualDuct_VLVStuck_Heating_100_.csv',
    './dataset/DDAHU/DualDuct_VLVStuck_Heating_100_', interval='5T')
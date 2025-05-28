import pandas as pd
from datetime import datetime
import os

# 时间判断函数优化
def is_working_time(dt):
    # 直接接收datetime对象而不是时间戳
    weekday = dt.weekday()
    hour = dt.hour

    if weekday < 5:  # 周一到周五
        return 6 <= hour < 18
    # elif weekday == 5:  # 周六
    #     return 6 <= hour < 18
    return False

def add_working_time_column(input_file, output_file=None):
    """
    读取CSV文件，添加一个表示是否是工作时间的列，并保存结果
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径，如果为None则覆盖原文件
    
    返回:
    处理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 确保Datetime列为日期时间格式
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 添加is_working列，直接传递datetime对象而不是timestamp
    df['is_working'] = df['Datetime'].apply(
        lambda x: 1 if is_working_time(x) else 0
    )
    
    # 重新排列列，使is_working列位于Datetime列之后
    cols = list(df.columns)
    datetime_idx = cols.index('Datetime')
    cols.remove('is_working')
    cols.insert(datetime_idx + 1, 'is_working')
    df = df[cols]
    
    # 保存结果
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"文件已保存至 {output_file}")
    
    return df

def batch_add_working_time_column(directory, keyword):
    """
    批量处理目录下所有包含关键词的CSV文件，生成带_working后缀的新文件
    """
    for filename in os.listdir(directory):
        if keyword in filename and filename.endswith('.csv'):
            input_file = os.path.join(directory, filename)
            # 生成带_working后缀的新文件名
            if filename.endswith('.csv'):
                output_file = os.path.join(
                    directory,
                    filename[:-4] + '_working.csv'
                )
            else:
                output_file = os.path.join(
                    directory,
                    filename + '_working'
                )
            print(f"处理文件: {input_file} -> {output_file}")
            add_working_time_column(input_file, output_file)

# 用法示例
if __name__ == "__main__":
    batch_add_working_time_column('./dataset/DDAHU', 'resampled_direct_5T')
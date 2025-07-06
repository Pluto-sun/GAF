import pandas as pd
from datetime import datetime
import os


def add_working_time_column(input_file, output_file=None, remove_start_points=2):
    """
    读取CSV文件，根据SYS_CTL列的值添加is_working列，并消除每天工作时段开始的前几个数据点

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径，如果为None则覆盖原文件
    remove_start_points: 每天工作时段开始时要消除的数据点数量，默认为2

    返回:
    处理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确保Datetime列为日期时间格式
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # 根据SYS_CTL列的值添加is_working列
    # 如果SYS_CTL列的值为1，则is_working设为1，否则设为0
    df["is_working"] = df["SYS_CTL"].apply(lambda x: 1 if x == 1 else 0)

    # 添加日期列用于分组
    df["date"] = df["Datetime"].dt.date

    # 消除每天工作时段开始的前几个数据点
    if remove_start_points > 0:
        for date in df["date"].unique():
            date_mask = df["date"] == date
            day_data = df[date_mask].copy()

            # 找到工作状态变化的位置（从0变到1）
            is_working_shift = day_data["is_working"].shift(1, fill_value=0)
            work_start_positions = day_data[
                (day_data["is_working"] == 1) & (is_working_shift == 0)
            ].index

            # 对每个工作时段开始位置，将前remove_start_points个点设为0
            for start_pos in work_start_positions:
                end_pos = min(start_pos + remove_start_points, len(df))
                # 确保这些位置原本是工作状态，才将其改为非工作状态
                for i in range(start_pos, end_pos):
                    if i < len(df) and df.loc[i, "is_working"] == 1:
                        df.loc[i, "is_working"] = 0

    # 删除临时的日期列
    df = df.drop("date", axis=1)

    # 重新排列列，使is_working列位于Datetime列之后
    cols = list(df.columns)
    datetime_idx = cols.index("Datetime")
    cols.remove("is_working")
    cols.insert(datetime_idx + 1, "is_working")
    df = df[cols]

    # 保存结果
    if output_file is None:
        output_file = input_file

    df.to_csv(output_file, index=False)
    print(f"文件已保存至 {output_file}")

    return df


def batch_add_working_time_column(
    directory, output_path, keyword=None, remove_start_points=2
):
    """
    批量处理目录下所有包含关键词的CSV文件，生成带_working后缀的新文件

    参数:
    directory: 输入文件目录
    keyword: 文件名关键词（可选，用于过滤文件）
    output_path: 输出文件目录
    remove_start_points: 每天工作时段开始时要消除的数据点数量，默认为2
    """
    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 获取所有符合条件的CSV文件
    all_csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and "working" not in filename:
            # 如果指定了关键词，检查文件名是否包含关键词
            if keyword is None or keyword in filename:
                all_csv_files.append(filename)
    
    if not all_csv_files:
        print(f"警告：在目录 {directory} 中未找到符合条件的CSV文件")
        return
    
    print(f"发现 {len(all_csv_files)} 个符合条件的CSV文件")
    
    # 第一步：检查哪些文件需要处理（输出文件不存在）
    files_to_process = []
    files_already_exist = []
    
    for filename in all_csv_files:
        input_file = os.path.join(directory, filename)
        # 生成带_working后缀的新文件名
        output_file = os.path.join(output_path, filename[:-4] + "_working.csv")
        
        if os.path.exists(output_file):
            files_already_exist.append(filename)
        else:
            files_to_process.append((input_file, output_file))
    
    # 输出检查结果
    print(f"需要处理的文件：{len(files_to_process)} 个")
    print(f"已存在的文件：{len(files_already_exist)} 个")
    
    if files_already_exist:
        print(f"跳过已存在的文件：{', '.join(files_already_exist)}")
    
    # 如果没有文件需要处理，直接返回
    if not files_to_process:
        print("所有文件都已处理完成，无需重新处理！")
        return
    
    # 第二步：只处理需要处理的文件
    print(f"开始处理 {len(files_to_process)} 个文件...")
    
    processed_count = 0
    for input_file, output_file in files_to_process:
        try:
            add_working_time_column(input_file, output_file, remove_start_points)
            processed_count += 1
            print(f"[{processed_count}/{len(files_to_process)}] 完成处理: {os.path.basename(input_file)}")
        except Exception as e:
            print(f"处理文件 {os.path.basename(input_file)} 时出错：{str(e)}")
    
    # 输出最终统计信息
    print(f"批量添加工作时间列完成！")
    print(f"成功处理文件数：{processed_count}")
    print(f"跳过文件数：{len(files_already_exist)}")
    print(f"输出目录：{output_path}")


# 用法示例
if __name__ == "__main__":
    batch_add_working_time_column(
        "./dataset/DDAHU/direct_5T",
        "./dataset/DDAHU/direct_5_working",
        remove_start_points=0,
    )

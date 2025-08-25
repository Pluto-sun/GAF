import os
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_single_file(file_info):
    """
    处理单个文件的重新采样
    """
    file_path, output_file, method, interval = file_info
    
    try:
        # 读取数据
        data = pd.read_csv(file_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
        
        # 重新采样
        if method == 'direct':
            resampled = data.resample(interval).asfreq()
        elif method == 'mean':
            resampled = data.resample(interval).mean()
        else:
            return f"错误：不支持的采样方法 {method}"
        
        # 保存结果
        resampled.to_csv(output_file)
        return f"完成处理 {os.path.basename(file_path)}"
        
    except Exception as e:
        return f"处理文件 {os.path.basename(file_path)} 时出错：{str(e)}"

def batch_resample_hvac_data(file_paths, method, interval='5T', max_workers=4):
    """
    批量重新采样HVAC数据。

    参数：
    - file_paths (str): 包含输入数据的根目录路径。
    - method (str): 采样方法，'direct' 或 'mean'。
    - interval (str): 重新采样的时间间隔（例如 '5T' 表示每5分钟，'10T' 表示每10分钟）。
    - max_workers (int): 最大并行工作线程数，默认为4。

    输出：
    - 为每个文件保存重新采样结果的CSV文件。
    """
    # 兼容采样间隔格式，将如'5T'自动转换为'5min'
    input_file_paths = os.path.join(file_paths, 'origin')
    output_path = os.path.join(file_paths, f'{method}_{interval}')
    interval_processed = re.sub(r'(\d+)T$', r'\1min', interval)
    
    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_file_paths) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"警告：在目录 {input_file_paths} 中未找到CSV文件")
        return
    
    print(f"发现 {len(csv_files)} 个CSV文件")
    
    # 第一步：检查哪些文件需要处理（输出文件不存在）
    files_to_process = []
    files_already_exist = []
    
    for file_name in csv_files:
        file_path = os.path.join(input_file_paths, file_name) 
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_path, f"{base_name}_{method}_{interval_processed}.csv")
        
        if os.path.exists(output_file):
            files_already_exist.append(file_name)
        else:
            files_to_process.append((file_path, output_file, method, interval_processed))
    
    # 输出检查结果
    print(f"需要处理的文件：{len(files_to_process)} 个")
    print(f"已存在的文件：{len(files_already_exist)} 个")
    
    if files_already_exist:
        print(f"跳过已存在的文件：{', '.join(files_already_exist)}")
    
    # 如果没有文件需要处理，直接返回
    if not files_to_process:
        print("所有文件都已处理完成，无需重新处理！")
        return
    
    # 第二步：只对需要处理的文件启动多进程处理
    print(f"开始并行处理 {len(files_to_process)} 个文件，使用 {max_workers} 个线程")
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_file, task): task for task in files_to_process}
        
        # 使用tqdm显示进度条
        with tqdm(total=len(files_to_process), desc="处理进度") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                print(result)
                pbar.update(1)
    
    print(f"批量重新采样完成！")
    print(f"处理文件数：{len(files_to_process)}")
    print(f"跳过文件数：{len(files_already_exist)}")
    print(f"输出目录：{output_path}")

# 示例调用
if __name__ == "__main__":
    batch_resample_hvac_data("./dataset/SAHU", "direct", interval='5T', max_workers=1)
    # batch_resample_hvac_data("./dataset/SAHU", "mean", interval='5T', max_workers=4)
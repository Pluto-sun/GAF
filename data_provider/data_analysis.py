import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy.fft import fft, fftfreq


# 1. 读取数据
file_path = './dataset/SAHU/AHU_annual.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 2. 选取信号列（可选：排除非信号列）
# 假设非信号列有 'Datetime', 'is_working', 'label' 等
exclude_cols = ['Datetime', 'is_working', 'label', 'segment_id', 'date', 'hour', 'ts', 'time_diff']
# exclude_cols = []
signal_cols = [col for col in df.columns if col not in exclude_cols]
signal_df = df[signal_cols]

# # 3. 计算相关性矩阵
# corr = signal_df.corr(method='pearson')

# # 4. 绘制热力图
# plt.figure(figsize=(16, 14))
# sns.heatmap(
#     corr,
#     xticklabels=signal_cols,
#     yticklabels=signal_cols,
#     cmap='coolwarm',
#     annot=False,  # 如果信号不多可以设为True显示数值
#     fmt=".2f",
#     square=True,
#     cbar_kws={"shrink": .8}
# )
# plt.title('Signal Correlation Heatmap', fontsize=18)
# plt.xticks(rotation=90, fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# # plt.show() 
# plt.savefig('./dataset/DDAHU/correlation_heatmap.png', dpi=300, bbox_inches='tight')

def plot_all_signals_fft(signal_df, time_col, save_path):
    n_signals = signal_df.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_signals / n_cols))
    plt.figure(figsize=(n_cols * 4, n_rows * 3))
    time = df[time_col].values if time_col in df.columns else np.arange(len(signal_df))
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0
    for idx, col in enumerate(signal_df.columns):
        sig = signal_df[col].values
        N = len(sig)
        yf = np.abs(fft(sig))[:N // 2]
        xf = fftfreq(N, dt)[:N // 2]
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.plot(xf, yf)
        plt.title(col)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
    plt.suptitle('FFT of All Signals', fontsize=20, y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_signals_cwt(signal_df, time_col, save_path):
    n_signals = signal_df.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_signals / n_cols))
    plt.figure(figsize=(n_cols * 4, n_rows * 3))
    time = df[time_col].values if time_col in df.columns else np.arange(len(signal_df))
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0
    widths = np.arange(1, 128)
    for idx, col in enumerate(signal_df.columns):
        sig = signal_df[col].values
        cwtmatr, freqs = pywt.cwt(sig, widths, 'morl', sampling_period=dt)
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(np.abs(cwtmatr), aspect='auto', extent=[0, len(sig)*dt, widths[-1], widths[0]], cmap='jet')
        plt.title(col)
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.tight_layout()
    plt.suptitle('CWT of All Signals', fontsize=20, y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ================== 主流程新增 ==================
# 5. 频域分析
# 假设时间戳列为 'Datetime'，如有不同请修改
plot_all_signals_fft(signal_df, time_col='Datetime', save_path='./data_provider/figures/fft/all_signals_fft.png')
plot_all_signals_cwt(signal_df, time_col='Datetime', save_path='./data_provider/figures/fft/all_signals_cwt.png')
# ===============================================
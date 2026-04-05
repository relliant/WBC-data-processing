import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='全身关节 FFT 频谱分析')
    parser.add_argument('--input', type=str, required=True, help='CSV 或 PKL 文件路径')
    parser.add_argument('--fps', type=int, default=30, help='采样频率 (FPS)')
    parser.add_argument('--max_freq', type=float, default=5.0, help='展示的最大频率 (默认 5Hz)')
    args = parser.parse_args()

    # 1. 加载数据
    df = pd.read_pickle(args.input) if args.input.endswith('.pkl') else pd.read_csv(args.input)

    # 提取后 20 列关节数据
    joint_data = df.iloc[:, 7:]
    joint_names = joint_data.columns
    n = len(df)
    fs = args.fps

    # 2. 设置中文显示与绘图风格
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-muted')  # 使用清爽的样式

    # 3. 创建 5x4 的大布局
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(22, 16))
    axes = axes.flatten()

    print(f"正在处理 {len(joint_names)} 个关节的 FFT...")

    for i, col in enumerate(joint_names):
        if i >= 20: break  # 确保不超出 20 个关节

        ax = axes[i]
        signal = joint_data[col].values
        # 去直流分量 (减均值)
        signal_detrended = signal - np.mean(signal)

        # FFT 计算
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(n, d=1 / fs)

        # 仅取正频率
        pos_mask = (fft_freq > 0) & (fft_freq <= args.max_freq)
        freqs = fft_freq[pos_mask]
        magnitudes = np.abs(fft_values[pos_mask]) * 2 / n

        # 绘图
        ax.plot(freqs, magnitudes, color='#2c7fb8', linewidth=1.2)

        # 标注主频峰值
        if len(magnitudes) > 0:
            max_idx = np.argmax(magnitudes)
            peak_freq = freqs[max_idx]
            ax.annotate(f'{peak_freq:.2f}Hz',
                        xy=(peak_freq, magnitudes[max_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red')

        ax.set_title(f'关节: {col}', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 仅在底部和左侧保留坐标轴标签，保持整洁
        if i >= 16: ax.set_xlabel('频率 (Hz)')
        if i % 4 == 0: ax.set_ylabel('振幅')

    plt.suptitle(f'全身关节频谱分析 (局部放大 0-{args.max_freq}Hz)\n文件: {os.path.basename(args.input)}',
                 fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存并显示
    output_name = f"fft_full_body_0-{int(args.max_freq)}hz.png"
    plt.savefig(output_name, dpi=200)
    print(f"分析完成！图片已保存为: {output_name}")
    plt.show()


if __name__ == "__main__":
    main()
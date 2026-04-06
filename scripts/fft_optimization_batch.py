import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

"""
批量对csv文件中存储的关节运动做轨迹平滑，或对单个文件做处理并画出图表
"""

def fourier_smooth_mirrored(data, cutoff_ratio):
    """使用镜像法 + 离散傅里叶变换(DFT) 进行低通滤波"""
    n = len(data)
    if n < 2: return data

    # 1. 镜像延拓
    mirrored_data = np.concatenate((data, data[::-1]))
    n_mirrored = len(mirrored_data)

    # 2. FFT 变换
    fft_coeffs = np.fft.fft(mirrored_data)

    # 3. 频率过滤
    cutoff_index = max(1, int(n_mirrored * cutoff_ratio / 2))
    mask = np.zeros(n_mirrored)
    mask[:cutoff_index] = 1
    mask[-cutoff_index:] = 1

    filtered_coeffs = fft_coeffs * mask

    # 4. IFFT 逆变换
    smoothed_mirrored_real = np.real(np.fft.ifft(filtered_coeffs))

    # 5. 截断保留原始长度
    return smoothed_mirrored_real[:n]


def plot_comparison(df_orig, df_smooth, cols, save_path=None, show=False):
    """独立的绘图逻辑"""
    num_joints = len(cols)
    n_cols = 4
    n_rows = int(np.ceil(num_joints / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(df_orig[col].values, color='#1f77b4', alpha=0.6, label='Original')
        ax.plot(df_smooth[col].values, color='#d62728', linewidth=1.5, label='Smoothed')
        # ax.set_title(f"Joint: {col}", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax.legend(loc='upper right', fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [Plot] 图表已保存至: {save_path}")
    if show:
        plt.show()
    plt.close(fig)  # 释放内存，防止批量处理时内存泄漏


def process_file(file_path, args):
    """处理单个文件的逻辑"""
    ext = file_path.suffix.lower()

    # 1. 加载数据
    if ext == '.csv':
        df = pd.read_csv(file_path)
    # elif ext == '.pkl':
    #     df = pd.read_pickle(file_path)
    else:
        return

    print(f"正在处理: {file_path.name}")
    smoothed_df = df.copy()

    # 2. 执行平滑
    cols_to_smooth = df.columns[args.skip:]
    if len(cols_to_smooth) == 0:
        print(f"  [Skip] {file_path.name} 没有可平滑的列")
        return

    for col in cols_to_smooth:
        original_values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        smoothed_df[col] = fourier_smooth_mirrored(original_values, args.cutoff)

    # 3. 确定输出路径
    if args.output and not os.path.isdir(args.output):
        output_path = args.output
    else:
        out_dir = Path(args.output) if args.output else file_path.parent
        output_path = out_dir / f"{file_path.stem}_smoothed{ext}"

    # 4. 保存文件
    if ext == '.csv':
        smoothed_df.to_csv(output_path, index=False)
    else:
        smoothed_df.to_pickle(output_path)
    print(f"  [Done] 数据已保存至: {output_path}")

    # 5. 绘图
    if args.plot:
        img_path = args.img_path
        if not img_path:
            img_path = output_path.with_suffix('.png')
        plot_comparison(df, smoothed_df, cols_to_smooth, save_path=img_path, show=args.show_plot)


def main():
    parser = argparse.ArgumentParser(description="批量平滑机器人轨迹 (支持 CSV)")

    # 输入输出
    parser.add_argument("--input", help="输入文件或文件夹路径")
    parser.add_argument("-o", "--output", help="输出路径或文件夹")

    # 平滑参数
    parser.add_argument("-c", "--cutoff", type=float, default=0.15, help="截止频率比例 (默认: 0.15)")
    parser.add_argument("-s", "--skip", type=int, default=7, help="跳过前几列 (默认: 7)")

    # 绘图参数
    parser.add_argument("-p", "--plot", action="store_true", help="是否生成对比图")
    parser.add_argument("--img_path", help="指定图片保存路径 (仅单文件处理时有效)")
    parser.add_argument("--show-plot", action="store_true", help="是否弹出窗口显示图片")

    args = parser.parse_args()

    input_path = Path(args.input)

    # 收集待处理文件
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process.extend(list(input_path.glob("*.csv")))
        files_to_process.extend(list(input_path.glob("*.pkl")))
    else:
        print(f"错误: 找不到路径 {args.input}")
        return

    if not files_to_process:
        print("未发现有效的 .csv 或 .pkl 文件")
        return

    print(f"找到 {len(files_to_process)} 个文件，准备开始处理...")

    # 创建输出目录
    if args.output and not Path(args.output).suffix:
        os.makedirs(args.output, exist_ok=True)

    for f in files_to_process:
        try:
            process_file(f, args)
        except Exception as e:
            print(f"处理文件 {f.name} 时出错: {e}")


if __name__ == "__main__":
    main()
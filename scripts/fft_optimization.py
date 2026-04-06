import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def fourier_smooth_mirrored(data, cutoff_ratio):
    """使用镜像法 + 离散傅里叶变换(DFT) 进行低通滤波"""
    n = len(data)

    # 1. 镜像延拓 (Symmetric Extension)
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
    smoothed_mirrored_data = np.fft.ifft(filtered_coeffs)
    smoothed_mirrored_real = np.real(smoothed_mirrored_data)

    # 5. 截断保留原始长度
    return smoothed_mirrored_real[:n]


def main():
    parser = argparse.ArgumentParser(description="使用镜像法+DFT平滑机器人关节轨迹数据并支持绘图")

    # 基础参数
    parser.add_argument("--input", help="输入的 CSV 文件路径")
    parser.add_argument("-o", "--output", help="输出的 CSV 文件路径 (默认: 原文件名_smoothed.csv)")
    parser.add_argument("-c", "--cutoff", type=float, default=0.15,
                        help="截止频率比例 (0.0 到 1.0，越小越平滑，默认: 0.15)")
    parser.add_argument("-s", "--skip", type=int, default=7,
                        help="跳过的前几列（通常是 Root 坐标，不进行平滑，默认: 7）")

    # 新增：绘图参数
    parser.add_argument("-p", "--plot", action="store_true",
                        help="是否在处理完成后显示平滑前后的对比图")
    parser.add_argument("--img_path", help="输出的图表路径")
    parser.add_argument("--plot-col", type=str, default=None,
                        help="指定要绘制对比图的列名 (默认: 被平滑的第一列)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到文件 {args.input}")
        return

    # 1. 读取数据
    df = pd.read_csv(args.input)
    smoothed_df = df.copy()

    print(f"正在处理: {args.input}")
    print(f"平滑系数 (Cutoff): {args.cutoff}")
    print("已启用: 镜像延拓法")

    # 2. 执行平滑
    cols_to_smooth = df.columns[args.skip:]
    num_joints = len(cols_to_smooth)
    if len(cols_to_smooth) == 0:
        print("警告: 根据 --skip 参数，没有列需要被平滑！")
        return

    for col in cols_to_smooth:
        original_values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        smoothed_df[col] = fourier_smooth_mirrored(original_values, args.cutoff)

    # 3. 保存 CSV
    output_path = args.output if args.output else args.input.replace(".csv", "_smoothed.csv")
    smoothed_df.to_csv(output_path, index=False)
    print(f"处理完成！数据已保存至: {output_path}")

    # 4. 绘制全关节网格对比图
    if args.plot:
        print("生成全关节对比图...")
        # 自动计算网格布局 (假设每行放 4 个子图)
        n_cols = 4
        n_rows = int(np.ceil(num_joints / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
        axes = axes.flatten()  # 展开为一维方便遍历

        for i, col in enumerate(cols_to_smooth):
            ax = axes[i]
            # 原始数据
            ax.plot(df[col].values, color='#1f77b4', alpha=1, label='Original')
            # 平滑后数据
            ax.plot(smoothed_df[col].values, color='#d62728', linewidth=1.5, label='Smoothed')

            ax.set_title(f"Joint: {col}", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            if i == 0:  # 只在第一个子图显示图例，节省空间
                ax.legend(loc='upper right', fontsize=8)

        # 隐藏多余的子图格子
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        # 保存图片
        img_path = args.img_path
        plt.savefig(img_path, dpi=150)
        print(f"对比图已保存至: {img_path}")

        # 弹出显示窗口
        plt.show()


if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def main():
    # 1. 参数配置
    parser = argparse.ArgumentParser(description='机器人关节数据多文件对比工具 (无表头版)')
    parser.add_argument('--input', type=str, nargs='+', required=True, help='输入一个或多个 CSV 文件路径')
    parser.add_argument('--fps', type=int, default=30, help='采样频率 (默认: 30)')
    parser.add_argument('--output', type=str, default='comparison_result.png', help='输出图片路径')
    parser.add_argument('--show', action='store_true', help='是否在绘图后弹窗显示')
    parser.add_argument('--skip', type=int, default=7, help='跳过前几列 (Root坐标，默认: 7)')

    args = parser.parse_args()

    # 2. 文件预检
    valid_files = []
    for pattern in args.input:
        # 处理通配符展开（以防某些系统终端不自动展开）
        import glob
        matched = glob.glob(pattern)
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        valid_files.extend(matched)

    if not valid_files:
        print("错误: 未找到任何有效的 CSV 文件，请检查路径。")
        return

    print(f"找到 {len(valid_files)} 个文件，准备开始绘制...")

    # 3. 确定关节范围 (读取第一个文件获取列数)
    try:
        sample_df = pd.read_csv(valid_files[0], header=None, nrows=1)
        total_cols = sample_df.shape[1]
        joint_indices = list(range(args.skip, total_cols))
        if not joint_indices:
            print(f"错误: 文件总列数为 {total_cols}，小于或等于跳过列数 {args.skip}。")
            return
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 4. 初始化画布 (5x4 布局)
    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 16), sharex=True)
    axes = axes.flatten()

    # 设置对比色和递减线宽 (确保重合时也能看清)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linewidths = [5, 3.5, 2, 1]  # 越靠后的文件线越细，叠在粗线上面

    legend_handles = []
    legend_labels = []

    # 5. 循环处理文件
    for f_idx, f_path in enumerate(valid_files):
        try:
            # 关键点：header=None 避免将第一行数值当做标题
            df = pd.read_csv(f_path, header=None)

            f_name = Path(f_path).name
            label = f_name.replace(".csv", "")

            time_seconds = df.index / args.fps
            color = colors[f_idx % len(colors)]
            lw = linewidths[f_idx % len(linewidths)] if f_idx < len(linewidths) else 1.0

            print(f"正在处理 [{f_idx + 1}/{len(valid_files)}]: {f_name}")
            print(f"  - 帧数: {len(df)} | 关节数: {len(joint_indices)}")

            first_line_obj = None

            for i, col_idx in enumerate(joint_indices):
                if i >= len(axes): break

                # 使用 .iloc 获取数据
                y_data = df.iloc[:, col_idx]

                line, = axes[i].plot(
                    time_seconds,
                    y_data,
                    color=color,
                    linewidth=lw,
                    alpha=0.8,
                    label=label if i == 0 else ""
                )

                if first_line_obj is None:
                    first_line_obj = line

                # 仅在处理第一个文件时设置背景和标题
                if f_idx == 0:
                    axes[i].set_title(f"Joint Index {col_idx}", fontsize=10, pad=2)
                    axes[i].grid(True, linestyle=':', alpha=0.4)
                    axes[i].tick_params(labelsize=8)

                if i >= (nrows * ncols - ncols):
                    axes[i].set_xlabel('Time (s)', fontsize=9)

            if first_line_obj:
                legend_handles.append(first_line_obj)
                legend_labels.append(label)

        except Exception as e:
            print(f"跳过文件 {f_path}，原因: {e}")

    # 6. 全局修饰
    plt.suptitle(f'Robot Joint Trajectory Comparison\nSource: {os.path.dirname(valid_files[0])}',
                 fontsize=20, y=0.98)

    # 动态调整图例位置
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='upper right',
                   bbox_to_anchor=(0.99, 0.97),
                   ncol=1,
                   fontsize=10,
                   frameon=True,
                   shadow=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # 7. 保存与展示
    plt.savefig(args.output, dpi=200)
    print(f"\n成功！对比图已保存至: {args.output}")

    if args.show:
        print("正在弹出窗口...")
        plt.show()


if __name__ == "__main__":
    main()
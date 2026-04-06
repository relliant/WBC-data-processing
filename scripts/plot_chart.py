import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def load_file(f_path, skip):
    """
    统一加载接口，返回 (dof_array: np.ndarray shape TxN, fps: int)
    - CSV: 跳过前 skip 列，其余列作为关节数据
    - PKL: 直接读取 dof_pos 字段 (shape TxN)，fps 从文件中读取
    """
    ext = Path(f_path).suffix.lower()
    if ext == '.pkl':
        with open(f_path, 'rb') as f:
            data = pickle.load(f)
        dof_pos = np.array(data['dof_pos'])   # TxN
        fps = int(data.get('fps', 30))
        return dof_pos, fps
    else:
        df = pd.read_csv(f_path, header=None)
        dof_array = df.iloc[:, skip:].to_numpy(dtype=float)
        return dof_array, None  # fps 由命令行参数决定


def main():
    # 1. 参数配置
    parser = argparse.ArgumentParser(description='机器人关节数据多文件对比工具 (支持 CSV / PKL)')
    parser.add_argument('--input', type=str, nargs='+', required=True, help='输入一个或多个 CSV / PKL 文件路径（支持通配符）')
    parser.add_argument('--fps', type=int, default=30, help='采样频率，仅对 CSV 文件生效 (默认: 30)')
    parser.add_argument('--output', type=str, default='comparison_result.png', help='输出图片路径')
    parser.add_argument('--show', action='store_true', help='是否在绘图后弹窗显示')
    parser.add_argument('--skip', type=int, default=7, help='CSV 文件跳过前几列 (Root坐标，默认: 7)')

    args = parser.parse_args()

    # 2. 文件预检
    valid_files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        valid_files.extend(matched)

    if not valid_files:
        print("错误: 未找到任何有效文件，请检查路径。")
        return

    print(f"找到 {len(valid_files)} 个文件，准备开始绘制...")

    # 3. 确定关节数 (读取第一个文件)
    try:
        sample_dof, _ = load_file(valid_files[0], args.skip)
        num_joints = sample_dof.shape[1]
        joint_indices = list(range(num_joints))
        if num_joints == 0:
            print("错误: 未检测到关节数据列。")
            return
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 4. 初始化画布 (5x4 布局)
    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 16), sharex=True)
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linewidths = [5, 3.5, 2, 1]

    legend_handles = []
    legend_labels = []

    # 5. 循环处理文件
    for f_idx, f_path in enumerate(valid_files):
        try:
            dof_array, file_fps = load_file(f_path, args.skip)
            fps = file_fps if file_fps is not None else args.fps

            f_name = Path(f_path).name
            label = Path(f_path).stem

            time_seconds = np.arange(len(dof_array)) / fps
            color = colors[f_idx % len(colors)]
            lw = linewidths[f_idx % len(linewidths)] if f_idx < len(linewidths) else 1.0

            print(f"正在处理 [{f_idx + 1}/{len(valid_files)}]: {f_name}")
            print(f"  - 帧数: {len(dof_array)} | 关节数: {dof_array.shape[1]} | FPS: {fps}")

            first_line_obj = None

            for i in range(min(len(joint_indices), len(axes))):
                if i >= dof_array.shape[1]:
                    break

                line, = axes[i].plot(
                    time_seconds,
                    dof_array[:, i],
                    color=color,
                    linewidth=lw,
                    alpha=0.8,
                )

                if first_line_obj is None:
                    first_line_obj = line

                if f_idx == 0:
                    axes[i].set_title(f"Joint {i}", fontsize=10, pad=2)
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
    # 隐藏多余子图
    for j in range(num_joints, len(axes)):
        axes[j].axis('off')
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
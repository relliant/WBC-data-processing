import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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


plt.style.use('seaborn-v0_8-whitegrid')


def main():
    # 1. 参数配置
    parser = argparse.ArgumentParser(description='机器人关节数据多文件对比工具 (支持 CSV / PKL)')
    parser.add_argument('--input', type=str, nargs='+', required=True, help='输入一个或多个 CSV / PKL 文件路径（支持通配符）')
    parser.add_argument('--fps', type=int, default=30, help='采样频率，仅对 CSV 文件生效 (默认: 30)')
    parser.add_argument('--output', type=str, default='comparison_result.png', help='输出图片路径')
    parser.add_argument('--show', action='store_true', help='是否在绘图后弹窗显示')
    parser.add_argument('--skip', type=int, default=7, help='CSV 文件跳过前几列 (Root坐标，默认: 7)')
    parser.add_argument('--joints', type=int, nargs=20, default=None, help='指定要绘制的4个关节索引 (默认: 0 1 2 3)')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'], help='绘图模式 (默认: 2d)')

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
        if num_joints == 0:
            print("错误: 未检测到关节数据列。")
            return
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 4. 确定要绘制的关节（全部关节）
    plot_joints = args.joints if args.joints else list(range(num_joints))

    colors = ['#670373', '#528aaa', '#c00405', '#b2cdce', '#fefff0', '#cab2d7', '#78969b', '#ff878a', '#343434']

    # 预加载所有文件数据
    file_data = []
    for f_path in valid_files:
        try:
            dof_array, file_fps = load_file(f_path, args.skip)
            fps = file_fps if file_fps is not None else args.fps
            file_data.append((f_path, dof_array, fps))
            print(f"已加载: {Path(f_path).name} | 帧数: {len(dof_array)} | 关节数: {dof_array.shape[1]} | FPS: {fps}")
        except Exception as e:
            print(f"跳过文件 {f_path}，原因: {e}")

    file_data = [(p, d[:int(fps * 10)], fps) for p, d, fps in file_data]

    ncols = 4
    nrows = (len(plot_joints) + ncols - 1) // ncols

    if args.mode == '3d':
        fig = plt.figure(figsize=(ncols * 4, nrows * 3))
        labels = [Path(p).stem for p, _, _ in file_data]
        for plot_i, joint_i in enumerate(plot_joints):
            ax = fig.add_subplot(nrows, ncols, plot_i + 1, projection='3d')
            for f_idx, (f_path, dof_array, fps) in enumerate(file_data):
                if joint_i >= dof_array.shape[1]:
                    continue
                t = np.arange(len(dof_array)) / fps
                ax.plot(t, np.full_like(t, f_idx), dof_array[:, joint_i],
                        color=colors[f_idx % len(colors)], linewidth=1)
            ax.set_title(f"Joint {joint_i}", fontsize=9, pad=2)
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.set_yticks(range(len(file_data)))
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_zlabel('rad', fontsize=7)
            ax.tick_params(labelsize=6)
        for plot_i in range(len(plot_joints), nrows * ncols):
            fig.add_subplot(nrows, ncols, plot_i + 1).set_visible(False)
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.2, nrows * 2.5))
        axes = axes.flatten()
        legend_handles, legend_labels = [], []

        for f_idx, (f_path, dof_array, fps) in enumerate(file_data):
            time_seconds = np.arange(len(dof_array)) / fps
            color = colors[f_idx % len(colors)]
            label = Path(f_path).stem
            first_line_obj = None

            for plot_i, joint_i in enumerate(plot_joints):
                if joint_i >= dof_array.shape[1]:
                    continue
                ax = axes[plot_i]
                y = dof_array[:, joint_i]
                line, = ax.plot(time_seconds, y, color=color, linewidth=1)
                if first_line_obj is None:
                    first_line_obj = line
                if f_idx == 0:
                    ax.set_title(f"Joint {joint_i}", fontsize=10, pad=4)
                    ax.grid(True, linestyle=':', alpha=0.4)
                    ax.tick_params(labelsize=8)
                    ax.margins(x=0)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.set_ylabel('rad', fontsize=9)

            if first_line_obj:
                legend_handles.append(first_line_obj)
                legend_labels.append(label)

        for ax in axes[len(plot_joints):]:
            ax.set_visible(False)

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper right',
                       bbox_to_anchor=(0.99, 0.99), ncol=1, fontsize=9, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
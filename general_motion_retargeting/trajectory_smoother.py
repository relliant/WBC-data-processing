import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class TrajectorySmoother:
    """
    机器人关节轨迹平滑工具类
    基于镜像延拓法 + 离散傅里叶变换 (DFT)
    """

    def __init__(self, cutoff=0.2, skip=7):
        """
        初始化平滑器
        :param cutoff: 截止频率比例 (0.0 到 1.0，越小越平滑，默认: 0.2)
        :param skip: 跳过的前几列（通常是 Root 坐标，不进行平滑，默认: 7）
        """
        self.cutoff = cutoff
        self.skip = skip

    @staticmethod
    def _fourier_smooth_mirrored(data, cutoff_ratio):
        """内部静态方法：使用镜像法 + DFT 进行低通滤波"""
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

    def smooth_arrays(self, data_list):
        """
        直接平滑数组列表或 2D Numpy 数组
        :param data_list: 形如 [array1, array2, ...] 的列表，或 shape 为 (num_joints, num_frames) 的 2D 数组
        :return: 与输入结构相同的平滑后数据
        """
        smoothed_list = []
        for arr in data_list:
            # 确保转为 1D float numpy 数组进行计算
            arr_np = np.asarray(arr, dtype=float)
            smoothed_arr = self._fourier_smooth_mirrored(arr_np, self.cutoff)
            smoothed_list.append(smoothed_arr)

        # 如果输入是 Numpy 2D 数组，则保持输出也是 Numpy 2D 数组
        if isinstance(data_list, np.ndarray) and data_list.ndim == 2:
            return np.array(smoothed_list)

        return smoothed_list

    def plot_arrays(self, original_list, smoothed_list, joint_names=None, show_plot=True):
        """
        绘制原生数组的对比图
        """
        num_joints = len(original_list)
        if num_joints == 0:
            return

        n_cols = 4
        n_rows = int(np.ceil(num_joints / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
        # 兼容只有一个关节（axes 不是数组）的情况
        if num_joints == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(num_joints):
            ax = axes[i]
            title = joint_names[i] if joint_names and i < len(joint_names) else f"Joint {i}"

            ax.plot(original_list[i], color='#1f77b4', alpha=1, label='Original')
            ax.plot(smoothed_list[i], color='#d62728', linewidth=1.5, label='Smoothed')

            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        for j in range(num_joints, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def smooth_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在内存中直接处理 DataFrame
        :param df: 原始数据的 DataFrame
        :return: 平滑后的 DataFrame
        """
        smoothed_df = df.copy()
        cols_to_smooth = df.columns[self.skip:]

        if len(cols_to_smooth) == 0:
            print("警告: 根据 skip 参数，没有列需要被平滑！")
            return smoothed_df

        for col in cols_to_smooth:
            original_values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            smoothed_df[col] = self._fourier_smooth_mirrored(original_values, self.cutoff)

        return smoothed_df

    def plot_comparison(self, original_df: pd.DataFrame, smoothed_df: pd.DataFrame,
                        img_path: str = None, show_plot: bool = True):
        """
        绘制全关节网格对比图
        :param original_df: 原始 DataFrame
        :param smoothed_df: 平滑后的 DataFrame
        :param img_path: 图片保存路径 (可选)
        :param show_plot: 是否在屏幕上弹出显示 (默认: True)
        """
        cols_to_smooth = original_df.columns[self.skip:]
        num_joints = len(cols_to_smooth)

        if num_joints == 0:
            return

        print("生成全关节对比图...")
        n_cols = 4
        n_rows = int(np.ceil(num_joints / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
        axes = axes.flatten()

        for i, col in enumerate(cols_to_smooth):
            ax = axes[i]
            # 原始数据
            ax.plot(original_df[col].values, color='#1f77b4', alpha=1, label='Original')
            # 平滑后数据
            ax.plot(smoothed_df[col].values, color='#d62728', linewidth=1.5, label='Smoothed')

            ax.set_title(f"Joint: {col}", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        # 隐藏多余的子图格子
        for j in range(len(cols_to_smooth), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        # 保存图片
        if img_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(img_path)), exist_ok=True)
            plt.savefig(img_path, dpi=150)
            print(f"对比图已保存至: {img_path}")

        # 弹出显示窗口
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # 如果不显示，清理内存

    def process_file(self, input_path: str, output_path: str = None,
                     plot: bool = False, img_path: str = None, show_plot: bool = True):
        """
        一键处理文件：读取 -> 平滑 -> 保存 -> (可选)绘图
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"错误: 找不到文件 {input_path}")

        print(f"正在处理: {input_path} (Cutoff: {self.cutoff})")

        # 1. 读取数据
        df = pd.read_csv(input_path)

        # 2. 执行平滑
        smoothed_df = self.smooth_dataframe(df)

        # 3. 保存 CSV
        if output_path is None:
            output_path = input_path.replace(".csv", "_smoothed.csv")
        smoothed_df.to_csv(output_path, index=False)
        print(f"处理完成！数据已保存至: {output_path}")

        # 4. 绘图
        if plot:
            default_img_path = img_path if img_path else input_path.replace(".csv", "_plot.png")
            self.plot_comparison(df, smoothed_df, img_path=default_img_path, show_plot=show_plot)
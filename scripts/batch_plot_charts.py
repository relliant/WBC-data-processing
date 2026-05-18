import glob
import argparse
import os
import subprocess
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='批量对同目录下的CSV文件绘制对比图')
    parser.add_argument('--folder', type=str, required=True, help='包含CSV文件的根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片的根目录')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--skip', type=int, default=7)
    args = parser.parse_args()

    # 按父目录分组
    groups = defaultdict(list)
    for csv_path in glob.glob(os.path.join(args.folder, '**', '*.csv'), recursive=True):
        parent = str(Path(csv_path).parent)
        groups[parent].append(csv_path)

    print(f"找到 {len(groups)} 个目录，共 {sum(len(v) for v in groups.values())} 个CSV文件")

    for parent_dir, csv_files in groups.items():
        rel = os.path.relpath(parent_dir, args.folder)
        out_path = os.path.join(args.output_dir, rel + '.png')

        cmd = [
            'python', 'scripts/plot_chart.py',
            '--input', *sorted(csv_files),
            '--output', out_path,
            '--fps', str(args.fps),
            '--skip', str(args.skip),
        ]
        print(f"绘制: {rel} ({len(csv_files)} 个文件) -> {out_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  失败: {result.stderr.strip()}")


if __name__ == '__main__':
    main()

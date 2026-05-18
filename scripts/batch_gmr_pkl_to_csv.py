import argparse
import pickle
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory (default: <folder>/csv)",
    )
    args = parser.parse_args()
    output_base = args.output_dir if args.output_dir else os.path.join(args.folder, "csv")

    # Collect all pkl files recursively
    pkl_files = []
    for root, dirs, files in os.walk(args.folder):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))

    if not pkl_files:
        print(f"No .pkl files found in {args.folder}")
        exit(0)

    print(f"Found {len(pkl_files)} .pkl files, starting conversion...")

    for i, pkl_path in enumerate(pkl_files):
        try:
            with open(pkl_path, "rb") as f:
                motion_data = pickle.load(f)

            dof_pos = motion_data["dof_pos"]
            frame_rate = motion_data["fps"]
            motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
            motion[:, :3] = motion_data["root_pos"]
            motion[:, 3:7] = motion_data["root_rot"]
            motion[:, 7:] = dof_pos

            if frame_rate > 30:
                # downsample to 30 fps
                downsample_factor = frame_rate / 30.0
                indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
                old_length = motion.shape[0]
                motion = motion[indices]
                print(f"  Downsampled from {old_length} to {motion.shape[0]} frames")

            rel_path = os.path.relpath(pkl_path, args.folder)
            csv_path = os.path.join(output_base, rel_path.replace(".pkl", ".csv"))
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            np.savetxt(csv_path, motion, delimiter=",")
            print(f"[{i+1}/{len(pkl_files)}] Saved to {csv_path}")

        except Exception as e:
            print(f"[{i+1}/{len(pkl_files)}] Error processing {pkl_path}: {e}")

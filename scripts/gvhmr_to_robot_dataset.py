import argparse
import pathlib
import os
import sys
import time
import pickle

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


def run_single_file(gvhmr_pred_file, robot, save_path, record_video=False, video_path=None, rate_limit=False, loop=False):
    smplx_folder = HERE / ".." / "assets" / "body_models"

    # Load GVHMR trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        gvhmr_pred_file, smplx_folder
    )

    # Align FPS
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    # Initialize retargeting
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
    )

    # In headless environments (no DISPLAY), disable viewer and video recording.
    has_display = bool(os.environ.get("DISPLAY"))
    enable_viewer = has_display
    if record_video and not has_display:
        print("[WARNING] DISPLAY is missing. Disable video recording in headless mode.")
        record_video = False

    robot_motion_viewer = None
    if enable_viewer:
        if video_path is None:
            video_path = f"videos/{robot}_{pathlib.Path(gvhmr_pred_file).stem}.mp4"
        video_dir = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        robot_motion_viewer = RobotMotionViewer(
            robot_type=robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=record_video,
            video_path=video_path,
        )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    qpos_list = []

    i = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    while True:
        if loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        if enable_viewer:
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

        smplx_frame_data = smplx_data_frames[i]
        qpos = retarget.retarget(smplx_frame_data)

        if enable_viewer:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=rate_limit,
            )

        qpos_list.append(qpos)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # save from wxyz to xyzw
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()


def find_target_pt_files(src_folder, pt_name):
    target_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in filenames:
            if filename == pt_name:
                target_files.append(os.path.join(dirpath, filename))
    return sorted(target_files)


def build_output_paths(src_folder, tgt_folder, input_pt_path, video_ext):
    rel_dir = os.path.relpath(os.path.dirname(input_pt_path), src_folder)
    if rel_dir == ".":
        rel_dir = ""
    motion_name = os.path.basename(os.path.dirname(input_pt_path))

    out_dir = os.path.join(tgt_folder, rel_dir)
    save_path = os.path.join(out_dir, f"{motion_name}.pkl")
    video_path = os.path.join(out_dir, f"{motion_name}.{video_ext}")
    return save_path, video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, required=True, help="Folder containing GVHMR outputs.")
    parser.add_argument("--tgt_folder", type=str, required=True, help="Folder to save retargeted outputs.")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong", "tienkung"
        ],
        default="tienkung",
    )
    parser.add_argument(
        "--pt_name",
        type=str,
        default="hmr4d_results.pt",
        help="Only process files with this exact file name.",
    )
    parser.add_argument("--override", default=False, action="store_true", help="Override existing output files.")
    parser.add_argument("--record_video", default=False, action="store_true", help="Record video if display is available.")
    parser.add_argument("--rate_limit", default=False, action="store_true", help="Enable rate limit in viewer.")
    parser.add_argument("--loop", default=False, action="store_true", help="Loop motion in viewer mode.")
    parser.add_argument("--video_ext", default="mp4", help="Video extension when --record_video is set.")

    args = parser.parse_args()

    src_folder = os.path.abspath(args.src_folder)
    tgt_folder = os.path.abspath(args.tgt_folder)
    os.makedirs(tgt_folder, exist_ok=True)

    input_files = find_target_pt_files(src_folder, args.pt_name)
    print(f"Found {len(input_files)} files named '{args.pt_name}' in {src_folder}")

    if len(input_files) == 0:
        print("[WARNING] No input files found. Nothing to do.")
        return

    success = 0
    skipped = 0
    failed = 0

    for idx, input_pt_path in enumerate(input_files, start=1):
        save_path, video_path = build_output_paths(src_folder, tgt_folder, input_pt_path, args.video_ext)

        if os.path.exists(save_path) and not args.override:
            skipped += 1
            print(f"[{idx}/{len(input_files)}] [SKIP] {save_path} already exists")
            continue

        print(f"[{idx}/{len(input_files)}] [RUN] {input_pt_path} -> {save_path}")
        try:
            run_single_file(
                gvhmr_pred_file=input_pt_path,
                robot=args.robot,
                save_path=save_path,
                record_video=args.record_video,
                video_path=video_path,
                rate_limit=args.rate_limit,
                loop=args.loop,
            )
            success += 1
            print(f"[{idx}/{len(input_files)}] [OK] {save_path}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{len(input_files)}] [FAIL] {input_pt_path}")
            print(f"Error: {e}")

    print("\nBatch retargeting finished.")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output root: {tgt_folder}")


if __name__ == "__main__":
    main()

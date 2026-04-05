#!/usr/bin/env python3
"""
Merge multiple PKL files into a single PKL mapping compatible with
`convert_motions.py`. This version treats `dof_pos` as the robot `joint_pos`
(i.e. exactly equivalent) and will copy/normalize it accordingly.

Usage:
  python gear_sonic_deploy/reference/wrap_pkls.py --input-dir /path/to/pkls --output merged.pkl
  python gear_sonic_deploy/reference/wrap_pkls.py file1.pkl file2.pkl --output merged.pkl
"""
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

try:
    import joblib
except Exception:
    joblib = None

import numpy as np

DEFAULT_BODY_COUNT = 14
DEFAULT_JOINT_COUNT = 29

ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

GROUPED_TO_MUJOCO_DOF = [
    # Source joint order from the retargeter (left leg, right leg, left arm, right arm, neck/head)
    # Target MuJoCo order in the G1 29-DOF model (legs, waist, left arm, right arm)
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    12,
    13,
    14,
]


def _reorder_joint_positions(joint_pos: np.ndarray, source_order: str) -> np.ndarray:
    if source_order == 'mujoco':
        return joint_pos
    if joint_pos.shape[1] != DEFAULT_JOINT_COUNT:
        return joint_pos

    if source_order == 'isaaclab':
        return joint_pos[:, ISAACLAB_TO_MUJOCO_DOF]
    if source_order == 'grouped':
        return joint_pos[:, GROUPED_TO_MUJOCO_DOF]

    raise ValueError(f"Unknown dof source order: {source_order}")


def load_pkl(path: Path) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, 'rb') as f:
        return pickle.load(f)


def _infer_timesteps(motion: Dict[str, Any]) -> int:
    for k in ("joint_pos", "dof_pos", "root_pos", "root_rot"):
        v = motion.get(k)
        if v is None:
            continue
        try:
            return int(getattr(v, 'shape', (len(v),))[0])
        except Exception:
            continue
    return 0


def _normalize_motion(motion: Dict[str, Any], dof_source_order: str = 'grouped') -> Dict[str, Any]:
    """
    Normalize a single motion dict:
      - Treat `dof_pos` as `joint_pos` (copy if joint_pos missing).
      - Ensure `joint_vel` exists (estimate if needed).
      - Ensure body arrays exist (fill zeros / unit quat).
    """
    m = dict(motion) if isinstance(motion, dict) else {}

    T = _infer_timesteps(m)

    # 1) joint_pos: 直接采用 dof_pos 顺序
    if 'joint_pos' not in m or m.get('joint_pos') is None:
        if 'dof_pos' in m and m.get('dof_pos') is not None:
            dof_pos = np.asarray(m['dof_pos'], dtype=np.float32)
            m['joint_pos'] = dof_pos
        else:
            m['joint_pos'] = np.zeros((T, DEFAULT_JOINT_COUNT), dtype=np.float32)
    else:
        m['joint_pos'] = np.asarray(m['joint_pos'], dtype=np.float32)

    # 1.5) body_pos_w/body_quat_w: 用 root_pos/root_rot 填充所有 body
    body_count = DEFAULT_BODY_COUNT
    if 'link_body_list' in m and m.get('link_body_list'):
        try:
            link_list = m.get('link_body_list')
            body_count = len(link_list) if link_list is not None else DEFAULT_BODY_COUNT
        except Exception:
            body_count = DEFAULT_BODY_COUNT

    if 'body_pos_w' not in m or m.get('body_pos_w') is None:
        if 'root_pos' in m and m.get('root_pos') is not None:
            root_pos = np.asarray(m['root_pos'], dtype=np.float32)
            m['body_pos_w'] = np.tile(root_pos[:, None, :], (1, body_count, 1))
        else:
            m['body_pos_w'] = np.zeros((m['joint_pos'].shape[0], body_count, 3), dtype=np.float32)
    else:
        m['body_pos_w'] = np.asarray(m['body_pos_w'], dtype=np.float32)

    if 'body_quat_w' not in m or m.get('body_quat_w') is None:
        if 'root_rot' in m and m.get('root_rot') is not None:
            root_rot = np.asarray(m['root_rot'], dtype=np.float32)
            # 原始四元数顺序即为 wxyz，无需转换
            root_rot_wxyz = root_rot
            m['body_quat_w'] = np.tile(root_rot_wxyz[:, None, :], (1, body_count, 1))
        else:
            quat = np.zeros((m['joint_pos'].shape[0], body_count, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            m['body_quat_w'] = quat
    else:
        m['body_quat_w'] = np.asarray(m['body_quat_w'], dtype=np.float32)

    # 2) joint_vel: estimate if missing
    if 'joint_vel' not in m or m.get('joint_vel') is None:
        if m['joint_pos'].shape[0] >= 2:
            # simple finite-difference / gradient estimate
            m['joint_vel'] = np.asarray(np.gradient(m['joint_pos'], axis=0), dtype=np.float32)
        else:
            m['joint_vel'] = np.zeros_like(m['joint_pos'], dtype=np.float32)
    else:
        m['joint_vel'] = np.asarray(m['joint_vel'], dtype=np.float32)

    # 3) body_count: try to infer from link_body_list, else use default
    body_count = DEFAULT_BODY_COUNT
    if 'link_body_list' in m and m.get('link_body_list'):
        try:
            link_list = m.get('link_body_list')
            body_count = len(link_list) if link_list is not None else DEFAULT_BODY_COUNT
        except Exception:
            body_count = DEFAULT_BODY_COUNT

    # 4) body_pos_w: try local_body_pos fallback, else zeros
    if 'body_pos_w' not in m or m.get('body_pos_w') is None:
        if 'local_body_pos' in m and m.get('local_body_pos') is not None:
            try:
                arr = np.asarray(m['local_body_pos'], dtype=np.float32)
                if arr.ndim == 3:
                    # Convert local pose to world with root_pos/root_rot if available
                    root_pos = np.asarray(m.get('root_pos', np.zeros((T, 3), dtype=np.float32)), dtype=np.float32)
                    root_rot = np.asarray(m.get('root_rot', np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (T, 1))), dtype=np.float32)

                    def quat_rotate(q, v):
                        # q: [x, y, z, w], v: [..., 3]
                        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
                        # quaternion-vector multiplication: p' = q * [v,0] * q_conj
                        t = 2.0 * (x * v[..., 0] + y * v[..., 1] + z * v[..., 2])
                        u = np.stack([w * v[..., 0] + y * v[..., 2] - z * v[..., 1],
                                      w * v[..., 1] + z * v[..., 0] - x * v[..., 2],
                                      w * v[..., 2] + x * v[..., 1] - y * v[..., 0]], axis=-1)
                        return v + t[..., None] * np.stack([x, y, z], axis=-1) + np.cross(np.stack([x, y, z], axis=-1), u)

                    if root_pos.shape[0] == arr.shape[0] and root_rot.shape[0] == arr.shape[0]:
                        world = np.zeros_like(arr)
                        for i in range(T):
                            world[i] = quat_rotate(root_rot[i], arr[i]) + root_pos[i]
                        m['body_pos_w'] = world
                    else:
                        m['body_pos_w'] = arr
                else:
                    m['body_pos_w'] = np.zeros((m['joint_pos'].shape[0], body_count, 3), dtype=np.float32)
            except Exception:
                m['body_pos_w'] = np.zeros((m['joint_pos'].shape[0], body_count, 3), dtype=np.float32)
        else:
            m['body_pos_w'] = np.zeros((m['joint_pos'].shape[0], body_count, 3), dtype=np.float32)
    else:
        m['body_pos_w'] = np.asarray(m['body_pos_w'], dtype=np.float32)

    # 5) body_quat_w: default to unit quaternion [1,0,0,0]
    if 'body_quat_w' not in m or m.get('body_quat_w') is None:
        base_quat = np.asarray(m.get('root_rot', np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (T, 1))), dtype=np.float32)
        # root_rot in source data appears as [x,y,z,w]; convert to MUJOCO [w,x,y,z]
        if base_quat.ndim == 2 and base_quat.shape[1] == 4:
            base_quat = base_quat[:, [3, 0, 1, 2]]
        quat = np.zeros((m['joint_pos'].shape[0], m['body_pos_w'].shape[1], 4), dtype=np.float32)
        if base_quat.shape[0] == m['joint_pos'].shape[0] and base_quat.shape[1] == 4:
            quat[:] = base_quat[:, None, :]
        else:
            quat[..., 0] = 1.0
        m['body_quat_w'] = quat
    else:
        m['body_quat_w'] = np.asarray(m['body_quat_w'], dtype=np.float32)

    # 6) body linear/angular velocities
    if 'body_lin_vel_w' not in m or m.get('body_lin_vel_w') is None:
        m['body_lin_vel_w'] = np.zeros((m['joint_pos'].shape[0], m['body_pos_w'].shape[1], 3), dtype=np.float32)
    else:
        m['body_lin_vel_w'] = np.asarray(m['body_lin_vel_w'], dtype=np.float32)

    if 'body_ang_vel_w' not in m or m.get('body_ang_vel_w') is None:
        m['body_ang_vel_w'] = np.zeros((m['joint_pos'].shape[0], m['body_pos_w'].shape[1], 3), dtype=np.float32)
    else:
        m['body_ang_vel_w'] = np.asarray(m['body_ang_vel_w'], dtype=np.float32)

    # Keep original dof_pos and other keys as-is (we copied dof_pos -> joint_pos if needed)
    return m


def merge_and_normalize(paths, output_path: Path, dof_source_order: str = 'grouped') -> None:
    merged: Dict[str, Any] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"Warning: {p} does not exist, skipping")
            continue
        print(f"Loading {p}")
        obj = load_pkl(p)

        if isinstance(obj, dict):
            sample_val = None
            if len(obj) > 0:
                sample_val = next(iter(obj.values()))

            if sample_val is None:
                name = p.stem
                normalized = _normalize_motion(obj, dof_source_order)
                merged[name] = normalized
            elif not isinstance(sample_val, dict):
                # single-motion dict
                name = p.stem
                normalized = _normalize_motion(obj, dof_source_order)
                merged[name] = normalized
            else:
                # obj is mapping of motions
                for key, val in obj.items():
                    name = key
                    if name in merged:
                        name = f"{p.stem}_{key}"
                        print(f"Key collision: renaming {key} -> {name}")
                    normalized = _normalize_motion(val, dof_source_order)
                    merged[name] = normalized
        else:
            # Non-dict (rare): wrap as single motion, treating object as dof_pos
            name = p.stem
            normalized = _normalize_motion({'dof_pos': obj})
            merged[name] = normalized

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'wb') as f:
        pickle.dump(merged, f)
    print(f"Wrote merged normalized PKL to {outp} (motions: {len(merged)})")


def collect_input_paths(args) -> list:
    paths = []
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            raise SystemExit(f"--input-dir {d} is not a directory")
        for p in sorted(d.glob('*.pkl')):
            paths.append(str(p))
    if args.files:
        for f in args.files:
            paths.append(f)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Wrap and normalize multiple motion PKLs into one mapping PKL for convert_motions.py")
    parser.add_argument('files', nargs='*', help='Explicit pkl files to include')
    parser.add_argument('--input-dir', help='Directory containing pkl files (will include *.pkl)')
    parser.add_argument('--output', required=True, help='Output merged pkl path')
    parser.add_argument(
        '--dof-source-order',
        choices=['isaaclab', 'grouped', 'mujoco'],
        default='grouped',
        help='Source dof order of dof_pos/joint_pos in input files',
    )
    args = parser.parse_args()

    paths = collect_input_paths(args)
    if not paths:
        raise SystemExit("No input PKL files found. Provide files or --input-dir")

    dof_source_order = getattr(args, 'dof_source_order', 'grouped')
    merge_and_normalize(paths, Path(args.output), dof_source_order)


if __name__ == '__main__':
    main()

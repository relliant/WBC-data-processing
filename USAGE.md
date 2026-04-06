# GMR Motion Retargeting — Usage Guide

## 1. Activate virtual environment
```bash
conda activate gmr
```

---

## 2. Retargeting: SMPL-X → Robot

### Single file (with visualization)

`smplx_to_robot.py` loads a single SMPL-X `.npz` file, retargets it to the target robot, visualizes the result in MuJoCo, and optionally saves the output as a `.pkl` file.

```bash
python scripts/smplx_to_robot.py \
    --smplx_file <path_to_smplx_data.npz> \
    --robot <robot_name> \
    --save_path <output.pkl> \
    --rate_limit
```

| Argument | Description |
|---|---|
| `--smplx_file` | Path to the SMPL-X `.npz` motion file |
| `--robot` | Target robot name (see supported robots below) |
| `--save_path` | Path to save the retargeted `.pkl` file (optional) |
| `--rate_limit` | Slow down playback to match the original motion FPS |
| `--loop` | Loop the motion in the viewer |
| `--record_video` | Record the MuJoCo viewer to a video file |
| `--fft` | Apply FFT-based low-pass smoothing to joint trajectories before saving |
| `--fft_cutoff` | Cutoff frequency ratio for FFT smoothing (0.0~1.0, smaller = smoother, default: `0.2`) |

Example:
```bash
# Basic retargeting with visualization
python scripts/smplx_to_robot.py \
    --smplx_file datasets/AMASS/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz \
    --robot tienkung \
    --save_path retargeting_results/stand_tienkung.pkl \
    --rate_limit

# With FFT smoothing (cutoff=0.1 for stronger smoothing)
python scripts/smplx_to_robot.py \
    --smplx_file datasets/AMASS/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz \
    --robot unitree_g1 \
    --save_path retargeting_results/stand_g1_smooth.pkl \
    --fft --fft_cutoff 0.1

# Record video
python scripts/smplx_to_robot.py \
    --smplx_file datasets/AMASS/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz \
    --robot tienkung \
    --save_path retargeting_results/stand_tienkung.pkl \
    --rate_limit --record_video
```

Videos are saved to `videos/`.

---

### Batch processing (dataset)

`smplx_to_robot_dataset.py` recursively processes all `.npz` files under `--src_folder`, retargets them, and saves `.pkl` files to `--tgt_folder` with the same directory structure. Supports multiprocessing and memory monitoring.

```bash
python scripts/smplx_to_robot_dataset.py \
    --src_folder <path_to_smplx_dataset/> \
    --tgt_folder <path_to_output_folder/> \
    --robot <robot_name>
```

| Argument | Description |
|---|---|
| `--src_folder` | Root folder containing SMPL-X `.npz` files |
| `--tgt_folder` | Root folder to save retargeted `.pkl` files |
| `--robot` | Target robot name |
| `--num_cpus` | Number of parallel worker processes (default: `4`) |
| `--override` | Re-process files even if the output `.pkl` already exists |
| `--fft` | Apply FFT-based low-pass smoothing to joint trajectories |
| `--fft_cutoff` | Cutoff frequency ratio for FFT smoothing (0.0~1.0, default: `0.2`) |

Example:
```bash
# Basic batch retargeting
python scripts/smplx_to_robot_dataset.py \
    --src_folder ~/datasets/AMASS/ \
    --tgt_folder retargeting_results/unitree_g1/ \
    --robot unitree_g1 \
    --num_cpus 8

# With FFT smoothing
python scripts/smplx_to_robot_dataset.py \
    --src_folder ~/datasets/AMASS/ \
    --tgt_folder retargeting_results/unitree_g1_smooth/ \
    --robot unitree_g1 \
    --num_cpus 8 \
    --fft --fft_cutoff 0.2
```

Output `.pkl` files contain: `fps`, `root_pos`, `root_rot`, `dof_pos`, `local_body_pos`, `link_body_list`.

---

### FFT trajectory smoothing

Both scripts support FFT-based low-pass filtering on joint position trajectories (`dof_pos`). The method uses mirrored symmetric extension before FFT to avoid boundary artifacts.

- `--fft`: enable smoothing (disabled by default)
- `--fft_cutoff`: cutoff ratio in `[0.0, 1.0]`
  - smaller → more aggressive smoothing, fewer high-frequency components retained
  - larger → closer to the original trajectory
  - recommended range: `0.1` (very smooth) to `0.3` (light smoothing)

To apply FFT smoothing to an already-saved CSV file:
```bash
# Single file
python scripts/fft_optimization.py \
    --input <input.csv> \
    --output <output_smoothed.csv> \
    --plot --img_path <plot.png>

# Batch
python scripts/fft_optimization_batch.py \
    --folder_path <folder_of_csvs/> \
    --output_path <output_folder/> -p
```

---

## 3. Retargeting: FBX (OptiTrack) → Robot

```bash
conda activate gmr_mocap
cd GMR/third_party

export LD_PRELOAD=/home/vega/anaconda3/envs/gmr_mocap/lib/libxml2.so
export PYTHONPATH=$PYTHONPATH:$(pwd)/poselib/skeleton/backend/fbx/

# Step 1: import FBX to pkl
python poselib/fbx_importer.py \
    --input ../datasets/Mocap/fbx/Hand.fbx \
    --output ../retargeting_results/fbx_to_pkl/Hand_motion.pkl \
    --root-joint root

cd GMR

# Step 2: convert skeleton joint names
python scripts/convert_fbx_skeleton_names.py \
    --input retargeting_results/fbx_to_pkl/Hand_motion.pkl \
    --output retargeting_results/fbx_to_pkl/converted/Hand_motion_converted.pkl

# Step 3: retarget to robot
conda activate gmr
python scripts/fbx_offline_to_robot.py \
    --motion_file retargeting_results/fbx_to_pkl/converted/Hand_motion_converted.pkl \
    --save_path retargeting_results/fbx_to_pkl/{robot}/Hand_final_{robot}.pkl \
    --rate_limit
```

---

## 4. Formalization: `.pkl` → `.csv`

```bash
# Batch convert all pkl files in a folder to CSV
python scripts/batch_gmr_pkl_to_csv.py --folder <path_to_pkl_files/>
```

Output CSVs are saved to `<path_to_pkl_files>/csv/`.

---

## 5. Visualization

### Single motion
```bash
python scripts/vis_robot_motion.py \
    --robot <robot_name> \
    --robot_motion_path <path_to_motion.pkl>
```

Add `--record_video --video_path <output.mp4>` to record.

### Folder of motions
```bash
python scripts/vis_robot_motion_dataset.py \
    --robot <robot_name> \
    --robot_motion_folder <path_to_folder/>
```

Keyboard controls in the MuJoCo viewer:

| Key | Action |
|---|---|
| `[` | Previous motion |
| `]` | Next motion |
| `space` | Toggle play / pause |

---

## 6. Plot joint trajectories

`plot_chart.py` plots joint position trajectories from one or more files for visual comparison. Supports both `.csv` and `.pkl` files.

```bash
python scripts/plot_chart.py \
    --input <file1.pkl> <file2.pkl> \
    --output comparison.png \
    --show
```

| Argument | Description |
|---|---|
| `--input` | One or more `.csv` or `.pkl` files (supports glob patterns) |
| `--output` | Output image path (default: `comparison_result.png`) |
| `--fps` | Sampling rate for CSV files (default: `30`); PKL files use their internal FPS |
| `--skip` | Number of leading columns to skip for CSV files (default: `7`, i.e. root pose) |
| `--show` | Pop up the plot window after saving |

Example — compare original vs smoothed:
```bash
python scripts/plot_chart.py \
    --input retargeting_results/stand_g1.pkl retargeting_results/stand_g1_smooth.pkl \
    --output plots/compare_smooth.png \
    --show
```

---

## Supported robots

| Name | DOF | Notes |
|---|---|---|
| `unitree_g1` | 29 | |
| `unitree_g1_with_hands` | 43 | With dexterous hands |
| `booster_t1` | — | Full-body |
| `booster_k1` | 22 | |
| `stanford_toddy` | — | Research platform |
| `fourier_n1` | — | |
| `fourier_gr3` | — | |
| `engineai_pm01` | — | |
| `kuavo_s45` | 28 | |
| `hightorque_hi` | 25 | |
| `galaxea_r1pro` | 24 | Wheeled humanoid |
| `tienkung` | — | |
| `walker` | — | |
| `openloong` | — | |
| `pnd_adam_lite` | — | |
| `berkeley_humanoid_lite` | — | |

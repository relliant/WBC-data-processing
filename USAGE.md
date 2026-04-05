# Common GMR retargeting USAGE

## 1.activate virtual environment
```bash
conda activate gmr
```

## 2.retargeting
### from oepn-sourcce datasets(SMPLX) to robot
```bash
# don't record video
python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit
# example
python scripts/smplx_to_robot.py --smplx_file datasets/AMASS/ACCAD/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz --robot tienkung --save_path retargeting_results/ --rate_limit

# retargeting from directory
python scripts/smplx_to_robot_dataset.py --src_folder <path_to_dir_of_smplx_data> --tgt_folder <path_to_dir_to_save_robot_data> --robot <robot_name>

# record video
python scripts/smplx_to_robot.py --smplx_file <path_to_smplx_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit --record_video
# example
python scripts/smplx_to_robot.py --smplx_file datasets/AMASS/ACCAD/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz --robot tienkung --save_path retargeting_results/ --rate_limit --record_video
```

#### Discrete Fourier Transform to Optimize Trajectory
```bash
python scripts/smplx_to_robot --smplx_file <path_to_smplx_data> --robot <path_to_robot_data> --save_path <path_to_save_robot_data.pkl> --rate_limit --fft
```
对已映射好的csv文件做轨迹平滑
```bash
python scripts/fft_optimization.py --input <csv_file_path.csv> --robot <robot_type> --save_path <path_to_save_robot_data.csv>
```

the **video directory** at `videos/`

after this procedure, you can see the **.pkl** files at `save_path`.

### from motion capture system to robot **TBD**

## 3.formalization(from `.pkl` to `.csv`)
```bash
# batch formalization
python scripts/batch_gmr_pkl_to_csv.py --folder <path_to_pkl_files>
# example
python scripts/batch_gmr_pkl_to_csv.py --folder retargeting_results/
```
the results after formalization located in the  `<path_to_pkl_files>/csv/`.


# Retargeting from FBX (OptiTrack) to Robot
```bash
conda activate gmr_mocap
cd GMR/third_party

export LD_PRELOAD=/home/vega/anaconda3/envs/gmr_mocap/lib/libxml2.so
export PYTHONPATH=$PYTHONPATH:$(pwd)/poselib/skeleton/backend/fbx/

python poselib/fbx_importer.py \
    --input ../datasets/Mocap/fbx/Hand.fbx \
    --output ../retargeting_results/fbx_to_pkl/Hand_motion.pkl \
    --root-joint root # root joint name maybe need modify
    
cd GMR
python scripts/convert_fbx_skeleton_names.py \
    --input retargeting_results/fbx_to_pkl/Hand_motion.pkl \
    --output retargeting_results/fbx_to_pkl/converted/Hand_motion_converted.pkl
    
conda activate gmr
python scripts/fbx_offline_to_robot.py \
    --motion_file retargeting_results/fbx_to_pkl/converted/Hand_motion_converted.pkl \
    --save_path retargeting_results/fbx_to_pkl/{robot}/Hand0103_final_{robot}.pkl \
    --rate_limit
```

# Visualize saved robot motion
Visualize a single motions:
```bash
python scripts/vis_robot_motion.py --robot <robot_name> --robot_motion_path <path_to_save_robot_data.pkl>
```
If you want to record video, add `--record_video` and `--video_path <your_video_path,mp4>`.

Visualize a folder of motions:
```bash
python scripts/vis_robot_motion_dataset.py --robot <robot_name> --robot_motion_folder <path_to_save_robot_data_folder>
```
After launching the MuJoCo visualization window and clicking on it, you can use the following keyboard controls::

- `[`: play the previous motion
- `]`: play the next motion
- `space`: toggle play/pause


# Plot the drafts
```bash
python scripts/fft_optimization.py --input retargeting_results/tienkung_lite/smplx_to_pkl/csv/D1_-_Urban_1_stageii_tienkung.csv --output retargeting_results/tienkung_lite/smplx_to_pkl/csv_smoothed/D1_-_Urban_1_stageii_tienkung.csv --plot --img_path retargeting_results/plots/D1_-_Urban_1_stageii_tienkung_Afer.png

python scripts/fft_optimization_batch.py --folder_path <single_csv_path.csv> --output_path <output_path.csv> -p
```
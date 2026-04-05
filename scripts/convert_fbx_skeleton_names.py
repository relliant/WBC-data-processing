import pickle
import sys
import os
import argparse


def fix_skeleton_names(input_path, output_path):
    print(f"🔄 Loading motion file: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # ---------------------------------------------------------
    # 骨骼名称映射表 (Mapping Table)
    # Key:   你的原始 .pkl 骨骼名
    # Value: GMR/Unitree G1 配置文件中需要的标准名 (Hips, LeftUpLeg...)
    # ---------------------------------------------------------
    raw_name_map = {
        # 根节点与躯干
        'root': 'Hips',
        'lowerback': 'Spine1',
        # 如果有 upperback/thorax，通常对应 Spine2/Spine3，视情况而定，
        # 但 G1 的 IK Table 主要用到了 Spine1 (torso_link)

        # 左腿 (Legs)
        'lhipjoint': 'LeftUpLeg',  # 对应 left_hip_roll_link
        'ltibia': 'LeftLeg',  # 对应 left_knee_link (胫骨的起点是膝盖)
        'lfoot': 'LeftFoot',  # 虽然 IK Table 没直接写，但标准骨架通常需要
        'ltoesX': 'LeftToeBase',  # 对应 left_toe_link

        # 右腿
        'rhipjoint': 'RightUpLeg',  # 对应 right_hip_roll_link
        'rtibia': 'RightLeg',  # 对应 right_knee_link
        'rfoot': 'RightFoot',
        'rtoesX': 'RightToeBase',  # 对应 right_toe_link

        # 左臂 (Arms)
        'lhumerus': 'LeftArm',  # 对应 left_shoulder_yaw_link (肱骨起点是肩)
        'lradius': 'LeftForeArm',  # 对应 left_elbow_link (桡骨起点是肘)
        'lwrist': 'LeftHand',  # 对应 left_wrist_yaw_link

        # 右臂
        'rhumerus': 'RightArm',  # 对应 right_shoulder_yaw_link
        'rradius': 'RightForeArm',  # 对应 right_elbow_link
        'rwrist': 'RightHand'  # 对应 right_wrist_yaw_link


    }

    # ---------------------------------------------------------
    # 骨骼名称映射表 - 针对 Mixamo 前缀进行清理
    # ---------------------------------------------------------
    mixamo_name_map = {
        # 根节点与躯干
        'mixamorig:Hips': 'Hips',
        'mixamorig:Spine': 'Spine',
        'mixamorig:Spine1': 'Spine1',
        'mixamorig:Spine2': 'Spine2',

        # 腿部 (Legs)
        'mixamorig:LeftUpLeg': 'LeftUpLeg',
        'mixamorig:LeftLeg': 'LeftLeg',
        'mixamorig:LeftFoot': 'LeftFoot',
        'mixamorig:LeftToeBase': 'LeftToeBase',

        'mixamorig:RightUpLeg': 'RightUpLeg',
        'mixamorig:RightLeg': 'RightLeg',
        'mixamorig:RightFoot': 'RightFoot',
        'mixamorig:RightToeBase': 'RightToeBase',

        # 手臂 (Arms)
        'mixamorig:LeftArm': 'LeftArm',
        'mixamorig:LeftForeArm': 'LeftForeArm',
        'mixamorig:LeftHand': 'LeftHand',

        'mixamorig:RightArm': 'RightArm',
        'mixamorig:RightForeArm': 'RightForeArm',
        'mixamorig:RightHand': 'RightHand',

        # 肩部 (Shoulders)
        'mixamorig:LeftShoulder': 'LeftShoulder',
        'mixamorig:RightShoulder': 'RightShoulder'
    }

    new_data = []

    # 遍历每一帧数据进行替换
    for frame_idx, frame_data in enumerate(data):
        new_frame = {}
        for old_name, value in frame_data.items():
            # 查找映射表，如果存在则替换，不存在则保留原名
            new_name = mixamo_name_map.get(old_name, old_name)
            new_frame[new_name] = value
        new_data.append(new_frame)

    # ---------------------------------------------------------
    # 简单的完整性检查
    # ---------------------------------------------------------
    if len(new_data) > 0:
        first_frame_keys = new_data[0].keys()
        # 检查 IK Config 中必须的几个关键骨骼是否已存在
        required_targets = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftArm', 'LeftForeArm']
        missing = [k for k in required_targets if k not in first_frame_keys]

        if missing:
            print(f"⚠️  Warning: Converted data is still missing key joints: {missing}")
            print("   (Please check if your source PKL uses different names)")
        else:
            print("✅ Structural check passed: Essential G1 joints found.")

    # 自动创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

    print(f"💾 Saved fixed motion to: {output_path}")
    print(f"📊 Total Frames: {len(new_data)}")


def main():
    parser = argparse.ArgumentParser(description="Fix skeleton bone names to match Unitree G1/GMR config.")

    # 添加命令行参数
    parser.add_argument("--input", type=str, required=True, help="Path to the source PKL file (from fbx_importer)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the renamed PKL file")

    args = parser.parse_args()

    fix_skeleton_names(args.input, args.output)


if __name__ == "__main__":
    main()
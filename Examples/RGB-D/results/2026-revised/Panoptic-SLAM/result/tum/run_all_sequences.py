#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# ============================================================================
# 配置区域：选择要处理的 TUM 序列
# 若要处理全部，设置 PROCESS_ALL = True
# ============================================================================
PROCESS_ALL = False   # True = 处理所有 rgbd_dataset_* 子目录；False = 只处理 SEQUENCES

# 手动指定要处理的序列（子目录名称）
SEQUENCES = [
    "rgbd_dataset_freiburg2_desk_with_person",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
]
# ============================================================================


def main():
    # 当前脚本应放在：
    # ~/Downloads/Panoptic-SLAM/Output/result/tum/
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    # 获取所有 TUM 结果子目录
    all_seq_dirs = [
        d for d in script_dir.iterdir()
        if d.is_dir() and d.name.startswith("rgbd_dataset_")
    ]

    if not all_seq_dirs:
        print("未找到任何 rgbd_dataset_* 子目录")
        return

    # 确定要处理的目录列表
    if PROCESS_ALL:
        target_dirs = sorted(all_seq_dirs)
        print(f"处理所有 {len(target_dirs)} 个 TUM 序列目录")
    else:
        target_dirs = [
            script_dir / name
            for name in SEQUENCES
            if (script_dir / name).is_dir()
        ]

        missing = [
            name for name in SEQUENCES
            if not (script_dir / name).is_dir()
        ]

        if missing:
            print("以下序列目录不存在，将跳过：")
            for name in missing:
                print(f"  - {name}")

        if not target_dirs:
            print("未找到任何匹配的 TUM 序列目录，请检查 SEQUENCES 或设置 PROCESS_ALL = True")
            return

        print(f"将处理以下 {len(target_dirs)} 个 TUM 序列:")
        for d in target_dirs:
            print(f"  - {d.name}")

    # 对每个选中的序列执行处理
    for seq_dir in target_dirs:
        print(f"\n--- 进入目录: {seq_dir.name} ---")
        os.chdir(seq_dir)

        # 当前目录:
        # Output/result/tum/rgbd_dataset_xxx/
        #
        # 向上 4 级到 Panoptic-SLAM/
        # 再进入 Dataset/TUM/当前序列/groundtruth.txt
        gt_rel = f"../../../../Dataset/TUM/{seq_dir.name}/groundtruth.txt"

        # 调用上一级 tum 目录中的 batch_evo_ape.py
        script_path = "../batch_evo_ape.py"

        # 处理 CameraTrajectory_01.txt 到 CameraTrajectory_10.txt
        cmd = [
            "python3",
            script_path,
            gt_rel,
            "CameraTrajectory",
            "1",
            "10",
        ]

        print(f"执行命令: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"错误: 序列 {seq_dir.name} 处理失败，退出码 {e.returncode}")
        finally:
            os.chdir(script_dir)

    print("\n所有选中的 TUM 序列处理完成。")


if __name__ == "__main__":
    main()

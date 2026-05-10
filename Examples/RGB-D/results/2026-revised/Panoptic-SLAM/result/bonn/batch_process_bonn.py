#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# -------------------------
# 手动选择要处理的序列
SEQUENCES = [
"rgbd_bonn_balloon"
,"rgbd_bonn_balloon2"
,"rgbd_bonn_balloon_tracking"
,"rgbd_bonn_balloon_tracking2"
,"rgbd_bonn_crowd"
,"rgbd_bonn_crowd2"
,"rgbd_bonn_crowd3"
,"rgbd_bonn_kidnapping_box"
,"rgbd_bonn_kidnapping_box2"
,"rgbd_bonn_moving_nonobstructing_box"
,"rgbd_bonn_moving_nonobstructing_box2"
,"rgbd_bonn_moving_obstructing_box"
,"rgbd_bonn_moving_obstructing_box2"
,"rgbd_bonn_person_tracking"
,"rgbd_bonn_person_tracking2"
,"rgbd_bonn_placing_nonobstructing_box"
,"rgbd_bonn_placing_nonobstructing_box2"
,"rgbd_bonn_placing_nonobstructing_box3"
,"rgbd_bonn_placing_obstructing_box"
,"rgbd_bonn_removing_nonobstructing_box"
,"rgbd_bonn_removing_nonobstructing_box2"
,"rgbd_bonn_removing_obstructing_box"
,"rgbd_bonn_static"
,"rgbd_bonn_static_close_far"
,"rgbd_bonn_synchronous"
,"rgbd_bonn_synchronous2"
]

# 根目录设置
PROJECT_ROOT = Path.home() / "Downloads" / "NGD-SLAM"
RESULT_ROOT = PROJECT_ROOT / "Results" / "Bonn"
BATCH_SCRIPT = PROJECT_ROOT / "batch_evo_ape.py"

START_RUN = 1
END_RUN = 11   # 执行 1-10

# -------------------------
def main():
    if not RESULT_ROOT.exists():
        RESULT_ROOT.mkdir(parents=True)

    # 确认 batch_evo_ape.py 存在
    if not BATCH_SCRIPT.is_file():
        print(f"错误: 找不到 batch_evo_ape.py -> {BATCH_SCRIPT}")
        return

    # 遍历选择的序列
    for seq in SEQUENCES:
        seq_dir = RESULT_ROOT / seq
        if not seq_dir.is_dir():
            print(f"警告: 序列文件夹不存在, 跳过 -> {seq_dir}")
            continue

        print(f"\n=== 处理序列: {seq} ===")

        # 遍历轨迹文件
        for run_idx in range(START_RUN, END_RUN):
            traj_file = seq_dir / f"CameraTrajectory_{run_idx:02d}.txt"
            if not traj_file.is_file():
                print(f"警告: 轨迹文件不存在, 跳过 -> {traj_file}")
                continue

            # groundtruth 文件
            gt_file = Path.home() / "Downloads" / "dataset" / "Bonn" / seq / "groundtruth.txt"
            if not gt_file.is_file():
                print(f"警告: Groundtruth 文件不存在, 跳过 -> {gt_file}")
                continue

            # 调用 batch_evo_ape.py
            cmd = [
                "python3",
                str(BATCH_SCRIPT),
                str(gt_file),
                "CameraTrajectory",
                str(run_idx),
                "1"  # 每次只处理当前轨迹文件
            ]
            print(f"执行命令: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"错误: 序列 {seq} Run {run_idx} 分析失败，退出码 {e.returncode}")
                continue

    print("\n所有选中序列处理完成。")

if __name__ == "__main__":
    main()

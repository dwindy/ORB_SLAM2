import subprocess
import re
import numpy as np
from pathlib import Path
import argparse
# 用户配置
#dataset_name = "rgbd_dataset_freiburg2_desk_with_person"  # 要测试的序列

import argparse

parser = argparse.ArgumentParser(description="Evaluate trajectory using evo_rpe")
parser.add_argument("dataset_name", type=str, help="Name of the dataset sequence, e.g., rgbd_dataset_freiburg2_desk_with_person")
args = parser.parse_args()

# 去掉尾部斜杠
dataset_name = args.dataset_name.rstrip("/")


base_gt = "/home/bingxin/Downloads/dataset/TUM"
base_result = "/home/bingxin/Downloads/ORB_SLAM2/Examples/RGB-D/results/2025-revised-result/TUM/"

gt_file = f"{base_gt}/{dataset_name}/groundtruth.txt"

metrics_order = ["max", "mean", "median", "min", "rmse", "sse", "std"]
all_values = {k: [] for k in metrics_order}

for i in range(1, 11):
    traj_file = f"{base_result}/{dataset_name}/CameraTrajectory_{i}.txt"
    cmd = [
        "evo_rpe", "tum", gt_file, "-vas", traj_file, "-r", "trans_part"
    ]
    print("Running:", " ".join(cmd))

    # 调用 evo_rpe
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout

    # 正则匹配每一行的数值
    for line in output.splitlines():
        for key in metrics_order:
            if line.strip().startswith(key):
                val = float(line.split()[-1])
                all_values[key].append(val)

# 按顺序组成矩阵
matrix = np.array([all_values[k] for k in metrics_order])

# 保存为 txt
out_file = Path(f"{dataset_name}_rpe_translation_summary.txt")
np.savetxt(out_file, matrix, fmt="%.6f")

print(f"Saved 7x10 results to {out_file}")


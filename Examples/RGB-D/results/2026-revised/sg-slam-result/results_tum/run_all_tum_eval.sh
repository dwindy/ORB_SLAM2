#!/usr/bin/env bash

set -euo pipefail

# =========================
# 可修改参数
# =========================

# SG-SLAM 的 TUM 结果总目录
RESULTS_ROOT="$HOME/Downloads/SG-SLAM/src/sg-slam/results_tum"

# TUM 数据集总目录（每个序列目录下应有 groundtruth.txt）
GT_ROOT="$HOME/Desktop/extract/TUM"

# 你现有的 batch evalue 脚本路径
BATCH_EVAL_PY="$HOME/Downloads/ORB_SLAM2/Examples/RGB-D/results-2026-vm-redo/Bonn/rgbd_bonn_crowd/Normal/batch_evo_ape.py"

# 从第几个 CameraTrajectory 开始
START_INDEX=1

# 一共评估多少个（例如 10 表示 11~20）
NUM_FILES=10

# 轨迹文件前缀
TRAJ_PREFIX="CameraTrajectory"

# Python 命令
PYTHON_BIN="python3"


# =========================
# 主逻辑
# =========================

if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "错误：RESULTS_ROOT 不存在：$RESULTS_ROOT"
    exit 1
fi

if [[ ! -d "$GT_ROOT" ]]; then
    echo "错误：GT_ROOT 不存在：$GT_ROOT"
    exit 1
fi

if [[ ! -f "$BATCH_EVAL_PY" ]]; then
    echo "错误：找不到 batch_evo_ape.py：$BATCH_EVAL_PY"
    exit 1
fi

echo "RESULTS_ROOT = $RESULTS_ROOT"
echo "GT_ROOT      = $GT_ROOT"
echo "BATCH_EVAL   = $BATCH_EVAL_PY"
echo "START_INDEX  = $START_INDEX"
echo "NUM_FILES    = $NUM_FILES"
echo

for seq_dir in "$RESULTS_ROOT"/*; do
    [[ -d "$seq_dir" ]] || continue

    seq_name="$(basename "$seq_dir")"
    gt_file="$GT_ROOT/$seq_name/groundtruth.txt"

    echo "========================================"
    echo "正在处理序列：$seq_name"
    echo "结果目录：$seq_dir"
    echo "GT文件：$gt_file"

    if [[ ! -f "$gt_file" ]]; then
        echo "警告：找不到 groundtruth.txt，跳过：$gt_file"
        echo
        continue
    fi

    cd "$seq_dir"

    $PYTHON_BIN "$BATCH_EVAL_PY" "$gt_file" "$TRAJ_PREFIX" "$START_INDEX" "$NUM_FILES"

    echo "完成：$seq_name"
    echo
done

echo "全部序列处理完成。"

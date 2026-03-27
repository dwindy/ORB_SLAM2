#!/bin/bash

set -u

# ========== 基本路徑設定 ==========
ORB_SLAM_DIR="/home/sin/Downloads/ORB_SLAM2"
DATA_ROOT="/home/sin/Desktop/extract/Bonn"
RESULT_ROOT="/home/sin/Downloads/ORB_SLAM2/Examples/RGB-D/results-2026-vm-redo/Bonn"

# 每個序列跑幾次
NUM_RUNS=1
START_INDEX=20

# Bonn 序列名列表（沿用你之前腳本裡實際啟用的那些）
SEQUENCES=(
    #balloon
    #balloon2
    #balloon_tracking
    #balloon_tracking2
    #crowd
    #crowd2
    #crowd3
   # kidnapping_box
    #kidnapping_box2
    #moving_nonobstructing_box
   # moving_nonobstructing_box2
    #moving_obstructing_box
    #moving_obstructing_box2
    #person_tracking
    #person_tracking2
    #placing_nonobstructing_box
    #placing_nonobstructing_box2
    #placing_nonobstructing_box3
    #placing_obstructing_box
    #removing_nonobstructing_box
    #removing_nonobstructing_box2
    #removing_obstructing_box
    static
    #static_close_far
    #synchronous
    #synchronous2
)

# 可執行程式與設定檔
EXEC="./rgbd_tum_yolo"
VOCAB="../../Vocabulary/ORBvoc.txt"
CONFIG="Bonn.yaml"
MODE_DATASET="Bonn"
MODE1="variance"
MODE2="mask"

# ========== 進入工作目錄 ==========
cd "$ORB_SLAM_DIR/Examples/RGB-D" || {
    echo "Error: cannot cd to $ORB_SLAM_DIR/Examples/RGB-D"
    exit 1
}

mkdir -p "$RESULT_ROOT"

echo "========================================"
echo "Batch Bonn run started"
echo "ORB_SLAM_DIR : $ORB_SLAM_DIR"
echo "DATA_ROOT    : $DATA_ROOT"
echo "RESULT_ROOT  : $RESULT_ROOT"
echo "NUM_RUNS     : $NUM_RUNS"
echo "START_INDEX  : $START_INDEX"
echo "========================================"

for SEQ in "${SEQUENCES[@]}"; do
    DATASET_DIR="$DATA_ROOT/rgbd_bonn_${SEQ}"
    ASSOC_FILE="$DATASET_DIR/associations.txt"
    SEQ_RESULT_DIR="$RESULT_ROOT/rgbd_bonn_${SEQ}/Normal"

    mkdir -p "$SEQ_RESULT_DIR"

    echo
    echo "########################################"
    echo "Sequence   : $SEQ"
    echo "Dataset    : $DATASET_DIR"
    echo "Assoc file : $ASSOC_FILE"
    echo "Save dir   : $SEQ_RESULT_DIR"
    echo "########################################"

    if [ ! -d "$DATASET_DIR" ]; then
        echo "Warning: dataset dir not found: $DATASET_DIR"
        continue
    fi

    if [ ! -f "$ASSOC_FILE" ]; then
        echo "Warning: associations.txt not found: $ASSOC_FILE"
        continue
    fi

    for ((offset=0; offset<NUM_RUNS; offset++)); do
        RUN_NO=$((START_INDEX + offset))
        IDX=$(printf "%02d" "$RUN_NO")

        echo
        echo "[$SEQ] Run $IDX (offset $((offset + 1)) / $NUM_RUNS)"

        rm -f CameraTrajectory.txt KeyFrameTrajectory.txt

        "$EXEC" "$VOCAB" "$CONFIG" "$DATASET_DIR/" "$ASSOC_FILE" "$MODE_DATASET" "$MODE1" "$MODE2"
        RET=$?

        if [ $RET -ne 0 ]; then
            echo "Warning: program failed on $SEQ run $IDX (exit code: $RET)"
            continue
        fi

        sleep 2

        if [ -f CameraTrajectory.txt ]; then
            mv CameraTrajectory.txt "$SEQ_RESULT_DIR/CameraTrajectory_${IDX}.txt"
            echo "Saved: $SEQ_RESULT_DIR/CameraTrajectory_${IDX}.txt"
        else
            echo "Warning: CameraTrajectory.txt not found for $SEQ run $IDX"
        fi

        if [ -f KeyFrameTrajectory.txt ]; then
            mv KeyFrameTrajectory.txt "$SEQ_RESULT_DIR/KeyFrameTrajectory_${IDX}.txt"
            echo "Saved: $SEQ_RESULT_DIR/KeyFrameTrajectory_${IDX}.txt"
        else
            echo "Warning: KeyFrameTrajectory.txt not found for $SEQ run $IDX"
        fi

        sleep 2
    done
done

echo
echo "========================================"
echo "All runs finished."
echo "Results saved in: $RESULT_ROOT"
echo "========================================"

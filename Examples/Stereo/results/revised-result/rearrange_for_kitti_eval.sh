#!/bin/bash

SRC="KITTI2"      
DST="results_for_eval2"

mkdir -p "$DST"

for run in {1..10}; do
    run_dir="$DST/ROVK-$run"
    mkdir -p "$run_dir"

    for dataset_dir in "$SRC"/*; do
        dataset=$(basename "$dataset_dir")
        src_file="$dataset_dir/CameraTrajectory_${run}.txt"
        dst_file="$run_dir/${dataset}.txt"

        if [[ -f "$src_file" ]]; then
            cp "$src_file" "$dst_file"
            echo "Copied $src_file -> $dst_file"
        else
            echo "Warning: $src_file does not exist"
        fi
    done
done

echo "Rearrangement completed. New structure in $DST"


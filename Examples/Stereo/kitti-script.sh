#this script is used for ablation experiment where the reviewer says even in TUM static mode, I should have results

#!/bin/bash

echo 'Hello World!'
echo 'Goint Run ORBSLAM2'

# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="/home/xin/Downloads/DATASET/KITTI/dataset/sequences/"
RESULT_PATH="results/KITTI/"

 DATASETS=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
 #DATASETS=("04")

# Change directory to ORB_SLAM2
cd "$ORB_SLAM_PATH/Examples/Stereo" || { echo "Failed to enter ORB_SLAM2 directory"; exit 1; }
echo 'Entered ORBSLAM ORIGIN Program Folder' - "$ORB_SLAM_PATH/Examples/Stereo"

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
    echo "Processing Dataset: $dataset"
    
	    # 选择配置文件
	if [[ "$dataset" == "00" || "$dataset" == "01" || "$dataset" == "02" ]]; then
	    yaml="KITTI00-02.yaml"
	elif [[ "$dataset" == "03" ]]; then
	    yaml="KITTI03.yaml"
	else
	    yaml="KITTI04-12.yaml"
	fi

    # Loop through sequences
    for i in {1..10}; do
        echo "Processing No $i"

        # Run ORB_SLAM2
        ./stereo_kitti_yolo ../../Vocabulary/ORBvoc.txt "$yaml" "$DATASET_PATH/$dataset/" variance || continue

        # give file with number
        camera_trajectory_filename="CameraTrajectory_$i.txt"
        #reprojErrorlog_filename="ReprojErrorLog_$i.txt"

        destination_folder="$RESULT_PATH/$dataset/"
        echo "Moving results to $destination_folder"
        
        # Check if the folder exists; if not, create it
	mkdir -p "$destination_folder"  # This will create the folder only if it doesn’t already exist

        # Copy trajectory files
        cp CameraTrajectory.txt "$destination_folder/$camera_trajectory_filename"
        #cp ReprojErrorLog.txt "$destination_folder/$reprojErrorlog_filename"

        echo "Results moved to $destination_folder"
        #echo $camera_processed_filename
        echo "---------------------------"
    done
    echo "---------------------------------------------------------------"
done

echo "Processing completed"

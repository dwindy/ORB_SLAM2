#this script is used for ablation experiment where the reviewer says even in TUM static mode, I should have results

#!/bin/bash

echo 'Hello World!'
echo 'Goint Run ORBSLAM2'

# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="//home/xin/Downloads/DATASET/TUM"
RESULT_PATH="results/Ablation/optical/TUM"

 DATASETS=("rgbd_dataset_freiburg3_sitting_halfsphere" "rgbd_dataset_freiburg3_sitting_rpy"
 "rgbd_dataset_freiburg3_sitting_static" "rgbd_dataset_freiburg3_sitting_xyz" "rgbd_dataset_freiburg3_walking_halfsphere"
 "rgbd_dataset_freiburg3_walking_rpy" "rgbd_dataset_freiburg3_walking_static" "rgbd_dataset_freiburg3_walking_xyz")

#DATASETS=("rgbd_dataset_freiburg2_desk_with_person" )

#DATASETS=("rgbd_dataset_freiburg3_sitting_halfsphere" "rgbd_dataset_freiburg3_sitting_rpy" "rgbd_dataset_freiburg3_sitting_xyz")

#DATASETS=("rgbd_dataset_freiburg3_walking_halfsphere" "rgbd_dataset_freiburg3_walking_rpy" "rgbd_dataset_freiburg3_walking_static" "rgbd_dataset_freiburg3_walking_xyz")

# Change directory to ORB_SLAM2
cd "$ORB_SLAM_PATH/Examples/RGB-D" || { echo "Failed to enter ORB_SLAM2 directory"; exit 1; }
echo 'Entered ORBSLAM ORIGIN Program Folder' - "$ORB_SLAM_PATH/Examples/RGB-D"

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
    echo "Processing Dataset: $dataset"

    # Loop through sequences
    for i in {1..10}; do
        echo "Processing No $i"

        # Run ORB_SLAM2
        ./rgbd_tum_yolo ../../Vocabulary/ORBvoc.txt TUM3.yaml "$DATASET_PATH/$dataset/" "$DATASET_PATH/$dataset/associate.txt" TUM euclidean

        # Process time stamp
        # python3 /home/xin/PycharmProjects/TumCamTrajTimeProcess/main.py CameraTrajectory.txt
	
	#optical-only or reproject-only	
	
        # give file with number
        key_trajectory_filename="KeyFrameTrajectory_$i.txt"
        camera_trajectory_filename="CameraTrajectory_$i.txt"
        reprojErrorlog_filename="OpticalFlowErrorLog_$i.txt"
        #camera_processed_filename="CameraTrajectoryprocessed_$i.txt"
        destination_folder="$RESULT_PATH/$dataset/"
        echo "Moving results to $destination_folder"
        
        # Check if the folder exists; if not, create it
	mkdir -p "$destination_folder"  # This will create the folder only if it doesn’t already exist

        # Copy trajectory files
        cp KeyFrameTrajectory.txt "$destination_folder/$key_trajectory_filename"
        cp CameraTrajectory.txt "$destination_folder/$camera_trajectory_filename"
        cp ReprojErrorLog.txt "$destination_folder/$reprojErrorlog_filename"
        #cp CameraTrajectoryprocessed.txt "$RESULT_PATH/$dataset/$camera_processed_filename"

        echo "Results moved to $destination_folder"
        #echo $camera_processed_filename
        echo "---------------------------"
    done
    echo "---------------------------------------------------------------"
done

echo "Processing completed"

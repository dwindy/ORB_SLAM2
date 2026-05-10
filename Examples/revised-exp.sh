#this script is used for ablation experiment where the reviewer says even in TUM static mode, I should have results

#!/bin/bash

echo 'Hello World!'
echo 'Going to Run ORBSLAM2'


##############################Bonn
# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="/xinfolder/Bonn"
RESULT_PATH="results/revised-result/Bonn"

DATASETS=(
    "rgbd_bonn_balloon"
    "rgbd_bonn_balloon2"
    "rgbd_bonn_balloon_tracking"
    "rgbd_bonn_balloon_tracking2"
    "rgbd_bonn_crowd"
    "rgbd_bonn_crowd2"
    "rgbd_bonn_crowd3"
    "rgbd_bonn_kidnapping_box"
   "rgbd_bonn_kidnapping_box2"
    "rgbd_bonn_moving_nonobstructing_box"
    "rgbd_bonn_moving_nonobstructing_box2"
    "rgbd_bonn_moving_obstructing_box"
    "rgbd_bonn_moving_obstructing_box2"
    "rgbd_bonn_person_tracking"
    "rgbd_bonn_person_tracking2"
    "rgbd_bonn_placing_nonobstructing_box"
    "rgbd_bonn_placing_nonobstructing_box2"
    "rgbd_bonn_placing_nonobstructing_box3"
    "rgbd_bonn_placing_obstructing_box"
   "rgbd_bonn_removing_nonobstructing_box"
    "rgbd_bonn_removing_nonobstructing_box2"
    "rgbd_bonn_removing_obstructing_box"
    "rgbd_bonn_static"
    "rgbd_bonn_static_close_far"
    "rgbd_bonn_synchronous"
    "rgbd_bonn_synchronous2"
)

# Change directory to ORB_SLAM2
cd "$ORB_SLAM_PATH/Examples/RGB-D" || { echo "Failed to enter ORB_SLAM2 directory"; exit 1; }
echo 'Entered ORBSLAM ORIGIN Program Folder' - "$ORB_SLAM_PATH/Examples/RGB-D"

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
    echo "Processing Dataset: $dataset"
    
	yaml="Bonn.yaml"

        destination_folder="$RESULT_PATH/$dataset/"
        echo "Moving results to $destination_folder"
      
        # Check if the folder exists; if not, create it
	mkdir -p "$destination_folder"  # This will create the folder only if it doesn’t already exist	

    # Loop through sequences
    for i in {1..10}; do
        echo "Processing No $i"
	
	echo $yaml
	echo $DATASET_PATH/$dataset/
	echo $DATASET_PATH/$dataset/associations.txt
	
        # Run ORB_SLAM2
        ./rgbd_tum_yolo ../../Vocabulary/ORBvoc.txt $yaml "$DATASET_PATH/$dataset/" "$DATASET_PATH/$dataset/associations.txt" Bonn variance mask || {
	    echo "Run failed on $dataset iteration $i"
	    continue
	}

	
        # give file with number
        key_trajectory_filename="KeyFrameTrajectory_$i.txt"
        camera_trajectory_filename="CameraTrajectory_$i.txt"

        # Copy trajectory files
        cp -f KeyFrameTrajectory.txt "$destination_folder/$key_trajectory_filename"
        cp -f CameraTrajectory.txt "$destination_folder/$camera_trajectory_filename"

        echo "Results moved to $destination_folder"
        echo "---------------------------"
    done
    echo "---------------------------------------------------------------"
done

echo "Bonn Processing completed"
###################################################################################


# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="/home/xin/Downloads/DATASET/TUM"
RESULT_PATH="results/revised-result/TUM"

DATASETS=("rgbd_dataset_freiburg2_desk_with_person" "rgbd_dataset_freiburg3_sitting_halfsphere" "rgbd_dataset_freiburg3_sitting_rpy"
"rgbd_dataset_freiburg3_sitting_static" "rgbd_dataset_freiburg3_sitting_xyz" "rgbd_dataset_freiburg3_walking_halfsphere"
"rgbd_dataset_freiburg3_walking_rpy" "rgbd_dataset_freiburg3_walking_static" "rgbd_dataset_freiburg3_walking_xyz")

# Change directory to ORB_SLAM2
cd "$ORB_SLAM_PATH/Examples/RGB-D" || { echo "Failed to enter ORB_SLAM2 directory"; exit 1; }
echo 'Entered ORBSLAM ORIGIN Program Folder' - "$ORB_SLAM_PATH/Examples/RGB-D"

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
    echo "Processing Dataset: $dataset"
    
    if [[ "$dataset" == rgbd_dataset_freiburg2* ]]; then
	    yaml="TUM2.yaml"
	else
    	yaml="TUM3.yaml"
	fi
	

        destination_folder="$RESULT_PATH/$dataset/"
        echo "Moving results to $destination_folder"
      
        # Check if the folder exists; if not, create it
	mkdir -p "$destination_folder"  # This will create the folder only if it doesn’t already exist	

    # Loop through sequences
    for i in {1..10}; do
        echo "Processing No $i"
        echo $yaml
        echo $DATASET_PATH/$dataset/
        echo $DATASET_PATH/$dataset/associate.txt

        # Run ORB_SLAM2
        ./rgbd_tum_yolo ../../Vocabulary/ORBvoc.txt $yaml "$DATASET_PATH/$dataset/" "$DATASET_PATH/$dataset/associate.txt" TUM variance mask || {
	    echo "Run failed on $dataset iteration $i"
	    continue
	}

	
        # give file with number
        key_trajectory_filename="KeyFrameTrajectory_$i.txt"
        camera_trajectory_filename="CameraTrajectory_$i.txt"

        # Copy trajectory files
        cp -f KeyFrameTrajectory.txt "$destination_folder/$key_trajectory_filename"
        cp -f CameraTrajectory.txt "$destination_folder/$camera_trajectory_filename"

        echo "Results moved to $destination_folder"
        echo "---------------------------"
    done
    echo "---------------------------------------------------------------"
done

echo "TUM Processing completed"


#########################################################################################





# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="/home/xin/Downloads/DATASET/KITTI/dataset/sequences/"
RESULT_PATH="results/revised-result/KITTI"

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

echo "Kitti Processing completed"

#this script is used for ablation experiment where the reviewer says even in TUM static mode, I should have results

#!/bin/bash

echo 'Hello World!'
echo 'Going to Run ORBSLAM2'


###################################################################################
# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="/home/xin/Downloads/DATASET/TUM"
RESULT_PATH="results/revised-result/TUM2"

DATASETS=("rgbd_dataset_freiburg3_walking_static" "rgbd_dataset_freiburg3_walking_xyz")

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

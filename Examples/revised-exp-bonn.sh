#this script is used for ablation experiment where the reviewer says even in TUM static mode, I should have results

#!/bin/bash

echo 'Hello World!'
echo 'Going to Run ORBSLAM2'


##############################Bonn
# Set paths
ORB_SLAM_PATH="/home/xin/Downloads/ORB_SLAM2"
DATASET_PATH="/xinfolder/Bonn"
RESULT_PATH="results/revised-result/Bonn2"
#Set1
DATASETS=(
	"rgbd_bonn_moving_nonobstructing_box"
#	"rgbd_bonn_static"
 #   "rgbd_bonn_crowd" "rgbd_bonn_crowd2" "rgbd_bonn_crowd3"
  #  "rgbd_bonn_balloon" "rgbd_bonn_balloon2"
   # "rgbd_bonn_balloon_tracking" "rgbd_bonn_balloon_tracking2"
    #"rgbd_bonn_kidnapping_box" "rgbd_bonn_kidnapping_box2"
    #"rgbd_bonn_moving_nonobstructing_box2"
   # "rgbd_bonn_moving_obstructing_box" "rgbd_bonn_moving_obstructing_box2"
   # "rgbd_bonn_person_tracking" "rgbd_bonn_person_tracking2"
   # "rgbd_bonn_placing_nonobstructing_box2" "rgbd_bonn_placing_nonobstructing_box3"
   # "rgbd_bonn_removing_nonobstructing_box" "rgbd_bonn_removing_nonobstructing_box2"
   # "rgbd_bonn_static_close_far"
   # "rgbd_bonn_synchronous2"
   #  "rgbd_bonn_synchronous"
   # "rgbd_bonn_placing_nonobstructing_box" "rgbd_bonn_placing_obstructing_box"  "rgbd_bonn_removing_obstructing_box"  
)
#set2
DATASETS2=("rgbd_bonn_moving_nonobstructing_box" "rgbd_bonn_synchronous")
#set3
#failed because of code bug
DATASETS3=("rgbd_bonn_placing_nonobstructing_box" "rgbd_bonn_placing_obstructing_box"  "rgbd_bonn_removing_obstructing_box"  "rgbd_bonn_static" )

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
    for i in {11..20}; do
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

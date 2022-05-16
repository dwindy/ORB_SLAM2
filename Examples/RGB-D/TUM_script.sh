#!/bin/bash

echo 'Hello World!'
echo 'Goint Run ORBSLAM2'

#cd ~/Downloads/ORB_SLAM2/Examples/RGB-D
#echo 'Entered ORBSLAM Program Folder'

cd /home/xin/Downloads/ORB_SLAM2/Examples/RGB-D
echo 'Entered ORBSLAM ORIGIN Program Folder'

address_names=(rgbd_dataset_freiburg3_nostructure_texture_far rgbd_dataset_freiburg3_nostructure_texture_near_withloop rgbd_dataset_freiburg3_structure_texture_far rgbd_dataset_freiburg3_structure_texture_near)
track_names=(TUM_NoStruct_Far TUM_NoStruct_Near TUM_Struct_Texture_Far TUM_Struct_Texture_Near)

#track index
for i in {0..3}
do
echo "${track_names[i]}"
#result file index
for j in {10..19}
do
echo "Processing ${track_names[i]} : $j"

./rgbd_tum ../../Vocabulary/ORBvoc.txt TUM3.yaml  /home/xin/Downloads/DATASET/TUM/structure/${address_names[i]} /home/xin/Downloads/DATASET/TUM/structure/${address_names[i]}/associations.txt

echo 'Going to Move Results'

address1="TUM/Structure/${track_names[i]}/KeyFrameTrajectory_"
address3=".txt"
address="$address1$j$address3"
echo "$address"
cp KeyFrameTrajectory.txt $address

address1="TUM/Structure/${track_names[i]}/CameraTrajectory_"
address="$address1$j$address3"
echo "$address"
cp CameraTrajectory.txt $address

address="TUM/Structure/${track_names[i]}/errorRecord.txt"
echo "$address"
mv errorRecord.txt $address

address="TUM/Structure/${track_names[i]}/errorRecordBundle.txt"
echo "$address"
mv errorRecordBundle.txt $address

sleep 60

done

done


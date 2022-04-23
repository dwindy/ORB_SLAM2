#!/bin/bash

echo 'Hello World!'
echo 'Goint Run ORBSLAM2'
sleep 5
#cd ~/Downloads/ORB_SLAM2/Examples/RGB-D
#echo 'Entered ORBSLAM Program Folder'

cd /home/xin/Downloads/ORB_SLAM2/Examples/RGB-D
echo 'Entered ORBSLAM ORIGIN Program Folder'


#traj2_frei_png
for i in {2..5}
do

echo "Processing $i"

./rgbd_tum ../../Vocabulary/ORBvoc.txt ICL-NUIM.yaml  /home/xin/Downloads/DATASET/ICL-NUIM/non-noise/traj2_frei_png /home/xin/Downloads/DATASET/ICL-NUIM/non-noise/traj2_frei_png/associations.txt

echo 'Going to Move Results'

address1="ICL-NUIM/traj2_frei_png/KeyFrameTrajectory_"
address3=".txt"
address="$address1$i$address3"
echo "$address"
cp KeyFrameTrajectory.txt $address

address1="ICL-NUIM/traj2_frei_png/CameraTrajectory_"
address="$address1$i$address3"
echo "$address"
cp CameraTrajectory.txt $address

sleep 60

done
#traj3_frei_png
for i in {2..5}
do

echo "Processing $i"

./rgbd_tum ../../Vocabulary/ORBvoc.txt ICL-NUIM.yaml  /home/xin/Downloads/DATASET/ICL-NUIM/non-noise/traj3_frei_png /home/xin/Downloads/DATASET/ICL-NUIM/non-noise/traj3_frei_png/associations.txt

echo 'Going to Move Results'

address1="ICL-NUIM/traj3_frei_png/KeyFrameTrajectory_"
address3=".txt"
address="$address1$i$address3"
echo "$address"
cp KeyFrameTrajectory.txt $address

address1="ICL-NUIM/traj3_frei_png/CameraTrajectory_"
address="$address1$i$address3"
echo "$address"
cp CameraTrajectory.txt $address

sleep 60

done

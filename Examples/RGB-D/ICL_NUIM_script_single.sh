#!/bin/bash

echo 'Hello World!'
echo 'Goint Run ORBSLAM2'

#cd ~/Downloads/ORB_SLAM2/Examples/RGB-D
#echo 'Entered ORBSLAM Program Folder'

cd /home/xin/Downloads/ORB_SLAM2/Examples/RGB-D
echo 'Entered ORBSLAM ORIGIN Program Folder'

#traj0_frei_png
trackName="traj0_frei_png"
for i in {9..15}
do
echo "Processing $trackName : $i"

./rgbd_tum ../../Vocabulary/ORBvoc.txt ICL-NUIM.yaml  /home/xin/Downloads/DATASET/ICL-NUIM/non-noise/$trackName /home/xin/Downloads/DATASET/ICL-NUIM/non-noise/$trackName/associations.txt

echo 'Going to Move Results'

address1="ICL-NUIM/$trackName/KeyFrameTrajectory_"
address3=".txt"
address="$address1$i$address3"
echo "$address"
cp KeyFrameTrajectory.txt $address

address1="ICL-NUIM/$trackName/CameraTrajectory_"
address="$address1$i$address3"
echo "$address"
cp CameraTrajectory.txt $address

address="ICL-NUIM/$trackName/errorRecord.txt"
#echo "$address"
#mv errorRecord.txt $address

address="ICL-NUIM/$trackName/errorRecordBundle.txt"
#echo "$address"
#mv errorRecordBundle.txt $address

sleep 30

done


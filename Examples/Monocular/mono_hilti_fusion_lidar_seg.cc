//
// Created by xin on 29/05/24.
//
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages_HILTI(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void LoadLaserscans_HILTI(const string strPathToSequence, vector<string> &vstrLiDARFilenames, const vector<long> &vstrLiDARTimes);

void LoadSegmentInfo(const string strPathToSequence, vector<string> &vstrSegmentInfoFilenames, const vector<long> &vstrCamTimes);

void loadKITTI(const string fileAddress, vector<vector<float>> &datas);

void loadPCDHILTI2023(string fileAddress, vector<vector<float>> &datas);

void LoadTimes(string stdTimeFrame, vector<long> &vCamTimestamps, vector<long> &vLiDTimestamps);

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<long> &vTimestamps, int camIndex);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    //Retrieve timeframes of Cam and LiDAR
    vector<long> vCamTimestamps;
    vector<long> vLiDTimestamps;
    LoadTimes(string(argv[3]), vCamTimestamps, vLiDTimestamps);

//    vector<string> vstrImageLeft;
//    vector<string> vstrImageRight;
//    vector<double> vTimestamps;
//    LoadImages_HILTI(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);
    // Retrieve paths to images and lidars
    vector<string> vstrImageLeft;
    LoadImages(string(argv[3]),vstrImageLeft, vCamTimestamps,0);
    // Retrieve paths to images and lidars
    vector<string> vstrImageRight;
    LoadImages(string(argv[3]),vstrImageRight, vCamTimestamps,1);

    const int nImages = vstrImageLeft.size();

    ///Added
    // Retrieve paths to lidars
    vector<string> vstrLiDARFilenames;
    LoadLaserscans_HILTI(string(argv[3]), vstrLiDARFilenames, vLiDTimestamps);
    // Retrieve paths to segment info
    vector<string> vstrSegFilenames;
    LoadSegmentInfo(string(argv[3]), vstrSegFilenames, vCamTimestamps);
    ///----------------------------------

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::Stereo_LiDAR_Seg,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vCamTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        ///Added
        //Read lidar from bin
        vector<vector<float>> liDARdata(1000000, vector<float>(4));//resize in the function
        //loadKITTI(vstrLiDARFilenames[ni],liDARdata);
        loadPCDHILTI2023(vstrLiDARFilenames[ni], liDARdata);
        ///----------------------------------

        // Pass the images to the SLAM system
        //SLAM.TrackStereo(imLeft,imRight,tframe);
        SLAM.TrackStereoLiDARSeg(imLeft, imRight, tframe, liDARdata, vstrImageLeft[ni], vstrSegFilenames[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = (vCamTimestamps[ni+1]-tframe)/10000000000.0;
        else if(ni>0)
            T = (tframe-vCamTimestamps[ni-1])/10000000000.0;

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}

void LoadImages_HILTI(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/synchronized_timestamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

//get lidar file names from image names
void LoadLaserscans_HILTI(const string strPathToSequence, vector<string> &vstrLiDARFilenames,
                          const vector<long> &vstrLiDARTimes) {
    string strPrefix = strPathToSequence + "/velodyne/points";
    const int frameNum = vstrLiDARTimes.size();
    vstrLiDARFilenames.resize(frameNum);
    for (int i = 0; i < frameNum; i++) {
        vstrLiDARFilenames[i] = strPrefix + to_string(vstrLiDARTimes[i]) + ".pcd";
    }
}

//get segment info file names from image names
void LoadSegmentInfo(const string strPathToSequence, vector<string> &vstrSegmentInfoFilenames,
                     const vector<long> &vstrCamTimes){
    string strPrefixLeft = strPathToSequence + "/cam0/segements/";
    const int frameNum = vstrCamTimes.size();
    vstrSegmentInfoFilenames.resize(frameNum);
    for (int i = 0; i < frameNum; i++) {
        vstrSegmentInfoFilenames[i] = strPrefixLeft + to_string(vstrCamTimes[i]) + ".txt";
    }
}

void loadKITTI(string fileAddress, vector<vector<float>> &datas) {
    fstream reader(fileAddress, ios::in);
    if (!reader.good())
        std::cerr << "Error: Unable to open file " << fileAddress << std::endl;
    int i = 0;
    float x, y, z, intensity;
    while (!reader.eof()) {
        reader.read(reinterpret_cast<char *>(&x), sizeof(float));
        reader.read(reinterpret_cast<char *>(&y), sizeof(float));
        reader.read(reinterpret_cast<char *>(&z), sizeof(float));
        reader.read(reinterpret_cast<char *>(&intensity), sizeof(float));
        if (x < 0 || x > 50) //remove lidar from car back and far away
            continue;
        datas[i][0] = x, datas[i][1] = y, datas[i][2] = z, datas[i][3] = intensity;
        i++;
    }
    reader.close();
    datas.resize(i);
}

void loadPCDHILTI2023(string fileAddress, vector<vector<float>> &datas) {
    fstream reader(fileAddress, ios::in);
    string readedLine;
    int linecounter = 0;
    int pointCounter = 0;
    while (getline(reader, readedLine)) {
        //cout << "get a line "<<readedLine << endl;
        linecounter++;
        if (linecounter > 11) {
            istringstream iss(readedLine);
            double x, y, z, intensity, timestamp;
            double ring; //beware of the loss of precission if change back to int
            //hilti2022 iss >> x >> y >> z >> intensity >> timestamp>>ring;
            //hilti2023  iss >> x >> y >> z >> intensity >> ring >> timestamp;
            iss >> x >> y >> z >> intensity >> ring >> timestamp;
            if (!(x == 0 && y == 0 && z == 0)) {
                vector<float> thisPoint = {float(x), float(y), float(z), float(intensity), float(ring), float(timestamp)};
                datas[pointCounter] = thisPoint;
                pointCounter++;
            }
        }
    }
    datas.resize(pointCounter);
    //cout<<"load "<<pointCounter<<" points from pcd "<<endl;
    reader.close();
}

void LoadTimes(string strPathToSequence, vector<long> &vCamTimestamps, vector<long> &vLiDTimestamps) {
    string strPathTimeFile = strPathToSequence + "/synchronized_timestamps.txt";
    ifstream reader(strPathTimeFile, ios::in);
    string aLine;
    if (reader.is_open()) {
        while (getline(reader, aLine)) {
            istringstream iss(aLine);
            long cam0Time, cam1Time, liDTime;
            double timeDiff0, timeDiff1, timeDiff2;
            if (iss >> cam0Time >>cam1Time >> liDTime >> timeDiff0 >> timeDiff1 >> timeDiff2) {
                vCamTimestamps.push_back(cam0Time);
                vLiDTimestamps.push_back(liDTime);
            }
        }
    } else {
        std::printf("time frame read error !!! \n");
        cout<<strPathTimeFile<<endl;
    }
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<long> &vTimestamps, int camIndex) {
    //string strPrefixLeft = strPathToSequence + "/data/";
    string strPrefixLeft;
    if(camIndex==0)
        strPrefixLeft = strPathToSequence + "/cam0/data_undis/";
    if(camIndex==1)
        strPrefixLeft = strPathToSequence + "/cam1/data_undis/";
    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);
    for (int i = 0; i < nTimes; i++) {
        vstrImageFilenames[i] = strPrefixLeft + to_string(vTimestamps[i]) + ".png";
    }
}

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
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void LoadLaserscans(const string &strPathToSequence, vector<string> &vstrLaserscanFilenames, vector<double> &vTimestamps, vector<double> &vTimestarts, vector<double> &vTimeends);

void readLaserPoints(string vstrScanFilename, vector<vector<double>> &laserPoints);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_image_sequence path_to_scan_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    ///Added Module
    //Retrieve paths to Laser Scans
    vector<string> vstrScanFilenames;
    vector<double> vLaserTimestamps;
    vector<double> vLaserStartTimes;
    vector<double> vLaserEndTimes;
    LoadLaserscans(string(argv[4]), vstrScanFilenames, vLaserTimestamps, vLaserStartTimes, vLaserEndTimes);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        ///Added module
        //Load scans of this frame
        vector<vector<double>> laserPoints;
        //1000000 is the KITTI readme file suggested number
        for(int i = 0; i<1000000; i++)
        {
            vector<double> point = {0,0,0,0};
            laserPoints.push_back(point);
        }
        readLaserPoints(vstrScanFilenames[ni], laserPoints);
        //store scan Middle time, start time and end time
        vector<double> laserTimes = {vLaserTimestamps[ni], vLaserStartTimes[ni], vLaserEndTimes[ni]};


        //SLAM.TrackMonocular(im,tframe);
        ///added module
        //Pass the image and lasers to the SLAM system
        SLAM.TrackMonucular(im,tframe,laserPoints,laserTimes);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

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
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    //string strPathTimeFile = strPathToSequence + "/times.txt";
    string strPathTimeFile = strPathToSequence + "/timestamp_processed.txt";
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

    //string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixLeft = strPathToSequence + "/data/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        //ss << setfill('0') << setw(6) << i;
        ss << setfill('0') << setw(10) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

/**
 * @brief Load Laser Scans
 * @param [in] strPathToSequence : the data folder address
 * @param [in,out] vstrLaserscanFilenames : the string vector contains each Scan File name
 * @param [in,out] vTimestamps : the double vector contains timestamps of each Scan file
 * @param [in,out] vTimestarts : the double vector contains start time of each Scan
 * @param [in,out] vTimeends : the double vector contains end time of each Scan
 */
void LoadLaserscans(const string &strPathToSequence, vector<string> &vstrLaserscanFilenames, vector<double> &vTimestamps, vector<double> &vTimestarts, vector<double> &vTimeends)
{
    //load scan times
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/timestamps_processed.txt";
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
    fTimes.close();

    //load scan start times
    string strPathStartTimeFile = strPathToSequence + "/timestamps_start_processed.txt";
    fTimes.open(strPathStartTimeFile.c_str());
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
            vTimestarts.push_back(t);
        }
    }
    fTimes.close();

    //load scan end times
    string strPathEndTimeFile = strPathToSequence + "/timestamps_end_processed.txt";
    fTimes.open(strPathEndTimeFile.c_str());
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
            vTimeends.push_back(t);
        }
    }
    fTimes.close();

    //load Laser Scan file names
    //string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixLeft = strPathToSequence + "/data/";

    const int nTimes = vTimestamps.size();
    vstrLaserscanFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        //ss << setfill('0') << setw(6) << i;
        ss << setfill('0') << setw(10) << i;
        vstrLaserscanFilenames[i] = strPrefixLeft + ss.str() + ".bin";
    }
}

/**
 * @brief Load laserpoint by given filename.
 * @param [in] vstrScanFilename : filename of laser scans
 * @param [in,out] laserPoints : laser points
 */
void readLaserPoints(string vstrScanFilename, vector<vector<double>> &laserPoints)
{
    ///Step 1 load laser points
    //allocate 4MB buffer (around ~130 * 4 * 4 KB)
    int32_t num = 1000000;
    float *data = (float *) malloc(num * sizeof(float));
    //pointers for reading laser point
    float *px = data + 0;
    float *py = data + 1;
    float *pz = data + 2;
    float *pr = data + 3;

    //load point cloud
    FILE *fstream;
    fstream = fopen(vstrScanFilename.c_str(), "rb");
    num = fread(data, sizeof(float), num, fstream)/4;
    for(int i=0; i<num;i++)
    {
        laserPoints[i][0] = *px;
        laserPoints[i][1] = *py;
        laserPoints[i][2] = *pz;
        laserPoints[i][3] = *pr;
        px+=4;py+=4;pz+=4;pr+=4;
    }
    fclose(fstream);
    //reset laserpoint vector size
    laserPoints.resize(num);

    ///Step2

}
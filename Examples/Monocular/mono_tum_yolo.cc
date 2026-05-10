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

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);
void LoadMasks(vector<string> vstrImageFilenames, vector<string> &vstrMaskFilenames);

void LoadClasses(vector<string> vstrImageFilenames, vector<string> &vstrClassFilenames, const string& folderaddress);

void LoadDepthZoe(vector<string> vstrImageFilenames, vector<string> &vstrDepthFilenames, const string &folder);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    ///Adds on. load class.txt address
    //vector<string> vstrMaskFilenames;
    //vstrMaskFilenames.resize(nImages);
    //LoadMasks(vstrImageFilenames, vstrMaskFilenames);
    vector<string> vstrClassFilenames;
    vstrClassFilenames.resize(nImages);
    LoadClasses(vstrImageFilenames, vstrClassFilenames,string(argv[3]));
//    vector<string> vstrDepthFilenames;
//    vstrDepthFilenames.resize(nImages);
//    LoadDepthZoe(vstrImageFilenames, vstrDepthFilenames,string(argv[3]));

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
    ///added
    //cv::Mat mask;
    cv::Mat imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];
        ///added
//        imD = cv::imread(string(argv[3])+"/"+vstrDepthFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif
        //1311870427.199132_seg.jpg 1311870427.199132_mask-0.png 1311870427.199132_class.txt
        ///adds on
        //// Read image from file
        //mask = cv::imread(string(argv[3])+"/"+vstrMaskFilenames[ni],CV_LOAD_IMAGE_GRAYSCALE);

//        // Pass the image to the SLAM system
//        SLAM.TrackMonocular(im,tframe);
        ///adds on
        SLAM.TrackMonocular(im,tframe,vstrClassFilenames[ni]);

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

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

/**
 * generate and store mask file names, giving image file names
 * @param vstrImageFilenames
 * @param vstrMaskFilenames
 */
void LoadMasks(vector<string> vstrImageFilenames, vector<string> &vstrMaskFilenames) {
    for (int i = 0; i < vstrImageFilenames.size(); i++) {
        string header = vstrImageFilenames[i].substr(3, 18);
        string maskFileName = "mask" + header + "_masks-merged.jpeg";
        //cout<<maskFileName<<endl;
        vstrMaskFilenames[i] = maskFileName;
    }
}

/**
 * generate class txt file names, giving image file names
 * @param vstrImageFilenames
 * @param vstrMaskFilenames
 */
void LoadClasses(vector<string> vstrImageFilenames, vector<string> &vstrClassFilenames, const string& folder) {
    for (int i = 0; i < vstrImageFilenames.size(); i++) {
        string header = vstrImageFilenames[i].substr(3, 18);
        string classFileName = folder + "/mask" + header + "_class.txt";
        //cout<<maskFileName<<endl;
        vstrClassFilenames[i] = classFileName;
    }
}

/**
 * generate depth png file names, giving image file names
 */
void LoadDepthZoe(vector<string> vstrImageFilenames, vector<string> &vstrDepthFilenames, const string &folder) {
    for (int i = 0; i < vstrImageFilenames.size(); i++) {
        string header = vstrImageFilenames[i].substr(3, 18);
        string depthFileName = folder + "/depth_Zoe" + header + ".png";
        vstrDepthFilenames[i] = depthFileName;
    }
}

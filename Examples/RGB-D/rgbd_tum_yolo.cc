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

void LoadImages(const string& strAssociationFilename, vector<string>& vstrImageFilenamesRGB,
                vector<string>& vstrImageFilenamesD, vector<double>& vTimestamps);

///adds on
void LoadClasses(vector<string> vstrImageFilenames, vector<string>& vstrClassFilenames, const string& folderaddress,
                 const string& dataset);

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        cerr << endl <<
            "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association dataset_name metric_type"
            << endl;
        cerr << "metric_type: variance | euclidean" << endl;
        cerr << "dataset_type: Bonn | TUM | Ulster" << endl;
        cerr <<
            "Example: ./rgbd_tum ../Vocabulary/ORBvoc.txt ../Examples/RGB-D/TUM1.yaml ../rgbd_dataset_freiburg1_xyz ../associations.txt TUM Variance"
            << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    ///adds on----------------------------------------
    ///object mask classes
    vector<string> vstrClassFilenames;
    vstrClassFilenames.resize(nImages);
    /// dataset name
    string dataset = string(argv[5]);
    if (dataset != "Bonn" && dataset != "TUM" && dataset != "Ulster")
    {
        cerr << "Invalid metric type: " << dataset << ". Use 'Bonn', 'Ulster' or 'TUM'." << endl;
        return 1;
    }
    /// metric method
    LoadClasses(vstrImageFilenamesRGB, vstrClassFilenames, string(argv[3]), dataset);
    string metric_type = string(argv[6]);
    if (metric_type != "variance" && metric_type != "euclidean")
    {
        cerr << "Invalid metric type: " << metric_type << ". Use 'variance' or 'euclidean'." << endl;
        return 1;
    }
    ///----------------------------------------------

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);

    ///set metric
    SLAM.SetMetricType(metric_type);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    int framecounter = 0;
    // Main loop
    cv::Mat imRGB, imD;
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        //SLAM.TrackRGBD(imRGB,imD,tframe);
        SLAM.TrackRGBD(imRGB, imD, tframe, vstrClassFilenames[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);

        framecounter++;
        //        cout << "framecounter " << framecounter << endl;
        //        if (framecounter == 5) {
        //            int pause = 0;
        //        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    //SLAM.SaveMapPoints();
    SLAM.SaveClusterDynamics();
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string& strAssociationFilename, vector<string>& vstrImageFilenamesRGB,
                vector<string>& vstrImageFilenamesD, vector<double>& vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}

/**
 * generate class txt file names, giving image file names
 * @param vstrImageFilenames
 * @param vstrMaskFilenames
 */
void LoadClasses(vector<string> vstrImageFilenames, vector<string>& vstrClassFilenames, const string& folder,
                 const string& dataset)
{
    for (int i = 0; i < vstrImageFilenames.size(); i++)
    {
        // //string header = vstrImageFilenames[i].substr(3, 20); //Ulster dataset lab4
        // //string header = vstrImageFilenames[i].substr(3, 17); //BONN dataset
        // string header = vstrImageFilenames[i].substr(3, 18); //TUM dataset
        // string classFileName = folder + "/mask" + header + "_class.txt";
        // //cout<<maskFileName<<endl;
        // vstrClassFilenames[i] = classFileName;
        string header;
        if (dataset == "Ulster")
            header = vstrImageFilenames[i].substr(3, 20);
        else if (dataset == "BONN")
            header = vstrImageFilenames[i].substr(3, 17);
        else if (dataset == "TUM")
            header = vstrImageFilenames[i].substr(3, 18);
        else
        {
            cerr << "Unknown dataset: " << dataset << ". Supported: TUM, BONN, Ulster" << endl;
            exit(1);
        }

        string classFileName = folder + "/mask" + header + "_class.txt";
        vstrClassFilenames[i] = classFileName;
    }
}

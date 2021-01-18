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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>

///added module
#include <math.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/segmentation/region_growing.h>
#include <pcl-1.8/pcl/search/search.h>
#include <pcl-1.8/pcl/search/kdtree.h>
#include <pcl-1.8/pcl/features/normal_3d.h>
//#include <pcl-1.8/pcl/visualization/cloud_viewer.h>
#include <pcl-1.8/pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, //系统实例?
                   ORBVocabulary* pVoc, //BOW字典
                   FrameDrawer *pFrameDrawer,
                   MapDrawer *pMapDrawer,
                   Map *pMap, //地图句柄
                   KeyFrameDatabase* pKFDB, //关键帧产生的词袋数据库
                   const string &strSettingPath,
                   const int sensor):
    mState(NO_IMAGES_YET),
    mSensor(sensor),
    mbOnlyTracking(false),
    mbVO(false), //当处于纯跟踪模式时候，这个变量表示了当前跟踪状态的好坏
    mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB),
    mpInitializer(static_cast<Initializer*>(NULL)),
    mpSystem(pSys),
    mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer),
    mpMapDrawer(pMapDrawer),
    mpMap(pMap),
    mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"]; //双目baseline * fx 50

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    ///added module
    //load Tcam_Lidar parameters
    cv::Mat Tcl = cv::Mat::eye(4,4,CV_64F);
    Tcl.at<double>(0,0) = fSettings["Rcl.11"];
    Tcl.at<double>(0,1) = fSettings["Rcl.12"];
    Tcl.at<double>(0,2) = fSettings["Rcl.13"];
    Tcl.at<double>(1,0) = fSettings["Rcl.21"];
    Tcl.at<double>(1,1) = fSettings["Rcl.22"];
    Tcl.at<double>(1,2) = fSettings["Rcl.23"];
    Tcl.at<double>(2,0) = fSettings["Rcl.31"];
    Tcl.at<double>(2,1) = fSettings["Rcl.32"];
    Tcl.at<double>(2,2) = fSettings["Rcl.33"];
    Tcl.at<double>(0,3) = fSettings["Tcl.1"];
    Tcl.at<double>(1,3) = fSettings["Tcl.2"];
    Tcl.at<double>(2,3) = fSettings["Tcl.3"];
    Tcl.copyTo(mTcamlid);

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"]; //每帧特征点数 1000
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"]; //图像金字塔的尺度 1.2
    int nLevels = fSettings["ORBextractor.nLevels"]; //金字塔层数 8
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"]; //fast初始阈值 20
    int fMinThFAST = fSettings["ORBextractor.minThFAST"]; //如果达不到足够的特征点数量，改用最小阈值 8

    //tracking过程使用的是left实例作为特征提取器
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //monocular初始化过程中使用这个实例作为特征提取器，注意两倍特征数
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        //判断一个3D点远近的阈值，mdf * 35 /fx 实际就是基线长度的xx倍
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

///added module
/**
 * Input image, image time, laser, laser times.
 * Constrcut Frame instance.
 * Run Track() and return Tcw.
 * @param im : passed image frame
 * @param timestamp : image frame time
 * @param lasers : passed laser points
 * @param laserTimes : laser middle time, start time and end time
 * @return Tcw
 */
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, const vector<vector<double>> &lasers, const vector<double> &laserTimes)
{
    //mImGray 是tracking class 的成员
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    //enum eTrackingState : Sys not ready -1, no img yet 0, not init 1, ok 2, lost 3
    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        //mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
        ///added module
        mCurrentFrame = Frame(mImGray, timestamp, lasers, laserTimes, mpIniORBextractor, mpORBVocabulary, mK, mTcamlid,
                              mDistCoef, mbf, mThDepth);
    else
        //mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
        ///added module
        mCurrentFrame = Frame(mImGray, timestamp, lasers, laserTimes, mpORBextractorLeft, mpORBVocabulary, mK, mTcamlid,
                              mDistCoef, mbf, mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    //mImGray 是tracking class 的成员
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    //enum eTrackingState : Sys not ready -1, no img yet 0, not init 1, ok 2, lost 3
    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


/////added module
//    void Tracking::ProjectLiDARtoImage() {
////    ///test
////    int testNum = 10;
////    vector<vector<double>> testPoints;
////    for(int i=0;i<testNum;i++)
////    {
////        vector<double> thisPoint = {1,0,-2};
////        thisPoint[0] +=i*2;
////        testPoints.push_back(thisPoint);
////    }
////    mCurrentFrame.mLaserPoints = testPoints;
//
//        ///project distorted Laser points to Image frame
//        int lsrPtNum = mCurrentFrame.mLaserPoints.size();
//
//        if (lsrPtNum > 0) {
//            cv::Mat P_rect_00 = cv::Mat::zeros(CvSize(4, 3), CV_64F);
//            P_rect_00.at<double>(0, 0) = (double) mK.at<float>(0, 0);
//            P_rect_00.at<double>(0, 2) = (double) mK.at<float>(0, 2);
//            P_rect_00.at<double>(1, 1) = (double) mK.at<float>(1, 1);
//            P_rect_00.at<double>(1, 2) = (double) mK.at<float>(1, 2);
//            P_rect_00.at<double>(2, 2) = 1;
//            cv::Mat R_rect_00 = cv::Mat::eye(CvSize(4, 4), CV_64F);
//
//            cv::Mat X(4, 1, CV_64F);//3D LiDAR point
//            cv::Mat Y(3, 1, CV_64F);//2D LiDAR projection
//            for (int li = 0; li < lsrPtNum; li++) {
//                // filter the not needed points
//                double maxX = 25.0, maxY = 6.0, minZ = -1.8;
//                if (mCurrentFrame.mLaserPoints[li][0] > maxX || mCurrentFrame.mLaserPoints[li][0] < 0.0
//                    || mCurrentFrame.mLaserPoints[li][1] > maxY || mCurrentFrame.mLaserPoints[li][1] < -maxY
//                    || mCurrentFrame.mLaserPoints[li][2] < minZ
//                    || mCurrentFrame.mLaserPoints[li][3] >
//                       -minZ) //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter distance can only detect 1.44+1.73 height
//                {
//                    continue;
//                }
//
//                X.at<double>(0, 0) = mCurrentFrame.mLaserPoints[li][0];
//                X.at<double>(1, 0) = mCurrentFrame.mLaserPoints[li][1];
//                X.at<double>(2, 0) = mCurrentFrame.mLaserPoints[li][2];
//                X.at<double>(3, 0) = 1;
//
//                //cout<<"LiDAR point "<<X.t()<<endl;
//                Y = P_rect_00 * R_rect_00 * mTcamlid * X;
//                //cout<<"Y "<<Y<<endl;
//                cv::Point pt;
//                pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
//                pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
//                //cout<<"image frame "<<pt.x<<" "<<pt.y<<endl;
//                if (pt.x < 0 || pt.x >= mImGray.cols || pt.y < 0 || pt.y >= mImGray.rows) {
//                    continue;
//                }
//                //distance as response
//                double responseVal = sqrt(
//                        X.at<double>(0, 0) * X.at<double>(0, 0) + X.at<double>(1, 0) * X.at<double>(1, 0) +
//                        X.at<double>(2, 0) * X.at<double>(2, 0));
//                cv::KeyPoint thisPoint(pt, 0, -1, responseVal, 0, -1);
//                //mCurrentFrame.mPjcLaserPts.push_back(thisPoint);
//                mCurrentFrame.mPjcLaserPts.push_back(pt);
//            }
//        }
//        ///project undistorted laser points
//        lsrPtNum = mCurrentFrame.mLaserPtsUndis.size();
//        if (lsrPtNum > 0) {
//            cv::Mat P_rect_00 = cv::Mat::zeros(CvSize(4, 3), CV_64F);
//            P_rect_00.at<double>(0, 0) = (double) mK.at<float>(0, 0);
//            P_rect_00.at<double>(0, 2) = (double) mK.at<float>(0, 2);
//            P_rect_00.at<double>(1, 1) = (double) mK.at<float>(1, 1);
//            P_rect_00.at<double>(1, 2) = (double) mK.at<float>(1, 2);
//            P_rect_00.at<double>(2, 2) = 1;
//            cv::Mat R_rect_00 = cv::Mat::eye(CvSize(4, 4), CV_64F);
//
//            cv::Mat X(4, 1, CV_64F);
//            cv::Mat Y(3, 1, CV_64F);
//            for (int li = 0; li < lsrPtNum; li++) {
//                // filter the not needed points
//                double maxX = 25.0, maxY = 6.0, minZ = -1.8;
//                if (mCurrentFrame.mLaserPtsUndis[li][0] > maxX || mCurrentFrame.mLaserPtsUndis[li][0] < 0.0
//                    || mCurrentFrame.mLaserPtsUndis[li][1] > maxY || mCurrentFrame.mLaserPtsUndis[li][1] < -maxY
//                    || mCurrentFrame.mLaserPtsUndis[li][2] < minZ
//                    || mCurrentFrame.mLaserPtsUndis[li][3] < 0.01) {
//                    continue;
//                }
//
//                X.at<double>(0, 0) = mCurrentFrame.mLaserPtsUndis[li][0];
//                X.at<double>(1, 0) = mCurrentFrame.mLaserPtsUndis[li][1];
//                X.at<double>(2, 0) = mCurrentFrame.mLaserPtsUndis[li][2];
//                X.at<double>(3, 0) = 1;
//
//                cout << "X " << X << endl;
//                Y = P_rect_00 * R_rect_00 * mTcamlid * X;
//                cout << "Y " << Y << endl;
//                cv::Point pt;
//                pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
//                pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
//                if (pt.x < 0 || pt.x > mImGray.cols || pt.y < 0 || pt.y > mImGray.rows) {
//                    //cout<<X.t()<<" | ";cout<<pt<<endl;
//                    continue;
//                }
//                //distance as response
//                double responseVal = sqrt(
//                        X.at<double>(0, 0) * X.at<double>(0, 0) + X.at<double>(1, 0) * X.at<double>(1, 0) +
//                        X.at<double>(2, 0) * X.at<double>(2, 0));
//                cv::KeyPoint thisPoint(pt, 0, -1, responseVal, 0, -1);
//                mCurrentFrame.mPjcLaserPtsUndis.push_back(thisPoint);
//            }
//        }
//    }

///**
// * project plane's 3D point to 2D frame
// */
//    void Tracking::ProjectPlanetoImage() {
//        cv::Mat P_rect_00 = cv::Mat::zeros(CvSize(4, 3), CV_64F);
//        P_rect_00.at<double>(0, 0) = (double) mK.at<float>(0, 0);
//        P_rect_00.at<double>(0, 2) = (double) mK.at<float>(0, 2);
//        P_rect_00.at<double>(1, 1) = (double) mK.at<float>(1, 1);
//        P_rect_00.at<double>(1, 2) = (double) mK.at<float>(1, 2);
//        P_rect_00.at<double>(2, 2) = 1;
//        cv::Mat R_rect_00 = cv::Mat::eye(CvSize(4, 4), CV_64F);
//        ///1st project plane normal first
//        int planNum = mCurrentFrame.mvPlanes.size();
//        for (int plni = 0; plni < planNum; plni++) {
//            cv::Mat X(4, 1, CV_64F);//3D LiDAR point
//            cv::Mat Y(3, 1, CV_64F);//2D LiDAR projection
//            cv::Point pt;
//            for (int pti = 0; pti < mCurrentFrame.mvPlanes[plni].pointList.size(); pti++) {
//                X.at<double>(0, 0) = mCurrentFrame.mvPlanes[plni].pointList[pti].x;
//                X.at<double>(1, 0) = mCurrentFrame.mvPlanes[plni].pointList[pti].y;
//                X.at<double>(2, 0) = mCurrentFrame.mvPlanes[plni].pointList[pti].z;
//                X.at<double>(3, 0) = 1;
//                //Y = P_rect_00 * R_rect_00 * mTcamlid * X;
//                Y = P_rect_00 * R_rect_00 * X;
//                pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
//                pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
//                mCurrentFrame.mvPlanes[plni].pointList2D.push_back(pt);
//                //cout<<"project point "<<X.t()<<" to "<<pt.x<<" "<<pt.y<<endl;
//            }
//        }
//    }

void Tracking::Track()
{
    ///added module
    ///project raw 3D LiDAR point to 2D image frame
    //ProjectLiDARtoImage();
    ///todo Should think about the low frequency of LiDAR plane extraction

    //Track包含估计运动和跟踪局部地图两个部分
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    //上锁，保证地图不发生变化
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    //* Step 1 初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();
        //更新绘制器中存储的最新状态
        mpFrameDrawer->Update(this);
        //这个状态量mState在上面的初始化函数中更新
        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        //临时变量，每个函数是否执行成功
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            //*Step 2 进入正常SLAM模式，包含地图更新
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                //* Step 2.1 检查并更新上一帧被替换的Mappoint
                //?主要是局部地图线城里面可能会对现有地图点进行替换
                CheckReplacedInLastFrame();

                //* Step 2.2 用参考关键帧来恢复位姿 <- 运动模型为空，说明初始化刚开始，或者已经丢失 || 当前帧 紧跟在 重定位帧之后
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    //系统刚初始化，没有速度，肯定进来这里
                    //用最近的关键帧来跟踪当前普通帧
                    //通过BoW在参考帧中找到当前帧的特征点匹配
                    //优化每个特征点的3D点在当前帧的重投影误差来得到位姿
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    //用最近的普通帧来跟踪当前普通帧
                    //恒速模型获得初始位姿
                    //投影匹配
                    //优化特征点对应的3D点获得位姿
                    bOK = TrackWithMotionModel();
                    //如果失败了，回去用参考关键帧
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated
            //*Step 2 跟丢了，重定位
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                //mbVO是纯定位模式才使用的变量
                //false表示此帧匹配了很多mappoint，跟踪正常
                //true表示匹配很少的mappoint，很不稳定
                if(!mbVO)
                {
                    //*Step 2.2 跟踪正常，跟踪
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                        //? 是不是要加上
                        // if(!bOK)
                        // bOK = TrackReferenceKeyFrame();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                //跟踪了很少的maoppoint 不稳定
                //既要跟踪又要重定位
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    //motion model跟踪结果
                    bool bOKMM = false;
                    //重定位结果
                    bool bOKReloc = false;
                    //运动模型构造的地图点
                    vector<MapPoint*> vpMPsMM;
                    //运动模型跟踪时候的outlier
                    vector<bool> vbOutMM;
                    //运动模型的位姿
                    cv::Mat TcwMM;
                    //*Step 2.3 运动模型跟踪
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        //恒速运动结束临时保存 后面的重定位会更新
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }

                    //*Step 2.4 重定位得到当前位姿
                    bOKReloc = Relocalization();

                    //*Step 2.5 跟组重定位和恒速模型的结果更新当前帧的跟踪结果
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        //如果当前匹配的mappoint很少，增加当前可是地图点的观测次数
                        //?必然True
                        //?是不是重复增加了观测次数？后面tracklocalmap函数包含这些动作
                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        //重定位成功，mbvo false
                        mbVO = false;
                    }
                    //两个过程成功一个即可
                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        //将最新的关键帧 作为 当前帧的.参考关键帧(有可能为空)
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        //*Step 3 在跟踪得到当前帧初始状态后 对local map进行跟踪 得到更多匹配 优化当前位姿
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        //*Step 4 更新显示线城的信息 比如图像 特征点 地图点
        // Update drawer
        mpFrameDrawer->Update(this);

        //只有跟踪成功的时候才考虑生成关键帧
        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            //*Step 5 更新恒速运动模型
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                //Velocity = Tcl = Tcw * Twl
                mVelocity = mCurrentFrame.mTcw * LastTwc;
                //cout << mCurrentFrame.mnId << " velocity " << mCurrentFrame.mTcw << endl;
                ///save keypoints
                fstream writer; string fileName = "data//keypoint//"+std::to_string(mCurrentFrame.mnId)+".txt";
                writer.open(fileName, std::ios::out);
                for (int ki = 0; ki < mCurrentFrame.mvpMapPoints.size(); ki++) {
                    if (mCurrentFrame.mvpMapPoints[ki]) {
                        writer << mCurrentFrame.mvpMapPoints[ki]->GetWorldPos().at<float>(0, 0) << " "
                               << mCurrentFrame.mvpMapPoints[ki]->GetWorldPos().at<float>(1, 0) << " "
                               << mCurrentFrame.mvpMapPoints[ki]->GetWorldPos().at<float>(2, 0) << " "
                               << mCurrentFrame.mvpMapPoints[ki]->mnId<<endl;
                    }
                }
                writer.close();
                ///save velocity
//                fstream writer;
//                string fileName = "data//velocity//" + std::to_string(mCurrentFrame.mnId) + ".txt";
//                cout << "file name " << fileName << endl;
//                writer.open(fileName, std::ios::out);
//                writer << mVelocity.at<float>(0, 0) << " " << mVelocity.at<float>(0, 1) << " "
//                       << mVelocity.at<float>(0, 2) << " " << mVelocity.at<float>(0, 3) << endl;
//                writer << mVelocity.at<float>(1, 0) << " " << mVelocity.at<float>(1, 1) << " "
//                       << mVelocity.at<float>(1, 2) << " " << mVelocity.at<float>(1, 3) << endl;
//                writer << mVelocity.at<float>(2, 0) << " " << mVelocity.at<float>(2, 1) << " "
//                       << mVelocity.at<float>(2, 2) << " " << mVelocity.at<float>(2, 3) << endl;
//                writer << mVelocity.at<float>(3, 0) << " " << mVelocity.at<float>(3, 1) << " "
//                       << mVelocity.at<float>(3, 2) << " " << mVelocity.at<float>(3, 3) << endl;
//                writer.close();
//                ///added module
//                ///undis LiDAR point with motion from vision
//                UndisLiDAR();
//                ///extract plane segement
//                RegionGrowing(mCurrentFrame,true);
////                //todo --- lidar VO
////                ///project plane
//                ProjectPlanetoImage();
////                ///update plane infor
//                mpFrameDrawer->UpdateLiDAR(this);
////                //todo associate the ORB feature with LiDAR points
            }
            else
            //否则速度为空
                mVelocity = cv::Mat();
            //更新显示的位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            //*Step 6 清楚观测不到的地图点
            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            //*Step 7 清楚恒速模型中 updatelastframe中临时添加的mappoints（仅双目和rgbd）
            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            //*Step 8 检测并插入关键帧，对于双目和rgbd会产生新的地图点
            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            //*Step 9 删除在BA中为outlier的点
            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        ///added Need to check if safe or not
        //*Step 4 更新显示线城的信息 比如图像 特征点 地图点
        // Update drawer
        mpFrameDrawer->Update(this);

        //*Step 10 如果初始化不久就跟踪失败 并且relocation也没搞定 就reset
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }
        //确保已经设置了参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        //保存上一帧的数据，当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    //*Step 11 记录位姿信息 用于最后保存所有轨迹
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        //Tcr = Tcw * Twr, Twr = Trw^-1
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        //如果当前帧位姿没有，即跟踪失败，相对位姿使用上一次的数据
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}//Tracking


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

/**
 * @brief 单目相机初始化成功后用三角化的点生成Mappoints地图点
 */
void Tracking::MonocularInitialization()
{
    //Step 1 mpInitializer不存在时，创建一个实例
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            //把当前帧赋给初始化帧
            mInitialFrame = Frame(mCurrentFrame);
            //把当前帧赋给上一帧
            mLastFrame = Frame(mCurrentFrame);
            //记录上一帧的特征点
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            //This will never work
            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            //初始化匹配结果 -1
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            //函数返回，下次进来执行else部分
            return;
        }
    }
    else
    {
        // Try to initialize
        //Step 2 如果当前帧的特征点太少 删除初始化器 
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        // Step 3 当真帧和初始帧匹配
        ORBmatcher matcher(0.9,                                        //最佳和次佳特征点评分的比值阈值，越大则最佳次佳的区分度越小
                           true);                                      //检查特征点的方向
        int nmatches = matcher.SearchForInitialization(mInitialFrame,  //初始帧
                                                       mCurrentFrame,  //当前帧
                                                       mvbPrevMatched, //初始帧的特征点
                                                       mvIniMatches,   //保存匹配关系，size = IniFrame kypt number
                                                       100);           //搜索框大小

        // Check if there are enough correspondences
        // Step 4 是否找到足够的匹配
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // Step 5 满足初始化条件，计算F H矩阵，得到帧间运动 初始化mapPoints
        if (mpInitializer->Initialize(mCurrentFrame,   //当前帧
                                      mvIniMatches,    //当前帧和参考帧的特征点匹配关系
                                      Rcw, tcw,        //初始化得到的世界相对相机的位姿
                                      mvIniP3D,        //三角化得到的空间点
                                      vbTriangulated)) //一个table，记录mvIniMatches里哪些点被三角化了。
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                //Step 6 删除有匹配关系但无法进行三角化的点
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // Step 7 初始化的第一帧作为世界坐标系，所以第一帧pose为I
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

//            ///added module
//            AssociateLiDARInit(10);

            // Step 8 创建初始化地图点MapPoints
            CreateInitialMapMonocular();
        }
    }
}

/**
 * given a origin point, a ray, a plane norn and plane point
 * return depth of intersect point
 */
    vector<float>
    Tracking::RayPlaneDis(vector<float> ray_dir, vector<float> origin, vector<float> PlaneN, vector<float> PlaneP) {
        ///ray = pt + t * dir
        ///plane = plane norm and plane pt
        ///t = (plane.n * plane.pt - plane.n * ray.origin) / (plane.n * ray_dir)
        ///intersect = origin + t * ray_dir
        cout<<" ray dir "<<ray_dir[0]<<" "<<ray_dir[1]<<" "<<ray_dir[2]<<" ";
        cout<<" origin pt "<<origin[0]<<" "<<origin[1]<<" "<<origin[2]<<" ";
        cout<<" Plane norm "<<PlaneN[0]<<" "<<PlaneN[1]<<" "<<PlaneN[2]<<" ";
        cout<<" Plane pt "<<PlaneP[0]<<" "<<PlaneP[1]<<" "<<PlaneP[2]<<" "<<endl;
        float t = (PlaneN[0] * PlaneP[0] + PlaneN[1] * PlaneP[1] + PlaneN[2] * PlaneP[2] - PlaneN[0] * origin[0] -
                   PlaneN[1] * origin[1] - PlaneN[2] * origin[2]) /
                  (PlaneN[0] * ray_dir[0] + PlaneN[1] * ray_dir[1] + PlaneN[2] * ray_dir[2]);
        cout<<" "<<origin[0] + t * ray_dir[0]<<" "<<origin[1] + t * ray_dir[1]<<" "<<origin[2] + t * ray_dir[2]<<endl;
        vector<float> intersectP;
        intersectP.push_back(origin[0] + t * ray_dir[0]);
        intersectP.push_back(origin[1] + t * ray_dir[1]);
        intersectP.push_back(origin[2] + t * ray_dir[2]);
        return intersectP;
    }

/**
 * find keypoint's nearby projected lidar point
 * retreve depth by fitting the nearby lidar point into a plane
 * ray of keypoint intersect with plane provides the depth.
 */
    void Tracking::AssociateLiDARInit(float pixelThres) {
        ///Step Traverse Mappoint, find closest 2d lidar Points (and related 3D lidar pts)
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] > -1) {
                cv::Point2d p2d = mCurrentFrame.mvKeysUn[mvIniMatches[i]].pt;
                vector<PtLsr> nearbyLsrPt;
                float maxZ = -999, minZ = 999;
                for (int j = 0; j < mCurrentFrame.mLaserPt_cam.size(); j++) {
                    if (mCurrentFrame.mLaserPt_cam[j].index2d >= 0) {
                        float disX = abs(p2d.x - mCurrentFrame.mLaserPt_cam[j].pt2d.x);
                        float disY = abs(p2d.y - mCurrentFrame.mLaserPt_cam[j].pt2d.y);
                        if (disX < pixelThres && disY < pixelThres && sqrt(disX * disX + disY * disY) < pixelThres) {
                            nearbyLsrPt.push_back(mCurrentFrame.mLaserPt_cam[j]);
                            if (maxZ < mCurrentFrame.mLaserPt_cam[j].pt3d.z)
                                maxZ = mCurrentFrame.mLaserPt_cam[j].pt3d.z;
                            if (minZ > mCurrentFrame.mLaserPt_cam[j].pt3d.z)
                                minZ = mCurrentFrame.mLaserPt_cam[j].pt3d.z;
                        }
                    }
                }
                ///group by Z depth. step 0.3m
                if (nearbyLsrPt.size() > 0) {
                    float stepZ = 0.3;
                    //cout<<"maxZ "<<maxZ <<" minZ "<<minZ<<endl;
                    float length = maxZ - minZ;
                    int groupNum = floor(length / stepZ + 1);
                    //cout<<"hist group num "<<groupNum<<endl;
                    vector<vector<PtLsr>> hist;
                    for (int g = 0; g < groupNum; g++) {
                        vector<PtLsr> thisHist;
                        hist.push_back(thisHist);
                    }
                    //cout << "step test" << endl;
                    for (int j = 0; j < nearbyLsrPt.size(); j++) {
                        //cout << "z - minZ : " << (nearbyLsrPt[j].pt3d.z + 0.00001 - minZ);
                        int index = floor((nearbyLsrPt[j].pt3d.z + 0.00001 - minZ) / stepZ);
                        //cout << " index: " << index << " | ";
                        hist[index].push_back(nearbyLsrPt[j]);//error?
                    }
                    //cout<<endl;
                    ///find group with min average distance
                    int minHistIndex = 0;
                    float mindis = 9999;
                    for (int g = 0; g < groupNum; g++) {
                        float disSum = 0;
                        for (int k = 0; k < hist[g].size(); k++) {
                            float dis = sqrt((p2d.x - hist[g][k].pt2d.x) * (p2d.x - hist[g][k].pt2d.x) +
                                             (p2d.y - hist[g][k].pt2d.y) * (p2d.y - hist[g][k].pt2d.y));
                            disSum += dis;
                        }
                        float disAvg = disSum / hist[g].size();
                        if (disAvg < mindis) {
                            minHistIndex = g;
                            mindis = mindis;
                        }
                    }
                    //cout<<"closest group "<<minHistIndex<<endl;
                    ///fit a plane with that group
                    if(hist[minHistIndex].size()>5)
                    {
                        ofstream writer;
                        string filename = "data//nearby//" + std::to_string(i) + ".txt";
                        cout<<filename<<endl;
                        writer.open(filename, ios::out);
                        writer << p2d.x << " " << p2d.y << " " << -999 << endl;
                        for (int j = 0; j < nearbyLsrPt.size(); j++) {
                            writer << nearbyLsrPt[j].pt3d.x << " " << nearbyLsrPt[j].pt3d.y << " " << nearbyLsrPt[j].pt3d.z
                                   << endl;
                        }

                        pcl::PointCloud<pcl::PointXYZ>::Ptr nearbyPoints2(new pcl::PointCloud<pcl::PointXYZ>);
                        nearbyPoints2->resize(hist[minHistIndex].size());
                        for(int pi =0; pi<hist[minHistIndex].size(); pi++)
                        {
                            nearbyPoints2->points[pi].x = hist[minHistIndex][pi].pt3d.x;
                            nearbyPoints2->points[pi].y = hist[minHistIndex][pi].pt3d.y;
                            nearbyPoints2->points[pi].z = hist[minHistIndex][pi].pt3d.z;
                        }
                        pcl::PointIndices inliersOutput;
                        Plane foundPlane;//not fill
                        int inPlaneNum = RANSACPlane(nearbyPoints2, foundPlane, inliersOutput);
                        //cout<<"inplane num "<<inPlaneNum;
                        ///Distance is the ray intersect with plane
                        ///Pt_img -> Pt_cam
                        vector<float> ray_dir;
                        cv::Mat Pt = cv::Mat::ones(3,1,CV_32F);
                        Pt.at<float>(0,0) = p2d.x;
                        Pt.at<float>(1,0) = p2d.y;
                        cv::Mat P_cam0 = mK.inv()*Pt;
                        ray_dir.push_back(P_cam0.at<float>(0,0));
                        ray_dir.push_back(P_cam0.at<float>(1,0));
                        ray_dir.push_back(P_cam0.at<float>(2,0));
                        vector<float> origin;
                        origin.push_back(0);
                        origin.push_back(0);
                        origin.push_back(0);
                        vector<float> planeN;
                        planeN.push_back(foundPlane.A);
                        planeN.push_back(foundPlane.B);
                        planeN.push_back(foundPlane.C);
                        vector<float> planeP;
                        planeP.push_back(nearbyPoints2->points[0].x);
                        planeP.push_back(nearbyPoints2->points[0].x);
                        planeP.push_back(nearbyPoints2->points[0].x);
                        vector<float> intersectP;
                        intersectP = RayPlaneDis(ray_dir, origin, planeN, planeP);
                        writer<<intersectP[0]<<" "<<intersectP[1]<<" "<<intersectP[2]<<endl;
                        writer.close();
                    }
                }
                //depth from Cur Frame
                //cout<<"init frame pt index "<<i;
//                int curLsrPtNum = mCurrentFrame.mLaserPt_cam.size();
//                pcl::PointCloud<pcl::PointXYZ>::Ptr nearbyPoints2(new pcl::PointCloud<pcl::PointXYZ>);
//                nearbyPoints2->resize(curLsrPtNum);
//                int index2 = mvIniMatches[i];
//                //cout<<" cur frame pt index2 "<<index2;
//                cv::KeyPoint kp2 = mCurrentFrame.mvKeysUn[index2];
//                int actualNum2 = 0;
//                for (int pi = 0; pi < curLidPtNum; pi++) {
//                    float distance = sqrt(
//                            (kp2.pt.x - mCurrentFrame.mPjcLaserPts[pi].x) *
//                            (kp2.pt.x - mCurrentFrame.mPjcLaserPts[pi].x)
//                            +
//                            (kp2.pt.y - mCurrentFrame.mPjcLaserPts[pi].y) *
//                            (kp2.pt.y - mCurrentFrame.mPjcLaserPts[pi].y));
//                    if (distance < pixelThres) {
//                        nearbyPoints2->points[actualNum2].x = mCurrentFrame.mLaserPt_cam[pi][0];
//                        nearbyPoints2->points[actualNum2].y = mCurrentFrame.mLaserPt_cam[pi][1];
//                        nearbyPoints2->points[actualNum2].z = mCurrentFrame.mLaserPt_cam[pi][2];
//                        actualNum2++;
//                    }
//                }
//                nearbyPoints2->resize(actualNum2);
                //cout<<" nearby laser num "<<actualNum2;
//                ///fit a plane with nearby lidar points
//                if (actualNum2 > 5) {
//                    pcl::PointIndices inliersOUT;
//                    Plane foundPlane;
//                    int inPlaneNum = RANSACPlane(nearbyPoints2, foundPlane, inliersOUT);
//                    //cout<<"inplane num "<<inPlaneNum;
//                    //todo distance is the ray intersect with plane
//                    vector<float> ray_dir;
//                    ray_dir.push_back(mvIniP3D[i].x);
//                    ray_dir.push_back(mvIniP3D[i].y);
//                    ray_dir.push_back(mvIniP3D[i].z);
//                    vector<float> origin;
//                    origin.push_back(0);
//                    origin.push_back(0);
//                    origin.push_back(0);
//                    vector<float> planeN;
//                    planeN.push_back(foundPlane.A);
//                    planeN.push_back(foundPlane.B);
//                    planeN.push_back(foundPlane.C);
//                    vector<float> planeP;
//                    planeP.push_back(foundPlane.pointList[0].x);
//                    planeP.push_back(foundPlane.pointList[0].y);
//                    planeP.push_back(foundPlane.pointList[0].z);
//                    float depth2 = RayPlaneDis(ray_dir, origin, planeN, planeP);
//                    cout << " " << mvIniP3D[i].x << " " << mvIniP3D[i].y << " " << mvIniP3D[i].z << endl;
//                }
            }
        }
    }

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // * Step 1 计算初始帧和当前帧的BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // * Step 2 关键帧插入地图
    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

//    ///added module
//    //LiDAR ICP on the init - cur frame
//    ///extract plane segement
//    RegionGrowing(mInitialFrame, false);
//    RegionGrowing(mCurrentFrame, false);
//    cv::Mat Tc1c2(4,4,CV_32F);
//    LidarICP(mInitialFrame,mCurrentFrame,Tc1c2);
//    cout<<"init map mono Lidar Tc1c2 "<<endl<<Tc1c2<<endl;
//    cv::Mat Tcw = mCurrentFrame.mTcw;
//    cout<<"init map mono Vision Tcw "<<endl<<Tcw<<endl;
//    ///calc and apply ratio
//    float lengthVision = sqrt(Tcw.at<float>(0, 3) * Tcw.at<float>(0, 3)
//                              + Tcw.at<float>(1, 3) * Tcw.at<float>(1, 3)
//                              + Tcw.at<float>(2, 3) * Tcw.at<float>(2, 3));
//    float lengthLiDAR = sqrt(Tc1c2.at<float>(0, 3) * Tc1c2.at<float>(0, 3)
//                              + Tc1c2.at<float>(1, 3) * Tc1c2.at<float>(1, 3)
//                              + Tc1c2.at<float>(2, 3) * Tc1c2.at<float>(2, 3));
//    float ratio = lengthLiDAR / lengthVision;
//    cout<<"lidar vision ratio "<<ratio<<endl;
////    for(size_t i=0; i<mvIniMatches.size();i++)
////    {
////        if(mvIniMatches[i]<0)
////            continue;
////        mvIniP3D[i].x = mvIniP3D[i].x * ratio;
////        mvIniP3D[i].y = mvIniP3D[i].y * ratio;
////        mvIniP3D[i].z = mvIniP3D[i].z * ratio;
////    }

    // * Step 3 用初始化得到的3D点来生成地图点MapPoints
    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        //直接把三角化的点赋值给Mat作为空间点的世界坐标
        cv::Mat worldPos(mvIniP3D[i]);
        // * Step 3.1 用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        //* Step 3.2 为这个MapPoint增加一些属性
        //关键帧instance增加地图点和相应的索引
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);
        //地图点增加关键帧和对应的索引
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);
        //从众多观测中挑选一个最具代表的描述子
        pMP->ComputeDistinctiveDescriptors();
        //更新该地图点的平均观测方向和观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //当前帧实例的Mappoints更新 Outlier更新
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //把地图点加入地图
        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    //* Step 3.3 更新关键帧之间的关系
    //当前关键帧和其他关键帧之间建边，边的权重是该帧和当前帧共视点数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    //* Step 4 全局BA
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    //* Step 5 取场景的中值深度，用于尺度归一化
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    cout<<"median depth "<<medianDepth<<endl;
    float invMedianDepth = 1.0f/medianDepth;
    //invMedianDepth = ratio;
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    //* STEP6 将两帧之间的变换 归一化到平均深度为1的尺度下
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    //* step7 把3D点也归一化到1
    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    //* Step 8 把关键帧插入局部地图，更新归一化后的位姿，局部地图点
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    //把得到的所有地图点存入局部地图
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);
    //todo ? for local optimization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


/**
 * @brief 对参考关键帧的MapPoints进行跟踪（一般是运动模式匹配到的特征点比较少，或者刚初始化完成）
 * 1.计算当前词包，将当前帧的特征点分配到指定层的nodes上
 * 2.对属于同一node的描述子进行匹配
 * 3.根据匹配对 估计当前帧的姿态
 * 4.根据姿态剔除错误匹配
 * @return 如果匹配数目大于10，返回True
*/
bool Tracking::TrackReferenceKeyFrame()
{
    //*STEP1 讲当前帧的描述子转化为BoW向量
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    //存储当前帧的特征点和3D地图点的匹配关系
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    //* Step3 将上一帧的位姿作为当前帧位姿的初始值 加速poseoptimization
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    //*Step4 优化重投影误差来（3D-2D）获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    //*Step 5 剔除outlier
    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    //*Step 1 计算上一帧在世界系下的位姿
    // Update pose according to reference keyframe
    //上一帧的参考帧
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    //Transfrom^lastframe_refence 从参考帧到上一帧的变换
    cv::Mat Tlr = mlRelativeFramePoses.back();

    //Tlw = Tlr * Trw  
    //?为什么视频说是上一帧在世界系下的位姿？应该类似getPose返回Tcw,这里是设置Tlw。
    mLastFrame.SetPose(Tlr*pRef->GetPose());

    //单目或者上一帧是关键帧，程序结束，就设置了当前帧的位姿
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    //*Step 2 生成临时的地图点
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor

    //*Step 2.1 得到上一帧有深度的地图点
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;
    //放到容器里面排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    //*Step 2.2: 找出不是地图点的生成临时地图点
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        //通过id找这个点是否是地图点
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
            //或者是地图点但是没被观测过
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            //特征点反投影到地图点
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            //插入到上一帧地图中
            //?可以直接用i下标访问？所以之前是null?
            mLastFrame.mvpMapPoints[i]=pNewMP;
            //加入到临时地图点容器中，再createnewkeyframe之前会清空
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }
        //深度超过mTheDepth（35倍基线）并且 超过100个
        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

/**
 * @brief 以匀速估计对上一帧进行跟踪
 * Step 1 更新上一帧的位姿，双目或者rgbd还会根据深度生成临时地图点
 * Step 2 根据上一帧特征点对地图点进行投影匹配
 * Step 3 优化位姿
 * Step 4 剔除outlier
 * @return 如果匹配数大于10，返回true
*/
bool Tracking::TrackWithMotionModel()
{
    //初始化matcher 0.9 是 最小距离小于次小距离0.9 | 检查旋转
    //? keyframe tracking 没这个matcher？
    ORBmatcher matcher(0.9,true);

    //* Step 1 更新上一帧的位姿
    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    //当前帧位姿=速度*上一帧位姿 //Tc2w
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    //清空当前帧的地图点
    //?why？不应该本来就是空的？
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    //*Step 2 根据上一帧的特征点对应地图点进行投影匹配
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    //*Step 3 优化当前位姿
    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    //*Step 4 剔除ouliter
    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    //纯跟踪模式以匹配数目来判断是否跟踪成功
    if(mbOnlyTracking)
    {
        //?mbVO是啥？
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }
    //*Step return
    return nmatchesMap>=10;
}

/**
 * @brief 对local map 的mappoints进行跟踪
 * 1. 更新局部关键帧（加入1共视关键帧，2共视关键帧的共视帧，3共视关键帧的父子帧）和局部地图点（前者新引入的地图点）
 * 2. 对局部mappoints进行投影匹配（排除掉视野范围外的等等）
 * 3. 根据匹配估计当前帧姿态
 * 4. 根据姿态剔除outlier
*/
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    //*Step 1 更新局部关键帧 mvpLocalKeyFrames 和 局部地图点 mvpLocalMapPoints
    UpdateLocalMap();

    //*Step 2 匹配局部地图中 与 当前帧 匹配的Mappoints
    SearchLocalPoints();

    //*Step 3 更新局部地图点后 更新位姿
    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    //*Step 4 更新当前帧的mappoints被观测程度
    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //当前帧的mappoints可以被当前帧观测到 被观测次数+1
            //*刚updatelocalmap做过一次+1了吧？ 这里加的是found，前面加的是visible
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            //? 是outlier并且是双目，就删除这个点
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    //*Step 5 根据匹配点数目和回环情况决定是否跟踪成功
    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/**
 * @brief 判断当前帧是否需要插入关键帧
 * step 1 纯VO模式不插入关键帧 如果局部地图线城被闭环检测使用 则不插入关键帧
 * step 2 如果距离上一次重定位比较远 或者 关键帧数目超出最大限制 不插入关键帧
 * step 3 得到参考关键帧跟踪到的地图点的数目
 * step 4 查询局部地图管理器是否繁忙
 * step 5 RGBD和stero，统计可以添加的有效地图点的总数 和 跟踪到的地图点数量
 * step 6 决策是否插入
 */ 
bool Tracking::NeedNewKeyFrame()
{
    //*Step VO模式不插入
    if(mbOnlyTracking)
        return false;
    //*Step 局部地图线城被闭环检测使用 不插入
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    //*Step 距离上次重定位比较远 或者 超出数量
    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    //*Step 参考关键帧跟踪到的地图点数量
    //地图点有个最小观测次数的要求nminobs = 3
    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    //*Step 查询局部地图管理器是否繁忙
    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    //*Step stero或者rgbd 统计可以添加的有校地图点总数喝跟踪到的地图点数量
    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            //深度
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                //有地图点而不是outlier
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }
    //如果跟踪到的地图点太少 同时 没有跟踪到的点太多 插入关键帧
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    //*Step  决策是否插入关键帧
    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    ////*Step 长时间没有插入
    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    //*Step  满足插入关键帧的最小间隔 并且 localmapper空闲
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //*Step 双目RGBD情况下当前帧跟踪到的点比参考帧的1.15倍少 或者 满足 needtiinsertclose
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    //*Step 跟踪到的点 < 参考帧的点 或者 needtoinsertclose TRUE 同时 跟跟踪到的点不是太少
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        //*Step mapping空闲
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            //不空闲就中断掉
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况 会同时创建新的mappoints
*/
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    //*Step 1 将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    //*Step 2 将当前关键帧设置为当前帧的参考关键帧
    mpReferenceKF = pKF; //参考关键帧成员，给后面的帧用
    mCurrentFrame.mpReferenceKF = pKF;

    //*Step 3 对于RGBstero 生成新得地图点
    //跟 tracking::udpatelastframe里面更新地图点类似
    if(mSensor!=System::MONOCULAR)
    {
        //更新几个位姿
        mCurrentFrame.UpdatePoseMatrices();

        //*Step 3.1 当前帧有深度值的特征点
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            //*Step 3.2 按照深度排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            //*Step 3.3 从中找出不是地图点的生成临时地图点
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                //如果这个点对应在上一帧的地图点中没有，或者创建后就没观测到，就生成临时地图点
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                //如果需要就创建地图点，这里是全局地图的地图点，用于跟踪
                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    //每次添加全局mappoint时候都要插入属性
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                //*Step 3.3 停止新建地图点需要满足以下条件
                //点的深度超过阈值
                //npoints超过100个
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    //*Step 插入关键帧
    //关键帧插入局部地图里面
    mpLocalMapper->InsertKeyFrame(pKF);
    //插入后允许局部建图线城停止
    mpLocalMapper->SetNotStop(false);
    //更新 当前帧成为新得关键帧 
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

///Added Module
/**
 * @brief Based on Velocity captured from VO
 * Undistort LiDAR point cloud
*/
//    void Tracking::UndisLiDAR() {
//        //laser time, start time, end time
//        double t_l = mCurrentFrame.mLaserTimes[0];
//        double t_ls = mCurrentFrame.mLaserTimes[1];
//        double t_le = mCurrentFrame.mLaserTimes[2];
//        //last vision frame time, current frame time.
//        double t_last = mLastFrame.mTimeStamp;
//        double t_cur = mCurrentFrame.mTimeStamp;
//        //vision frame delta time.
//        double deltaTime = t_cur - t_last;
//        cv::Mat V = cv::Mat::zeros(3, 1, CV_64F);
//        V.at<double>(0, 0) = mVelocity.at<float>(0, 3) / deltaTime;
//        V.at<double>(1, 0) = mVelocity.at<float>(1, 3) / deltaTime;
//        V.at<double>(2, 0) = mVelocity.at<float>(2, 3) / deltaTime;
//        //cout << "Velocity " <<endl<< V << endl;
//        double timeToProj = t_cur - t_l;
//        cv::Mat timeToProjM = cv::Mat::zeros(3, 1, CV_64F);
//        timeToProjM.at<double>(0, 0) = timeToProj;
//        timeToProjM.at<double>(1, 0) = timeToProj;
//        timeToProjM.at<double>(2, 0) = timeToProj;
//        //cout << "timeToProjM " <<endl<< timeToProjM << endl;
//        cv::Mat P_undis = cv::Mat::zeros(3, 1, CV_64F);
//        cv::Mat P_distor = cv::Mat::zeros(3, 1, CV_64F);
//        for (int i = 0; i < mCurrentFrame.mLaserPt_cam.size(); i++) {
//            P_distor.at<double>(0, 0) = mCurrentFrame.mLaserPt_cam[i][0];
//            P_distor.at<double>(1, 0) = mCurrentFrame.mLaserPt_cam[i][1];
//            P_distor.at<double>(2, 0) = mCurrentFrame.mLaserPt_cam[i][2];
//            //cout << "P_distor " <<endl<< P_distor << endl;
//            if (t_l < t_cur)
//                ///t_last ---> t_ls ---> t_l ---> t_cur ---> t_le
//                ///forward from t_1 to t_cur
//                P_undis = P_distor + V.mul(timeToProjM);
//            else
//                ///t_last ---> t_ls ---> t_cur ---> t_l ---> t_le
//                ///backward from t_1 to t_cur
//            if (t_l > t_cur)
//                P_undis = P_distor - V.mul(timeToProjM);
//            else
//                ///t_l==t_cur
//                P_undis = P_distor;
//            //cout << "P_undis " << endl << P_undis << endl;
//            vector<double> undisPoint{P_undis.at<double>(0, 0), P_undis.at<double>(1, 0), P_undis.at<double>(2, 0)};
//            mCurrentFrame.mLaserPtsUndis.push_back(undisPoint);
//        }
//    }

///added module
//    void Tracking::RegionGrowing(Frame &inputFrame, bool Undistored) {
//        pcl::PointCloud<pcl::PointXYZ>::Ptr LiDARCloud(new pcl::PointCloud<pcl::PointXYZ>);
//        if (Undistored) {
//            int LiDARNum = inputFrame.mLaserPtsUndis.size();
//            LiDARCloud->points.resize(LiDARNum);
//            int actualCounter = 0;
//            for (int li = 0; li < LiDARNum; li++) {
//                LiDARCloud->points[actualCounter].x = inputFrame.mLaserPtsUndis[li][0];
//                LiDARCloud->points[actualCounter].y = inputFrame.mLaserPtsUndis[li][1];
//                LiDARCloud->points[actualCounter].z = inputFrame.mLaserPtsUndis[li][2];
//                actualCounter++;
//            }
//            LiDARCloud->points.resize(actualCounter);
//        } else {
//            int LiDARNum = inputFrame.mLaserPt_cam.size();
//            LiDARCloud->points.resize(LiDARNum);
//            int actualCounter = 0;
//            for (int li = 0; li < LiDARNum; li++) {
//                LiDARCloud->points[actualCounter].x = inputFrame.mLaserPt_cam[li][0];
//                LiDARCloud->points[actualCounter].y = inputFrame.mLaserPt_cam[li][1];
//                LiDARCloud->points[actualCounter].z = inputFrame.mLaserPt_cam[li][2];
//                actualCounter++;
//            }
//            LiDARCloud->points.resize(actualCounter);
//        }
//        //cout<<"lidar cloud actual number "<<LiDARCloud->points.size()<<endl;
//        ///estimating normals for each point
//        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
//        normal_estimator.setSearchMethod(tree);
//        normal_estimator.setInputCloud(LiDARCloud);
//        normal_estimator.setKSearch(50);
//        normal_estimator.compute(*normals);
//        ///region growing
//        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
//        reg.setMinClusterSize(1000);
//        reg.setMaxClusterSize(100000);
//        reg.setSearchMethod(tree);
//        reg.setNumberOfNeighbours(100);//too little will cause run time error
//        reg.setInputCloud(LiDARCloud);
//        reg.setInputNormals(normals);
//        reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
//        reg.setCurvatureThreshold(2.0);
//        ///extract each cluster
//        clock_t startTime = clock();
//        std::vector<pcl::PointIndices> clusters;
//        reg.extract(clusters);
//        clock_t endTime = clock();
//        double timeUsed = double(endTime - startTime) / CLOCKS_PER_SEC;
//        //cout << "Region Growing " << timeUsed << " sec ";
//        ///call RANSAC plane fitting
//        for (int ci = 0; ci < clusters.size(); ci++) {
//            pcl::PointCloud<pcl::PointXYZ>::Ptr thisCloud(new pcl::PointCloud<pcl::PointXYZ>);
//            thisCloud->points.resize(clusters[ci].indices.size());
//            thisCloud->height = 1;
//            thisCloud->width = clusters[ci].indices.size();
//            //cout<<" | cluster contains"<<clusters[ci].indices.size();
//            for (int index = 0; index < clusters[ci].indices.size(); index++) {
//                thisCloud->points[index].x = LiDARCloud->points[clusters[ci].indices[index]].x;
//                thisCloud->points[index].y = LiDARCloud->points[clusters[ci].indices[index]].y;
//                thisCloud->points[index].z = LiDARCloud->points[clusters[ci].indices[index]].z;
//                //cout<<"cluster point "<<thisCloud->points[index].x<<" "<<thisCloud->points[index].y<<" "<<thisCloud->points[index].z<<endl;
//            }
//            Plane foundPlane;
//            startTime = clock();
//            pcl::PointIndices inliersOUT;
//            int inPlaneNum = RANSACPlane(thisCloud, foundPlane, inliersOUT);
//            endTime = clock();
//            double timeUsed = double(endTime - startTime) / CLOCKS_PER_SEC;
//            //cout << " RANSAC plane " << timeUsed << " sec. Inliners num: " << inPlaneNum;
//            if (inPlaneNum > 0) {
//                int planeID = inputFrame.mvPlanes.size();
//                foundPlane.PlaneId = planeID;
//                inputFrame.mvPlanes.push_back(foundPlane);
//            }
//        }
//        //cout << endl;
//
//        ///save LiDAR planes
////    fstream writer;
////    string fileName = "data//lidar//"+std::to_string(mCurrentFrame.mnId) + ".txt";
////    writer.open(fileName,std::ios::out);
////    for(int plni = 0; plni<mCurrentFrame.mvPlanes.size();plni++)
////    {
////        for(int pi=0;pi<mCurrentFrame.mvPlanes[plni].pointList.size();pi++)
////        {
////            writer<<mCurrentFrame.mvPlanes[plni].pointList[pi].x<<" "<<mCurrentFrame.mvPlanes[plni].pointList[pi].y<<" "<<mCurrentFrame.mvPlanes[plni].pointList[pi].z<<endl;
////        }
////    }
////    writer.close();
//    }

int Tracking::RANSACPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Plane &foundPlane, pcl::PointIndices &inliersOutput)
{
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = inputCloud.makeShared();
    pcl::ModelCoefficients::Ptr  coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //create the segmentation objects
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    //Optional
    seg.setOptimizeCoefficients(true);
    //Mandatory
    seg.setMethodType(pcl::SACMODEL_PLANE);
    seg.setModelType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.001);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.size()==0)
        return 0;
    inliersOutput = *inliers;
    foundPlane.A = coefficients->values[0];foundPlane.B = coefficients->values[1];
    foundPlane.C = coefficients->values[2];foundPlane.D = coefficients->values[3];
    double sumX =0,sumY=0,sumZ=0;
    for(int i = 0; i < inliers->indices.size();i++)
    {
        double x = cloud->points[inliers->indices[i]].x;
        double y = cloud->points[inliers->indices[i]].y;
        double z = cloud->points[inliers->indices[i]].z;
        sumX += x;
        sumY += y;
        sumZ += z;
        //foundPlane.points3D.push_back(cv::Point3d(x,y,z));
        //cout<<"inliner push back "<<x<<" "<<y<<" "<<z<<endl;
    }
    foundPlane.centreP = cv::Point3d (sumX/inliers->indices.size(), sumY/inliers->indices.size(),sumZ/inliers->indices.size());
    return inliers->indices.size();
}

//void Tracking::LidarICP(Frame &inputFrame1, Frame &inputFrame2, cv::Mat &transformation)
//{
//    int numInit = 30000;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
//    cloud1->points.resize(numInit);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
//    cloud2->points.resize(numInit);
//    int actualNum = 0;
//    for (int plnIndex1 = 0; plnIndex1 < inputFrame1.mvPlanes.size(); plnIndex1++) {
//        for (int ptIndex = 0; ptIndex < inputFrame1.mvPlanes[plnIndex1].pointList.size(); ptIndex++) {
//            cloud1->points[actualNum].x = inputFrame1.mvPlanes[plnIndex1].pointList[ptIndex].x;
//            cloud1->points[actualNum].y = inputFrame1.mvPlanes[plnIndex1].pointList[ptIndex].y;
//            cloud1->points[actualNum].z = inputFrame1.mvPlanes[plnIndex1].pointList[ptIndex].z;
//            actualNum++;
//        }
//    }
//    cloud1->points.resize(actualNum);
//    int actualNum2 = 0;
//    for (int plnIndex2 = 0; plnIndex2 < inputFrame2.mvPlanes.size(); plnIndex2++) {
//        for (int ptIndex = 0; ptIndex < inputFrame2.mvPlanes[plnIndex2].pointList.size(); ptIndex++) {
//            cloud2->points[actualNum2].x = inputFrame2.mvPlanes[plnIndex2].pointList[ptIndex].x;
//            cloud2->points[actualNum2].y = inputFrame2.mvPlanes[plnIndex2].pointList[ptIndex].y;
//            cloud2->points[actualNum2].z = inputFrame2.mvPlanes[plnIndex2].pointList[ptIndex].z;
//            actualNum2++;
//        }
//    }
//    cloud2->points.resize(actualNum2);
//    ///ICP
////    Eigen::Matrix4f init;
////    cv::Mat pose2 = inputFrame2.GetPose();
////    //Todo float or double?
////    init << pose2.at<float>(0,0), pose2.at<float>(0,1), pose2.at<float>(0,2), pose2.at<float>(0,3),
////            pose2.at<float>(1,0), pose2.at<float>(1,1), pose2.at<float>(1,2), pose2.at<float>(1,3),
////            pose2.at<float>(2,0), pose2.at<float>(2,1), pose2.at<float>(2,2), pose2.at<float>(2,3),
////            pose2.at<float>(3,0), pose2.at<float>(3,1), pose2.at<float>(3,2), pose2.at<float>(3,3);
//    pcl::PointCloud<pcl::PointXYZ> cloudRegistered;
//    cloudRegistered.points.resize(actualNum);
//    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> ICPer;
//    ICPer.setInputSource(cloud1);
//    ICPer.setInputTarget(cloud2);
//    //icper.setMaxCorrespondenceDistance(1);
//    //icper.setTransformationEpsilon(1e-8);//?
//    //icper.setEuclideanFitnessEpsilon(0.01);
//    ICPer.setMaximumIterations(100);
//    ICPer.align(cloudRegistered);
//    Eigen::Matrix4f Tc1c2 = ICPer.getFinalTransformation();
//    transformation.at<float>(0,0) = Tc1c2(0,0);transformation.at<float>(0,1) = Tc1c2(0,1);transformation.at<float>(0,2) = Tc1c2(0,2);transformation.at<float>(0,3) = Tc1c2(0,3);
//    transformation.at<float>(1,0) = Tc1c2(1,0);transformation.at<float>(1,1) = Tc1c2(1,1);transformation.at<float>(1,2) = Tc1c2(1,2);transformation.at<float>(1,3) = Tc1c2(1,3);
//    transformation.at<float>(2,0) = Tc1c2(2,0);transformation.at<float>(2,1) = Tc1c2(2,1);transformation.at<float>(2,2) = Tc1c2(2,2);transformation.at<float>(2,3) = Tc1c2(2,3);
//    transformation.at<float>(3,0) = Tc1c2(3,0);transformation.at<float>(3,1) = Tc1c2(3,1);transformation.at<float>(3,2) = Tc1c2(3,2);transformation.at<float>(3,3) = Tc1c2(3,3);
//}

/**
 * This function will associate the ORB feature and Plane
 * TODO: what if a vision feature point is close to two plane?
 * TODO: what if a plane contains no ORB feature?
 */
//    bool Tracking::associateVisionLiDAR() {
//        //Step 1 : associate in 2D
//        int keyPtNum = mCurrentFrame.mvKeysUn.size();
//        int planeNum = mCurrentFrame.mvPlanes.size();
//        for (int kpIndex = 0; kpIndex < keyPtNum; kpIndex++) {
//            double minDis = 65535;
//            int foundPlaneIndex = -1;
//            int foundLiDARPtIndex = -1;
//            for (int plnIndex = 0; plnIndex < planeNum; plnIndex++) {
//                int planePtNum = mCurrentFrame.mvPlanes[plnIndex].pointList2D.size();//todo check if the 3D num match 2D num (sometime projection out of boundires)
//                for (int ldPtIndex = 0; ldPtIndex < planePtNum; ldPtIndex++) {
//                    double xdiff = mCurrentFrame.mvKeysUn[kpIndex].pt.x -
//                                   mCurrentFrame.mvPlanes[plnIndex].pointList2D[ldPtIndex].x;
//                    double ydiff = mCurrentFrame.mvKeysUn[kpIndex].pt.y -
//                                   mCurrentFrame.mvPlanes[plnIndex].pointList2D[ldPtIndex].y;
//                    double distance = sqrt(xdiff * xdiff + ydiff * ydiff);
//                    if (distance < 5 && distance < minDis) {
//                        minDis = distance;
//                        foundPlaneIndex = plnIndex;
//                        foundLiDARPtIndex = ldPtIndex;
//                    }
//                }
//            }
//            //Step 2: associate in 3D
//            if(mCurrentFrame.mvpMapPoints[kpIndex]!=NULL)//todo 是这么用的吗？
//            {
//                mCurrentFrame.mvPlanes[foundPlaneIndex].vpMapPointMatches.push_back(mCurrentFrame.mvpMapPoints[kpIndex]);
//                mCurrentFrame.mvPlanes[foundPlaneIndex].mindices.push_back(kpIndex);
//            }
//        }
//        //test distance to plane
//        for(int plnIndex = 0; plnIndex < planeNum; plnIndex++)
//        {
//            int mpNum = mCurrentFrame.mvPlanes[plnIndex].vpMapPointMatches.size();
//            double fenmu = sqrt(mCurrentFrame.mvPlanes[plnIndex].A * mCurrentFrame.mvPlanes[plnIndex].A
//                                + mCurrentFrame.mvPlanes[plnIndex].B * mCurrentFrame.mvPlanes[plnIndex].B
//                                + mCurrentFrame.mvPlanes[plnIndex].C * mCurrentFrame.mvPlanes[plnIndex].C);
//            for(int mpIndex = 0; mpIndex < mpNum; mpIndex++)
//            {
//                //todo establish plane world pose.
//                //mCurrentFrame.mvPlanes[plnIndex].A*mCurrentFrame.mvPlanes[plnIndex].vpMapPointMatches[mpIndex]->GetWorldPos())
//            }
//        }
//    }

void Tracking::SearchLocalPoints()
{
    //*Step1 遍历当前帧的mappoints，标记这些点不参与之后的搜索
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                //该点被当前帧观测到了，观测次数+1
                pMP->IncreaseVisible();
                //记录当前帧ID
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                //该点将来不用于投影，因为已经匹配过
                //指的是之前的关键帧tracking，速度模型tracking和重定位tracking
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    //*Step 2 将所有的局部地图点 投影到当前帧 
    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        //是否在视野中
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            //增加被观测的次数
            pMP->IncreaseVisible();
            //记录投影匹配的数目
            nToMatch++;
        }
    }

    //*Step 3 投影匹配
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
            //对局部地图点中新增的地图点进行投影匹配
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

/**
 *@brief 更新localmap
 包括：1.k1个关键帧,k2个临近关键帧和参考关键帧
 2. 由这些关键帧观测到的Mappoints
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    //红色地图点
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    //*Step1 清空局部mappoints
    mvpLocalMapPoints.clear();

    //*Step1 遍历局部关键帧 将这些帧的地图点插入进来
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                //类似参考关键帧的id记录，这里地图点也要记录自己被哪个帧参考了。
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 *@brief 更新局部关键帧
 遍历当前帧的mappoints，将观测到这些mappoints的关键帧和相邻关键帧以及父子关键帧，作为mvplocalkeyframes
 */
void Tracking::UpdateLocalKeyFrames()
{
    //*Step 1 遍历当前帧的地图点 记录所有能观测到这些地图点的关键帧
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                //观测到该点的的KF和该点在KF中的索引
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    //it->first = keyframe
                    //keyframeCounter本身没有初始化，如果it->first存在则++，不存在就增加一组键值对。
                    //所以同一个关键帧看到的地图点，都会累加到这个关键帧（it->first）的计数
                    //最后得到的就是各个关键帧的共视程度
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    //*Step 2 更新局部关键帧 mvplocalkeyframes，有三个策略添加
    mvpLocalKeyFrames.clear();
    //申请三倍内存
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    //*Step 2.1 遍历刚找到的有共视局部关键帧
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        //找到最高的共视度
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        //把共视局部关键帧 添加进 局部关键帧
        mvpLocalKeyFrames.push_back(it->first);
        //记录当前帧ID，表示这个关键帧已经是当前帧（mnId）的局部关键帧了
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    //*Step 2.2 遍历已经有的局部关键帧，将它们的共视度前10的帧插入 局部关键帧
    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        //*Step 2.2.1 共视关键帧的共视邻居
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                //? 邻居关键帧的mnTrackReferenceForFrame 会等于其他帧嘛？（只有null和当前帧两种？）
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    //*只用了最高的邻居？
                    break;
                }
            }
        }
        //*Step 2.2.2 共视关键帧的子关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        //*Step 2.2.3 共视关键帧的父关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                //! 这个break导致只有第一个共视关键帧的邻居和父子会加入到局部关键帧，后面的共视不关键帧不贡献这些帧
                break;
            }
        }

    }

    //*Step 3 当前帧的参考帧就是与自己共视程度最高的关键帧
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @brief 重定位函数
 * Step 1: 计算当前帧的Bow
 * Step 2: 找到与当前帧相似的候选关键帧
 * Step 3: Bow匹配
 * Step 4: EPnP
 * Step 5：PoseOptimization
 * Step 6: 如果inliner少，通过投影的方式对之前未匹配的点进行匹配，再优化求解
 * @return True | false
*/
bool Tracking::Relocalization()
{
    //*Step 1 计算当前帧BoW
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    //*Step 2 找候选关键帧
    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);
    //每个关键帧的结算器
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);
    //每个关键帧和当前帧的特征点匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);
    //放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;
    //*Step 3 遍历所有关键帧 通过BoW进行快速匹配 用匹配结果初始化PnP Solver
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    //*Step 4 开始找匹配关键帧了-Epnp找初始位姿
    //重定位一定要准确，所以很严格
    while (nCandidates > 0 && !bMatch)
    {
        //所有候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            //内点标记
            vector<bool> vbInliers;
            //内点数
            int nInliers;
            //RANSAC标记
            bool bNoMore;

            //*Step 4.1 EPnP计算姿态 迭代5次
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            //*Step 4.2 优化epnp结果，优化epnp结果，优化后inliner>50，结束。小于50，投影匹配增加匹配点，再优化。
            //*再优化后inliner>50，结束。30<inliner<50,再投影匹配再优化。
            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                //存内点
                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        //通过当前帧和参考帧的特征点匹配关系MapPointMatches[i][j]
                        //赋给当前帧的MapPoints成员
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }
                //优化
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                //内点太少，则跳过不优化，Tcw还是保留了
                if(nGood<10)
                    continue;

                //删除outlier
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                //*Step 4.3 如果inliner太少，则投影关键帧的地图点到当前帧，搜索新得匹配
                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        //3D-2D pnp BA 优化
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        ////*Step 4.4 BA后inliner不是很多，也不是太少
                        //再进行投影匹配，用更小的搜索框口和距离阈值
                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                //删除outlier
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        //匹配成功，当前帧已经有位姿，把当前帧ID记录到mnLastRelocFrameId。
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM

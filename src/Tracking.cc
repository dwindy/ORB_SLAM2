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

/////Added module
//#include "tic_toc.h"
//#include "lidarFactor.hpp"

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, //系统实例
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
    mnLastRelocFrameId(0),
    mbLiDARInit(false), //Added module, for lidar init
    mLiDARState(NOT_INITIALIZED)
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

    ///-----added module
    ///load Tcam_Lidar parameters
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

    //init T^world_cur
//    q_w_curr = Eigen::Quaterniond (1, 0, 0, 0);
//    t_w_curr = Eigen::Vector3d (0, 0, 0);
//    para_q[0] = 0; para_q[1] = 0; para_q[2] = 0; para_q[3] = 1; //{0, 0, 0, 1};
//    para_t[0] = 0; para_t[1] = 0; para_t[2] = 0; //{0, 0, 0};
//    q_last_curr = Eigen::Quaterniond (1, 0, 0, 0);
//    t_last_curr = Eigen::Vector3d (0, 0, 0);
////    q_last_curr = Eigen::Map<Eigen::Quaterniond> (para_q);
////    t_last_curr = Eigen::Map<Eigen::Vector3d> (para_t);
////    laserCloudCornerLast = new pcl::PointCloud<PointType>();
////    laserCloudSurfLast = new pcl::PointCloud<PointType>();
////    laserCloudFullRes = new pcl::PointCloud<PointType>();
    ///-----

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

//Tracking::Tracking(
//            System *pSys,                       //系统实例
//            ORBVocabulary* pVoc,                //BOW字典
//            FrameDrawer *pFrameDrawer,          //帧绘制器
//            MapDrawer *pMapDrawer,              //地图点绘制器
//            Map *pMap,                          //地图句柄
//            KeyFrameDatabase* pKFDB,            //关键帧产生的词袋数据库
//            const string &strSettingPath,       //配置文件路径
//            const int sensor):                  //传感器类型
//            mState(NO_IMAGES_YET),                              //当前系统还没有准备好
//            mSensor(sensor),
//            mbOnlyTracking(false),                              //处于SLAM模式
//            mbVO(false),                                        //当处于纯跟踪模式的时候，这个变量表示了当前跟踪状态的好坏
//            mpORBVocabulary(pVoc),
//            mpKeyFrameDB(pKFDB),
//            mpInitializer(static_cast<Initializer*>(NULL)),     //暂时给地图初始化器设置为空指针
//            mpSystem(pSys),
//            mpViewer(NULL),                                     //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
//            mpFrameDrawer(pFrameDrawer),
//            mpMapDrawer(pMapDrawer),
//            mpMap(pMap),
//            mnLastRelocFrameId(0)                               //恢复为0,没有进行这个过程的时候的默认值
//    {
//        // Load camera parameters from settings file
//        // Step 1 从配置文件中加载相机参数
//        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
//        float fx = fSettings["Camera.fx"];
//        float fy = fSettings["Camera.fy"];
//        float cx = fSettings["Camera.cx"];
//        float cy = fSettings["Camera.cy"];
//
//        //     |fx  0   cx|
//        // K = |0   fy  cy|
//        //     |0   0   1 |
//        //构造相机内参矩阵
//        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
//        K.at<float>(0,0) = fx;
//        K.at<float>(1,1) = fy;
//        K.at<float>(0,2) = cx;
//        K.at<float>(1,2) = cy;
//        K.copyTo(mK);
//
//        // 图像矫正系数
//        // [k1 k2 p1 p2 k3]
//        cv::Mat DistCoef(4,1,CV_32F);
//        DistCoef.at<float>(0) = fSettings["Camera.k1"];
//        DistCoef.at<float>(1) = fSettings["Camera.k2"];
//        DistCoef.at<float>(2) = fSettings["Camera.p1"];
//        DistCoef.at<float>(3) = fSettings["Camera.p2"];
//        const float k3 = fSettings["Camera.k3"];
//        //有些相机的畸变系数中会没有k3项
//        if(k3!=0)
//        {
//            DistCoef.resize(5);
//            DistCoef.at<float>(4) = k3;
//        }
//        DistCoef.copyTo(mDistCoef);
//
//        // 双目摄像头baseline * fx 50
//        mbf = fSettings["Camera.bf"];
//
//        float fps = fSettings["Camera.fps"];
//        if(fps==0)
//            fps=30;
//
//        // Max/Min Frames to insert keyframes and to check relocalisation
//        mMinFrames = 0;
//        mMaxFrames = fps;
//
//        //输出
//        cout << endl << "Camera Parameters: " << endl;
//        cout << "- fx: " << fx << endl;
//        cout << "- fy: " << fy << endl;
//        cout << "- cx: " << cx << endl;
//        cout << "- cy: " << cy << endl;
//        cout << "- k1: " << DistCoef.at<float>(0) << endl;
//        cout << "- k2: " << DistCoef.at<float>(1) << endl;
//        if(DistCoef.rows==5)
//            cout << "- k3: " << DistCoef.at<float>(4) << endl;
//        cout << "- p1: " << DistCoef.at<float>(2) << endl;
//        cout << "- p2: " << DistCoef.at<float>(3) << endl;
//        cout << "- fps: " << fps << endl;
//
//        // 1:RGB 0:BGR
//        int nRGB = fSettings["Camera.RGB"];
//        mbRGB = nRGB;
//
//        if(mbRGB)
//            cout << "- color order: RGB (ignored if grayscale)" << endl;
//        else
//            cout << "- color order: BGR (ignored if grayscale)" << endl;
//
//        // Load ORB parameters
//
//        // Step 2 加载ORB特征点有关的参数,并新建特征点提取器
//
//        // 每一帧提取的特征点数 1000
//        int nFeatures = fSettings["ORBextractor.nFeatures"];
//        // 图像建立金字塔时的变化尺度 1.2
//        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
//        // 尺度金字塔的层数 8
//        int nLevels = fSettings["ORBextractor.nLevels"];
//        // 提取fast特征点的默认阈值 20
//        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
//        // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
//        int fMinThFAST = fSettings["ORBextractor.minThFAST"];
//
//        // tracking过程都会用到mpORBextractorLeft作为特征点提取器
//        mpORBextractorLeft = new ORBextractor(
//                nFeatures,      //参数的含义还是看上面的注释吧
//                fScaleFactor,
//                nLevels,
//                fIniThFAST,
//                fMinThFAST);
//
//        // 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
//        if(sensor==System::STEREO)
//            mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
//
//        // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
//        if(sensor==System::MONOCULAR)
//            mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
//
//        cout << endl  << "ORB Extractor Parameters: " << endl;
//        cout << "- Number of Features: " << nFeatures << endl;
//        cout << "- Scale Levels: " << nLevels << endl;
//        cout << "- Scale Factor: " << fScaleFactor << endl;
//        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
//        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
//
//        if(sensor==System::STEREO || sensor==System::RGBD)
//        {
//            // 判断一个3D点远/近的阈值 mbf * 35 / fx
//            //ThDepth其实就是表示基线长度的多少倍
//            mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
//            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
//        }
//
//        if(sensor==System::RGBD)
//        {
//            // 深度相机disparity转化为depth时的因子
//            mDepthMapFactor = fSettings["DepthMapFactor"];
//            if(fabs(mDepthMapFactor)<1e-5)
//                mDepthMapFactor=1;
//            else
//                mDepthMapFactor = 1.0f/mDepthMapFactor;
//        }
//
//    }

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
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, const vector<vector<double>> &lasers)
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
        mCurrentFrame = Frame(mImGray, timestamp, lasers, mpIniORBextractor, mpORBVocabulary, mK, mTcamlid,
                              mDistCoef, mbf, mThDepth);
    else
        //mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
        ///added module
        mCurrentFrame = Frame(mImGray, timestamp, lasers, mpORBextractorLeft, mpORBVocabulary, mK, mTcamlid,
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

void Tracking::Track()
{

    //Track包含估计运动和跟踪局部地图两个部分
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    //上锁，保证地图不发生变化(localMapping and loopClosure)
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    //* Step 1 初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            //MonocularInitialization();
            MonoLiDARInitialization();
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
                    //系统刚初始化or没有速度/系统刚刚relocal 2帧之内
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
                    //优化特征点对应的3D点获得位姿,No Bow
                    bOK = TrackWithMotionModel();
                    //如果失败了，回去用参考关键帧
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else//mState==LOST
            {
                //计算当前帧的Bow找到共词关键帧(DetectRelocalizationCandidates
                //遍历上述候选，用Bow匹配找到足够多的匹配
                //inline足够, EPnP
                //BA PoseOptimization
                //如果inline少，通过投影的方式对之前未匹配的点进行匹配，再优化求解(SearchByProjection
                //匹配结果再BA PoseOptimization
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
                        //question 是不是要启用
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
                        //question? 必然True
                        //question? 是不是重复增加了观测次数？后面tracklocalmap函数包含这些动作
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
            if(bOK){
                //post-process following above three track functions: trackReference trackMotion, reLocalization.
                //search level 1 co-visible keyframes by currentFrame's mapPoints.
                //search level 2 co-visible, parent, child keyframes of level 1 keyframes
                //extract mappoint from above keyframes, as temporary local map, removing the out of view points.
                //project the rest to local frame, match by SearchByProjection
                //BA, PoseOptimization
                bOK = TrackLocalMap();
//                cout<<"mCurrentFrame.mnId Twc: "<<mCurrentFrame.mnId<<endl;
//                cout<<mCurrentFrame.mTcw.inv()<<endl;
                if (!bOK) {//check why lost
                    //write lost frame's key point
                    ofstream writer;
                    writer.open("lostFrameKeys.txt", ios::out);
                    for (int i = 0; i < mCurrentFrame.mvKeysUn.size(); i++) {
                        if (mCurrentFrame.mvpMapPoints[i])//if keypoint matched with mappoints
                        {
                            writer << i << " " << mCurrentFrame.mvKeysUn[i].pt.x
                                   << " " << mCurrentFrame.mvKeysUn[i].pt.y // u,v
                                   << " " << mCurrentFrame.mvORBAttributions[i].ID // ID
                                   << " " << mCurrentFrame.mvORBAttributions[i].depth // ID
                                   << " " << mCurrentFrame.mvORBAttributions[i].depthSource << " "; //depthSource
                            if (mCurrentFrame.mvORBAttributions[i].depthSource == 1) {//Plane
                                int tmpPlaneID = mCurrentFrame.mvORBAttributions[i].floorPlaneID;
                                writer << mCurrentFrame.mvORBAttributions[i].floorPlaneID
                                       << " " << mCurrentFrame.mvPlanes[tmpPlaneID].A
                                       << " " << mCurrentFrame.mvPlanes[tmpPlaneID].B
                                       << " " << mCurrentFrame.mvPlanes[tmpPlaneID].C
                                       << " " << mCurrentFrame.mvPlanes[tmpPlaneID].D << " ";
                            }
                            if (mCurrentFrame.mvORBAttributions[i].depthSource == 2) {
                                int tmpLineID = mCurrentFrame.mvORBAttributions[i].LSDlineID;
                                writer << mCurrentFrame.mvORBAttributions[i].LSDlineID
                                       << " " << mCurrentFrame.mvLines[tmpLineID].pt3dStart.x
                                       << " " << mCurrentFrame.mvLines[tmpLineID].pt3dStart.y
                                       << " " << mCurrentFrame.mvLines[tmpLineID].pt3dStart.z
                                       << " " << mCurrentFrame.mvLines[tmpLineID].pt3dEnd.x
                                       << " " << mCurrentFrame.mvLines[tmpLineID].pt3dEnd.y
                                       << " " << mCurrentFrame.mvLines[tmpLineID].pt3dEnd.z << " ";
                            }
                            if (mCurrentFrame.mvORBAttributions[i].depthSource == 3) {
                                int tmpLiDARID = mCurrentFrame.mvORBAttributions[i].LiDARPtID;
                                writer << mCurrentFrame.mvORBAttributions[i].LiDARPtID
                                       << " " << mCurrentFrame.mvORBAttributions[i].LiDARPt->pt3d.x
                                       << " " << mCurrentFrame.mvORBAttributions[i].LiDARPt->pt3d.y
                                       << " " << mCurrentFrame.mvORBAttributions[i].LiDARPt->pt3d.z << " ";
                            }
                            writer<<endl;
                        }

                    }
                    writer.close();
                    int pause = 1;
                }
            }
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
        //If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            //*Step 5 更新恒速运动模型
            // Update motion model
            if (!mLastFrame.mTcw.empty()) {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                //Velocity = Tcl = Tcw * Twl
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            } else            //否则速度为空
                mVelocity = cv::Mat();
            //更新显示的位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            //*Step 6 清除观测不到的地图点
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

            //*Step 7 清除恒速模型中 upDateLastFrame中临时添加的mapPoints（仅stereo和rgbd）
            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            //*Step 8 检测并插入关键帧，对于stereo和rgbd会产生新的地图点
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

        ///added module - Need to check if safe or not --- why I add this?
//        //*Step 4 更新显示线城的信息 比如图像 特征点 地图点
//        // Update drawer
//        mpFrameDrawer->Update(this);

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


/**
 * LiDAR Mode Init first, then Camera Mode Init
 * Because we need depth infor for Mono Feature Points
 */
 /*
void Tracking::LiDARInit()
{
    //Step 1 mpInitializer不存在时，创建一个实例
    if (!mpInitializer) {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100) {
            //把当前帧赋给初始化帧
            mInitialFrame = Frame(mCurrentFrame);
            //把当前帧赋给上一帧
            mLastFrame = Frame(mCurrentFrame);
            //记录上一帧的特征点 //not in use in LiDAR mode
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            //This will never work
            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            //初始化匹配结果 -1
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            //函数返回，下次进来执行else部分
            return;
        }
    } else {
        //Try to initital
        ///Step 2.1 get LiDAR feature from last frame, set up KdTree for it
        pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp->clear();
        surfPointsLessFlat->clear();
        int lastCornerNum = mInitialFrame.mLaserLessCorner_cam.size();
        int lastSurfNum = mInitialFrame.mLaserLessFlat_cam.size();
        cornerPointsLessSharp->resize(lastCornerNum);
        surfPointsLessFlat->resize(lastSurfNum);
        for (size_t i = 0; i < lastCornerNum; i++) {
            cornerPointsLessSharp->points[i].x = mInitialFrame.mLaserLessCorner_cam[i].pt3d.x;
            cornerPointsLessSharp->points[i].y = mInitialFrame.mLaserLessCorner_cam[i].pt3d.y;
            cornerPointsLessSharp->points[i].z = mInitialFrame.mLaserLessCorner_cam[i].pt3d.z;
            cornerPointsLessSharp->points[i].intensity = mInitialFrame.mLaserLessCorner_cam[i].intensity;
        }
        cout<<"cornerPointsLessSharp init "<<cornerPointsLessSharp->size()<<endl;
        for (size_t i = 0; i < lastSurfNum; i++) {
            surfPointsLessFlat->points[i].x = mInitialFrame.mLaserLessFlat_cam[i].pt3d.x;
            surfPointsLessFlat->points[i].y = mInitialFrame.mLaserLessFlat_cam[i].pt3d.y;
            surfPointsLessFlat->points[i].z = mInitialFrame.mLaserLessFlat_cam[i].pt3d.z;
            surfPointsLessFlat->points[i].intensity = mInitialFrame.mLaserLessFlat_cam[i].intensity;
        }
        cout<<"surfPointsLessFlat init "<<surfPointsLessFlat->size()<<endl;
//        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
//        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        kdtreeCornerLast.setInputCloud(cornerPointsLessSharp);
        kdtreeSurfLast.setInputCloud(surfPointsLessFlat);
        ///Step 2.2 get LiDAR feature from current frame
        pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
        cornerPointsSharp->clear();
        surfPointsFlat->clear();
        int curCornerNum = mCurrentFrame.mLaserCorner_cam.size();
        int curSurfNum = mCurrentFrame.mLaserFlat_cam.size();
        cornerPointsSharp->resize(curCornerNum);
        surfPointsFlat->resize(curSurfNum);
        for (size_t i = 0; i < lastCornerNum; i++) {
            cornerPointsSharp->points[i].x = mCurrentFrame.mLaserCorner_cam[i].pt3d.x;
            cornerPointsSharp->points[i].y = mCurrentFrame.mLaserCorner_cam[i].pt3d.y;
            cornerPointsSharp->points[i].z = mCurrentFrame.mLaserCorner_cam[i].pt3d.z;
            cornerPointsSharp->points[i].intensity = mCurrentFrame.mLaserCorner_cam[i].intensity;
        }
        cout<<"cornerPointsSharp "<<cornerPointsSharp->size()<<endl;
        for (size_t i = 0; i < lastSurfNum; i++) {
            surfPointsFlat->points[i].x = mCurrentFrame.mLaserFlat_cam[i].pt3d.x;
            surfPointsFlat->points[i].y = mCurrentFrame.mLaserFlat_cam[i].pt3d.y;
            surfPointsFlat->points[i].z = mCurrentFrame.mLaserFlat_cam[i].pt3d.z;
            surfPointsFlat->points[i].intensity = mCurrentFrame.mLaserFlat_cam[i].intensity;
        }
        cout<<"surfPointsFlat "<<surfPointsFlat->size()<<endl;

        //for test/
        pcl::PointXYZI pointSel;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        pointSel.x = 5;pointSel.y = 1;pointSel.z = -1.5;
        kdtreeCornerLast.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

        ///Step 3 time to init!
//        cereInit(cornerPointsLessSharp, surfPointsLessFlat,
//                 &kdtreeCornerLast, &kdtreeSurfLast,
//                 cornerPointsSharp, surfPointsFlat);
    }
}
*/

///**
// * Should be switch to G2O in the future
// */
/*
    bool Tracking::cereLiDAR(pcl::PointCloud<PointType>::Ptr laserCloudCornerLast,
                            pcl::PointCloud<PointType>::Ptr laserCloudSurfLast,
                            pcl::PointCloud<PointType>::Ptr cornerPointsSharp,
                            pcl::PointCloud<PointType>::Ptr surfPointsFlat) {

        //Step1 fill up the kdtree
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        kdtreeCornerLast.setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast.setInputCloud(laserCloudSurfLast);

        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();
        TicToc t_opt;
        // q_curr_last(x, y, z, w), t_curr_last
        double para_q[4] = {0, 0, 0, 1};
        double para_t[3] = {0, 0, 0};
//        cout<<"LiDAR para_q "<<para_q[0]<<" "<<para_q[1]<<" "<<para_q[2]<<" "<<para_q[3]<<endl;
//        cout<<"LiDAR para_t "<<para_t[0]<<" "<<para_t[1]<<" "<<para_t[2]<<" "<<para_t[3]<<endl;

        for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter) {
            int corner_correspondence = 0;
            int plane_correspondence = 0;

            //ceres::LossFunction *loss_function = NULL;
            // 定义一下ceres的核函数
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            // 由于旋转不满足一般意义的加法，因此这里使用ceres自带的local param
            ceres::LocalParameterization *q_parameterization =
                    new ceres::EigenQuaternionParameterization();
            ceres::Problem::Options problem_options;

            ceres::Problem problem(problem_options);
            // 待优化的变量是帧间位姿，平移和旋转，这里旋转使用四元数来表示
            problem.AddParameterBlock(para_q, 4, q_parameterization);
            problem.AddParameterBlock(para_t, 3);

            pcl::PointXYZI pointSel;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            TicToc t_data;
            // find correspondence for corner features
            // 寻找角点的约束
            for (int i = 0; i < cornerPointsSharpNum; ++i) //select one sharp point from current frame
            {
                // 运动补偿
                TransformToStart(&(cornerPointsSharp->points[i]),
                                 &pointSel); //project the sharp point to frame start time
                // cout<<pointSel<<endl;
                // 在上一帧所有角点构成的kdtree中寻找距离当前帧最近的一个点
                kdtreeCornerLast.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
//                cout<<"pointSel "<<pointSel.x<<" "<<pointSel.y<<" "<<pointSel.z<<" to "
//                        <<laserCloudCornerLast->points[pointSearchInd[0]].x<<" "<<laserCloudCornerLast->points[pointSearchInd[0]].y<<" "<<laserCloudCornerLast->points[pointSearchInd[0]].z
//                        <<" dis "<<pointSearchSqDis[0]<<endl;
                int closestPointInd = -1, minPointInd2 = -1;
                // 只有小于给定门限才认为是有效约束
                if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                    closestPointInd = pointSearchInd[0];    // 对应的最近距离的索引取出来
                    // 找到其所在线束id，线束信息藏在intensity的整数部分
                    int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                    double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                    // search in the direction of increasing scan line
                    // 寻找角点，在刚刚角点id上下分别继续寻找，目的是找到最近的角点，由于其按照线束进行排序，所以就是向上找
                    for (int j = closestPointInd + 1; j < (int) laserCloudCornerLast->points.size(); ++j) {
                        // if in the same scan line, continue
                        // 不找同一根线束的
                        if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                            continue;

                        // if not in nearby scans, end the loop
                        // 要求找到的线束距离当前线束不能太远
                        if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                            break;
                        // 计算和当前找到的角点之间的距离
                        double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                            (laserCloudCornerLast->points[j].x - pointSel.x) +
                                            (laserCloudCornerLast->points[j].y - pointSel.y) *
                                            (laserCloudCornerLast->points[j].y - pointSel.y) +
                                            (laserCloudCornerLast->points[j].z - pointSel.z) *
                                            (laserCloudCornerLast->points[j].z - pointSel.z);
                        // 寻找距离最小的角点及其索引
                        if (pointSqDis < minPointSqDis2) {
                            // find nearer point
                            // 记录其索引
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                    }

                    // search in the direction of decreasing scan line
                    // 同样另一个方向寻找对应角点
                    for (int j = closestPointInd - 1; j >= 0; --j) {
                        // if in the same scan line, continue
                        if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                            continue;

                        // if not in nearby scans, end the loop
                        if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                            break;

                        double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                            (laserCloudCornerLast->points[j].x - pointSel.x) +
                                            (laserCloudCornerLast->points[j].y - pointSel.y) *
                                            (laserCloudCornerLast->points[j].y - pointSel.y) +
                                            (laserCloudCornerLast->points[j].z - pointSel.z) *
                                            (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (pointSqDis < minPointSqDis2) {
                            // find nearer point
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                    }
                }
                // 如果这个角点是有效的角点
                if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                {
                    // 取出当前点和上一帧的两个角点
                    Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                               cornerPointsSharp->points[i].y,
                                               cornerPointsSharp->points[i].z);
                    Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                 laserCloudCornerLast->points[closestPointInd].y,
                                                 laserCloudCornerLast->points[closestPointInd].z);
                    Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                 laserCloudCornerLast->points[minPointInd2].y,
                                                 laserCloudCornerLast->points[minPointInd2].z);

                    double s;
                    if (DISTORTION)
                        s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) /
                            SCAN_PERIOD;
                    else
                        s = 1.0;
                    ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b,
                                                                                 s);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    corner_correspondence++;
                }
            }
            // find correspondence for plane features
            for (int i = 0; i < surfPointsFlatNum; ++i) {
                TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                // 先寻找上一帧距离这个面点最近的面点
                kdtreeSurfLast.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                // 距离必须小于给定阈值
                if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
                    // 取出找到的上一帧面点的索引
                    closestPointInd = pointSearchInd[0];

                    // get closest point's scan ID
                    // 取出最近的面点在上一帧的第几根scan上面
                    int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                    double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                    // 额外在寻找两个点，要求，一个点和最近点同一个scan，另一个是不同scan
                    // search in the direction of increasing scan line
                    // 按照增量方向寻找其他面点
                    for (int j = closestPointInd + 1; j < (int) laserCloudSurfLast->points.size(); ++j) {
                        // if not in nearby scans, end the loop
                        // 不能和当前找到的上一帧面点线束距离太远
                        if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                            break;
                        // 计算和当前帧该点距离
                        double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                            (laserCloudSurfLast->points[j].x - pointSel.x) +
                                            (laserCloudSurfLast->points[j].y - pointSel.y) *
                                            (laserCloudSurfLast->points[j].y - pointSel.y) +
                                            (laserCloudSurfLast->points[j].z - pointSel.z) *
                                            (laserCloudSurfLast->points[j].z - pointSel.z);

                        // if in the same or lower scan line
                        // 如果是同一根scan且距离最近
                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID &&
                            pointSqDis < minPointSqDis2) {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                            // if in the higher scan line
                            // 如果是其他线束点
                        else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID &&
                                 pointSqDis < minPointSqDis3) {
                            minPointSqDis3 = pointSqDis;
                            minPointInd3 = j;
                        }
                    }

                    // search in the direction of decreasing scan line
                    // 同样的方式，去按照降序方向寻找这两个点
                    for (int j = closestPointInd - 1; j >= 0; --j) {
                        // if not in nearby scans, end the loop
                        if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                            break;

                        double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                            (laserCloudSurfLast->points[j].x - pointSel.x) +
                                            (laserCloudSurfLast->points[j].y - pointSel.y) *
                                            (laserCloudSurfLast->points[j].y - pointSel.y) +
                                            (laserCloudSurfLast->points[j].z - pointSel.z) *
                                            (laserCloudSurfLast->points[j].z - pointSel.z);

                        // if in the same or higher scan line
                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID &&
                            pointSqDis < minPointSqDis2) {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        } else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID &&
                                   pointSqDis < minPointSqDis3) {
                            // find nearer point
                            minPointSqDis3 = pointSqDis;
                            minPointInd3 = j;
                        }
                    }
                    // 如果另外找到的两个点是有效点，就取出他们的3d坐标
                    if (minPointInd2 >= 0 && minPointInd3 >= 0) {

                        Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                   surfPointsFlat->points[i].y,
                                                   surfPointsFlat->points[i].z);
                        Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                     laserCloudSurfLast->points[closestPointInd].y,
                                                     laserCloudSurfLast->points[closestPointInd].z);
                        Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                     laserCloudSurfLast->points[minPointInd2].y,
                                                     laserCloudSurfLast->points[minPointInd2].z);
                        Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                     laserCloudSurfLast->points[minPointInd3].y,
                                                     laserCloudSurfLast->points[minPointInd3].z);

                        double s;
                        if (DISTORTION)
                            s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) /
                                SCAN_PERIOD;
                        else
                            s = 1.0;
                        // 构建点到面的约束
                        ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a,
                                                                                      last_point_b, last_point_c, s);
                        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        plane_correspondence++;
                    }
                }
            }
            //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
            printf("data association time %f ms \n", t_data.toc());
            cout<<"corner_corres "<<corner_correspondence<<" plane_corre "<<plane_correspondence<<endl;
            // 如果总的约束太少，就打印一下
            if ((corner_correspondence + plane_correspondence) < 10)
            {
                printf("less correspondence! *************************************************\n");
                return false;
            }
            // 调用ceres求解器求解
            TicToc t_solver;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            printf("solver time %f ms \n", t_solver.toc());
        }
        printf("optimization twice time %f \n", t_opt.toc());
        //map the cere parameter to member
        q_last_curr = Eigen::Map<Eigen::Quaterniond> (para_q);
        t_last_curr = Eigen::Map<Eigen::Vector3d> (para_t);
//        cout<<"LiDAR para_q "<<para_q[0]<<" "<<para_q[1]<<" "<<para_q[2]<<" "<<para_q[3]<<endl;
//        cout<<"LiDAR para_t "<<para_t[0]<<" "<<para_t[1]<<" "<<para_t[2]<<" "<<para_t[3]<<endl;
        // 这里的w_curr 实际上是 w_last
        t_w_curr = t_w_curr + q_w_curr * t_last_curr;
        q_w_curr = q_w_curr * q_last_curr;
        cout<<"LiDAR init Trans "<<endl<<t_w_curr<<endl;
        cout<<"LiDAr init Quate "<<endl;
        cout<<q_w_curr.toRotationMatrix()<<endl;
        return true;
    }
    */


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
 * @brief Monocular plus LiDAR initialization function
 */
    void Tracking::MonoLiDARInitialization(){
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
                ///Added lidar module
                mbLiDARInit = false;
                ///-----
                mLiDARState = NOT_INITIALIZED;
                return;
            }

            // Find correspondences
            // Step 3 当真帧和初始帧匹配
            ORBmatcher matcher(0.9,                                        //最佳和次佳特征点评分的比值阈值，越大则最佳次佳的区分度越小
                               true);                                      //检查特征点的方向
            int nmatches = matcher.SearchForInitialization(mInitialFrame,  //初始帧
                                                           mCurrentFrame,  //当前帧
                                                           mvbPrevMatched, //初始帧的特征点
                                                           mvIniMatches,   //保存匹配关系，size = IniFrame keypt number
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
//                //Print point with depth and point pair with second frame
//                //index larger than 1000 could not be matched by second frame? why?
//                for(int i=0;i<mInitialFrame.mvORBAttributions.size();i++){
//                    if(mInitialFrame.mvORBAttributions[i].depthSource>-1){
//                        cout<<i<<endl;
//                    }
//                }
//                cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
//                for(int i=0;i<mvIniMatches.size();i++){
//                    if(mvIniMatches[i]>-1){
//                        cout<<i<<endl;
//                    }
//                }
                /**
                * Store key points used for initialization
                */
                ofstream outer1, outer2,outer3;
                string fileName1 = to_string(mInitialFrame.mnId), fileName2 = to_string(mCurrentFrame.mnId);
                cout<<endl<<"init by frame "<<fileName1<<" "<<fileName2<<endl;
                outer1.open(fileName1 + ".txt",ios::out);
                outer2.open(fileName2 + ".txt",ios::out);
                outer3.open("mvIniP3D.txt",ios::out);
                for (int i = 0; i < mvIniMatches.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if(curIndex>=0){
                        //cout<<"mvIniMatches ["<<i<<"] = "<<mvIniMatches[i]<<endl;
                        outer1 << i << " " << mInitialFrame.mvKeys[i].pt.x << " " << mInitialFrame.mvKeys[i].pt.y << endl;
                        outer2 << curIndex << " " << mCurrentFrame.mvKeys[curIndex].pt.x << " " << mCurrentFrame.mvKeys[curIndex].pt.y << endl;
                        outer3 << mvIniP3D[i].x<<" "<< mvIniP3D[i].y<<" "<< mvIniP3D[i].z<<endl;
                    }
                }
                outer1.close();
                outer2.close();
                outer3.close();

                ///Added Module ------ After triangulated 3d map points. now LiDAR information get involves
                ///Step Fusion 1 --- if a point, in initial frame, successfully triangulated, and has depth from LiDAR
                //cout<<"mInitialFrame.mvORBAttributions.size() "<<mInitialFrame.mvORBAttributions.size()<<" mvIniMatches.size "<<mvIniMatches.size()<<endl;
                vector<vector<double>> ratiosXYZ;
                ///Step Fusion 1.1 store ratios over each axis
                int count1  = 0, count2 = 0;
                for(int i=0;i<mInitialFrame.mvORBAttributions.size();i++){
                    if(mInitialFrame.mvORBAttributions[i].depthSource>-1)
                        count1++;
                }
                cout<<"init frame with depth point num "<<count1<<endl;
                for(int i=0;i<mCurrentFrame.mvORBAttributions.size();i++){
                    if(mCurrentFrame.mvORBAttributions[i].depthSource>-1)
                        count2++;
                }

//                cv::Mat image0 = cv::imread("000000.png", CV_LOAD_IMAGE_UNCHANGED);
//                cv::Mat im_clone = image0.clone();//note this is image two
//                cv::cvtColor(image0, im_clone, CV_GRAY2BGR);
//                for (int i = 0; i < mvIniMatches.size(); i++) {
//                    if (mvIniMatches[i] > -1) {//this is a ORB feature
//                        cv::circle(im_clone, cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x,
//                                                       mInitialFrame.mvORBAttributions[i].keyPt.pt.y), 1,
//                                   cv::Scalar(0, 0, 255), -1);//red
//                        if (mInitialFrame.mvORBAttributions[i].depthSource == 1) {
//                            cv::rectangle(im_clone, cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x - 2,
//                                                              mInitialFrame.mvORBAttributions[i].keyPt.pt.y - 2),
//                                          cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x + 2,
//                                                    mInitialFrame.mvORBAttributions[i].keyPt.pt.y + 2),
//                                          cv::Scalar(0, 255, 0), -1);//green
//                        }
//                        if (mInitialFrame.mvORBAttributions[i].depthSource == 2) {
//                            int lineID = mInitialFrame.mvORBAttributions[i].LSDlineID;
//                            float sx = mInitialFrame.mvLines[lineID].LSD.startPointX;
//                            float sy = mInitialFrame.mvLines[lineID].LSD.startPointY;
//                            float ex = mInitialFrame.mvLines[lineID].LSD.endPointX;
//                            float ey = mInitialFrame.mvLines[lineID].LSD.endPointY;
//                            cv::line(im_clone, cv::Point(sx,sy),cv::Point(ex,ey),cv::Scalar(255,153,255),1);
//                            cv::rectangle(im_clone, cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x - 2,
//                                                              mInitialFrame.mvORBAttributions[i].keyPt.pt.y - 2),
//                                          cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x + 2,
//                                                    mInitialFrame.mvORBAttributions[i].keyPt.pt.y + 2),
//                                          cv::Scalar(255, 0, 127), -1);//purple
//                        }
//                        if (mInitialFrame.mvORBAttributions[i].depthSource == 3) {
//                            cv::rectangle(im_clone, cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x - 2,
//                                                              mInitialFrame.mvORBAttributions[i].keyPt.pt.y - 2),
//                                          cv::Point(mInitialFrame.mvORBAttributions[i].keyPt.pt.x + 2,
//                                                    mInitialFrame.mvORBAttributions[i].keyPt.pt.y + 2),
//                                          cv::Scalar(255, 255, 0), -1);//light blue
//                        }
//                    }
//                }
//                for (int i = 0; i < mInitialFrame.mLaserPt_cam.size(); i++) {
//                    cv::circle(im_clone, cv::Point(mInitialFrame.mLaserPt_cam[i].pt2d.x,
//                                                   mInitialFrame.mLaserPt_cam[i].pt2d.y), 1,
//                               cv::Scalar(0, 255, 255));
//                }
//                string windowName = "check fusion " + mInitialFrame.mnId;
//                cv::imshow(windowName, im_clone);
//                cv::waitKey(0);

                cout<<"feature number, ini "<<mInitialFrame.mvORBAttributions.size()<<" cur "<<mCurrentFrame.mvORBAttributions.size()<<endl;
                cout<<"init frame with depth point num "<<count1<<endl;
                cout<<"current frame with depth point num "<<count2<<endl;
                int matchNum = 0;
                for (int i = 0; i < mvIniMatches.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if (curIndex >= 0) {
//                        cout<<i<<" match to "<<curIndex<<" with estimated "<<mInitialFrame.mvORBAttributions[i].p3d_est.x<<" "
//                        <<mInitialFrame.mvORBAttributions[i].p3d_est.y<<" "
//                        <<mInitialFrame.mvORBAttributions[i].p3d_est.z<<" triangulated "
//                        <<mvIniP3D[i].x<<" "<<mvIniP3D[i].y<<" "<<mvIniP3D[i].z<<endl;
                        matchNum++;
                    }
                }
                int matchwithdepthnum = 0;
                int depthPlnCounter =0, depthLinCounter = 0, depthPtCounter = 0;
                for (int i = 0; i < mInitialFrame.mvORBAttributions.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if (curIndex >= 0) {
                        if (mInitialFrame.mvORBAttributions[i].depthSource > -1) {
                            if (mInitialFrame.mvORBAttributions[i].depthSource == 1)
                                depthPlnCounter++;
                            if (mInitialFrame.mvORBAttributions[i].depthSource == 2)
                                depthLinCounter++;
                            if (mInitialFrame.mvORBAttributions[i].depthSource == 3)
                                depthPtCounter++;
//                            cout << "ID " << mInitialFrame.mvORBAttributions[i].ID
//                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.x
//                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.y
//                                 << " " << mInitialFrame.mvORBAttributions[i].depthSource
//                                 << " " << mInitialFrame.mvORBAttributions[i].p3d_est
//                                 << " ORBSLAM2 : "
//                                 //<< mInitialFrame.mvORBAttributions[i].p3d_tri << endl;
//                                 << mvIniP3D[i].x << " " << mvIniP3D[i].y << " " << mvIniP3D[i].z << endl;
                            vector<double> thisRatio;
                            thisRatio.push_back(mInitialFrame.mvORBAttributions[i].p3d_est.x / mvIniP3D[i].x);
                            thisRatio.push_back(mInitialFrame.mvORBAttributions[i].p3d_est.y / mvIniP3D[i].y);
                            thisRatio.push_back(mInitialFrame.mvORBAttributions[i].p3d_est.z / mvIniP3D[i].z);
                            ratiosXYZ.push_back(thisRatio);
                            matchwithdepthnum++;
                        }
                    }
                }
                for (int i = 0; i < mInitialFrame.mvORBAttributions.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if (curIndex >= 0) {
                        if (mInitialFrame.mvORBAttributions[i].depthSource ==1) {
                            cout << "ID " << mInitialFrame.mvORBAttributions[i].ID
                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.x
                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.y
                                 << " " << mInitialFrame.mvORBAttributions[i].depthSource
                                 << " " << mInitialFrame.mvORBAttributions[i].p3d_est
                                 << " ORBSLAM2 : "
                                 //<< mInitialFrame.mvORBAttributions[i].p3d_tri << endl;
                                 << mvIniP3D[i].x << " " << mvIniP3D[i].y << " " << mvIniP3D[i].z << endl;
                        }
                    }
                }
                for (int i = 0; i < mInitialFrame.mvORBAttributions.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if (curIndex >= 0) {
                        if (mInitialFrame.mvORBAttributions[i].depthSource ==2) {
                            cout << "ID " << mInitialFrame.mvORBAttributions[i].ID
                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.x
                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.y
                                 << " " << mInitialFrame.mvORBAttributions[i].depthSource
                                 << " " << mInitialFrame.mvORBAttributions[i].p3d_est
                                 << " ORBSLAM2 : "
                                 //<< mInitialFrame.mvORBAttributions[i].p3d_tri << endl;
                                 << mvIniP3D[i].x << " " << mvIniP3D[i].y << " " << mvIniP3D[i].z << endl;
                        }
                    }
                }
                for (int i = 0; i < mInitialFrame.mvORBAttributions.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if (curIndex >= 0) {
                        if (mInitialFrame.mvORBAttributions[i].depthSource ==3) {
                            cout << "ID " << mInitialFrame.mvORBAttributions[i].ID
                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.x
                                 << " " << mInitialFrame.mvORBAttributions[i].keyPt.pt.y
                                 << " " << mInitialFrame.mvORBAttributions[i].depthSource
                                 << " " << mInitialFrame.mvORBAttributions[i].p3d_est
                                 << " ORBSLAM2 : "
                                 //<< mInitialFrame.mvORBAttributions[i].p3d_tri << endl;
                                 << mvIniP3D[i].x << " " << mvIniP3D[i].y << " " << mvIniP3D[i].z << endl;
                        }
                    }
                }
                cout<<"matched "<<matchNum<<" match with depth num "<<matchwithdepthnum<<endl;
                cout<<"plane point "<<depthPlnCounter<<" line point "<<depthLinCounter<<" point point "<<depthPtCounter<<endl;
                size_t ratiosNum = ratiosXYZ.size();
                ///Step Fusion 1.2 select ratios with small variance
                double meanXratio=0,meanYratio=0,meanZratio=0;
                int validRatioNum = 0;
                for(int i=0;i<ratiosNum;i++){
                    double variance = -1, mean = 0;
                    mean = (ratiosXYZ[i][0]+ratiosXYZ[i][1]+ratiosXYZ[i][2])/3;
                    variance = sqrt((ratiosXYZ[i][0] - mean) * (ratiosXYZ[i][0] - mean)
                                    + (ratiosXYZ[i][1] - mean) * (ratiosXYZ[i][1] - mean)
                                    + (ratiosXYZ[i][2] - mean) * (ratiosXYZ[i][2] - mean));
                    //cout<<"ratio "<<ratiosXYZ[i][0]<<" "<<ratiosXYZ[i][1]<<" "<<ratiosXYZ[i][2]<<" mean "<<mean<<" variance "<<variance<<endl;
                    if (variance < 0.005) {//remove candidate with large ratio variance
                        meanXratio += ratiosXYZ[i][0];
                        meanYratio += ratiosXYZ[i][1];
                        meanZratio += ratiosXYZ[i][2];
                        validRatioNum++;
                    }
                }
                //get the mean ratio over each axis
                meanXratio = meanXratio/validRatioNum;
                meanYratio = meanYratio/validRatioNum;
                meanZratio = meanZratio/validRatioNum;
                cout<<"meanXratio "<<meanXratio<<" meanYratio "<<meanYratio<<" meanZratio "<<meanZratio<<endl;
                ///Step Fusion 1.3 Apply ratio to All ini3dpoints
                for (int i = 0; i < mvIniP3D.size(); i++) {
                    int curIndex = mvIniMatches[i];
                    if(curIndex>-1){
                        if (mInitialFrame.mvORBAttributions[i].depthSource > -1) {
//                        mvIniP3D[i].x = mInitialFrame.mvORBAttributions[i].p3d_est.x;
//                        mvIniP3D[i].y = mInitialFrame.mvORBAttributions[i].p3d_est.y;
//                        mvIniP3D[i].z = mInitialFrame.mvORBAttributions[i].p3d_est.z;
                            mvIniP3D[i].x = mvIniP3D[i].x * meanXratio;
                            mvIniP3D[i].y = mvIniP3D[i].y * meanYratio;
                            mvIniP3D[i].z = mvIniP3D[i].z * meanZratio;
                        } else {
                            mvIniP3D[i].x = mvIniP3D[i].x * meanXratio;
                            mvIniP3D[i].y = mvIniP3D[i].y * meanYratio;
                            mvIniP3D[i].z = mvIniP3D[i].z * meanZratio;
                        }
                    }
                }

                ///Step Fusion 2. BA. in the create map function

                // Set Frame Poses
                // Step 7 初始化的第一帧作为世界坐标系，所以第一帧pose为I
                mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(Tcw.rowRange(0,3).col(3));
                mCurrentFrame.SetPose(Tcw);

                ///Added module-----
                //There should be a ratio between mono and lidar
                cout<<"init Tcw from Mono Triangulated but not scaled with lidar :"<<endl<<Tcw.inv()<<endl;
                //------------------

                // Step 8 创建初始化地图点MapPoints
                CreateInitialMapMonocular();
            }
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
            ///Added lidar module
            mbLiDARInit = false;
            mLiDARState = NOT_INITIALIZED;
            return;
        }

        // Find correspondences
        // Step 3 当真帧和初始帧匹配
        ORBmatcher matcher(0.9,                                        //最佳和次佳特征点评分的比值阈值，越大则最佳次佳的区分度越小
                           true);                                      //检查特征点的方向
        int nmatches = matcher.SearchForInitialization(mInitialFrame,  //初始帧
                                                       mCurrentFrame,  //当前帧
                                                       mvbPrevMatched, //初始帧的特征点
                                                       mvIniMatches,   //保存匹配关系，size = IniFrame keypt number
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

            /**
            * Store keypoints used for initialization
            */
            ofstream outer1, outer2,outer3;
            string fileName1 = to_string(mInitialFrame.mnId), fileName2 = to_string(mCurrentFrame.mnId);
            cout<<endl<<"init by frame "<<fileName1<<" "<<fileName2<<endl;
            outer1.open(fileName1 + ".txt",ios::out);
            outer2.open(fileName2 + ".txt",ios::out);
            outer3.open("mvIniP3D.txt",ios::out);
            for (int i = 0; i < mvIniMatches.size(); i++) {
                int curIndex = mvIniMatches[i];
                if(curIndex>=0){
                    //cout<<"mvIniMatches ["<<i<<"] = "<<mvIniMatches[i]<<endl;
                    outer1 << i << " " << mInitialFrame.mvKeys[i].pt.x << " " << mInitialFrame.mvKeys[i].pt.y << endl;
                    outer2 << curIndex << " " << mCurrentFrame.mvKeys[curIndex].pt.x << " " << mCurrentFrame.mvKeys[curIndex].pt.y << endl;
                    outer3 << mvIniP3D[i].x<<" "<< mvIniP3D[i].y<<" "<< mvIniP3D[i].z<<endl;
                }
            }
            outer1.close();
            outer2.close();
            outer3.close();

            // Set Frame Poses
            // Step 7 初始化的第一帧作为世界坐标系，所以第一帧pose为I
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            ///Added module-----
            //There should be a ratio between mono and lidar
            cout<<"Tcw "<<endl<<Tcw.inv()<<endl;
            //------------------

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
        //地图点增加关键帧和对应的索引69
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
    //cout<<"median depth "<<medianDepth<<endl;
    float invMedianDepth = 1.0f/medianDepth;
    //invMedianDepth = ratio;
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }


    ///Added Module --- with LiDAR, no need unify

    //* STEP 6 将两帧之间的变换 归一化到平均深度为1的尺度下
    // Scale initial baseline
//    cv::Mat Tc2w = pKFcur->GetPose();
//    cout<<"Tcw2 before medianDepth "<<endl<<Tc2w<<endl;
//    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
//    pKFcur->SetPose(Tc2w);
//    cout<<"Tcw2 after medianDepth "<<endl<<Tc2w<<endl;

    //* step 7 把3D点也归一化到1
    // Scale points
//    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
//    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
//    {
//        if(vpAllMapPoints[iMP])
//        {
//            MapPoint* pMP = vpAllMapPoints[iMP];
//            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
//        }
//    }

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
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    cout<<"after first Global BA Twc:"<<endl<<pKFcur->GetPose().inv()<<endl;
    int pause = 0;
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
    ///Added for check
//    int monoMatchNum = 0, planeMatchNum = 0, lineMatchNum = 0, pointMatchNum = 0;
//    for (int i = 0; i < vpMapPointMatches.size(); i++) {
//        if (mCurrentFrame.mvORBAttributions[i].depthSource == 1 && vpMapPointMatches[i]) {
//            planeMatchNum++;
//            cout<<"curF "<<i<<" depth source "<<mCurrentFrame.mvORBAttributions[i].depthSource<<" to keyF "<<vpMapPointMatches[i]<<endl;
//        }
//        if (mCurrentFrame.mvORBAttributions[i].depthSource == 2 && vpMapPointMatches[i]) {
//            lineMatchNum++;
//            cout<<"curF "<<i<<" depth source "<<mCurrentFrame.mvORBAttributions[i].depthSource<<" to keyF "<<vpMapPointMatches[i]<<endl;
//        }
//        if (mCurrentFrame.mvORBAttributions[i].depthSource == 3 && vpMapPointMatches[i]) {
//            pointMatchNum++;
//            cout<<"curF "<<i<<" depth source "<<mCurrentFrame.mvORBAttributions[i].depthSource<<" to keyF "<<vpMapPointMatches[i]<<endl;
//        }
//        if (mCurrentFrame.mvORBAttributions[i].depthSource == -1 && vpMapPointMatches[i]) {
//            monoMatchNum++;
//            cout<<"curF "<<i<<" depth source "<<mCurrentFrame.mvORBAttributions[i].depthSource<<" to keyF "<<vpMapPointMatches[i]<<endl;
//        }
//    }
//    cout<<"normal "<<monoMatchNum<<" planematch "<<planeMatchNum<<" lineMatch "<<lineMatchNum<<" pointMatch "<<pointMatchNum<<endl;

    if(nmatches<15)
        return false;

    //存储当前帧的特征点和3D地图点的匹配关系
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    //* Step3 将上一帧的位姿作为当前帧位姿的初始值 加速poseoptimization
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    //*Step4 优化重投影误差来（3D-2D）获得位姿
    //Optimizer::PoseOptimization(&mCurrentFrame);//Just optimize current Frame.
    Optimizer::FusionPoseOptimization(&mCurrentFrame);//Just optimize current Frame.

//    cout<<"mCurrentFrame : "<<mCurrentFrame.mnId<<" "<<endl<<mCurrentFrame.mTcw<<endl;
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

    cout<<"frame "<<mCurrentFrame.mnId<<" TrackReferenceKeyFrame() nmatchesMap "<<nmatchesMap<<endl;
    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    //*Step 1 计算上一帧在世界系下的位姿
    // Update pose according to reference keyframe
    //上一帧的参考帧
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    //Transfrom^lastframe_refence 从参考帧到上一帧的变换
    cv::Mat Tlr = mlRelativeFramePoses.back();//T from ref to lastFrame

    //Tlw = Tlr * Trw  
    //?为什么视频说是上一帧在世界系下的位姿？应该类似getPose返回Tcw,这里是设置Tlw。
    //是世界坐标在上一帧的位姿
    mLastFrame.SetPose(Tlr*pRef->GetPose());

    //单目或者上一帧是关键帧，程序结束，等于只设置了当前帧的位姿
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    //*Step 2 生成临时的地图点
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor

    //*Step 2.1 得到上一帧有深度的地图点,不一定是地图点
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            //pair(depth,point_index)
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

            //（临时）插入到上一帧地图中
            //?可以直接用i下标访问？所以之前是null?
            mLastFrame.mvpMapPoints[i]=pNewMP;

            //加入到临时地图点容器中，为了将来createnewkeyframe清空这些临时地图点用
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }
        //深度超过mTheDepth（35倍基线）并且 超过100个(因为depth sort过了，100个之后的点可能比较远了)
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
    ORBmatcher matcher(0.9,true);

    //* Step 1 更新上一帧的位姿
    //1. Update last frame pose according to its reference keyframe
    //2. Create "visual odometry" points for RGBD/Sterep mode, if in Localization Mode
    UpdateLastFrame();

    //当前帧位姿=速度*上一帧位姿 //Tcw = Tcl * Tlw;
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    //清空当前帧的地图点
    //?why？不应该本来就是空的？
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=30;//th=15;
    else
        th=7;
    //*Step 2 根据上一帧的特征点对应地图点进行投影匹配
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<5)//20
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<5){//20
        cout<<"TrackWithMotionModel() nmatches : "<<nmatches<<endl;
        return false;
    }


    //*Step 3 优化当前位姿
    // Optimize frame pose with all matches
//    Optimizer::PoseOptimization(&mCurrentFrame);
    Optimizer::FusionPoseOptimization(&mCurrentFrame);

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
    cout<<"frame ID "<<mCurrentFrame.mnId<<" TrackWithMotionModel() nmatchesMap "<<nmatchesMap<<endl;
    return nmatchesMap>=5; //10

}

/**
 * @brief Track LiDAR motion with 10hz fequencey
 * should be same function as LiDAR init
 * @return
 */
//bool Tracking::TrackLiDAR(){
//
//        ///Step 1 get LiDAR feature from last frame, set up KdTree for it
//        pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
//        pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
//        cornerPointsLessSharp->clear();
//        surfPointsLessFlat->clear();
//        int lastCornerNum = mLastFrame.mLaserLessCorner_cam.size();
//        int lastSurfNum = mLastFrame.mLaserLessFlat_cam.size();
//        cornerPointsLessSharp->resize(lastCornerNum);
//        surfPointsLessFlat->resize(lastSurfNum);
//        for (size_t i = 0; i < lastCornerNum; i++) {
//            cornerPointsLessSharp->points[i].x = mLastFrame.mLaserLessCorner_cam[i].pt3d.x;
//            cornerPointsLessSharp->points[i].y = mLastFrame.mLaserLessCorner_cam[i].pt3d.y;
//            cornerPointsLessSharp->points[i].z = mLastFrame.mLaserLessCorner_cam[i].pt3d.z;
//            cornerPointsLessSharp->points[i].intensity = mLastFrame.mLaserLessCorner_cam[i].intensity;
//        }
//        cout<<"last cornerPointsLessSharp "<<cornerPointsLessSharp->size()<<endl;
//        for (size_t i = 0; i < lastSurfNum; i++) {
//            surfPointsLessFlat->points[i].x = mLastFrame.mLaserLessFlat_cam[i].pt3d.x;
//            surfPointsLessFlat->points[i].y = mLastFrame.mLaserLessFlat_cam[i].pt3d.y;
//            surfPointsLessFlat->points[i].z = mLastFrame.mLaserLessFlat_cam[i].pt3d.z;
//            surfPointsLessFlat->points[i].intensity = mLastFrame.mLaserLessFlat_cam[i].intensity;
//        }
//        cout<<"last surfPointsLessFlat "<<surfPointsLessFlat->size()<<endl;
//
//        pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeCornerLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
//        pcl::KdTreeFLANN<pcl::PointXYZI> kdtreeSurfLast; //(new pcl::KdTreeFLANN<pcl::PointXYZI>());
//        kdtreeCornerLast.setInputCloud(cornerPointsLessSharp);
//        kdtreeSurfLast.setInputCloud(surfPointsLessFlat);
//        ///Step 2 get LiDAR feature from current frame
//        pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
//        pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
//        cornerPointsSharp->clear();
//        surfPointsFlat->clear();
//        int curCornerNum = mCurrentFrame.mLaserCorner_cam.size();
//        int curSurfNum = mCurrentFrame.mLaserFlat_cam.size();
//        cornerPointsSharp->resize(curCornerNum);
//        surfPointsFlat->resize(curSurfNum);
//        for (size_t i = 0; i < curCornerNum; i++) {
//            cornerPointsSharp->points[i].x = mCurrentFrame.mLaserCorner_cam[i].pt3d.x;
//            cornerPointsSharp->points[i].y = mCurrentFrame.mLaserCorner_cam[i].pt3d.y;
//            cornerPointsSharp->points[i].z = mCurrentFrame.mLaserCorner_cam[i].pt3d.z;
//            cornerPointsSharp->points[i].intensity = mCurrentFrame.mLaserCorner_cam[i].intensity;
//        }
//        cout<<"cur cornerPointsSharp "<<cornerPointsSharp->size()<<endl;
//        for (size_t i = 0; i < curSurfNum; i++) {
//            surfPointsFlat->points[i].x = mCurrentFrame.mLaserFlat_cam[i].pt3d.x;
//            surfPointsFlat->points[i].y = mCurrentFrame.mLaserFlat_cam[i].pt3d.y;
//            surfPointsFlat->points[i].z = mCurrentFrame.mLaserFlat_cam[i].pt3d.z;
//            surfPointsFlat->points[i].intensity = mCurrentFrame.mLaserFlat_cam[i].intensity;
//        }
//        cout<<"cur surfPointsFlat "<<surfPointsFlat->size()<<endl;
//
//        ///Step 3 time to solve it!
//        //KDtree put outside
//        cereLiDAR(cornerPointsLessSharp, surfPointsLessFlat,
//                  cornerPointsSharp, surfPointsFlat);
//}

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

    ///*Step 1 更新局部关键帧 mvpLocalKeyFrames 和 局部地图点 mvpLocalMapPoints
    UpdateLocalMap();

    ///*Step 2 匹配局部地图中 与 当前帧 匹配的Mappoints
    SearchLocalPoints();

    ///*Step 3 更新局部地图点后 更新位姿
    // Optimize Pose
    //Optimizer::PoseOptimization(&mCurrentFrame);
    Optimizer::FusionPoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    ///*Step 4 更新当前帧的mappoints被观测程度
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
            //question monocular do nothing?

        }
    }

    cout<<"track localmap mnMatchesInliers "<<mnMatchesInliers<<endl;

    ///*Step 5 根据匹配点数目和回环情况决定是否跟踪成功
    // Decide if the tracking was successful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<20) //50
        return false;

    if(mnMatchesInliers<5) //30
    {
        cout<<"TrackLocalMap() "<<mnMatchesInliers<<endl;
        return false;
    }

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
    ///*Step 1 VO模式不插入
    if(mbOnlyTracking)
        return false;
    ///*Step 1 局部地图线城被闭环检测使用 不插入
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    ///*Step 2 距离上次重定位比较近 或者 超出数量
    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    ///*Step 3 参考关键帧跟踪到的地图点数量
    //地图点有个最小观测次数的要求nminobs = 3
    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    ///*Step 4 查询局部地图管理器是否繁忙
    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    ///*Step 5 stero或者rgbd 统计可以添加的有校地图点总数和跟踪到的地图点数量
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

    ///*Step 6  决策是否插入关键帧
    ///*Step 6.1
    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    ////*Step 6.2 长时间没有插入
    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    ///*Step 6.3  满足插入关键帧的最小间隔 并且 localmapper空闲
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    ///*Step 6.4 双目RGBD情况下当前帧跟踪到的点比参考帧的0.25倍少 或者 满足 bNeedToInsertClose
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    ///*Step 6.5 跟踪到的点 < 参考帧的点 或者 needtoinsertclose TRUE 同时 跟跟踪到的点不是太少
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise, send a signal to interrupt BA
        //*Step mapping空闲
        ///*Step 6.6 localmappingidle free, return ture
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
                ///*Step 6.7 mpLocalMapper contains KeyframesInQueue < 3, return ture
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                //note no need for insert keyframe for monocular if localmapping thread is busy.
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

    ///*Step 1 将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    ///*Step 2 将当前关键帧设置为当前帧的参考关键帧
    mpReferenceKF = pKF; //参考关键帧成员，给后面的帧用
    mCurrentFrame.mpReferenceKF = pKF; //note this will be updated by updatelocalkeyframe function, a most co-visible frame will be set as reference KF for current frame.

    ///*Step 3 对于RGB/Stereo 生成新得地图点
    //跟 tracking::udpatelastframe()里面更新地图点类似
    if(mSensor!=System::MONOCULAR)
    {
        //更新几个位姿
        mCurrentFrame.UpdatePoseMatrices();

        ///*Step 3.1 当前帧有深度值的特征点.
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest. LESS?
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
            ///*Step 3.2 按照深度排序, closer is better
            sort(vDepthIdx.begin(),vDepthIdx.end());

            ///*Step 3.3 从中找出不是地图点的生成临时地图点
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
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);//NULL the old one, create new instead
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
                    //note. in trackWithMotion function -> updateLastFrame function -> just create new mappoint but do not add attributions above

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                ///*Step 3.3 停止新建地图点需要满足以下条件
                //点的深度超过阈值
                //npoints超过100个
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    ///*Step 4 插入关键帧
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
//
void Tracking::SearchLocalPoints()
{
    ///*Step1 遍历当前帧的mappoints，标记这些点不参与之后的搜索
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
        int matchNum = matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
        cout<<"frame ID "<<mCurrentFrame.mnId<<" TrackLocalMap()->SearchByProjection() match result "<<matchNum<<endl;
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

    //*Step1 遍历局部关键帧,将这些帧的地图点插入进来
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            //类似参考关键帧的id记录，这里地图点也要记录自己被哪个帧参考了,避免重复添加
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
    ///*Step 1 遍历当前帧的地图点 记录所有能观测到这些地图点的关键帧
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
                    //keyframeCounter本身没有初始化，如果it->first存在则value++，不存在就增加一组<keyframe,value>对。
                    //所以每次这个关键帧看到一个地图点，都会累加到这个关键帧（it->first）的计数(value)
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

    ///*Step 2.1 遍历刚找到的有共视局部关键帧
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
        //记录当前帧ID，表示这个pKF关键帧已经是当前帧currentFrame（mnId）的局部关键帧了
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;//待会就不重复添加了
    }

    ///*Step 2.2 遍历已经有的局部关键帧，将它们的共视度前10的帧插入 局部关键帧
    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        ///*Step 2.2.1 共视关键帧的共视邻居
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
                    //? 只用了最高的邻居？
                    break;
                }
            }
        }
        ///*Step 2.2.2 共视关键帧的子关键帧
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
                    //? only take the first child?
                    break;
                }
            }
        }
        ///*Step 2.2.3 共视关键帧的父关键帧
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
        //Note. mpReferenceKF = pKFmax. which is the most co-visible keyframe of current frame.
        //in Keyframe::UpdateConnections() confirms the keyframe's parent and child keyframes.
        //parent: the most co-visible keyframe of this keyframe.
        //child: the child of above parent's most co-visible keyframe.

    }

    ///*Step 3 当前帧的参考帧就是与自己共视程度最高的关键帧
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
    ///*Step 1 计算当前帧BoW
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    ///*Step 2 找候选关键帧
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
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300,
                                             4,//min number of one sample, 3 for Sim3 4 for EPnP,
                                             0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    ///*Step 4 开始找匹配关键帧了-Epnp找初始位姿
    //重定位一定要准确，所以很严格
    while (nCandidates > 0 && !bMatch)
    {
        //所有候选关键帧---the first qualify KeyFrame will break the for loop and stop while loop by bMatch
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

            ///*Step 4.1 EPnP计算姿态 迭代5次
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            ///*Step 4.2 优化epnp结果，优化后inliner>50，结束。小于50，投影匹配增加匹配点，再优化。
            ///*再优化后inliner>50，结束。30<inliner<50,再投影匹配再优化。
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
                //优化3D-2D PnP BA
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
                        //再进行投影匹配，用更小的搜索框口和距离阈值(因为刚pnp之后的位姿更准确了)
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

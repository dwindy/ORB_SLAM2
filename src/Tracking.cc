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
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

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
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                //Velocity = Tcl = Tcw * Twl
                mVelocity = mCurrentFrame.mTcw*LastTwc;
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

            //*Step 7 清楚恒速模型中 updatelastfrane中临时添加的mappoints（仅双目和rgbd）
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
            //把当前帧赋给初始化帧（为何要用Frame套上？类型转换？）
            mInitialFrame = Frame(mCurrentFrame);
            //用当前帧更新上一帧，用在哪？
            mLastFrame = Frame(mCurrentFrame);
            //记录上一帧的特征点
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            //不会工作的判断语句
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
        int nmatches = matcher.SearchForInitialization(mInitialFrame,  //参考帧
                                                       mCurrentFrame,  //当前帧
                                                       mvbPrevMatched, //参考帧的特征点
                                                       mvIniMatches,   //保存匹配关系，size = IniFrame point number
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

            // Step 8 创建初始化地图点MapPoints
            CreateInitialMapMonocular();
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
    //*Step 5 取场景的中值深度，用于尺度归一化
    //?似乎只是为了绘制的时候看起来大小合适？
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

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

    //当前帧位姿=速度*上一帧位姿
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

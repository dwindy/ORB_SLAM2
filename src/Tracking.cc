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
    Tracking::Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
                       KeyFrameDatabase* pKFDB, const string& strSettingPath, const int sensor):
        mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
        mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
        mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
    {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3,CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

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
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::MONOCULAR)
            mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD)
        {
            mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::YOLOZOE)
        {
            mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD)
        {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        ///adds on
        if (sensor == System::YOLOZOE)
        {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }
        kfGlobalID = 0;
    }

    void Tracking::SetLocalMapper(LocalMapping* pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing* pLoopClosing)
    {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer* pViewer)
    {
        mpViewer = pViewer;
    }


    cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft, const cv::Mat& imRectRight, const double& timestamp)
    {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    ///added module
    cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft, const cv::Mat& imRectRight, const double& timestamp,
                                      string classAddress)
    {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        // mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK,
        //                       mDistCoef, mbf, mThDepth);
        ///added module
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
                              mK,
                              mDistCoef, mbf, mThDepth, classAddress);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD, const double& timestamp)
    {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
            imDepth.convertTo(imDepth,CV_32F, mDepthMapFactor);

        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD, const double& timestamp,
                                    const string classAddress, const string objType)
    {
        mImGray.copyTo(mImGray_prev);
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
            imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

        //mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        ///adds on
        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth, classAddress, objType);


        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im, const double& timestamp)
    {
        mImGray = im;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                                  mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    ///adds on
    cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im, const double& timestamp, const string classAddress)
    {
        mImGray = im;
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        ///adds on --- load merged masks
        //        mImMask = mask;
        //        if (mImMask.channels() == 3) {
        //            if (mbRGB)
        //                cvtColor(mImMask, mImMask, CV_RGB2GRAY);
        //            else
        //                cvtColor(mImMask, mImMask, CV_BGR2GRAY);
        //        }

        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            //            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
            ///adds on
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth,
                                  classAddress);
        else
            //            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,mThDepth);
            ///adds on
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth,
                                  classAddress);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    ///Addes on
    //returns the center point of a mask
    cv::Point2f getCentroid(const cv::Mat& mask)
    {
        cv::Moments moments = cv::moments(mask, true);
        cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);
        return centroid;
    }

    void Tracking::Track()
    {
        if (mState == NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED)
        {
            if (mSensor == System::STEREO || mSensor == System::RGBD
                || mSensor == System::YOLOZOE) ///adds on
                StereoInitialization();
            else
                MonocularInitialization();

            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        }
        else
        {

            // System is initialized. Track Frame.
            bool bOK;

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking)
            {
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.

                if (mState == OK)
                {
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    CheckReplacedInLastFrame();

                    ///Adds on Kalman filter tracking
                    trackKalmanFilter();
                    ///-----------------------------------

                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                    else
                    {
                        bOK = TrackWithMotionModel();
                        if (!bOK)
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

                if (mState == LOST)
                {
                    bOK = Relocalization();
                }
                else
                {
                    if (!mbVO)
                    {
                        // In last frame we tracked enough MapPoints in the map

                        if (!mVelocity.empty())
                        {
                            bOK = TrackWithMotionModel();
                        }
                        else
                        {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                    else
                    {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint*> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;
                        if (!mVelocity.empty())
                        {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }
                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc)
                        {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if (mbVO)
                            {
                                for (int i = 0; i < mCurrentFrame.N; i++)
                                {
                                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        }
                        else if (bOKReloc)
                        {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }

            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking)
            {
                if (bOK)
                    bOK = TrackLocalMap();
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            mpFrameDrawer->Update(this);

            // If tracking were good, check if we insert a keyframe
            if (bOK)
            {
                // Update motion model
                if (!mLastFrame.mTcw.empty())
                {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                }
                else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++)
                {
                    MapPoint* pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                if (NeedNewKeyFrame())
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST)
            {
                if (mpMap->KeyFramesInMap() <= 5)
                {
                    cout << "Track lost soon after initialisation, reseting..." << endl;
                    mpSystem->Reset();
                    return;
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty())
        {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
            ///added
            Frame* newFrame = new Frame(mCurrentFrame);
            allFrames.push_back(newFrame); //is this released?
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }
    }


    void Tracking::StereoInitialization()
    {
        if (mCurrentFrame.N > 500)
        {
            // Set Frame pose to the origin
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            // Create KeyFrame
            KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and associate to KeyFrame
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;

                    ///Adds on - init the mappoint's dynamic rate and given label
                    pNewMP->AddObservationStatic(pKFini, i);
                    pNewMP->label = pKFini->mvKeysLabels[i];
                    ///------------------------------------------------------
                }
            }

            cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;

            ///Added Module----init Kalman filter
            for (int i = 0; i < mCurrentFrame.maskCentres.size(); i++)
            {
                KalmanFilter* newKF = new KalmanFilter(kfGlobalID, mCurrentFrame.mvClusterLabels[i],
                                                       mCurrentFrame.maskCentres[i].x, mCurrentFrame.maskCentres[i].y);
                allKFs.push_back(newKF);
                kfGlobalID++;
                mCurrentFrame.mvKalFilts[i] = newKF; //init frame, same order as mvClusterlabels
                //newKF->frame = &mCurrentFrame;
                //newKF->allFrames.push_back(&mCurrentFrame);//Error. becuase mCurrentFrame while be overwrite every time when new frame constructed.
            }
            int pause = 1;
            ///---------------------------------
        }
    }

    void pointMean(vector<cv::Point2f> point_prev, cv::Point2f& meanPoint)
    {
        // Initialize sum vectors
        cv::Point2f sumPoint(0, 0);
        // Calculate the sum of vectors
        for (int i = 0; i < point_prev.size(); ++i)
        {
            sumPoint += point_prev[i];
        }
        // Calculate the mean vector
        meanPoint = sumPoint / static_cast<float>(point_prev.size());
    }

    void vectorMean(vector<cv::Point2f> point_prev, vector<cv::Point2f> point_cur, cv::Point2f& meanVector)
    {
        // Ensure that both vectors have the same size
        if (point_prev.size() != point_cur.size())
        {
            // Handle the case where the vectors have different sizes
            std::cerr << "vectorMean() Error: Vectors must have the same size." << std::endl;
            return;
        }

        // Initialize sum vectors
        cv::Point2f sumVector(0, 0);

        // Calculate the sum of vectors
        for (int i = 0; i < point_prev.size(); ++i)
        {
            sumVector += point_cur[i] - point_prev[i];
        }

        // Calculate the mean vector
        meanVector = sumVector / static_cast<float>(point_prev.size());
    }

    void vectorVariance(vector<cv::Point2f> point_prev, vector<cv::Point2f> point_cur, cv::Point2f meanVector,
                        cv::Point2f& varianceVector)
    {
        // Ensure that both vectors have the same size
        if (point_prev.size() != point_cur.size())
        {
            // Handle the case where the vectors have different sizes
            std::cerr << "vectorVariance() Error: Vectors must have the same size." << std::endl;
            return;
        }
        // Initialize sum squared differences vector
        cv::Point2f sumSquaredDifferences(0, 0);
        // Calculate the sum of squared differences
        for (int i = 0; i < point_prev.size(); ++i)
        {
            cv::Point2f difference = point_cur[i] - point_prev[i] - meanVector;
            sumSquaredDifferences += cv::Point2f(difference.x * difference.x, difference.y * difference.y);
        }
        // Calculate the variance vector
        varianceVector = sumSquaredDifferences / static_cast<float>(point_prev.size());
    }

    void Tracking::CheckOpticalFlowDynamic(Frame& curF, KeyFrame& lastKeyF)
    {
        //cout << "Enter func CheckOpticalFlowDynamic(Frame& curF, KeyFrame& lastKeyF)" << endl;
        ///Step 0 prepare last frame key features
        int opflowFeatureNum = 2000;
        cv::goodFeaturesToTrack(lastKeyF.frameImGray, lastKeyF.mvOpFlwKyPt, opflowFeatureNum, 0.01, 3.0);
        //cout<<"lastKeyF.mvOpFlwKyPt size "<<lastKeyF.mvOpFlwKyPt.size()<<endl;
        int opfeatureNumbefore = lastKeyF.mvOpFlwKyPt.size();
        ///Step 1 calc the optical flow vector
        vector<uchar> status;
        vector<float> err;
        vector<cv::Point2f> matchedPt;
        cv::calcOpticalFlowPyrLK(lastKeyF.frameImGray, curF.frameImGray, lastKeyF.mvOpFlwKyPt, matchedPt,
                                 status, err, cv::Size(21, 21), 3);
        //cout<<"lastKeyF.mvOpFlwKyPt size "<<lastKeyF.mvOpFlwKyPt.size()<<endl;
        int opfeatureNumafter = lastKeyF.mvOpFlwKyPt.size();
        if (opfeatureNumbefore != opfeatureNumafter)
            cout << "opfeatureNumbefore!=opfeatureNumafter in CheckOpticalFlowDynamic() function" << endl;
        ///Step 2, allocate the current optical flow feature to each cluster to current frame
        curF.mvOpFlowKyClusters = vector<int>(matchedPt.size(), -1); //cluster index of each optical flow point
        curF.mvOpFlowKyLabels = vector<int>(matchedPt.size(), -1); //labels of each optical flow point
        std::vector<vector<cv::Point2f>> pointOfClusters_cur(curF.mvClusterLabels.size());
        std::vector<vector<cv::Point2f>> pointOfClusters_last(curF.mvClusterLabels.size());
        //process background
        vector<cv::Point2f> pointofBackground_cur;//for background
        vector<cv::Point2f> pointofBackground_last;
        for (int i = 0; i < matchedPt.size(); i++)
        {
            if (status[i])
            {
                bool inMask = false;
                for (int j = 0; j < curF.allMasks.size(); j++)
                {
                    //cout<<"matchedPt[i].y, matchedPt[i].x "<<matchedPt[i].y<<" "<<matchedPt[i].x<<endl;
                    int x = int(matchedPt[i].x), y = int(matchedPt[i].y);
                    if (int(curF.allMasks[j].at<uchar>(y, x)) > 0)
                    {
                        ///Step 2.1 for each key point, store the cluster index and object label
                        int ptLabel = curF.mvClusterLabels[j];
                        curF.mvOpFlowKyLabels[i] = ptLabel;
                        curF.mvOpFlowKyClusters[i] = j;
                        ///Step 2.2 for each cluster, store the optical features
                        pointOfClusters_cur[j].push_back(matchedPt[i]);
                        pointOfClusters_last[j].push_back(lastKeyF.mvOpFlwKyPt[i]);

                        inMask = true;
                    }
                }
                //process background
                if (!inMask)
                {
                    pointofBackground_cur.push_back(matchedPt[i]);
                    pointofBackground_last.push_back(lastKeyF.mvOpFlwKyPt[i]);
                }
            }
        }

        ///Step 3, for each cluster, calc the vector mean and vector variance

        //Store variance vectors of each cluster
        curF.mvOpFlow_VarianceVecs_ofClusters.assign(pointOfClusters_cur.size(), cv::Point2f(0, 0));
        //Store Variance Scalar of each cluster
        curF.mvOptflwVarianceofClusters.assign(pointOfClusters_cur.size(), 0.0f);
        //Store the distance of each pair of optflow of each fluster
        curF.mvOptflow_Distances_OfClusters.clear();
        curF.mvOptflow_Distances_OfClusters.resize(pointOfClusters_cur.size());
        curF.mvMean_OptFlowVector_ofClusters.clear();
        // vector<cv::Point2f> clusterMeans, clusterVariance, clusterMeanPoints; //no necessary need this anymore
        for (int i = 0; i < pointOfClusters_cur.size(); i++)
        {
            if (pointOfClusters_cur[i].empty())
                continue;

            //store the moving distance of each flow point in this cluster
            for (int j = 0; j < pointOfClusters_cur[i].size(); j++)
            {
                cv::Point2f diff = pointOfClusters_cur[i][j] - pointOfClusters_last[i][j];
                float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
                curF.mvOptflow_Distances_OfClusters[i].push_back(dist);
            }

            //mean error vector
            cv::Point2f meanVector(0, 0), varianceVector(0, 0);
            cv::Point2f meanPoint(0, 0);
            //in case not matching
            if (pointOfClusters_cur[i].size() == 0)
                continue;
            pointMean(pointOfClusters_cur[i], meanPoint);
            vectorMean(pointOfClusters_last[i], pointOfClusters_cur[i], meanVector);
            vectorVariance(pointOfClusters_last[i], pointOfClusters_cur[i], meanVector, varianceVector);
            curF.mvMean_OptFlowVector_ofClusters.push_back(meanVector);
            //clusterMeans.push_back(meanVector);
            // clusterVariance.push_back(varianceVector);
            // clusterMeanPoints.push_back(meanPoint);

            curF.mvOpFlow_VarianceVecs_ofClusters[i] = varianceVector;
            float variance = 0;
            //variance = sqrt(varianceVector.x * varianceVector.x + varianceVector.y * varianceVector.y);
            variance = sqrt(varianceVector.x + varianceVector.y);
            curF.mvOptflwVarianceofClusters[i] = variance;

            if (std::isnan(varianceVector.x) || std::isnan(varianceVector.y))
            {
                cout << "if(std::isnan(varianceVector.x)||std::isnan(varianceVector.y)) is TRUE !!!" << endl;
                int testpausepoint = 1;
            }
        }

        //process background
        curF.mvOptflow_Distances_OfBackground.clear();
        for (int i=0;i<pointofBackground_cur.size();i++)
        {
            //store the moving distance of each flow point in this cluster
                cv::Point2f diff = pointofBackground_cur[i] - pointofBackground_last[i];
                float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
                curF.mvOptflow_Distances_OfBackground.push_back(dist);
        }
        //mean error vector
        if (!pointofBackground_cur.empty())
        {
            cv::Point2f meanVector(0, 0), varianceVector(0, 0);
            cv::Point2f meanPoint(0, 0);
            pointMean(pointofBackground_cur, meanPoint);
            vectorMean(pointofBackground_last, pointofBackground_cur, meanVector);
            vectorVariance(pointofBackground_last, pointofBackground_cur, meanVector, varianceVector);
            curF.mMean_OptFlowVector_ofBackground = meanVector;
            curF.mOptFlow_VarianceVec_OfBackground = varianceVector;
            float variance = 0;
            //variance = sqrt(varianceVector.x * varianceVector.x + varianceVector.y * varianceVector.y);
            variance = sqrt(varianceVector.x + varianceVector.y);
            curF.mScalarOptFlowVarianceOfBackground = variance;
        }
        else
        {
            curF.mMean_OptFlowVector_ofBackground = cv::Point2f(0, 0);
            curF.mOptFlow_VarianceVec_OfBackground = cv::Point2f(0, 0);
            curF.mScalarOptFlowVarianceOfBackground = 0.0f;
        }


        //print check
        for (int i = 0; i < curF.mvOptflwVarianceofClusters.size(); i++)
            cout << curF.mvOptflwVarianceofClusters[i] << " ";
        std::cout << endl;

        // // --- Optical flow cluster visualization ---
        // cv::Mat vis;
        // if (!curF.frameImGray.empty())
        //     cv::cvtColor(curF.frameImGray, vis, cv::COLOR_GRAY2BGR);
        // else
        //     vis = cv::Mat(480,640,CV_8UC3, cv::Scalar(255,255,255));
        //
        // for (size_t i = 0; i < pointOfClusters_cur.size(); i++)
        // {
        //     const auto &curPts  = pointOfClusters_cur[i];
        //     const auto &lastPts = pointOfClusters_last[i];
        //     const int n = curPts.size();
        //     if (n < 2) continue;
        //
        //     // 累加计算均值
        //     cv::Point2f sumCur(0,0), sumLast(0,0), sumDir(0,0);
        //     for (int j = 0; j < n; j++)
        //     {
        //         cv::Point2f p0 = lastPts[j];
        //         cv::Point2f p1 = curPts[j];
        //         sumLast += p0;
        //         sumCur  += p1;
        //         sumDir  += (p1 - p0);
        //
        //         // 画光流向量
        //         cv::line(vis, p0, p1, cv::Scalar(0,255,0), 1);
        //         cv::circle(vis, p1, 2, cv::Scalar(0,0,255), -1);
        //     }
        //
        //     cv::Point2f meanLast = sumLast * (1.0f/n);
        //     cv::Point2f meanCur  = sumCur  * (1.0f/n);
        //     cv::Point2f meanDir  = sumDir  * (1.0f/n);
        //     float meanLen = std::sqrt(meanDir.x*meanDir.x + meanDir.y*meanDir.y);
        //
        //     // 计算方差
        //     float var = 0.0f;
        //     for (int j = 0; j < n; j++)
        //     {
        //         cv::Point2f err = ((curPts[j]-lastPts[j]) - meanDir);
        //         var += err.x*err.x + err.y*err.y;
        //     }
        //     var = std::sqrt(var / n);
        //
        //     // 绘制 cluster 均值中心、均值方向和方差线
        //     cv::Point2f center = meanCur;
        //     cv::circle(vis, center, 4, cv::Scalar(0,0,255), -1);
        //     cv::line(vis, center, center + meanDir, cv::Scalar(0,0,255), 2);
        //     if (meanLen > 1e-3)
        //     {
        //         cv::Point2f varEnd = center + (meanDir/meanLen) * var;
        //         cv::line(vis, center, varEnd, cv::Scalar(255,0,255), 2);
        //     }
        // }
        //
        // cv::imshow("OpticalFlowClusters", vis);
        // cv::waitKey(0);

        int pause = 1;
    }

    void Tracking::CheckOpticalFlowDynamic(Frame& curF, Frame& lastF)
    {
        //cout<<"Enter func CheckOpticalFlowDynamic(Frame& curF, Frame& lastF)"<<endl;
        ///Step 0 prepare last frame key features
        int opflowFeatureNum = 2000;
        cv::goodFeaturesToTrack(lastF.frameImGray, lastF.mvOpFlwKyPt, opflowFeatureNum, 0.01, 3.0);
        //cout<<"lastKeyF.mvOpFlwKyPt size "<<lastKeyF.mvOpFlwKyPt.size()<<endl;
        int opfeatureNumbefore = lastF.mvOpFlwKyPt.size();
        ///Step 1 calc the optical flow vector
        vector<uchar> status;
        vector<float> err;
        vector<cv::Point2f> matchedPt;
        cv::calcOpticalFlowPyrLK(lastF.frameImGray, curF.frameImGray, lastF.mvOpFlwKyPt, matchedPt,
                                 status, err, cv::Size(21, 21), 3);
        //cout<<"lastKeyF.mvOpFlwKyPt size "<<lastKeyF.mvOpFlwKyPt.size()<<endl;
        int opfeatureNumafter = lastF.mvOpFlwKyPt.size();
        if (opfeatureNumbefore != opfeatureNumafter)
            cout << "opfeatureNumbefore!=opfeatureNumafter in CheckOpticalFlowDynamic() function" << endl;
        ///Step 2, allocate the current optical flow feature to current clusters
        curF.mvOpFlowKyClusters = vector<int>(matchedPt.size(), -1);
        curF.mvOpFlowKyLabels = vector<int>(matchedPt.size(), -1);
        std::vector<vector<cv::Point2f>> pointOfClusters_cur(curF.mvClusterLabels.size());
        std::vector<vector<cv::Point2f>> pointOfClusters_last(curF.mvClusterLabels.size());
        //cout << "CheckOpticalFlowDynamic function matchedPt " << matchedPt.size() << endl;
        //process background
        vector<cv::Point2f> pointofBackground_cur;//for background
        vector<cv::Point2f> pointofBackground_last;
        for (int i = 0; i < matchedPt.size(); i++)
        {
            if (status[i])
            {
                bool inMask = false;
                for (int j = 0; j < curF.allMasks.size(); j++)
                {
                    int x = int(matchedPt[i].x), y = int(matchedPt[i].y);
                    if (int(curF.allMasks[j].at<uchar>(y, x)) > 0)
                    {
                        ///Step 2.1 for each key point, store the cluster index and object label
                        int ptLabel = curF.mvClusterLabels[j];
                        curF.mvOpFlowKyLabels[i] = ptLabel;
                        curF.mvOpFlowKyClusters[i] = j;
                        ///Step 2.2 for each cluster, store the optical features
                        pointOfClusters_cur[j].push_back(matchedPt[i]);
                        pointOfClusters_last[j].push_back(lastF.mvOpFlwKyPt[i]);
                        inMask = true;
                    }
                }
                //process background
                if (!inMask)
                {
                    pointofBackground_cur.push_back(matchedPt[i]);
                    pointofBackground_last.push_back(lastF.mvOpFlwKyPt[i]);
                }
            }
        }
        ///Step 3, for each cluster, calc the vector mean and vector variance
        curF.mvOpFlow_VarianceVecs_ofClusters.assign(pointOfClusters_cur.size(), cv::Point2f(0, 0));
        curF.mvOptflwVarianceofClusters.assign(pointOfClusters_cur.size(), 0.0f);
        vector<cv::Point2f> clusterMeans, clusterVariance, clusterMeanPoints;
        curF.mvOptflow_Distances_OfClusters.clear();
        curF.mvOptflow_Distances_OfClusters.resize(pointOfClusters_cur.size());
        curF.mvMean_OptFlowVector_ofClusters.clear();
        for (int i = 0; i < pointOfClusters_cur.size(); i++)
        {
            if (pointOfClusters_cur[i].empty())
                continue;

            //store the moving distance of each flow point in this cluster
            for (int j = 0; j < pointOfClusters_cur[i].size(); j++)
            {
                cv::Point2f diff = pointOfClusters_cur[i][j] - pointOfClusters_last[i][j];
                float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
                curF.mvOptflow_Distances_OfClusters[i].push_back(dist);
            }

            //mean error vector
            cv::Point2f meanVector(0, 0), varianceVector(0, 0);
            cv::Point2f meanPoint(0, 0);

            if (pointOfClusters_cur[i].size() == 0) //in case not matching
                continue;

            pointMean(pointOfClusters_cur[i], meanPoint);
            vectorMean(pointOfClusters_last[i], pointOfClusters_cur[i], meanVector);
            vectorVariance(pointOfClusters_last[i], pointOfClusters_cur[i], meanVector, varianceVector);
            curF.mvMean_OptFlowVector_ofClusters.push_back(meanVector);
            //clusterMeans.push_back(meanVector);
            clusterVariance.push_back(varianceVector);
            clusterMeanPoints.push_back(meanPoint);
            curF.mvOpFlow_VarianceVecs_ofClusters[i] = varianceVector;
            //error --- need update threshold or experiment results?
            //float variance = sqrt(varianceVector.x * varianceVector.x + varianceVector.y * varianceVector.y);
            float variance = sqrt(varianceVector.x + varianceVector.y );
            curF.mvOptflwVarianceofClusters[i] = variance;
            if (std::isnan(varianceVector.x) || std::isnan(varianceVector.y))
                cout << "if(std::isnan(varianceVector.x)||std::isnan(varianceVector.y)) is TRUE!!!" << endl;


            ///-----draw
            // static int flowFrameCount = 0; // 静态变量，只在前20帧保存
            //
            // if (flowFrameCount < 20)
            // {
            //     cv::Mat vis;
            //     if (!curF.frameImGray.empty())
            //         cv::cvtColor(curF.frameImGray, vis, cv::COLOR_GRAY2BGR);
            //     else
            //         vis = cv::Mat(480,640,CV_8UC3, cv::Scalar(255,255,255));
            //
            //     for (size_t i = 0; i < matchedPt.size(); i++)
            //     {
            //         if (status[i])
            //         {
            //             cv::Point2f p0 = lastF.mvOpFlwKyPt[i];
            //             cv::Point2f p1 = matchedPt[i];
            //             cv::line(vis, p0, p1, cv::Scalar(0,255,255), 1);      // 黄线
            //             cv::circle(vis, p1, 2, cv::Scalar(0,255,255), -1);    // 黄点
            //         }
            //     }
            //
            //     // 保存图像
            //     std::string filename = "save_figs/opt_" + std::to_string(flowFrameCount) + ".png";
            //     cv::imwrite(filename, vis);
            //
            //     flowFrameCount++;
            // }

            int pause = 1;
        }

        //process background
        curF.mvOptflow_Distances_OfBackground.clear();
        for (int i=0;i<pointofBackground_cur.size();i++)
        {
            //store the moving distance of each flow point in this cluster
            cv::Point2f diff = pointofBackground_cur[i] - pointofBackground_last[i];
            float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
            curF.mvOptflow_Distances_OfBackground.push_back(dist);
        }
        //mean error vector
        if (!pointofBackground_cur.empty())
        {
            cv::Point2f meanVector(0, 0), varianceVector(0, 0);
            cv::Point2f meanPoint(0, 0);
            pointMean(pointofBackground_cur, meanPoint);
            vectorMean(pointofBackground_last, pointofBackground_cur, meanVector);
            vectorVariance(pointofBackground_last, pointofBackground_cur, meanVector, varianceVector);
            curF.mMean_OptFlowVector_ofBackground = meanVector;
            curF.mOptFlow_VarianceVec_OfBackground = varianceVector;
            //float variance = 0;
            //variance = sqrt(varianceVector.x * varianceVector.x + varianceVector.y * varianceVector.y);
            float variance = sqrt(varianceVector.x + varianceVector.y );
            curF.mScalarOptFlowVarianceOfBackground = variance;
        }
        else
        {
            curF.mMean_OptFlowVector_ofBackground = cv::Point2f(0, 0);
            curF.mOptFlow_VarianceVec_OfBackground = cv::Point2f(0, 0);
            curF.mScalarOptFlowVarianceOfBackground = 0.0f;
        }

    }

    /**
     * This function check the variance of re-projection error
     * mark the large variance cluster(label) with dynamic flag
     * @param initF
     * @param curF
     * @param initMatches
     */
    void Tracking::FilterOutDynamicMatchs(Frame& initF, Frame& curF, vector<int>& initMatches)
    {
        for (int i = 0; i < initMatches.size(); i++)
        {
            int labeli = initF.mvKeysLabels[i];
            int indexj = initMatches[i];
            if (indexj != -1)
            {
                int labelj = curF.mvKeysLabels[indexj];
                if (labeli != labelj)
                {
                    cout << "init Frame pt index " << i << " label " << labeli << " dismatch curF " << indexj
                        << " label " << labelj << " | initMatches marked -1 " << endl;
                    initMatches[i] = -1;
                }
                if (labeli == labelj)
                {
                    //cout << "init Frame pt index " << i << " label " << labeli << " match curF " << indexj << " label "
                    //     << labelj << endl;
                    if (labelj == 0 && labeli == 0)
                    {
                        initMatches[i] = -1;
                    }
                }
            }
        }
        int pause = 0;
    }

    float calculateMean(const std::vector<float>& data)
    {
        float sum = 0.0;
        for (float value : data)
        {
            sum += value;
        }
        return sum / static_cast<float>(data.size());
    }

    double calculateStandardDeviation(const std::vector<float>& data, float mean)
    {
        float sumSquaredDiff = 0.0;
        for (float value : data)
        {
            float diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return std::sqrt(sumSquaredDiff / static_cast<float>(data.size()));
    }

    /*
     * Calc the variance mean, And CV(coefficient variance)
     */
    void varianceAndMean(vector<float> data, float& mean, float& std, float& CV)
    {
        if (data.size() > 1)
        {
            mean = calculateMean(data);
            std = calculateStandardDeviation(data, mean);
            CV = std / mean;
        }
        else
        {
            mean = data[0];
            std = 0;
            CV = 0;
        }
    }

    void filterByVariance(vector<float> data, float varianceThres, vector<bool>& dynamicFlag)
    {
        // Identify outliers
        for (int i = 0; i < data.size(); i++)
        {
            if (data[i] > varianceThres)
            {
                dynamicFlag[i] = true;
            }
        }
    }

    /* collecting keypoint and mappoint for each cluster
     * check the distance for each pair of keypoint and mappoing
     * calc mean error, variance of each cluster
     */
    void Tracking::CheckReprojectDynamic(Frame& F)
    {
        F.mvRePjtMeanofClusters.clear();
        F.mvRePjtVarianceofClusters.clear();
        F.mvRePjtErrorsOfClusters.clear();

        ///Step 1. Store data : labels of each cluster, keypoint index in each mask, mappoint in each masks
        int keyNumCur = F.mvKeysUn.size();
        //todo change to keypoint*?
        vector<vector<int>> ptIndexOfEachCluster(F.mvClusterLabels.size()); //keypoint indexes of each cluster
        vector<vector<MapPoint*>> mapPointsOfEachMasks(F.mvClusterLabels.size()); //mapPoint of each cluster
        vector<int> ptIndexOfBackGround; //store background points for variance unifying
        vector<MapPoint*> mapPointsOfBackground;
        for (int i = 0; i < keyNumCur; i++)
        {
            //Store keypoint to each cluster
            if (F.mvpMapPoints[i])
            {
                int clusteri = F.mvKeysClusters[i]; //todo to check
                if (clusteri > -1)
                {
                    ptIndexOfEachCluster[clusteri].push_back(i);
                    mapPointsOfEachMasks[clusteri].push_back(F.mvpMapPoints[i]);
                }
                //int label = F.mvClusterLabels[clusteri]; ///error. i forget comment the first line why orb has no this error while boon met?
                //if(label==0)
                //    cout<<"pt "<<F_cur.mvKeysUn[i].pt<<" is person point "<<endl;
                else
                {
                    ptIndexOfBackGround.push_back(i);
                    mapPointsOfBackground.push_back(F.mvpMapPoints[i]);
                }
            }
        }
        //cout<<"-------------"<<endl;
        //check cluster and label
        //        for(int i=0;i<ptIndexOfEachCluster.size();i++)
        //            cout<<"label "<<F_cur.mvClusterLabels[i]<<" ptIndexOfEachCluster i "<<i<<" "<<ptIndexOfEachCluster[i].size()<<endl;

        ///Step 2. PROJECT FROM MAP TO LOCAL image for pixel error

        // cv::Mat imgClone;
        // cv::cvtColor(F.frameImGray, imgClone, cv::COLOR_GRAY2BGR);

        vector<vector<float>> errorOfClusters(F.mvClusterLabels.size());
        if (mVelocity.empty())
            mVelocity = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat Tcw;
        if (mLastFrame.mTcw.empty()) //in relocate case?
            Tcw = mVelocity * cv::Mat::eye(4, 4, CV_32F);
        else
            Tcw = mVelocity * mLastFrame.mTcw;

        //process points in clusters
        for (int i = 0; i < ptIndexOfEachCluster.size(); i++)
        {
            int numofClu = ptIndexOfEachCluster[i].size();
            for (int j = 0; j < numofClu; j++)
            {
                MapPoint* mp = mapPointsOfEachMasks[i][j];
                cv::Mat pose = mp->GetWorldPos();
                cv::Mat pose_h(4, 1,CV_32F);
                pose.copyTo(pose_h.rowRange(0, 3));
                pose_h.at<float>(3, 0) = 1.0f;
                // cout<<"pose_h"<<endl<<pose_h<<endl;
                // cout<<"Tcw"<<endl<<Tcw<<endl;
                cv::Mat pose_cur = Tcw * pose_h;
                cv::Point2f predict = F.project2image(pose_cur);
                cv::Point2f observe = F.mvKeysUn[ptIndexOfEachCluster[i][j]].pt;
                float distance = sqrt((predict.x - observe.x)
                    * (predict.x - observe.x)
                    + (predict.y - observe.y)
                    * (predict.y - observe.y));
                // cout<<"label "<<allLabels[i]<<" distance "<<distance<<endl;
                errorOfClusters[i].push_back(distance);
                ///show it
                // cv::circle(imgClone, predict, 2, cv::Scalar(0, 255, 255), -1);
                // cv::circle(imgClone, observe, 2, cv::Scalar(0, 255, 0), -1);
                // cv::line(imgClone, predict, observe, cv::Scalar(0, 255, 255), 2);
            }
            // cout<<"cluster indenx "<<i<<" size "<<numofClu<<" label "<<F_cur.mvClusterLabels[i]<<endl;
        }
        // cv::imshow("Reprojection Errors", imgClone);
        // cv::waitKey(0);
        F.mvRePjtErrorsOfClusters = errorOfClusters;

        //process the point in backgrounds
        vector<float> errorOfBackground(ptIndexOfBackGround.size());
        for (int i = 0; i < ptIndexOfBackGround.size(); i++)
        {
            MapPoint* mp = mapPointsOfBackground[i];
            cv::Mat pose = mp->GetWorldPos();
            cv::Mat pose_h(4, 1,CV_32F);
            pose.copyTo(pose_h.rowRange(0, 3));
            pose_h.at<float>(3, 0) = 1.0f;
            cv::Mat pose_cur = Tcw * pose_h;
            cv::Point2f predict = F.project2image(pose_cur);
            cv::Point2f observe = F.mvKeysUn[ptIndexOfBackGround[i]].pt;
            float distance = sqrt((predict.x - observe.x)
                * (predict.x - observe.x)
                + (predict.y - observe.y)
                * (predict.y - observe.y));
            // cout<<"label "<<allLabels[i]<<" distance "<<distance<<endl;
            errorOfBackground[i] = distance;
        }
        F.mvRePjtErrorsofBackground = errorOfBackground;


        ///// --- 可视化重投影误差和统计信息 ---

        // // 准备图像
        // if (!F.frameImGray.empty())
        //     cv::cvtColor(F.frameImGray, imgClone, cv::COLOR_GRAY2BGR);
        // else
        //     imgClone = cv::Mat(480, 640,CV_8UC3, cv::Scalar(255, 255, 255));
        //
        // for (int i = 0; i < ptIndexOfEachCluster.size(); i++)
        // {
        //     const int num = ptIndexOfEachCluster[i].size();
        //     if (num < 2) continue;
        //
        //     cv::Point2f sumPredict(0, 0), sumObserve(0, 0), sumDir(0, 0);
        //
        //     // 先绘制每对点并累计均值
        //     for (int j = 0; j < num; j++)
        //     {
        //         MapPoint* mp = mapPointsOfEachMasks[i][j];
        //         cv::Mat pose = mp->GetWorldPos();
        //         cv::Mat pose_h(4, 1,CV_32F);
        //         pose.copyTo(pose_h.rowRange(0, 3));
        //         pose_h.at<float>(3, 0) = 1.0f;
        //         cv::Mat pose_cur = Tcw * pose_h;
        //
        //         cv::Point2f predict = F.project2image(pose_cur);
        //         cv::Point2f observe = F.mvKeysUn[ptIndexOfEachCluster[i][j]].pt;
        //
        //         // 累加
        //         sumPredict += predict;
        //         sumObserve += observe;
        //         sumDir += (observe - predict);
        //
        //         // 画连线和点
        //         cv::circle(imgClone, predict, 2, cv::Scalar(0, 255, 255), -1);
        //         cv::circle(imgClone, observe, 2, cv::Scalar(0, 255, 0), -1);
        //         cv::line(imgClone, predict, observe, cv::Scalar(0, 255, 255), 1);
        //     }
        //
        //     // 计算均值
        //     cv::Point2f meanPredict = sumPredict * (1.0f / num);
        //     cv::Point2f meanObserve = sumObserve * (1.0f / num);
        //     cv::Point2f meanDir = sumDir * (1.0f / num);
        //     float meanErr = std::sqrt(meanDir.x * meanDir.x + meanDir.y * meanDir.y);
        //
        //     // 计算方差（标准差）
        //     float var = 0.0f;
        //     for (int j = 0; j < num; j++)
        //     {
        //         MapPoint* mp = mapPointsOfEachMasks[i][j];
        //         cv::Mat pose = mp->GetWorldPos();
        //         cv::Mat pose_h(4, 1,CV_32F);
        //         pose.copyTo(pose_h.rowRange(0, 3));
        //         pose_h.at<float>(3, 0) = 1.0f;
        //         cv::Mat pose_cur = Tcw * pose_h;
        //
        //         cv::Point2f predict = F.project2image(pose_cur);
        //         cv::Point2f observe = F.mvKeysUn[ptIndexOfEachCluster[i][j]].pt;
        //         cv::Point2f errVec = (observe - predict) - meanDir;
        //         var += (errVec.x * errVec.x + errVec.y * errVec.y);
        //     }
        //     var = std::sqrt(var / num);
        //
        //     // 绘制均值中心和方向
        //     cv::Point2f center = meanObserve; // 也可以用 (meanPredict+meanObserve)*0.5
        //     cv::Scalar col(0, 0, 255);
        //
        //     cv::circle(imgClone, center, 4, col, -1); // 中心
        //     cv::line(imgClone, center, center + meanDir, col, 2); // 均值方向
        //     if (meanErr > 1e-3)
        //     {
        //         cv::Point2f varEnd = center + (meanDir / meanErr) * var;
        //         cv::line(imgClone, center, varEnd, cv::Scalar(255, 0, 255), 2); // 方差线
        //     }
        // }
        //
        // // 显示
        // cv::imshow("Reprojection Clusters", imgClone);
        // cv::waitKey(0);
        // ///-----------------------------------------------


        ///Step 3. Calc the coefficient variance for each cluster

        for (int i = 0; i < errorOfClusters.size(); i++)
        {
            float mean = 0, variance = 0, CV = 0; //coefficient variance
            if (errorOfClusters[i].size() > 0)
            {
                varianceAndMean(errorOfClusters[i], mean, variance, CV);
            }
            F.mvRePjtMeanofClusters.push_back(mean);
            F.mvRePjtVarianceofClusters.push_back(variance);
            //int label = F.mvClusterLabels[i];
            //cout << "label " << label << " size " << errorOfClusters[i].size() << " mean " << mean << " variance " << variance << " CV " << CV << endl;
        }

        //process background
        float mean = 0, variance = 0, CV = 0;
        if (errorOfClusters.size() > 0)
            varianceAndMean(errorOfBackground, mean, variance, CV);
        F.mvRePjtMeanofBackground = mean;
        F.mvRePjtVarianceofBackground = variance;

        ///mark dynamics --- leaves this to determinedynamic()
        //        for (int i = 0; i < dynamicFlags.size(); i++) {
        //            if (dynamicFlags[i]) {
        //                for (int j = 0; j < ptIndexOfEachCluster[i].size(); j++) {
        //                    //darw cluster i
        //                    //cv::circle(imgClone, F_cur.mvKeysUn[ptIndexOfEachCluster[i][j]].pt, 2, cv::Scalar(0, 0, 255), -1);
        //                    //mark dynamic
        //                    int ptIndex = ptIndexOfEachCluster[i][j];
        //                    F.mvKeysDynamic[ptIndex] = true;
        //                }
        //            }
        //        }
        ////        cv::imshow("test", imgClone);
        ////        cv::waitKey(0);
        ////        cout<<"---------------------------------------------------"<<endl;
        int pause = 0;
    }

    // void normalizeData(std::vector<float> data, std::vector<float>& normalizedData)
    // {
    //     //find the max
    //     float maxData = *max_element(data.begin(), data.end());
    //     //normalize each element
    //     for (float value : data)
    //     {
    //         normalizedData.push_back(value / maxData);
    //     }
    // }
    //
    // void normalizeVector(std::vector<cv::Point2f> data, std::vector<cv::Point2f>& normalizedData)
    // {
    //     //find the max
    //     float maxX = 0.0, maxY = 0.0;
    //     //normalize each element
    //     for (cv::Point2f element : data)
    //     {
    //         if (element.x > maxX)
    //             maxX = element.x;
    //         if (element.y > maxY)
    //             maxY = element.y;
    //     }
    //     for (cv::Point2f element : data)
    //     {
    //         normalizedData.push_back(cv::Point2f(element.x / maxX, element.y / maxY));
    //     }
    // }

    /*
     * combine the reproject variance and opticalflow vector variance of each cluster
     * and determining a dynamic flag for each keypoints
     */
    void Tracking::DetermineDynamics(Frame& curF)
    {
        if (mpSystem->mMetricType == "variance")
        {
            //Step 1 Normalize variance among clusters and background -> for confidence score method
            vector<float> allReVar = curF.mvRePjtVarianceofClusters;
            allReVar.push_back(curF.mvRePjtVarianceofBackground);//push background variance into all variances
            vector<float> allOptVar = curF.mvOptflwVarianceofClusters;
            allOptVar.push_back(curF.mScalarOptFlowVarianceOfBackground);//push background variance into all variances
            //wow new func
            float minRe = *min_element(allReVar.begin(), allReVar.end());
            float maxRe = *max_element(allReVar.begin(), allReVar.end());
            float minOpt = *min_element(allOptVar.begin(), allOptVar.end());
            float maxOpt = *max_element(allOptVar.begin(), allOptVar.end());
            //define a func
            auto normalize = [](float v, float vmin, float vmax) {
                return (v - vmin) / (vmax - vmin + 1e-6f);
            };
            //Apply normalization
            curF.mvNormalized_RePjtVarianceofClusters.clear();
            curF.mvNormalized_RePjtVarianceofClusters.resize(curF.mvRePjtVarianceofClusters.size());
            ///variance to confidence score
            curF.C_o.assign(curF.mvOptflwVarianceofClusters.size(), 1.0f);
            curF.C_r.assign(curF.mvRePjtVarianceofClusters.size(), 1.0f);
            curF.C_fused.assign(curF.mvOptflwVarianceofClusters.size(), 1.0f);


            float beta = 5.0f;
            for (int i = 0; i < curF.mvRePjtVarianceofClusters.size(); i++)
            {
                curF.mvNormalized_RePjtVarianceofClusters[i] = normalize(
                    curF.mvRePjtVarianceofClusters[i], minRe, maxRe);
                curF.C_r[i] = std::exp(-beta * curF.mvNormalized_RePjtVarianceofClusters[i]);
            }
            curF.mvNormalized_RePjtVarianceofBackground = normalize(curF.mvRePjtVarianceofBackground, minRe, maxRe);

            curF.mvNormalized_OptflwVarianceofClusters.clear();
            curF.mvNormalized_OptflwVarianceofClusters.resize(curF.mvOptflwVarianceofClusters.size());
            for (int i = 0; i < curF.mvOptflwVarianceofClusters.size(); i++)
            {
                curF.mvNormalized_OptflwVarianceofClusters[i] = normalize(
                    curF.mvOptflwVarianceofClusters[i], minOpt, maxOpt);
                curF.C_o[i] = std::exp(-beta * curF.mvNormalized_OptflwVarianceofClusters[i]);
            }
            curF.mNormalized_ScalarOptFlowVarianceOfBackground = normalize(
                curF.mScalarOptFlowVarianceOfBackground, minOpt, maxOpt);
            /// 保存 Confidence Score 到文件
            // {
            //     std::ofstream fout("confidence_log.txt", std::ios::app);
            //     if (!fout.is_open())
            //     {
            //         std::cerr << "Cannot open confidence_log.txt" << std::endl;
            //     }
            //     else
            //     {
            //         float beta = 5.0f; // 控制置信度衰减速度，可调
            //         for (int i = 0; i < curF.mvClusterLabels.size(); i++)
            //         {
            //             float c_r = std::exp(-beta * curF.mvNormalized_RePjtVarianceofClusters[i]);
            //             float c_o = std::exp(-beta * curF.mvNormalized_OptflwVarianceofClusters[i]);
            //             float c_fused = c_r * c_o;
            //
            //             fout << curF.mnId << " "                       // 帧号
            //                  << curF.mvClusterLabels[i] << " "         // object label
            //                  << c_r << " "                             // reprojection confidence
            //                  << c_o << " "                             // optical flow confidence
            //                  << c_fused << " "                         // fused confidence
            //                  << 0                                      // type 0: 普通 object
            //                  << "\n";
            //         }
            //
            //         // 背景
            //         float c_r_bg = std::exp(-beta * curF.mvNormalized_RePjtVarianceofBackground);
            //         float c_o_bg = std::exp(-beta * curF.mNormalized_ScalarOptFlowVarianceOfBackground);
            //         float c_fused_bg = c_r_bg * c_o_bg;
            //
            //         fout << curF.mnId << " "
            //              << -1 << " "                        // 背景 label
            //              << c_r_bg << " "
            //              << c_o_bg << " "
            //              << c_fused_bg << " "
            //              << 1                                // type 1: 背景
            //              << "\n";
            //
            //         fout.close();
            //     }
            // }


            ///Store variance
            // {
            //     // 打开文件（追加模式）
            //     std::ofstream fout("variance_log.txt", std::ios::app);
            //     if (!fout.is_open())
            //     {
            //         std::cerr << "Cannot open variance_log.txt" << std::endl;
            //     }
            //     else
            //     {
            //         // 保存每个 cluster 的 variance
            //         for (int i = 0; i < curF.mvRePjtVarianceofClusters.size(); i++)
            //         {
            //             fout << curF.mnId << " "
            //                  << curF.mvClusterLabels[i] << " "
            //                  << curF.mvRePjtVarianceofClusters[i] << " "
            //                  << curF.mvOptflwVarianceofClusters[i] << " "
            //                  << curF.mvNormalized_RePjtVarianceofClusters[i] << " "
            //                  << curF.mvNormalized_OptflwVarianceofClusters[i] << " "
            //                  << 0  // 0 表示普通 cluster
            //                  << "\n";
            //         }
            //
            //         // 保存背景的 variance，索引用 -1
            //         fout << curF.mnId << " "
            //              << -1 << " "
            //              << curF.mvRePjtVarianceofBackground << " "
            //              << curF.mScalarOptFlowVarianceOfBackground << " "
            //              << curF.mvNormalized_RePjtVarianceofBackground << " "
            //              << curF.mNormalized_ScalarOptFlowVarianceOfBackground << " "
            //              << 1  // 1 表示背景
            //              << "\n";
            //
            //         fout.close();
            //     }
            // }

            //note --- statistic from walking_halfsphere
            //reprojection error variance
            //mask label -1 mean 5.158 25% 3.573 50% 4.926 75% 6.414
            //mask label 0 mean 7.046 25% 1.743 50% 8.181 75% 10.944
            //optical flow error variance
            //mask label -1 mean 4.830 25% 2.167 50% 3.672 75% 6.405
            //mask label 0 mean 5.121 25% 1.248 50% 2.496 75% 5.401
            //reprojection error variance normalized
            //mask label -1 mean 0.481 25% 0.301 50% 0.412 75% 0.613
            //masl label 0 mean 0.591 25% 0.148 50% 0.695 75% 1.000
            //optical flow error variance normalized
            //mask label -1 mean 0.655 25% 0.397 50% 0.662 75% 1.000
            //mask label 0 mean 0.495 25% 0.170 50% 0.420 75% 0.907

            //step 2 assign dynamic flags
            vector<bool> clusterDynamicFlags(curF.mvClusterLabels.size(), false);
            for (int i = 0; i < clusterDynamicFlags.size(); i++)
            {
                // if (curF.mvRePjtVarianceofClusters[i] > 5.0 && curF.mvOptflwVarianceofClusters[i] > 1.0
                //     || curF.mvRePjtVarianceofClusters[i] > 10.0
                //     || curF.mvOptflwVarianceofClusters[i] > 10.0)
                // {
                //
                //     //Note First way. Old direct way.
                //     //TUM and most Bonn
                //     /// testing other value for checking the dynamic determing ability and capture screenshot for special case
                //     //if (curF.mvRePjtVarianceofClusters[i] > 5.0 && varianceVector > 1.0 ) { //Bonn move obstruct
                //     //if (curF.mvRePjtVarianceofClusters[i] > 5.0 && varianceVector > 2.0 ) { //Bonn rgbd_bonn_synchronous
                //     //if(curF.mvClusterLabels[i]==0){
                //     //if(curF.mvRePjtVarianceofClusters[i] > 5.0){
                //     // if(varianceVector > 1.0){
                //     clusterDynamicFlags[i] = true;
                //     curF.mvClusterDynamic[i] = true;
                // }
                // if (!(curF.mvOptflwVarianceofClusters[i] >= 0 && curF.mvRePjtVarianceofClusters[i] >= 0))
                //     int pause = 1;

                // //Note Second way. normalized variance
                // if (curF.mvNormalized_RePjtVarianceofClusters[i]>0.591
                //     &&curF.mvNormalized_OptflwVarianceofClusters[i]>0.495
                //     ||curF.mvNormalized_RePjtVarianceofClusters[i]>0.9
                //     ||curF.mvNormalized_OptflwVarianceofClusters[i]>0.9)
                // {
                //     clusterDynamicFlags[i] = true;
                //     curF.mvClusterDynamic[i] = true;
                // }

                curF.C_fused[i] = curF.C_r[i] * curF.C_o[i];

                //Note Third way. confidence score
                if (curF.C_r[i] < 0.0521 //exo(0.591 * -5)
                    && curF.C_o[i] < 0.0842 //exp(0.495 * -5)
                    || curF.C_r[i] < 0.0111 //exp(0.9*-5)
                    || curF.C_o[i] < 0.0111) //exp(0.9*-5)
                {
                    clusterDynamicFlags[i] = true;
                    curF.mvClusterDynamic[i] = true;
                }
            }
            //cout<<curF.mnId<<"curF.mvOptflwVarianceofClusters "<<curF.mvOptflwVarianceofClusters.size()<<" curF.mvRePjtVarianceofClusters "<<curF.mvRePjtVarianceofClusters.size()<<endl;
            //apply to each keypoint
            for (int i = 0; i < curF.mvKeysClusters.size(); i++)
            {
                int clusterIndex = curF.mvKeysClusters[i];
                if (clusterIndex > -1 && clusterDynamicFlags[clusterIndex])
                {
                    curF.mvKeysDynamic[i] = true;
                }
            }
        }
        else if (mpSystem->mMetricType == "euclidean")
        {
            vector<bool> clusterDynamicFlags(curF.mvClusterLabels.size(), false);

            /// 打开文件，以追加模式，每次写一行
            // std::ofstream fout("OpticalFlowRropejctErrorLog.txt", std::ios::app);
            // if (!fout.is_open())
            // {
            //     std::cerr << "Cannot open OpticalFlowErrorLog.txt" << std::endl;
            // }

            // // 每帧开始写一行: FrameId:
            // fout << curF.mnId << " ";
            for (int i = 0; i < clusterDynamicFlags.size(); i++)
            {
                // // 输出 label 和对应误差，用冒号分隔，用逗号隔开不同cluster
                // fout << curF.mvClusterLabels[i] << ":" << curF.mvRePjtMeanofClusters[i];
                // // cout << curF.mvClusterLabels[i]<< " : " << curF.mvRePjtMeanofClusters[i]<<" ";
                // if (i != clusterDynamicFlags.size() - 1)
                //     fout << ","; // cluster之间加逗号
                // else
                //     fout << std::endl; // 每帧结束换行

                //if (curF.mvRePjtMeanofClusters[i] > 7.04)//based on walking halfsphere error report
                float distance = sqrt(
                    curF.mvMean_OptFlowVector_ofClusters[i].x * curF.mvMean_OptFlowVector_ofClusters[i].x
                    + curF.mvMean_OptFlowVector_ofClusters[i].y * curF.mvMean_OptFlowVector_ofClusters[i].y);
                //if (distance > 7.192087)//based on walking halfsphere error report
                if (distance > 7.192087 && curF.mvRePjtMeanofClusters[i] > 7.04)
                {
                    clusterDynamicFlags[i] = true;
                    curF.mvClusterDynamic[i] = true;
                }
                if (!(curF.mvOptflwVarianceofClusters[i] >= 0 && curF.mvRePjtVarianceofClusters[i] >= 0))
                    int pause = 1;
            }
            // // cout << endl;
            // fout.close();
            //cout<<curF.mnId<<"curF.mvOptflwVarianceofClusters "<<curF.mvOptflwVarianceofClusters.size()<<" curF.mvRePjtVarianceofClusters "<<curF.mvRePjtVarianceofClusters.size()<<endl;
            //apply to each keypoint
            for (int i = 0; i < curF.mvKeysClusters.size(); i++)
            {
                int clusterIndex = curF.mvKeysClusters[i];
                if (clusterIndex > -1 && clusterDynamicFlags[clusterIndex])
                {
                    curF.mvKeysDynamic[i] = true;
                }
            }
        }

        ///add a Bayesian Belief Update block
        if (mpSystem->mMetricType == "variance")
        {
            for (int i = 0; i < curF.mvClusterLabels.size(); i++)
            {
                //Step 1 get current confidence
                float cr = curF.C_r[i];
                float co = curF.C_o[i];
                float c_static = curF.C_fused[i]; //confidence is the probability of static
                float c_dynamic = 1.0f - c_static;
                //Step 2 find corresponding kalman filter instance
                KalmanFilter* kf = curF.mvKalFilts[i];
                if (!kf) continue; //should always has a kalman filter instance

                //Step 3 get last confience
                float bel_stc_prev = kf->confidence_Pre;
                float bel_dyn_prev = 1.0f - bel_stc_prev;

                //Step 4 normalized posterior probability
                float bel_static_un = c_static * bel_stc_prev;
                float bel_dynamic_un = c_dynamic * bel_dyn_prev;
                float S = bel_static_un + bel_dynamic_un;

                //Step 5 normalization
                float bel_static = bel_static_un / (S + 1e-6f);
                //kf->static_prob[] = bel_static; //store a copy of bel_static in the kalman instance?
                //Step 6 apply update
                curF.C_fused[i] = bel_static;
                kf->confidence_Pre = bel_static;
            }
        }
        ///----------------------------
        // // 转成彩色图
        // // --- 可视化每个 cluster 的 ORB variance 向量 + mask ---
        // // --- 准备基础底图（灰度转彩色） ---
        // cv::Mat base;
        // if (!curF.frameImGray.empty())
        //     cv::cvtColor(curF.frameImGray, base, cv::COLOR_GRAY2BGR);
        // else
        //     base = cv::Mat(480, 640,CV_8UC3, cv::Scalar(255, 255, 255));
        //
        // // 定义颜色表和透明度
        // std::vector<cv::Scalar> colorTable = {
        //     cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
        //     cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
        //     cv::Scalar(128, 0, 255), cv::Scalar(255, 128, 0), cv::Scalar(0, 128, 255),
        //     cv::Scalar(128, 255, 0)
        // };
        // double alpha = 0.3;
        //
        // // 把 mask 叠加到 base 上，生成两张相同的底图
        // cv::Mat vis_orb = base.clone();
        // cv::Mat vis_flow = base.clone();
        // for (size_t i = 0; i < curF.allMasks.size(); i++)
        // {
        //     cv::Mat colorLayer(base.size(), base.type(), cv::Scalar(0, 0, 0));
        //     cv::Scalar color = colorTable[i % colorTable.size()];
        //     colorLayer.setTo(color, curF.allMasks[i]);
        //     cv::addWeighted(colorLayer, alpha, vis_orb, 1.0, 0, vis_orb);
        //     cv::addWeighted(colorLayer, alpha, vis_flow, 1.0, 0, vis_flow);
        // }
        //
        // // --- 1. 绘制 ORB variance (绿色箭头) 到 vis_orb ---
        // for (size_t i = 0; i < curF.mvClusterLabels.size(); i++)
        // {
        //     cv::Point2f center(0, 0);
        //     int count = 0;
        //     if (i < curF.allMasks.size() && !curF.allMasks[i].empty())
        //     {
        //         for (int r = 0; r < curF.allMasks[i].rows; r++)
        //         {
        //             const uchar* ptr = curF.allMasks[i].ptr<uchar>(r);
        //             for (int c = 0; c < curF.allMasks[i].cols; c++)
        //             {
        //                 if (ptr[c] > 0)
        //                 {
        //                     center.x += c;
        //                     center.y += r;
        //                     count++;
        //                 }
        //             }
        //         }
        //     }
        //
        //     if (count > 0)
        //     {
        //         center.x /= count;
        //         center.y /= count;
        //
        //         cv::Point2f varVec = curF.mvClusterOpFlowVariance[i];
        //         cv::Point2f endPoint = center + varVec * 5.0f;
        //
        //         cv::arrowedLine(vis_orb, center, endPoint, cv::Scalar(0, 255, 0), 2,
        //                         cv::LINE_AA, 0, 0.2);
        //     }
        // }
        //
        // // --- 2. 绘制 Optical Flow variance (黄色箭头) 到 vis_flow ---
        // for (size_t i = 0; i < curF.mvClusterLabels.size(); i++)
        // {
        //     cv::Point2f center(0, 0);
        //     int count = 0;
        //     if (i < curF.allMasks.size() && !curF.allMasks[i].empty())
        //     {
        //         for (int r = 0; r < curF.allMasks[i].rows; r++)
        //         {
        //             const uchar* ptr = curF.allMasks[i].ptr<uchar>(r);
        //             for (int c = 0; c < curF.allMasks[i].cols; c++)
        //             {
        //                 if (ptr[c] > 0)
        //                 {
        //                     center.x += c;
        //                     center.y += r;
        //                     count++;
        //                 }
        //             }
        //         }
        //     }
        //
        //     if (count > 0)
        //     {
        //         center.x /= count;
        //         center.y /= count;
        //
        //         // 均值方向和 variance 大小
        //         cv::Point2f meanVec = (i < curF.mvOptflwMeanofClusters.size())
        //                                   ? curF.mvOptflwMeanofClusters[i]
        //                                   : cv::Point2f(0, 0);
        //         float varLen = (i < curF.mvOptflwVarianceofClusters.size()) ? curF.mvOptflwVarianceofClusters[i] : 0.0f;
        //
        //         float norm = std::sqrt(meanVec.x * meanVec.x + meanVec.y * meanVec.y);
        //         cv::Point2f dir = (norm > 1e-5) ? (meanVec / norm) : cv::Point2f(0, 0);
        //         cv::Point2f endPoint = center + dir * varLen * 5.0f;
        //
        //         cv::arrowedLine(vis_flow, center, endPoint, cv::Scalar(0, 255, 255), 2,
        //                         cv::LINE_AA, 0, 0.2);
        //     }
        // }
        //
        // // --- 保存和显示 ---
        // if (curF.mnId <= 20)
        // {
        //     std::string saveName1 = "save_figs/ORB_variance" + std::to_string(curF.mnId) + ".png";
        //     std::string saveName2 = "save_figs/OptFlow_variance" + std::to_string(curF.mnId) + ".png";
        //     cv::imwrite(saveName1, vis_orb);
        //     cv::imwrite(saveName2, vis_flow);
        // }
        //
        // cv::imshow("ORB Variance Vectors", vis_orb);
        // cv::imshow("Optical Flow Variance Vectors", vis_flow);
        // // cv::waitKey(0);


        int pause = 1;
    }

    void Tracking::RecordClusterDynamics(Frame& F)
    {
        ofstream writer;
        writer.open(F.clusterDynamicName);
        for (int i = 0; i < F.mvClusterDynamic.size(); i++)
        {
            writer << F.mvClusterDynamic[i] << " ";
        }
        writer.close();
    }

    /*
     * Update the mappoint's dynamics / static observation for mappoint
     */
    //todo
    //    void Tracking::UpdateMapDynamics(Frame &F_cur, Map &mpMap) {
    //        for (int i = 0; i < F_cur.mvKeysDynamic.size(); i++) {
    //            if (F_cur.mvpMapPoints[i]) {
    //                if (F_cur.mvKeysDynamic[i])
    //                    F_cur.mvpMapPoints[i]->AddObservationDynamic();
    //                else
    //                    F_cur.mvpMapPoints[i]->AddObservationStatic();
    //            }
    //        }
    //    }

    ///added-----------------------
    void Tracking::kalmanFilterUpdateDynamics()
    {
        for (int i = 0; i < allKFs.size(); i++)
        {
            allKFs[i]->updateDynamics();
        }
    }

    void Tracking::MonocularInitialization()
    {
        if (!mpInitializer)
        {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > 100)
            {
                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
                if (mpInitializer)
                    delete mpInitializer;
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }
        }
        else
        {
            // Try to initialize
            if ((int)mCurrentFrame.mvKeys.size() <= 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer*>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }
            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);
            ///adds on
            int Counter = 0;
            for (int i = 0; i < mvIniMatches.size(); i++)
                if (mvIniMatches[i] > -1)
                    Counter = Counter + 1;
            cout << "before filter " << Counter << endl;
            FilterOutDynamicMatchs(mInitialFrame, mCurrentFrame, mvIniMatches);
            Counter = 0;
            for (int i = 0; i < mvIniMatches.size(); i++)
                if (mvIniMatches[i] > -1)
                    Counter = Counter + 1;
            cout << "after filter " << Counter << endl;
            // Check if there are enough correspondences
            if (nmatches < 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer*>(NULL);
                return;
            }
            ///adds on ends

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                CreateInitialMapMonocular();
            }
        }
    }

    void Tracking::CreateInitialMapMonocular()
    {
        // Create KeyFrames
        KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);


        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);

            ///Added module
            int matchedPtIndex = mvIniMatches[i];
            int label = mLastFrame.mvKeysLabels[matchedPtIndex];
            bool soft = mLastFrame.mvKeysSoft[matchedPtIndex];
            pMP->label = label;
            pMP->soft = soft;
            pMP->AddObservationStatic(pKFini, i);
            pMP->AddObservationStatic(pKFcur, mvIniMatches[i]);
            ///---------------------------------------------------
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
        {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                MapPoint* pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }

    void Tracking::CheckReplacedInLastFrame()
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            MapPoint* pMP = mLastFrame.mvpMapPoints[i];

            if (pMP)
            {
                MapPoint* pRep = pMP->GetReplaced();
                if (pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

    ///adds on
    bool checkIfExistingMatches(int obserIndice, double distance, vector<vector<double>>& kf_observation_pairs)
    {
        bool flag = false;
        for (int n = 0; n < kf_observation_pairs.size(); n++)
        {
            if (kf_observation_pairs[n][0] == obserIndice)
            {
                if (kf_observation_pairs[n][1] > distance)
                {
                    kf_observation_pairs[n][0] = -1;
                    kf_observation_pairs[n][1] = 10000;
                }
                else
                {
                    flag = true;
                }
            }
        }
        return flag;
    }

    bool Tracking::TrackReferenceKeyFrame()
    {
        std::cout << "TrackReferenceKeyFrame start, mCurrentFrame.mnId=" << mCurrentFrame.mnId << std::endl;

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint*> vpMapPointMatches;
        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        // ///Added module--------------------Logically I should put this block here, need wait mvpmappoints has values-------------------
        // CheckReprojectDynamic(mCurrentFrame);
        // CheckOpticalFlowDynamic(mCurrentFrame, *mpReferenceKF);
        // DetermineDynamics(mCurrentFrame);
        // //RecordClusterDynamics(mCurrentFrame);
        // ///---------------------------------------------------

        if (nmatches < 15)
            return false;

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        ///Added module---------------------------------------
        CheckReprojectDynamic(mCurrentFrame);
        CheckOpticalFlowDynamic(mCurrentFrame, *mpReferenceKF);
        DetermineDynamics(mCurrentFrame);
        //RecordClusterDynamics(mCurrentFrame);
        ///---------------------------------------------------

        //Optimizer::PoseOptimization(&mCurrentFrame);
        Optimizer::PoseOptimization_dynamic(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        ///---------------draw ORB-------------
        // cv::Mat imgShow;
        // cv::cvtColor(mCurrentFrame.frameImGray, imgShow, cv::COLOR_GRAY2BGR);
        //
        // for (int i = 0; i < mCurrentFrame.N; i++)
        // {
        //     if (mCurrentFrame.mvpMapPoints[i])
        //     {
        //         cv::Point2f ptCurrent = mCurrentFrame.mvKeysUn[i].pt;
        //
        //         // 找到该 MapPoint 在参考关键帧中的索引
        //         MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        //         int idxRef = pMP->GetIndexInKeyFrame(mpReferenceKF);
        //
        //         if(idxRef >= 0)
        //         {
        //             // 用参考关键帧中的特征点坐标作为对应点
        //             cv::Point2f ptRef = mpReferenceKF->mvKeysUn[idxRef].pt;
        //
        //             // 画当前帧的点
        //             cv::circle(imgShow, ptCurrent, 2, cv::Scalar(0,255,0), -1);
        //
        //             // 画参考关键帧的点
        //             cv::circle(imgShow, ptRef, 2, cv::Scalar(0,255,0), -1);
        //
        //             // 画连线
        //             cv::line(imgShow, ptCurrent, ptRef, cv::Scalar(0,255,0), 1);
        //         }
        //     }
        // }
        // std::string saveName = "save_figs/ORB_" + std::to_string(mCurrentFrame.mnId) + ".png";
        // cv::imwrite(saveName, imgShow);
        //
        // cv::imshow("TrackRefKF Matches", imgShow);
        // cv::waitKey(0);
        // ///--------------------------------

        std::cout << "TrackReferenceKeyFrame end, mvOptflwVarianceofClusters size="
            << mCurrentFrame.mvOptflwVarianceofClusters.size() << std::endl;

        return nmatchesMap >= 10;
    }

    ///adds on ---- kalman filter
    void Tracking::trackKalmanFilter()
    {
        ///step 1 store the indices of matched observation, and distance. same size as allKFs
        vector<vector<double>> kf_observation_pairs(allKFs.size());
        for (int k = 0; k < allKFs.size(); k++)
        {
            vector<double> tempPair{-1, 10000};
            kf_observation_pairs[k] = tempPair;
        }
        ///step 2 search the nearest observation for each kf instance
        for (int k = 0; k < allKFs.size(); k++)
        {
            if (allKFs[k]->lost)
                continue;
            ///step 2.1 predict status
            allKFs[k]->predictStatus();
            ///step 2.2 search the observation
            cv::Point2f centre_predict(allKFs[k]->X.at<double>(0, 0), allKFs[k]->X.at<double>(1, 0));
            int minIndex = -1;
            float minDistance = 50;
            bool found = false;
            for (int o = 0; o < mCurrentFrame.maskCentres.size(); o++)
            {
                float distance = cv::norm(centre_predict - mCurrentFrame.maskCentres[o]);
                if (distance < minDistance)
                {
                    //note: placing the checking function in the loop,
                    //note: would enable the kf to check for next close observation
                    //note: if this observation has been taken.
                    bool taken = checkIfExistingMatches(o, distance, kf_observation_pairs);
                    if (!taken)
                    {
                        minDistance = distance;
                        minIndex = o;
                        found = true;
                    }
                }
            }
            if (found)
            {
                kf_observation_pairs[k][0] = minIndex;
                kf_observation_pairs[k][1] = minDistance;
            }
            else
            {
                allKFs[k]->lostCounter++;
                if (allKFs[k]->lostCounter > 3)
                    allKFs[k]->lost = true;
            }
        }
        ///step 3 update status of each kf instance
        vector<int> obserMatchFlags(mCurrentFrame.mvClusterLabels.size(), false);
        for (int k = 0; k < allKFs.size(); k++)
        {
            int index = kf_observation_pairs[k][0];
            if (index > -1)
            {
                cv::Point2f observationPoint = mCurrentFrame.maskCentres[index];
                cv::Mat thisObserv = (cv::Mat_<double>(2, 1) << observationPoint.x, observationPoint.y);
                allKFs[k]->updateStatus(thisObserv);
                obserMatchFlags[index] = true;

                //store the dynamic status
                if (mCurrentFrame.mvClusterDynamic[index])
                    allKFs[k]->dynamicsHistory.push_back(1);
                else
                    allKFs[k]->dynamicsHistory.push_back(0);
                //connecting currentFrame to this kalman filter instance
                mCurrentFrame.mvKalFilts[index] = allKFs[k];
                //allKFs[k]->allFrames.push_back(&mCurrentFrame); Error. becuase mCurrentFrame while be overwrite every time when new frame constructed.
            }
            else
                allKFs[k]->lostCounter++;
        }

        ///step 4 add unmatched observation as new kf instance
        for (int l = 0; l < mCurrentFrame.mvClusterLabels.size(); l++)
        {
            if (!obserMatchFlags[l])
            {
                cv::Point2f observationPoint = mCurrentFrame.maskCentres[l];
                KalmanFilter* newKF = new KalmanFilter(kfGlobalID, mCurrentFrame.mvClusterLabels[l],
                                                       observationPoint.x, observationPoint.y);
                allKFs.push_back(newKF);
                //newKF->allFrames.push_back(&mCurrentFrame); Error. becuase mCurrentFrame while be overwrite every time when new frame constructed.
                kfGlobalID++;
                //store the dynamic status
                if (mCurrentFrame.mvClusterDynamic[l])
                    newKF->dynamicsHistory.push_back(1);
                else
                    newKF->dynamicsHistory.push_back(0);
                //store the kalman filter instance to frame
                mCurrentFrame.mvKalFilts[l] = newKF;
            }
        }
    }

    ///---------------------------

    /*
     * update the Transform of ref to last keyframe
     * create new Mappoint for last frame and push to mlpTemporalPoints[]
     */
    void Tracking::UpdateLastFrame()
    {
        // Update pose according to reference keyframe
        KeyFrame* pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back(); //Transform ref to last

        mLastFrame.SetPose(Tlr * pRef->GetPose()); //get T world to last

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++)
        {
            float z = mLastFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint* pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
            }

            if (bCreateNew)
            {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }
    }

    bool Tracking::TrackWithMotionModel()
    {
        //std::cout << "TrackWithMotionModel start, mCurrentFrame.mnId=" << mCurrentFrame.mnId << std::endl;

        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        ///adds on
        CheckReprojectDynamic(mCurrentFrame);
        CheckOpticalFlowDynamic(mCurrentFrame, mLastFrame);
        DetermineDynamics(mCurrentFrame);
        //RecordClusterDynamics(mCurrentFrame);
        ///adds on end--------------------------------------


        ///---------------draw ORB-------------
        // cv::Mat imgShow;
        // cv::cvtColor(mCurrentFrame.frameImGray, imgShow, cv::COLOR_GRAY2BGR);
        //
        // for (int i = 0; i < mCurrentFrame.N; i++)
        // {
        //     if (mCurrentFrame.mvpMapPoints[i])
        //     {
        //         cv::Point2f ptCurrent = mCurrentFrame.mvKeysUn[i].pt;
        //
        //         // 找到该 MapPoint 在参考关键帧中的索引
        //         MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        //         int idxRef = pMP->GetIndexInKeyFrame(mpReferenceKF);
        //
        //         if(idxRef >= 0)
        //         {
        //             // 用参考关键帧中的特征点坐标作为对应点
        //             cv::Point2f ptRef = mpReferenceKF->mvKeysUn[idxRef].pt;
        //
        //             // 画当前帧的点
        //             cv::circle(imgShow, ptCurrent, 2, cv::Scalar(0,255,0), -1);
        //
        //             // 画参考关键帧的点
        //             cv::circle(imgShow, ptRef, 2, cv::Scalar(0,255,0), -1);
        //
        //             // 画连线
        //             cv::line(imgShow, ptCurrent, ptRef, cv::Scalar(0,255,0), 1);
        //         }
        //     }
        // }
        // std::string saveName = "save_figs/ORB_" + std::to_string(mCurrentFrame.mnId) + ".png";
        // cv::imwrite(saveName, imgShow);
        //
        // cv::imshow("TrackRefKF Matches", imgShow);
        // cv::waitKey(0);
        // ///--------------------------------



        // If few matches, uses a wider window search
        if (nmatches < 20)
        {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
        }

        if (nmatches < 20)
            return false;

        // Optimize frame pose with all matches
        //Optimizer::PoseOptimization(&mCurrentFrame);
        ///adds on
        Optimizer::PoseOptimization_dynamic(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        if (mbOnlyTracking)
        {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }


        //std::cout << "TrackWithMotionModel end, mvOptflwVarianceofClusters size="
        //       << mCurrentFrame.mvOptflwVarianceofClusters.size() << std::endl;

        return nmatchesMap >= 10;
    }

    bool Tracking::TrackLocalMap()
    {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.

        UpdateLocalMap();

        SearchLocalPoints();

        //todo --- check dynamic here? no need
        //CheckReprojectDynamic(mCurrentFrame);
        //checkoptical();
        //CheckOpticalFlowDynamic()

        // Optimize Pose
        //Optimizer::PoseOptimization(&mCurrentFrame);
        Optimizer::PoseOptimization_dynamic(&mCurrentFrame);

        mnMatchesInliers = 0;
        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking)
                    {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    }
                    else
                        mnMatchesInliers++;
                }
                else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }


    bool Tracking::NeedNewKeyFrame()
    {
        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;
        if (mSensor != System::MONOCULAR)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nTrackedClose++;
                    else
                        nNonTrackedClose++;
                }
            }
        }

        bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
        //Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

        if ((c1a || c1b || c1c) && c2)
        {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle)
            {
                return true;
            }
            else
            {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR)
                {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
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

    /*
     * Create new KayFrame from CurrentFrame
     * Add new Mappoint from keyPoints.(with depth sensor)
     */
    void Tracking::CreateNewKeyFrame()
    {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        mpReferenceKF = pKF; //Tracking instance's mpReferenceKF
        mCurrentFrame.mpReferenceKF = pKF; //CurrentFrame's mpReferenceKF

        if (mSensor != System::MONOCULAR)
        {
            mCurrentFrame.UpdatePoseMatrices();

            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            vector<pair<float, int>> vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty())
            {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++)
                {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1)
                    {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    }

                    if (bCreateNew)
                    {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i); //todo think about when using zoe depth.
                        MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;

                        ///adds on - init the mappoint dynamic rate - given label
                        if (pKF->mvKeysDynamics[i])
                            pNewMP->AddObservationDynamic(pKF, i);
                        else
                            pNewMP->AddObservationStatic(pKF, i);
                        pNewMP->label = pKF->mvKeysLabels[i];
                        ///adds on - end
                    }
                    else
                    {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints()
    {
        // Do not search map points already matched
        for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.
                                             end();
             vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (pMP)
            {
                if (pMP->isBad())
                {
                    *vit = static_cast<MapPoint*>(NULL);
                }
                else
                {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5))
            {
                pMP->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0)
        {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD)
                th = 3;
            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    /* 1. mpMap->SetReferenceMapPoints(mvpLocalMapPoints); ---for visualization
     * 2. UpdateLocalKeyFrames(); --- fill up mvpLocalKeyFrames
     * 3. UpdateLocalPoints(); --- fill up mvpLocalMapPoints
     */
    void Tracking::UpdateLocalMap()
    {
        // This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    //fill up mvpLocalMapPoints from mvpLocalKeyFrames
    void Tracking::UpdateLocalPoints()
    {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++)
        {
            KeyFrame* pKF = *itKF;
            const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++)
            {
                MapPoint* pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad())
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    //fill up mvpLocalKeyFrames
    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame*, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad())
                {
                    const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++)
        {
            KeyFrame* pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max)
            {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++)
        {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame* pKF = *itKF;

            const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++)
            {
                KeyFrame* pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad())
                {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame*> spChilds = pKF->GetChilds();
            for (set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
            {
                KeyFrame* pChildKF = *sit;
                if (!pChildKF->isBad())
                {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame* pParent = pKF->GetParent();
            if (pParent)
            {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        if (pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization()
    {
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<PnPsolver*> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint*>> vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++)
        {
            KeyFrame* pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else
            {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15)
                {
                    vbDiscarded[i] = true;
                    continue;
                }
                else
                {
                    PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nKFs; i++)
            {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver* pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty())
                {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint*> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++)
                    {
                        if (vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        }
                        else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50)
                    {
                        int nadditional = matcher2.
                            SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                        if (nadditional + nGood >= 50)
                        {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(
                                    mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                                // Final optimization
                                if (nGood + nadditional >= 50)
                                {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50)
                    {
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        if (!bMatch)
        {
            return false;
        }
        else
        {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }
    }

    void Tracking::Reset()
    {
        cout << "System Reseting" << endl;
        if (mpViewer)
        {
            mpViewer->RequestStop();
            while (!mpViewer->isStopped())
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

        if (mpInitializer)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        if (mpViewer)
            mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string& strSettingPath)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3,CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool& flag)
    {
        mbOnlyTracking = flag;
    }
} //namespace ORB_SLAM

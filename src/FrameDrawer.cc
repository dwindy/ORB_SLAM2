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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

///added module
//cv::Mat FrameDrawer::DrawLiDAR()
//{
//    cv::Mat im = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
//    mIm.copyTo(im);
//    ///added module
//    //draw projected laser points
//    int PjcLsrNum = mvPjcLsrPts.size();
//    if(PjcLsrNum>0)
//    {
//        for(int i=0;i<PjcLsrNum;i++)
//        {
//            //color
//            float maxVal = 20.0;
//            int red = min(255, (int) (255 * abs((mvPjcLsrPts[i].response - maxVal) / maxVal)));
//            int green = min(255, (int) (255 * (1 - abs((mvPjcLsrPts[i].response - maxVal) / maxVal))));
//            cv::circle(im,mvPjcLsrPts[i].pt,2,cv::Scalar(0,0,0),-1);
//        }
//    }
//    PjcLsrNum = mvPjcLsrPtsUndis.size();
//    if(PjcLsrNum>0)
//    {
//        for(int i=0;i<PjcLsrNum;i++)
//        {
//            //color
//            float maxVal = 20.0;
//            int red = min(255, (int) (255 * abs((mvPjcLsrPtsUndis[i].response - maxVal) / maxVal)));
//            int green = min(255, (int) (255 * (1 - abs((mvPjcLsrPtsUndis[i].response - maxVal) / maxVal))));
//            cv::circle(im,mvPjcLsrPtsUndis[i].pt,2,cv::Scalar(255,255,255),-1);
//        }
//    }
//    return im;
//}

    cv::Mat FrameDrawer::DrawFrame() {
        cv::Mat im;
        vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
        vector<int> vMatches; // Initialization: correspondences with reference keypoints
        vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
        vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
        int state; // Tracking state
        ///Added module
        vector<cv::Point2d> vCurrentLsrPt; //Laser point in current frame
        vector<cv::Point2d> vCurrentLsrCorner; //Laser point in current frame
        vector<cv::Point2d> vCurrentLsrLessCorner; //Laser point in current frame
        vector<cv::Point2d> vCurrentLsrFlat; //Laser point in current frame
        vector<cv::Point2d> vCurrentLsrLessFlat; //Laser point in current frame
        vector<int> vDepthSource;

        //Copy variables within scoped mutex
        {
            unique_lock<mutex> lock(mMutex);
            state = mState;
            if (mState == Tracking::SYSTEM_NOT_READY)
                mState = Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);

            if (mState == Tracking::NOT_INITIALIZED) {
                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
            } else if (mState == Tracking::OK) {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
            } else if (mState == Tracking::LOST) {
                vCurrentKeys = mvCurrentKeys;
            }
            ///added module
            vCurrentLsrPt = mvPjcLsrPts;
            vCurrentLsrCorner = mvPjcLsrCorner;
            vCurrentLsrLessCorner = mvPjcLsrLessCorner;
            vCurrentLsrFlat = mvPjcLsrFlat;
            vCurrentLsrLessFlat = mvPjcLsrLessFlat;
            vDepthSource = mvDepthSource;
        } // destroy scoped mutex -> release mutex

        if (im.channels() < 3) //this should be always true
            cvtColor(im, im, CV_GRAY2BGR);

        //Draw
        if (state == Tracking::NOT_INITIALIZED) //INITIALIZING
        {
            for (unsigned int i = 0; i < vMatches.size(); i++) {
                if (vMatches[i] >= 0) {
                    cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt,
                             cv::Scalar(0, 255, 0));
                }
            }
        } else if (state == Tracking::OK) //TRACKING
        {
            mnTracked = 0;
            mnTrackedVO = 0;
            const float r = 5;
            const int n = vCurrentKeys.size();
            for (int i = 0; i < n; i++) {
                if (vbVO[i] || vbMap[i]) {
                    cv::Point2f pt1, pt2;
                    pt1.x = vCurrentKeys[i].pt.x - r;
                    pt1.y = vCurrentKeys[i].pt.y - r;
                    pt2.x = vCurrentKeys[i].pt.x + r;
                    pt2.y = vCurrentKeys[i].pt.y + r;

                    // This is a match to a MapPoint in the map
                    if (vbMap[i]) {
                        if (vDepthSource[i] == 1) {
                            cv::rectangle(im, pt1, pt2, cv::Scalar(255, 128, 0)); //gray-blue
                            cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 128, 0), -1);
                            mnTracked++;
                        } else {
                            if (vDepthSource[i] == 2) {
                                cv::rectangle(im, pt1, pt2, cv::Scalar(255, 0, 127)); //purple
                                cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 127), -1);
                                mnTracked++;
                            } else {
                                if (vDepthSource[i] == 3) {
                                    cv::rectangle(im, pt1, pt2, cv::Scalar(0, 0, 255)); //red
                                    cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 0, 255), -1);
                                    mnTracked++;
                                } else {
                                    cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0)); //green
                                    cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
                                    mnTracked++;
                                }
                            }
                        }
                    } else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        cv::rectangle(im, pt1, pt2, cv::Scalar(255, 0, 0));
                        cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 0), -1);
                        mnTrackedVO++;
                    }
                }
            }
            ///added module : draw projected raw laser points
//        int PjcLsrNum = vCurrentLsrPt.size();
//        if (PjcLsrNum > 0) {
//            for (int i = 0; i < PjcLsrNum; i++) {
//                float maxVal = 20.0;
////                int red = min(255, (int) (255 * abs((mvPjcLsrPts[i].response - maxVal) / maxVal)));
////                int green = min(255, (int) (255 * (1 - abs((mvPjcLsrPts[i].response - maxVal) / maxVal))));
//                cv::circle(im, vCurrentLsrPt[i], 2, cv::Scalar(0, 255, 255), -1);
//            }
//        }
//            int PjcLsrCorNum = vCurrentLsrCorner.size();
//            if (PjcLsrCorNum > 0) {
//                for (int i = 0; i < PjcLsrCorNum; i++) {
//                    cv::circle(im, vCurrentLsrCorner[i], 2, cv::Scalar(60, 20, 220), -1);
//                }
//            }
//            int PjcLsrLessCorNum = vCurrentLsrLessCorner.size();
//            if (PjcLsrLessCorNum > 0) {
//                for (int i = 0; i < PjcLsrLessCorNum; i++) {
//                    cv::circle(im, vCurrentLsrLessCorner[i], 2, cv::Scalar(180, 105, 255), -1);
//                }
//            }
//            int PjcLsrFltNum = vCurrentLsrFlat.size();
//            if (PjcLsrFltNum > 0) {
//                for (int i = 0; i < PjcLsrFltNum; i++) {
//                    cv::circle(im, vCurrentLsrFlat[i], 2, cv::Scalar(255, 0, 0), -1);
//                }
//            }
//            int PjcLsrLessFltNum = vCurrentLsrLessFlat.size();
//            if (PjcLsrLessFltNum > 0) {
//                for (int i = 0; i < PjcLsrLessFltNum; i++) {
//                    cv::circle(im, vCurrentLsrLessFlat[i], 2, cv::Scalar(255, 255, 0), -1);
//                }
//            }
        }
        cv::Mat imWithInfo;
        DrawTextInfo(im, state, imWithInfo);

        return imWithInfo;
    }


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

    void FrameDrawer::Update(Tracking *pTracker) {
        unique_lock<mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys; //pass tracker->fames' keypoint to framedrawer.keypoints
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N, false);
        mvbMap = vector<bool>(N, false);
        mbOnlyTracking = pTracker->mbOnlyTracking;
        ///Added Module: pass tracker->frame.mLaserPt_cam[].pt2d to frameDrawer.mvPjcLsrPts
        mvPjcLsrPts.clear();
        for (int i = 0; i < pTracker->mCurrentFrame.mLaserPt_cam.size(); i++) {
            mvPjcLsrPts.push_back(pTracker->mCurrentFrame.mLaserPt_cam[i].pt2d);
        }
        mvPjcLsrCorner.clear();
        for (int i = 0; i < pTracker->mCurrentFrame.mLaserCorner_cam.size(); i++) {
            mvPjcLsrCorner.push_back(pTracker->mCurrentFrame.mLaserCorner_cam[i].pt2d);
        }
        mvPjcLsrLessCorner.clear();
        for (int i = 0; i < pTracker->mCurrentFrame.mLaserLessCorner_cam.size(); i++) {
            mvPjcLsrLessCorner.push_back(pTracker->mCurrentFrame.mLaserLessCorner_cam[i].pt2d);
        }
        mvPjcLsrFlat.clear();
        for (int i = 0; i < pTracker->mCurrentFrame.mLaserFlat_cam.size(); i++) {
            mvPjcLsrFlat.push_back(pTracker->mCurrentFrame.mLaserFlat_cam[i].pt2d);
        }
        mvPjcLsrLessFlat.clear();
        for (int i = 0; i < pTracker->mCurrentFrame.mLaserLessFlat_cam.size(); i++) {
            mvPjcLsrLessFlat.push_back(pTracker->mCurrentFrame.mLaserLessFlat_cam[i].pt2d);
        }
        mvDepthSource.clear();
        for (int i = 0; i < pTracker->mCurrentFrame.mvORBAttributions.size(); i++) {
            mvDepthSource.push_back(pTracker->mCurrentFrame.mvORBAttributions[i].depthSource);
        }
        ///-------------------------------------------------------------------------------------


        if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED) {
            mvIniKeys = pTracker->mInitialFrame.mvKeys;
            mvIniMatches = pTracker->mvIniMatches;
        } else if (pTracker->mLastProcessedState == Tracking::OK) {
            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if (pMP) {
                    //if (!pTracker->mCurrentFrame.mvbOutlier[i]) { ///My comments to check why my withdetph Feature no working.
                        if (pMP->Observations() > 0)
                            mvbMap[i] = true;
                        else
                            mvbVO[i] = true;
                    //}
                }
            }
        }
        mState = static_cast<int>(pTracker->mLastProcessedState);

//        cout << "pTracker->mCurrentFrame.mvKeys " << pTracker->mCurrentFrame.mvKeys.size()
//             << " " << pTracker->mCurrentFrame.mvORBAttributions.size() << endl;
//        for (int i = 0; i < pTracker->mCurrentFrame.mvKeys.size(); i++) {
//            if (pTracker->mCurrentFrame.mvpMapPoints[i])
//                cout << pTracker->mCurrentFrame.mvKeys[i].pt.x << " " << pTracker->mCurrentFrame.mvKeys[i].pt.y
//                     << " mvbMap " << mvbMap[i] << " depth source "
//                     << pTracker->mCurrentFrame.mvORBAttributions[i].depthSource << endl;
//        }
        int pause = 1;
    }

} //namespace ORB_SLAM

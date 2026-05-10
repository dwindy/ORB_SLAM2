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
    FrameDrawer::FrameDrawer(Map* pMap) : mpMap(pMap)
    {
        mState = Tracking::SYSTEM_NOT_READY;
        mIm = cv::Mat(480, 640,CV_8UC3, cv::Scalar(0, 0, 0));
    }

    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
        vector<int> vMatches; // Initialization: correspondences with reference keypoints
        vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
        vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
        int state; // Tracking state
        ///Added Module
        int frameID;
        vector<bool> vbKeysSoft;
        vector<int> viKeysLabels;
        vector<bool> vbKeysDynamic;
        vector<float> vfKeysRjtVar;
        vector<float> vfKeysOptVar;
        vector<cv::Point2f> vpCentreMasks;
        vector<bool> vbObjDynamic;
        vector<bool> vbObjDynamicRep;
        vector<bool> vbObjDynamicOpj;

        //Copy variables within scoped mutex
        {
            unique_lock<mutex> lock(mMutex);
            state = mState;
            if (mState == Tracking::SYSTEM_NOT_READY)
                mState = Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);

            if (mState == Tracking::NOT_INITIALIZED)
            {
                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
            }
            else if (mState == Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
                ///Added Module
                vbKeysSoft = mvKeysSoft;
                viKeysLabels = mvKeysLabels;
                vbKeysDynamic = mvKeysDynamics;
                vfKeysRjtVar = mvKeyRjtVars;
                vfKeysOptVar = mvKeyOptVars;
                vpCentreMasks = mvCentersOfMasks;
                vbObjDynamic = mvObjDynamics;
                vbObjDynamicRep = mvObjDynamicsRep;
                vbObjDynamicOpj = mvObjDynamicsOpj;
                frameID = mFrameID;
                ///--------------------------
            }
            else if (mState == Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
            }
        } // destroy scoped mutex -> release mutex

        if (im.channels() < 3) //this should be always true
            cvtColor(im, im, CV_GRAY2BGR);

        //Draw
        if (state == Tracking::NOT_INITIALIZED) //INITIALIZING
        {
            for (unsigned int i = 0; i < vMatches.size(); i++)
            {
                if (vMatches[i] >= 0)
                {
                    cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt,
                             cv::Scalar(0, 255, 0));
                }
            }
        }
        else if (state == Tracking::OK) //TRACKING
        {
            mnTracked = 0;
            mnTrackedVO = 0;
            const float r = 5;
            const int n = vCurrentKeys.size();
            for (int i = 0; i < n; i++)
            {
                if (vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1, pt2;
                    pt1.x = vCurrentKeys[i].pt.x - r;
                    pt1.y = vCurrentKeys[i].pt.y - r;
                    pt2.x = vCurrentKeys[i].pt.x + r;
                    pt2.y = vCurrentKeys[i].pt.y + r;

                    // This is a match to a MapPoint in the map
                    if (vbMap[i])
                    {
                        ///Added Modules
                        if (vbKeysDynamic[i])
                        {
                            //if dynamic point, draw red
                            //cv::rectangle(im, pt1, pt2, cv::Scalar(0, 0, 255));
                            //cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 0, 255), -1);
                            mnTracked++;
                        }
                        else
                        {
                            //-----
                            //normal case
                            //cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0));
                            //cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
                            mnTracked++;
                        }
                    }
                    else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        cv::rectangle(im, pt1, pt2, cv::Scalar(255, 0, 0));
                        cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 0), -1);
                        mnTrackedVO++;
                    }
                }
            }
            ///Added Module ///todo add empty opt or rjt check
            //draw variance of each label
            //cout<<"frameID: "<<frameID<<" vfKeysOptVar size "<<vfKeysOptVar.size()<<" vfKeysRjtVar size "<<vfKeysRjtVar.size()<<endl;

            // //for drawing, prepare for paper
            // cv::Mat repimage, opjimage,combinedImage;
            // combinedImage = im.clone();
            // repimage = im.clone();
            // opjimage = im.clone();
            // int timer = 3;
            for (int i = 0; i < vpCentreMasks.size(); i++)
            {
                if (vbObjDynamic[i])
                {
                    cv::drawMarker(im, vpCentreMasks[i], cv::Scalar(51, 255, 255), 1, vfKeysOptVar[i], 2); //yellow
                    cv::drawMarker(im, vpCentreMasks[i], cv::Scalar(255, 0, 255), 1, vfKeysRjtVar[i], 2); //purple
                }
                else
                {
                    cv::circle(im, vpCentreMasks[i], vfKeysOptVar[i], cv::Scalar(51, 255, 255), 2); //yellow
                    cv::circle(im, vpCentreMasks[i], vfKeysRjtVar[i], cv::Scalar(255, 0, 255), 2); //purple
                }

                // //for drawing, prepare for paper
                //  if (frameID==2)
                //  {
                //      cout<<vpCentreMasks[i]<<endl;
                //      string texttoput = std::to_string(vpCentreMasks[i].x);
                //      //cv::putText(combinedImage,texttoput,vpCentreMasks[i],1,1,cv::Scalar(0,0,200),2);
                //      if (vpCentreMasks[i].x>=20 && vpCentreMasks[i].x<=28)//left chair
                //      {
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //      }
                //      if (vpCentreMasks[i].x>=115 && vpCentreMasks[i].x<=120)//left human
                //      {
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //      }
                //      if (vpCentreMasks[i].x>=200 && vpCentreMasks[i].x<=205)//left keyboard
                //      {
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //      }
                //      if (vpCentreMasks[i].x>=205 && vpCentreMasks[i].x<=215)//left monitor
                //      {
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::circle(opjimage, vpCentreMasks[i], vfKeysOptVar[i] * timer, cv::Scalar(51,255,255), 2); //yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysOptVar[i] * timer, cv::Scalar(51,255,255), 2); //yellow
                //      }
                //      if (vpCentreMasks[i].x>=260 && vpCentreMasks[i].x<=265)//left mouse
                //      {
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //      }
                //      if (vpCentreMasks[i].x>=336 && vpCentreMasks[i].x<=340)//centre book
                //      {
                //          cv::drawMarker(repimage,vpCentreMasks[i],cv::Scalar(255, 0, 255),1,vfKeysRjtVar[i] * timer,2);//purple
                //          cv::circle(opjimage, vpCentreMasks[i], vfKeysOptVar[i] * timer, cv::Scalar(51,255,255), 2); //yellow
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(255, 0, 255),1,vfKeysRjtVar[i] * timer,2);//purple
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysOptVar[i] * timer, cv::Scalar(51,255,255), 2); //yellow
                //      }
                //      if (vpCentreMasks[i].x>=430 && vpCentreMasks[i].x<=435)//right monitor
                //      {
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //         cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //      }
                //      if (vpCentreMasks[i].x>=450 && vpCentreMasks[i].x<=458)//right human
                //      {
                //          cv::drawMarker(repimage,vpCentreMasks[i],cv::Scalar(255, 0, 255),1,vfKeysRjtVar[i] * timer,2);//purple
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(255, 0, 255),1,vfKeysRjtVar[i] * timer,2);//purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //      }
                //      if (vpCentreMasks[i].x>=510 && vpCentreMasks[i].x<=515)//back monitor?
                //      {
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::circle(opjimage, vpCentreMasks[i], vfKeysOptVar[i] * timer, cv::Scalar(51,255,255), 2); //yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysOptVar[i] * timer, cv::Scalar(51,255,255), 2); //yellow
                //      }
                //      if (vpCentreMasks[i].x>=553 && vpCentreMasks[i].x<=558)//right chair
                //      {
                //          cv::circle(repimage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(opjimage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //          cv::circle(combinedImage, vpCentreMasks[i], vfKeysRjtVar[i] * timer, cv::Scalar(255, 0, 255), 2); //purple
                //          cv::drawMarker(combinedImage,vpCentreMasks[i],cv::Scalar(51,255,255),1,vfKeysOptVar[i] * timer,2);//yellow
                //      }
                //
                //      std::string saveName1 = "opj-image" + std::to_string(frameID) + ".png";
                //      cv::imwrite(saveName1, opjimage);
                //      std::string saveName2 = "rep-image" + std::to_string(frameID) + ".png";
                //      cv::imwrite(saveName2, repimage);
                //      std::string saveName = "combined-image" + std::to_string(frameID) + ".png";
                //      cv::imwrite(saveName, combinedImage);
                //      int pause = 1;
                // }
            }
        }

        cv::Mat imWithInfo;
        DrawTextInfo(im, state, imWithInfo);

        return imWithInfo;
    }


    void FrameDrawer::DrawTextInfo(cv::Mat& im, int nState, cv::Mat& imText)
    {
        stringstream s;
        if (nState == Tracking::NO_IMAGES_YET)
            s << " WAITING FOR IMAGES";
        else if (nState == Tracking::NOT_INITIALIZED)
            s << " TRYING TO INITIALIZE ";
        else if (nState == Tracking::OK)
        {
            if (!mbOnlyTracking)
                s << "SLAM MODE |  ";
            else
                s << "LOCALIZATION | ";
            int nKFs = mpMap->KeyFramesInMap();
            int nMPs = mpMap->MapPointsInMap();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
            if (mnTrackedVO > 0)
                s << ", + VO matches: " << mnTrackedVO;
        }
        else if (nState == Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if (nState == Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
        im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
        imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
        cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1,
                    cv::Scalar(255, 255, 255), 1, 8);
    }

    void FrameDrawer::Update(Tracking* pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
        ///Added Module------------------------------
        mvKeysLabels = pTracker->mCurrentFrame.mvKeysLabels;
        mvKeysSoft = pTracker->mCurrentFrame.mvKeysSoft;
        mvKeysDynamics = pTracker->mCurrentFrame.mvKeysDynamic;
        mvKeyRjtVars = pTracker->mCurrentFrame.mvRePjtVarianceofClusters;
        mvKeyOptVars = pTracker->mCurrentFrame.mvOptflwVarianceofClusters;
        mvCentersOfMasks = pTracker->mCurrentFrame.maskCentres;
        mvObjDynamics = pTracker->mCurrentFrame.mvClusterDynamic;
        mvObjDynamicsRep = pTracker->mCurrentFrame.mvClusterDynamicRep;
        mvObjDynamicsOpj = pTracker->mCurrentFrame.mvClusterDynamicOpj;
        mFrameID = pTracker->mCurrentFrame.mnId;
        ///------------------------------------------
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N, false);
        mvbMap = vector<bool>(N, false);
        mbOnlyTracking = pTracker->mbOnlyTracking;

        if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED)
        {
            mvIniKeys = pTracker->mInitialFrame.mvKeys;
            mvIniMatches = pTracker->mvIniMatches;
        }
        else if (pTracker->mLastProcessedState == Tracking::OK)
        {
            for (int i = 0; i < N; i++)
            {
                MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        if (pMP->Observations() > 0)
                            mvbMap[i] = true;
                        else
                            mvbVO[i] = true;
                    }
                }
            }
        }
        mState = static_cast<int>(pTracker->mLastProcessedState);
    }
} //namespace ORB_SLAM

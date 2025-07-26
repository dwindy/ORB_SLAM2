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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}

//void MapDrawer::DrawMapPoints()
//{
//    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
//    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();
//
//    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());
//
//    if(vpMPs.empty())
//        return;
//
//    glPointSize(mPointSize);
//    glBegin(GL_POINTS);
//    glColor3f(0.0,0.0,0.0);
//
//    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
//    {
//        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
//            continue;
//        cv::Mat pos = vpMPs[i]->GetWorldPos();
//        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
//    }
//    glEnd();
//
//    glPointSize(mPointSize);
//    glBegin(GL_POINTS);
//    glColor3f(1.0,0.0,0.0);
//
//    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
//    {
//        if((*sit)->isBad())
//            continue;
//        cv::Mat pos = (*sit)->GetWorldPos();
//        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
//
//    }
//
//    glEnd();
//}

    void MapDrawer::DrawMapPoints() {
        const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
        const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();
        set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        if (vpMPs.empty())
            return;

        ///Added Module
        vector<vector<float>> colorCharts;
        colorCharts.push_back({0.0 / 255.0, 0.0 / 255, 0.0 / 255});           // Black
        colorCharts.push_back({255.0 / 255, 218.0 / 255, 185.0 / 255});     // Peach Puff
        colorCharts.push_back({105.0 / 255, 105.0 / 255, 105.0 / 255});     // Dim Gray
        colorCharts.push_back({255.0 / 255, 69.0 / 255, 0.0 / 255});       // Red-Orange
        colorCharts.push_back({255.0 / 255, 215.0 / 255, 0.0 / 255});      // Gold
        colorCharts.push_back({112.0 / 255, 128.0 / 255, 144.0 / 255});    // Slate Gray
        colorCharts.push_back({25.0 / 255, 25.0 / 255, 112.0 / 255});      // Midnight Blue
        colorCharts.push_back({230.0 / 255, 230.0 / 255, 250.0 / 255});    // Lavender
        colorCharts.push_back({143.0 / 255, 188.0 / 255, 143.0 / 255});    // Dark Sea Green
        colorCharts.push_back({205.0 / 255, 92.0 / 255, 92.0 / 255});     // Indian Red
        colorCharts.push_back({153.0 / 255, 50.0 / 255, 204.0 / 255});    // Dark Orchid
        colorCharts.push_back({0.0 / 255, 250.0 / 255, 154.0 / 255});     // Medium Spring Green
        colorCharts.push_back({123.0 / 255, 104.0 / 255, 238.0 / 255});   // Medium Slate Blue
        colorCharts.push_back({250.0 / 255, 128.0 / 255, 114.0 / 255});   // Salmon
        colorCharts.push_back({240.0 / 255, 128.0 / 255, 128.0 / 255});   // Light Coral
        colorCharts.push_back({102.0 / 255, 205.0 / 255, 170.0 / 255});   // Medium Aquamarine
        colorCharts.push_back({0.0 / 255, 206.0 / 255, 209.0 / 255});   // Dark Turquoise
        colorCharts.push_back({218.0 / 255, 112.0 / 255, 214.0 / 255});   // Orchid
        colorCharts.push_back({47.0 / 255, 79.0 / 255, 79.0 / 255});       // Dark Green Gray
        colorCharts.push_back({0.0 / 255, 0.0 / 255, 128.0 / 255});        // Dark Blue
        colorCharts.push_back({0.0 / 255, 255.0 / 255, 127.0 / 255});      // Spring Green
        colorCharts.push_back({255.0 / 255, 255.0 / 255, 0.0 / 255});      // Yellow
        colorCharts.push_back({255.0 / 255, 0.0 / 255, 147.0 / 255});      // Pink
        colorCharts.push_back({0.0 / 255, 128.0 / 255, 128.0 / 255});      // Teal
        colorCharts.push_back({160.0 / 255, 32.0 / 255, 240.0 / 255});     // Purple
        colorCharts.push_back({75.0 / 255, 0.0 / 255, 130.0 / 255});      // Indigo
        colorCharts.push_back({255.0 / 255, 255.0 / 255, 255.0 / 255});      // Cyan
        colorCharts.push_back({34.0 / 255, 139.0 / 255, 34.0 / 255});      // Forest Green
        colorCharts.push_back({255.0 / 255, 128.0 / 255, 0.0 / 255});     // Orange
        colorCharts.push_back({255.0 / 255, 0.0 / 255, 69.0 / 255});      // Orange Red
        colorCharts.push_back({210.0 / 255, 105.0 / 255, 30.0 / 255});     // Chocolate
        colorCharts.push_back({100.0 / 255, 149.0 / 255, 237.0 / 255});     // Cornflower Blue
        colorCharts.push_back({85.0 / 255, 107.0 / 255, 47.0 / 255});     // Dark Olive Green
        colorCharts.push_back({188.0 / 255, 143.0 / 255, 143.0 / 255});     // Rosy Brown
        colorCharts.push_back({72.0 / 255, 61.0 / 255, 219.0 / 255});      //  Dark Slate Blue
        colorCharts.push_back({147.0 / 255, 112.0 / 255, 219.0 / 255});     // Medium Purple
        colorCharts.push_back({0.0 / 255, 139.0 / 255, 139.0 / 255});     // Dark Cyan
        colorCharts.push_back({178.0 / 255, 34.0 / 255, 34.0 / 255});     //  FireBrick
        colorCharts.push_back({32.0 / 255, 178.0 / 255, 170.0 / 255});     //  Light Sea Green
        colorCharts.push_back({199.0/255,21.0/255,133.0/255}); //Medium Violet Red
        colorCharts.push_back({135.0/255,206.0/255,250.0/255}); //Light Sky Blue;
        colorCharts.push_back({189.0 / 255, 183.0 / 255, 107.0 / 255});  // Dark Khaki (for label 8)
        colorCharts.push_back({139.0 / 255, 0.0 / 255, 0.0 / 255});      // Dark Red (for label 13)
        colorCharts.push_back({176.0 / 255, 196.0 / 255, 222.0 / 255});  // Light Steel Blue (for label 14)
        colorCharts.push_back({123.0 / 255, 104.0 / 255, 238.0 / 255});  // Medium Slate Blue (for label 24)
        colorCharts.push_back({72.0 / 255, 61.0 / 255, 219.0 / 255});   // Dark Slate Blue (for label 30)
        colorCharts.push_back({0.0 / 255, 139.0 / 255, 139.0 / 255});   // Dark Cyan (for label 33)
        colorCharts.push_back({255.0 / 255, 160.0 / 255, 122.0 / 255});  // Light Salmon (for label 69)
        colorCharts.push_back({255.0 / 255, 105.0 / 255, 180.0 / 255});  // Hot Pink added 76 - 6th Nov 2024
        colorCharts.push_back({144.0 / 255, 238.0 / 255, 144.0 / 255});  // Light Green added 44
        colorCharts.push_back({255.0 / 255, 140.0 / 255, 0.0 / 255});  // Dark Orange added 43
        colorCharts.push_back({70.0 / 255, 130.0 / 255, 180.0 / 255});  // Steel Blue 47
        colorCharts.push_back({255.0 / 255, 99.0 / 255, 71.0 / 255}); // Tomato 2
        colorCharts.push_back({46.0 / 255, 139.0 / 255, 87.0 / 255}); // Sea Green 49
        colorCharts.push_back({0.0 / 255, 255.0 / 255, 0.0 / 255}); // Bright Aqua (new for label 15)
        colorCharts.push_back({200.0 / 255, 0.0 / 255, 255.0 / 255}); // Vivid Magenta (new for label 79)



        //store label to color map
        vector<vector<int>> mPofColorPoints;
        vector<int> mPofNormalPoints;
        for (int i = 0; i < colorCharts.size(); i++) {
            vector<int> colorPoints;
            mPofColorPoints.push_back(colorPoints);
        }
        map<int, int> label2colorMap;
        //Manually initial the map
        label2colorMap.insert(pair<int, int>(-1, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(0, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(1, label2colorMap.size()));
        //if color change alot in future diagram for semanticSLAM paper. comment this No2 --- 7 Nov
        label2colorMap.insert(pair<int, int>(2, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(3, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(4, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(6, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(8, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(7, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(13, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(14, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(15, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(16, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(24, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(25, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(26, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(27, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(28, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(29, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(30, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(32, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(33, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(36, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(39, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(41, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(43, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(44, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(45, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(47, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(49, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(56, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(57, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(58, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(60, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(61, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(62, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(63, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(64, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(65, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(66, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(67, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(68, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(69, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(71, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(72, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(73, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(74, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(75, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(76, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(77, label2colorMap.size()));
        label2colorMap.insert(pair<int, int>(79, label2colorMap.size()));
        for (int i = 0; i < vpMPs.size(); i++) {
            int label = vpMPs[i]->label;
            auto iter = label2colorMap.find(label);
            if (iter == label2colorMap.end()) {
                cout << "label " << label << " not in the color map" << endl;
            } else {
                int colorIndex = iter->second;
                mPofColorPoints[colorIndex].push_back(i);
            }
        }
//        for(auto iter:label2colorMap)
//            cout<<"label "<<iter.first<<" color index "<<iter.second<<endl;
        //draw by each color group
        for (int i = 0; i < mPofColorPoints.size(); i++) {
            glPointSize(10);//mPointSize

            //glColor3f(colorCharts[i][0], colorCharts[i][1], colorCharts[i][2]);

            glBegin(GL_POINTS);
            for (int j = 0; j < mPofColorPoints[i].size(); j++) {
                cv::Mat pos = vpMPs[mPofColorPoints[i][j]]->GetWorldPos();
                glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
                float static_rate = float(vpMPs[mPofColorPoints[i][j]]->staticObs) / (float(vpMPs[mPofColorPoints[i][j]]->staticObs) + float(vpMPs[mPofColorPoints[i][j]]->dynamicObs));
                glColor4f(colorCharts[i][0], colorCharts[i][1], colorCharts[i][2],static_rate);
            }
            glEnd();
        }
//        for (int i = 0; i < mPofNormalPoints.size(); i++) {
//            glPointSize(mPointSize);
//            glBegin(GL_POINTS);
//            glColor3f(0, 0, 0);
//            cv::Mat pos = vpMPs[mPofNormalPoints[i]]->GetWorldPos();
//            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
//            glEnd();
//        }
    }

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM

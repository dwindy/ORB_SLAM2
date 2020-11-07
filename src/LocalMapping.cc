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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

    LocalMapping::LocalMapping(Map *pMap, const float bMonocular) : mbMonocular(bMonocular),
                                                                    mbResetRequested(false),
                                                                    mbFinishRequested(false), //请求停止当前线程的标志，只是请求，终止取决于mbFinised
                                                                    mbFinished(true),         //判断最红localmapping::run()是否完成的标志
                                                                    mpMap(pMap),
                                                                    mbAbortBA(false),                         //是否停止BA
                                                                    mbStopped(false), mbStopRequested(false), //外部线程调用，true表示外部线程请求停止local mapping
                                                                    mbNotStop(false),                         //true 表示不要停止 localmapping线程，因为要插入关键帧了，需要mbStepped结合使用
                                                                    mbAcceptKeyFrames(true)                   //是否允许接受关键帧，trakcing和local mapping之间的关键帧调度
    {
    }

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

//主函数
void LocalMapping::Run()
{
    //localmapping是否完成
    mbFinished = false;

    while(1)
    {   
        //*Step 1 告诉Tracking线程that，LocalMapping线程正处于繁忙状态
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        //等待处理的关键帧列表不能为空
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            //*Step 2 处理列表中的关键帧，包括计算BoW，更新观测，描述子，共视图，插入到地图等
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            //*Step 3 根据地图点的观测情况剔除质量不好的地图点
            // Check recent MapPoints
            MapPointCulling();

            //*Step 4 当前关键帧与相邻关键帧通过三角化产生“新的”地图点
            // Triangulate new MapPoints
            CreateNewMapPoints();

            //处理完队列最后一个关键帧
            if(!CheckNewKeyFrames())
            {
                //*Step 5 检查并融合当前关键帧与相邻关键帧（二级相邻）中重复的地图点
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            //终止BA标志
            mbAbortBA = false;

            //已经处理完列表，并且闭环检测没有要求停止
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                //*Step 6 当局部地图点中的关键帧大于2个的时候进行局部地图BA
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    //第二个参数mbAbortBA传的指针，所以可以即时停止
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                //*Step 7 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                //该关键帧90%的地图点可以被其他关键帧观测到
                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            //*Step 8 将当前帧插入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        //列表处理完了，终止当前线程
        else if(Stop())
        {
            //没有终止就等待
            //?哪里会让这里没有终止？
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            //确认终止break这个while
            if(CheckFinish())
                break;
        }

        //查看是否有复位线程的请求
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        //结束
        if(CheckFinish())
            break;

        //?似乎是3秒local mapping就run一次？
        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理列表中的关键帧，包括：计算BoW，更新观测，描述子，共视图，插入到地图等
*/
void LocalMapping::ProcessNewKeyFrame()
{
    //*Step 1 从缓冲队列中取出一帧关键帧
    //该关键帧队列是tracking线程向loacalmapping中插入的关键帧
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        //取出然后pop掉
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    //*STep 2 计算该关键帧的特征点BoW信息
    //?trakcing线程没算过嘛？
    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    //*Step 3 当前处理关键帧中有效的地图点，更新normal，描述子信息
    //?tracklocalmap中 和 当前帧新匹配上的地图点 和 当前关键帧 进行关联
    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                //该地图点是否在关键帧中
                //?为什么会有个地图点不在关键帧中，所以是来自于匹配上的帧？
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {   
                    //如果地图点不是来自当前帧的观测，为当前地图点添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    //获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    //更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    //如果关键帧包含了这个地图点，但是地图点却没有包含关键帧的信息
                    //因为这些地图点可能来自双目或者RGBD跟中过程新生成的地图点，或者是CreateNewMapPoints中三角化产生
                    //将上述地图点放入mlpRecentAddedMapPoints，等到后续mappointculling函数检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    //*Step 4 更新关键帧间的共视图
    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    //*Step 5 将关键帧插入到地图里面
    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 *@brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增地图点
 *mlpRecentAddedMapPoints 存储新增的地图点
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    //*Step 1 不同相机不同阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    //*Step 2 遍历新添加的mappoints
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            //*Step 2.1 坏点直接删除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            //*Step 2.2 跟踪到该MapPoints的frame数比预计可观测到该MapPoint的frame数的比例小于25% 删除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            //*Step 2.3 从该点建立开始 到现在已经过了不小于2个关键帧
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            //*Step 2.4 从建立该点开始 已经过了3个关键帧而没有被剔除 
            //好点 不setbadflag，从检测列表里面剔除
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

/*
当前关键帧与相邻关键帧生成新的地图点
*/
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;

    //*Step 1 当前关键帧中前nn个共视度高的关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    //ORB matcher setting 最佳 < 0.6 * 次佳 ?,不查旋转
    ORBmatcher matcher(0.6,false);

    //取出Tcw
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    //光心坐标Ow1
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    //深度验证用得比例 1.5是经验值
    //1.5*1.2
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    //*Step 2 遍历相邻关键帧
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        //因为要处理20帧比较耗时间
        //新的关键要处理则return
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        //邻居关键帧光心Ow2
        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        //基线向量
        cv::Mat vBaseline = Ow2-Ow1;
        //基线距离
        const float baseline = cv::norm(vBaseline);

        //*Step 3 双目的话，运动T要大于相机本身baseline
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            //邻居关键帧深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            //baseline与景深比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        //*Step 4 根据两个关键帧得位姿计算他们之间的F矩阵
        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        //*Step 5 通过BoW对两帧未匹配的特征点快速匹配，用极限约束抑制离群点，生成新的匹配点对。
        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;
        
        //*Step 6 对每对匹配通过三角化生成3D点
        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {   
            //*Step 6.1 取出匹配特征点
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            //双目深度值 单目为-1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            //*Step 6.2 匹配点反投影得到视察角
            //特征点反投影，得到各自相机系下一个非归一化方向向量
            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);
            //把这个点（射线）从相机坐标系转到世界坐标系
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            //向量夹角
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
            //+1是为了初始化成一个大值
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
            //*Step 6.3 双目得到视差角，单目不做特殊操作
            if(bStereo1)
                //如果是双目，用双目的3D点啊基线什么的算出两个相机的视差角，比三角化计算的可靠
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            //*Step 6.4 三角化恢复3D点
            //视差角小用三角法恢复，视差角大用双目恢复
            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                //如果是双目 用视差角更大的双目信息来恢复，直接用3D点反投影
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();
            
            //*Step 6.5 检测3D点在相机前方
            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //*Step 6.6 3D点在当前帧的重投影误差
            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            //单目
            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                //2自由度一个像素对应5.991
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            //双目
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //3D点在另一帧的检测
            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //*Step 6.7 尺度连续性
            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            //两个光心到点的向量的长度的比例 跟 金字塔比例不应该差太多
            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            //*Step 6.8 构造MapPoint
            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            //*Step 6.9 添加地图点属性
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            //*Step 6.10 待检测队列
            //将来用mappointculling检验
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    //*Step1 当前关键帧在共视图中前nn排名的共视关键帧
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
        //第一级相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    //*Step2 存储第一级相邻关键帧和第二级相邻关键帧
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        //当前关键帧ID 不等于 相邻关键帧fusetarget 
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        //标记已经target fuse
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        //二级取5个
        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            //二级相邻关键帧 没有加入target 也不是 当前帧
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    //*Step 3 当前关键帧的地图点 和 每一个相邻关键帧的地图点进行融合
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    //*Step 4 相邻关键帧 与 当前关键帧融合
    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    //*Step 4.1 遍历相邻关键帧，收集地图点存进vpFuseCandidates
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }
    //*Step 4.2 相邻关键帧 跟 当前关键帧 地图点融合
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

    //*Step 5 更新当前关键帧的地图点的描述子，深度，主观测方向等。
    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                //所有找到pMP的关键帧中，最佳的描述子
                pMP->ComputeDistinctiveDescriptors();
                //更新平均观测方向和观测距离
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    //*Step 6 更新当前帧的Mapppoints后更新与其他帧的连接关系。
    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

//有R有T。通过定义算E
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

//设置mbAcceptKeyFrames标志
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

/**
 * @brief 有90%的地图点可以可以被其他关键帧（至少三个）观测到
*/
void LocalMapping::KeyFrameCulling()
{
    //包含的变量：
    //mpCurrentKeyFrame：当前关键帧
    //pKF：当前关键帧CurrentKEyFrame的某个共视关键帧，判断的就是某个共视帧是否冗余
    //vpMapPoints：pKF对应的所有地图点
    //pMP：vpMapPoints中某个地图点
    //observations：所有能观测到pMP的关键帧
    //pKFi:observations中的某个关键帧
    //scaleLeveli:pKFi的金字塔尺度
    //scalelevel:pkf的金字塔尺度

    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    //*Step 1 根据共视图提取所有共视关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        //*Step 2 提取每个共视关键帧的地图点
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        //某个点的被观测次数
        int nObs = 3;
        //观测次数阈值
        const int thObs=nObs;
        //冗余观测点的数目
        int nRedundantObservations=0;
        int nMPs=0;
        //*Step 3 遍历该共视关键帧的所有地图点，判断是否90%以上的地图点能被其他至少3个关键帧观测到
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {   
                        //双目仅仅考虑近处的地图点（基线35倍）
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    //观测到该地图点的相机数目，单目1，双目2
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        //所有可以观测到这个地图点的关键帧的集合
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        //遍历这个集合
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            //相同或者更低的层级看见
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  
        //*Step 4 当前关键帧90%以上的有效地图点被判断为冗余
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM

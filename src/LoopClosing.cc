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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        //*Step 1 查看闭环检测队列mlpLoopKeyFrameQueue中有没有关键帧
        //LoopClosing中的关键帧是localmapping发送过来的
        //insertkeyframe将关键帧插入闭环检测队列mlpLoopKeyFrameQueue
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            //*Step 2 
            // Detect loop candidates and check co-visibility consistency
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        //*Step 1 从队列里面取出一个关键帧，作为当前检测闭环关键帧
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        //设置这个关键帧不要被删除,by其他线程
        mpCurrentKF->SetNotErase();
    }

    //*Step 2 如果距离上次闭环没有太久（10帧），或者map中关键帧总共还没有10帧，不做
    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    //*Step 3 遍历当前闭环检测关键帧的所有连接（>15个共视点）关键帧，计算当前关键帧与每个共视关键帧的bow相似度得分，并且得到最低得分minScore
    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    //*Step 4 在所有关键帧中找出闭环候选帧（不和当前闭环检测关键帧连接）
    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    //*Step 5 在候选帧中检测具有连续性的候选帧
    //1。每个候选关键帧将与自己相连的关键帧构成一个“子候选组spCandidateGroup"，vpCandidateKFs-->spCandidateGroup
    //2。检测“子候选组”中每一个关键帧是否存在于上次“连续组”，如果存在则nCurrentConsistency是上次连续组的连续性+1，并将该“子候选组”和连续性成对放入“当前连续组vCurrentConsistentGroups”
    //3。如果nCurrentConsistency大于等于3，那么该“子候选组”代表的候选帧过关，进入mvpEnoughConsistentCandidates

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a co-visibility group (keyframes connected to the loop candidate in the co-visibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;

    //Size是上次闭环检测到的子连续组vector size，bool表示当前候选组中是否有和该组相同的一个关键帧
    //即表示当前候选组和上次的某个子侯选组时候包含同一个关键帧
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    //*Step 5.1 遍历闭环候选关键帧
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];
        //*Step 5.2 自己以及相连的关键帧构成一个子候选组spCandidateGroup
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        //*Step 5.3 遍历 前一次 闭环检测到的“子连续组”
        //上次闭环检测到连续组std::vector<MConsistentGroup> mvConsistentGroups
        //其中ConsistentGroup定义为 typedef pair<set<KeyFrame*>, int> ConsistentGroup
        //其中ConsistentGroup.first 对应每个连续组中的关键帧，.second为每个连续组的序号（强度）
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            //取出其中一个连续组的关键帧集合set
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            //*Step 5.4 遍历每个当前的“子候选组”，检测子候选组中的每一个关键帧 在 子连续组 中是否存在
            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            //*Step 5.5 如果当前子候选组 和 之前的子连续组 存在连续，检查连续条件
            if(bConsistent)
            {
                //子持续组.second是已经连续的几次。
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                //如果上述连续关系还没有记录到vector<bool> spCurrentConsistentGroups里面，记录一下
                //如果上述连续关系还没有记录到vector<bool> vbConsistentGroup，记录一下
                if(!vbConsistentGroup[iG])
                {
                    //创建一个新的连续组cg，包含当前子候选组spCandidateGroup和继承+1来的连续性
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    //存入当前连续组的vector中
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoids to include the same group more than once
                }
                //如果当前候选组的连续性 大于 阈值3 并且 足够连续的标记false
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    //最后插入的还是闭环候选帧
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoids to insert the same candidate more than once
                    //Note because 一个闭环候选关键帧(所组成的候选组)，可能跟之前的子连续组vector里的多个子连续租产生连续性
                }
            }
        }

        //*Step 5.6 如果该 子候选组 的所有关键帧 都和上次闭环不连续，即vCurrentConsistentGroups没有添加任何连续关系
        //创建一个新的连续组cg，包含当前子候选组spCandidateGroup和连续性0
        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    //当前连续性组取代上次连续性组,恩make sense 上一次的子连续组要是这次没连接上，就抛弃了，只有连续观测到连续的能保存到
    //Update Co-visibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;

    //把当前闭环检测关键帧插入到关键帧数据库中
    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);
    //Question this hasn't been added before it comes to loopClosure thread?

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }
    //执行不到这里
    mpCurrentKF->SetErase();
    return false;
}

/**
 * @brief 计算当前关键帧的上一步闭环候选帧的sim3变换
 * 1.遍历闭环候选帧集，筛选出和当前帧匹配的特征点数>20的候选帧集合，并为每一个候选帧构造一个sim3solver
 * 2.对每个候选帧率sim3solver迭代匹配，直到有一个候选帧匹配成功或者全部失败
 * 3.取出闭环匹配上关键帧的相连关键帧，得到他们的mappoints放入mvploopmappoints
 * 4.将闭环匹配上关键帧以及相连关键帧的mappoints投影到当前关键帧进行投影匹配
 * 5.判断当前帧与检测出的所有闭环关键帧是否有足够多的mappoints匹配
 * 6.清空mvpenoughconsistentcandidates
 */
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;// using vector to store each sim3Solver of each ConsistentCandidate
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    //*Step 1. 遍历闭环候选帧集 筛选匹配特征点>20个的帧率 构建sim3solver
    for(int i=0; i<nInitialCandidates; i++)
    {
        //*Step 1.1 取一个候选
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        //避免localMapping线程keyFrameCulling把它当作冗余剔除了
        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        //*Step 1.2 将当前帧mpCurrentKF 与 闭环候选真pKF匹配
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            //*Step 1.3 构造sim3solver
            //mbFixScale是单目fix，7自由度，双目6自由度。
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    //*Step 2 对于每一个候选帧 用sim3solver迭代匹配，直到一个成功或者全部失败
    // Perform alternatively RANSAC iterations for each candidate
    // until one is successful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;//each the best solution or not

            //*Step 2.1 取回sim3solver 开始迭代
            Sim3Solver* pSolver = vpSim3Solvers[i];
            //5次迭代，拿到candidate pKF到当前帧mpCurrentKF的Sim3变换T12
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            //如果有Scm，继续优化
            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                //取出内点
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                //*Step 2.2 通过求解sim3变换，引导关键帧匹配，弥补Step1中的漏匹配（bow漏的）
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                //*Step 2.3 通过新的匹配 优化sim3 只要有一个候选帧通过sim3的求解与优化，就跳出停止对其他候选帧的判断
                //opencv的Mat矩阵转换成Eigen的Matrix
                //gScm: 候选关键帧到当前帧的sim3变换
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is successful stop Ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true; // do not check next candidate keyframe
                    mpMatchedKF = pKF;
                    //gSmw: sim3 from world to candidate keyframe m. scale = because the candidate keyframe pKF is under world coordination as well, the scale is the same?
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    mg2oScw = gScm*gSmw;//get sim3 from world to current keyframe
                    mScw = Converter::toCvMat(mg2oScw);//record mScw
                    mvpCurrentMatchedPoints = vpMapPointMatches;//record matched mapPoints (modified by optimizeSim3 function above
                    break;
                }
            }
        }
    }

    if(!bMatch)//no good candidate keyframe found, erase everything
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();//this keyframe no need to join loop-closure in the future
        return false;
    }

    //Step 3 : 取出与当前帧闭环匹配上的关键帧及其共视关键帧，以及这些共视关键帧的地图点
    //Retrieve MapPoints seen in Loop Keyframe and neighbors
    //mpMatchedKF and mpMatchedKF's co-visible keyframe into vpLoopConnectedKFs,
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    //put vpLoopConnectedKF's mapPoint into mvpLoopMapPoints
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId) {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF = mpCurrentKF->mnId;//Mark so not repeat add
                }
            }
        }
    }

    //Step 4: 将闭环关键帧及其共视关键帧的所有地图点投影到当前关键帧进行投影匹配
    //根据投影查找更多的匹配（成功的闭环需要足够多的匹配
    //根据sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，搜索新的匹配
    //mvpCurrentMatchedPoints是经过前面searchbySim3得到的匹配对，这里就略过不再匹配了
    //Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    //Step 5: 统计当前帧与检测出的所有闭环关键帧的匹配地图点数目，如果超过40个说明成功
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])//if ture
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        //if good, only keep current matched candidate keyframe, erase other candidate
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

/**
 * @brief correct loop
 * 1. 通过求解的sim3以及相对姿态关系，调整与当前相连的关键帧的位姿，以及这些关键帧观测到的地图点（相连关键帧－－－当前帧）
 * 2. 将闭环帧以及闭环帧相连的关键帧的地图点和当前帧相连的关键帧的点进行匹配（当前帧+相连关键帧－－－闭环帧+相连关键帧）
 * 3. 通过MapPoints的匹配关系更新这些关键帧之间的项链关系，即更新covisibility graph
 * 4. 对Essntial graph(pose graph）进行优化，Mappoint的位置根据优化后的位姿做出相对应的调整
 * 5. 创建线程进行全局Bundle Adjustment
 */
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;
    //Step 0: 结束局部地图线程，全局BA，为闭环矫正做准备
    //Step 1: 根据共视关系更新当前帧与其他关键帧之间的连接
    //Step 2: 通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及他们的MapPoints.
    //Step 3: 检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突中的MapPoints进行替换或填补
    //Step 4: 通过将闭环时相连关键帧的mvpLoopMApPoints投影到这些关键帧中，进行MapPoints检查与替换
    //Step 5: 更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints#融合而新得到的连接关系
    //Step 6: 进行EssentialGraph优化，loopConnection是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系。
    //Step 7: 添加当前帧与闭环匹配帧之间的边（这个连接关系不优化
    //Step 8: 新建一个线程用于全局BA

    //g2oSic: 当前关键帧mpCurrentKF到其共视关键帧pKFi的Sim3相对变换
    //mg2oScw: 世界坐标系到当前关键帧的sim3变换
    //g2oCorrectedSiw: 世界坐标系到当前关键帧共视关键帧的sim3变换

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    //Step 0: 结束局部地图线程，全局BA，为闭环矫正做准备
    //avoid localMapping thread insertKeyFrame function inserts new keyframe
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //Step 1: 根据共视关系更新当前帧与其他关键帧之间的连接
    //因为之前闭环检测，计算Sim3中改变了该关键帧的地图点，所以需要更新
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    //Step 2: 通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及他们的MapPoints.
    //当前帧与世界坐标系之间的Sim3变换在computeSim3函数中已经确定并且优化
    //通过相对位姿关系，可以确定这些相连关键帧与世界坐标系之间的sim3变换

    //取出当前关键帧的共视关键帧，和当前关键帧组成“当前连接关键帧组”
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);
    //map<keyframe, sim3>
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;//首先将当前关键帧和对应sim3存进去.这个被认为是准确的，优化以此为基础
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    //action for mapPoints
    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        //Step 2.1 通过mg2oScw（认为是准确的）来进行位姿传播，得到当前关键帧的共视关键帧的世界系下sim3位姿（还没修正)
        //遍历“当前连接关键帧组”
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                //get Tic from mpCurrentKF to co-visible pKFi
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                //get g2oSic: sim3 from mpCurrentKF to its co-visible pKFi
                //the scale is believed unchanged in local small co-visible region. set scale 1.0
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;//QUESTION? if the scale is not 1, the scale is multiplied ?
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;//this g2oSic is get from mg2oScw (which is optimized from compute sim3 function earlier), so we believe this g2oSic is correct as well
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;//this is get directly from existing poses, to be corrected.

            //NOTE: the scale (s) of g2oScw has been optimized before, the Tic is believed scale(1) unchanged. so the whole scaled is corrected.
            //NOTE: the scale of g2oSiw is directly get from existing pose, which scale(1) is not corrected yet.
            //NOTE: the error is the scale difference between g2oScw and g2oSiw.
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        //Step 2.2: 用前面得到的矫正后的当前关键帧的共视关键帧的位姿，修正这些关键帧的地图点
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)//Mark. not repeat action in the future
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                //get MapPoint pose under world (P3Dw), transform (by map function of uncorrected g2oSiw) it to i-th connected keyframe,
                //then transform(by map func of corrected g2oCorrectedSwi) it back to world again.
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));//map function return s*(r*xyz) + t;
                //Question? using Tiw is same to gwoSiw?

                //update the mapPoint's location change
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            //Step 2.3: 将共视关键帧的sim3转换为SE3，根据更新的sim3，更新关键帧的位姿
            //其实就是把刚才临时存入keyframeandpose的pose应用到每一个共视关键帧
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();//automatically unify the rotation. and get scale.
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        //Step 3: 检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突中的MapPoints进行替换或填补
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];//this is optimized after computesim3 function,
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //Step 4: 通过将闭环时相连关键帧的mvpLoopMApPoints投影到这些关键帧中，进行MapPoints检查与替换
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    //Step 5: 更新当前关键帧之间的共视相连关系，得到因闭环时mappoint融合而心得到连接关系。
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;
    //Step 5.1: traverse current keyframe's connected keyframes 当前帧相连关键帧组(一级相连)
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        //Step 5.2 : get Co-visible Keyframes of current pKFi(2级相连)
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        //Step 5.3 : 更新一级相连关键帧的连接关系（会把当前关键帧添加进去，因为地图点已经更新or替换了）
        pKFi->UpdateConnections();
        //Step 5.4: 取出该帧更新后的连接关系
        //QUESTION? GetVectorCovisibleKeyFrames and GetConnectedKeyFrames?
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        //Step 5.5: 从更新后的连接关系中erase闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        //Step 5.6: 从更新后的连接关系中erase闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
        //NOTE 等于对于当前帧相连关键帧组中的某个关键帧，它在闭环之前的相连关系和它跟当前关键帧的关系都被删除了，剩下的存粹是闭环之后新添加的相连关系
    }

    // Optimize graph
    //Step 6: 进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    //Step 7 添加当前帧与闭环匹配帧之间的边（这个关系不优化
    //ERROR 这两句代码应该在otimizeessentialgraph之前，因为该函数step4.2优化了这个关系
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    //Step 8: create a new thread for global BA
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();
    //QUESTION why release local mapping thread?

    mLastLoopKFid = mpCurrentKF->mnId;   
}

/**
 * @brief 将闭环相连关键帧组mvpLoopMapPoints 投影到当前关键帧组中，进行匹配，新增或者替换当前关键帧组中KF的地图点
 * 因为闭环相连关键帧组mvpLoopMapPoints在地图中存在的时间比较久，经历多次优化，相对更准确。
而当前关键帧中组中的关键帧是最近计算的，可能有累积误差。
 //QUESTION? 应该说当前关键帧组中的关键帧累计误差更多吧。
 //QUESTION? 但是刚才不是有correctedsim3了吗？ -> 所以这里更新的是这些keyframe下的mappoints
 * @param CorrectedPosesMap
 */
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    //Step 1 traverse corrected currentkeyframe at its connect keyframes
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);
        //Step 2 Project mvpLoopMapPoints (from loop-closure candidate keyframe and its connected keyframes) into pKF, check and fuse
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        //Step 3 apply the mappoint fusions
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

//input name loopKF, actually given current keyframe ID
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    //Step 1 carry out gloabl BA
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            //Only contains the first keyframe
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());
            //Step 2 traverse gloabl map's spanning tree keyframe
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                //traverse child keyframe
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {//get old T_child_father
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        //update child's TcwGBA by adding father's GBA
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;//mark avoid repeat action
                        //NOTE each child node is updated except the root node

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
            //Step 3 traver each mappoint and update it
            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                //not every mappoint joined BA, but mappoint need update after BA
                //if this mappoint joned BA, just use GBA result
                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                { //If this mappoint didn't join GBA, using referenc keyframe GBA to update
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)//if referamce keyframe didn't join GBA. ignore
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM

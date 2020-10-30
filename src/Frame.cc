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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //分两个线程提取
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    //特征点匹配，计算深度放进mvDpeth
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef,
             const float &bf,                                                                                      //baseline * f
             const float &thDepth)                                                                                 //区分远近点的深度阈值
    : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)), //单目没有右相机
      mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}
/**
 * @brief 单目帧构造函数
 * 
 * @param[in] imGray //灰度图
 * @param[in] timeStamp //时间戳
 * @param[in & out] extractor //ORB特征点提取器的句柄
 * @param[in] voc //ORB字典句柄
 * @param[in] K //相机内参矩阵
 * @param[in] bf //baseline*f
 * @param[int]thDepth //区分远近点的深度阈值
 */
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef,
             const float &bf,
             const float &thDepth)
    : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
      mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    //Step 1 帧ID增加
    mnId=nNextId++;

    // Scale Level Info
    //Step 2 图像金字塔参数
    mnScaleLevels = mpORBextractorLeft->GetLevels(); //层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor(); //缩放因子
    mfLogScaleFactor = log(mfScaleFactor); //缩放因子的自然数对数
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors(); //缩放因子 again?
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors(); //缩放因子的倒数
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares(); //sigma^2
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares(); //sigma^2倒数

    // ORB extraction
    //Step 3 提取特征点 0 左图 1 右图
    ExtractORB(0,imGray); //提取ORB特征

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    //Step 4 OpenCV的去畸变函数
    UndistortKeyPoints();

    // Set no stereo information
    //单目，右边图像的对应点和深度都赋-1
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    //初始化本帧的地图点-给null
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    //初始化outlier，给false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    //标志位，只在第一帧或者相机标定参数变化后执行
    if(mbInitialComputations)
    {
        //计算去畸变图像的边界
        ComputeImageBounds(imGray);

        //一个图像像素相当于多少个图像网格列（grid cols）/ (col length)
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        //一个图像像素相当于多少个图像网格行（grid rows）/ (row height)
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    //计算baseline，单目用不到其实
    mb = mbf/fx;

    //把特征点分配到网格中，默认64/48
    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    //预分配，总特征点x0.5分配到各个网格
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    //对mGrid这个二位数组的每一个vector元素遍历，预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        //输入特征点，输出网格坐标。
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            //把特征点索引值放进网格中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        /*
        仿函数
        */
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

/**
 * @brief 地图点是否在事业内
 * 计算该地图点的世界坐标
 * 1 该点在当前相机的深度>0
 * 2 投影到像素平面，要在范围内
 * 3 mappoint到相机中心的距离，是否在尺度变化的距离内
 * 4 当前视角和平均观测方向的夹角的余弦值，要大于阈值
 * 地图点到光心的距离来预测一个尺度（仿造特征点金字塔层级）
 * 记录一些参数
 * @param[in] pMP 当前地图点
 * @param[in] viewingCosLimit 余弦阈值
 * @return true false
*/
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    //*Step 获得地图点世界坐标
    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    //*Step 1 深度判断
    // Check positive depth
    if(PcZ<0.0f)
        return false;

    //*Step 2 成像平面内判断
    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    //*Step 3 地图点到相机中心的距离 在尺度变化内
    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    //mOW 当前相机光心在世界坐标系下的坐标
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

    //*Step 4 相机视角 和 地图点平均观测方向 的夹角的余弦
   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal(); //在updatenormanddepth中更新

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    //*Step 根据地图点到光心的距离预测一个尺度
    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);
    //*Step 一些信息
    // Data used by the tracking
    pMP->mbTrackInView = true;
    //像素横坐标 左图
    pMP->mTrackProjX = u;
    //bf/z视察 即得到像素右视图坐标
    pMP->mTrackProjXR = u - mbf*invz;
    //像素纵坐标 左图
    pMP->mTrackProjY = v;
    //预测到的层级
    pMP->mnTrackScaleLevel= nPredictedLevel;
    //包留视角和平均观测方向的夹角
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    /*
    计算x,y半径r上下左右边界所在行列对应的网格index
    例，从mnMinX开始 每mfGridElementWidthInv个像素 就是一个cell, 找到x-r所在的cell
    */
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    //for what?
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);
    //Step 2 遍历所有cell，寻找匹配上的特征点
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            //在frame函数中特征点已经被丢到mGrid里面
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                //通过vCell存的索引取到特征点
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    //why check maxlevel >=0?
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
                //在r半径范围内
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if (mBowVec.empty())
    {
        //将描述子转换成DBoW需要的格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc, //当前描述子Vector
                                   mBowVec,      //输出词袋向量，记录的是单词的id以及对应权重TF-IDF值
                                   mFeatVec,     //输出，记录node id以及对应的图像feature对应的索引
                                   4);           //节点向前数的层数
    }
}

/**
 * @brief 用内参对特征去畸变，结果保存在mvKeysUn中
 */
void Frame::UndistortKeyPoints()
{
    //mDistCoef存储了opencv指定格式的去畸变参数(k1,k2,p1,p2,k3)
    //第一个参数为0一般意味着没有畸变
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    //reshape(int cn, int rows=0),cn为通道数，rows=0表示不修改rows数
    //为了使用opencv去畸变函数，需要将原矩阵改成2通道（对应x,y）
    mat=mat.reshape(2);
    cv::undistortPoints(mat, //输入
    mat, //输出
    mK, //相机内参
    mDistCoef, //畸变参数
    cv::Mat(), //空矩阵即可
    mK); //新内参矩阵（加入了畸变参数进去）
    //恢复到单通道
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N); //给vector大小
    for(int i=0; i<N; i++)
    {
        //只更新kp.pt和坐标（包留了特征点的其他信息）
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    //把原图像的四个顶点去畸变
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    /*
    1.行特征点统计，统计right上每一行上的特征点集，便于行/极线搜索。
    2.粗匹配，在left第i行的orb特征点pi,在right第i行上搜索对应qi。
    3.精确匹配，在qi半径r范围内进行块匹配（归一化SAD）
    4.亚像素精度优化
    5.最优时差/深度选择，通过WTA获得最佳匹配
    6.删除outliers，块匹配相似度阈值判断，归一化SAD最小，并不一定是正确匹配。（光照纹理灯）
    输出：稀疏特征点视察图/深度图mvDepth匹配结果mvuRight
    */

    //右图特征点的索引和depth
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    //orb相似度阈值
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    //第0层金字塔的rows
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    //二维vector存储每一行特征点的列坐标？小六这里写错了吧？似乎只是把特征点索引iR塞进row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    //Step 1. 行特征点统计，因为有金字塔，一个特征点可能存在于多行
    for(int iR=0; iR<Nr; iR++)
    {
        //取出特征点row行值
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        //加上一个r浮动值-跟尺度相关
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        //似乎只是把特征点索引iR塞进row table
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    //Step 2->3 粗匹配and精匹配
    // Set limits for search
    //不会在整行里面找，只在pi.x左右找
    //Z = f * b / d 
    //Depth = focal * baseline / disparity | disparity = Ul-Ur
    //maxd = baseline * focal / minZ
    //minD = baseline * focal / maxZ
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    //精匹配用来保存相似度和索引
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        //NOTICE,the row table contains index of right feature
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        //极线搜索的范围U range, x range, col range
        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        //初始化orb descriptor distance and INDEX
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];
            
            //只在金字塔相邻层找
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;
            //只在极线的一定范围内
            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        //Step 3. 精确匹配
        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            //window width
            const int w = 5;
            //提取左图特征点[scaleduL,scaledvL]为中心半径w范围的图像块patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            //亮度值归一化
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            //滑动的索引量range(-L,L)
            int bestincR = 0;
            //滑动大小 slide size
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            //计算滑动窗口的滑动范围的边界，因为是块匹配，需要算上图像块的尺寸
            //列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
            //列方向终点 endu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            //似乎L=5 W=5只管了右边是否越界
            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                /* 举个栗子，左边特征点P_l(12,13)，上面取得IL块为（7:18,8:19）。
                右边特征点为P_r(12,14)，取第一个块为(7:18, 4:15)，最后一个块为(7:18,14:25)。就是以14为中心，x坐标从4到25滑动。
                */
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                //sad 计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }
                //存入相似度
                vDists[L+incR] = dist;
            }

            //这个越界判断的目的是？
            if(bestincR==-L || bestincR==L)
                continue;

            //Step 4 亚像素插值
            //直接套了opencv sgbm插值公式
            //disparity d_* = d - ( (d+ - d-) / 2*(d+ + d- - 2*d) )
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            //修正量应该在[-1,1]之间
            if(deltaR<-1 || deltaR>1)
                continue;

            //修正量加入到scaleduR0中
            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);
            
            //保存深度和视察（不保存匹配结果？）
            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    //Step 6. 去除outlier
    //判断条件 norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    //sort了，第一个<thDist则都小于treshold
    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM

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

////Copy Constructor
//    Frame::Frame(const Frame &frame)
//            :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
//             mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
//             mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
//             mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
//             mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
//             mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
//             mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
//             mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
//             mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
//             mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
//             mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
//    {
//        for(int i=0;i<FRAME_GRID_COLS;i++)
//            for(int j=0; j<FRAME_GRID_ROWS; j++)
//                mGrid[i][j]=frame.mGrid[i][j];
//
//        if(!frame.mTcw.empty())
//            SetPose(frame.mTcw);
//    }

    ///TODO - update the Frame Copy Constructor
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
             mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
             //added LiDAR and Segment modules
             mTcamlid(frame.mTcamlid), mvORBAttributions(frame.mvORBAttributions),
             mSegmentsInfo(frame.mSegmentsInfo), mvLiDARPoints(frame.mvLiDARPoints),
             mvPlanes(frame.mvPlanes)
    {
        for(int i=0;i<FRAME_GRID_COLS;i++)
            for(int j=0; j<FRAME_GRID_ROWS; j++)
                mGrid[i][j]=frame.mGrid[i][j];

        if(!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }


    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
                 ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
              mpReferenceKF(static_cast<KeyFrame *>(NULL)) {
        // Frame ID
        mnId = nNextId++;

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
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight);
        threadLeft.join();
        threadRight.join();

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        //特征点匹配，计算深度放进mvDpeth
        ComputeStereoMatches();

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);


        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imLeft);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    ///Added Module-------------------------------------
    Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
                 ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
                 ///Added----------
                 const vector<vector<float>> &LiDARRaw, const string &ImageFileNAme, const string &SegFileAddress,
                 const cv::Mat &mTcamlidInput)
                 ///---------------
            : mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
              mpReferenceKF(static_cast<KeyFrame *>(NULL)) {
        // Frame ID
        mnId = nNextId++;

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
        thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
        thread threadRight(&Frame::ExtractORB, this, 1, imRight);
        threadLeft.join();
        threadRight.join();

        N = mvKeys.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ///Added modes--------------------
        mTcamlid = cv::Mat::eye(4,4,CV_64F);
        mTcamlidInput.copyTo(mTcamlid);
        mTcamlid.convertTo(mTcamlid, CV_32F);
        ///init ORB attributions
        mvORBAttributions.resize(N, nullptr); // Resize and initialize with nullptr
        for (int i = 0; i < N; ++i) {
            mvORBAttributions[i] = new mORBAttribution(); // Create new mORBAttribution objects
        }
        for(int i=0; i<N; i++){
            mvORBAttributions[i]->keyPt = &mvKeysUn[i];
            mvORBAttributions[i]->orbID = i;
        }
        ///read image segmentation
        readSegmentsInfo(mSegmentsInfo, SegFileAddress);
        vector<cv::Mat*> segmentImages(mSegmentsInfo.size());
        cout<<"segmentImage number "<<segmentImages.size()<<endl;
        readSegmentImages(segmentImages, SegFileAddress);

        ///Pair ORB and segments
        alignORBSegments(segmentImages);

        ///Read LiDAR construct LiDAR member
        processLiDARPts(LiDARRaw);
        ProjectLiDARtoCamtoImage_KITTI(imLeft.cols,imLeft.rows);

        ///Pair LiDAR and Segments
        groupLiDARandSegment(segmentImages);
        planeFitEachSegment();
        cout<<"mvPlanes num "<<mvPlanes.size()<<endl;
        ORBdetphFromPlane(mvPlanes, mSegmentsInfo, segmentImages);
        ORBdetphFromSegments(mvLiDARPoints, mSegmentsInfo, segmentImages, imLeft);
        ComputeStereoFromFusion(mvORBAttributions);
        //MarkStereoFromFusion(mvORBAttributions);
        ///---------------------------

        //特征点匹配，计算深度放进mvDpeth
        //ComputeStereoMatches();
        ComputeStereoMatches(mvORBAttributions);

        /// Deallocate memory for each cv::Mat object
        for (size_t i = 0; i < segmentImages.size(); ++i) {
            delete segmentImages[i];
        }
        /// Clear the vector
        segmentImages.clear();
        for (int i=0;i<mvLiDARPoints.size();i++){
            delete mvLiDARPoints[i];
        }
        mvLiDARPoints.clear();
        ///-------------------------------------------------

        ///Test code
        for(int i=0;i<mvDepth.size();i++){
            if(mvDepth[i]>-1 && mvORBAttributions[i]->depthSource==-1){
                cout<<" i "<<i<<" depth "<<mvDepth[i]<<" mvuRight "<<mvuRight[i]<<" depthsour "<<mvORBAttributions[i]->depthSource<<endl;
            }
        }
        ///Test code
        ///---------------

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvbOutlier = vector<bool>(N, false);


        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imLeft);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }
    ///-------------------------------------------------

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

    //TODO check if the added vector<*> member has been initialized
    ///Added --- for rgbd
    //add lidar, image segmentation
//    Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
//                 ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef,
//                 const float &bf, const float &thDepth,
//                 ///Added
//                 const vector<vector<float>> &LiDARRaw, const string &ImageFileNAme, const string &SegFileAddress,
//                 const cv::Mat &TCamLid)
//                 ///------
//            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
//              mpORBextractorRight(static_cast<ORBextractor *>(NULL)), //单目没有右相机
//              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth) {
//        // Frame ID
//        mnId = nNextId++;
//        // Scale Level Info
//        mnScaleLevels = mpORBextractorLeft->GetLevels();
//        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
//        mfLogScaleFactor = log(mfScaleFactor);
//        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
//        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
//        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
//        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
//        // ORB extraction
//        ExtractORB(0, imGray);
//        N = mvKeys.size();
//        if (mvKeys.empty())
//            return;
//        UndistortKeyPoints();
//
//        ///Added modes--------------------
//        ///init ORB attributions
//        mvORBAttributions.resize(N);
//        for(int i=0; i<N; i++){
//            mvORBAttributions[i]->keyPt = &mvKeysUn[i];
//            mvORBAttributions[i]->orbID = i;
//        }
//        ///read image segmentation
//        readSegmentsInfo(mSegmentsInfo, SegFileAddress);
//        vector<cv::Mat*> segmentImages(mSegmentsInfo.size());
//        cout<<"segmentImage number "<<segmentImages.size()<<endl;
//        readSegmentImages(segmentImages, SegFileAddress);
//        ///Read LiDAR construct LiDAR member
//        processLiDARPts(LiDARRaw);
//        ProjectLiDARtoCamtoImage_KITTI(imGray.cols,imGray.rows);
//        ///Pair LiDAR and Segments
//        groupLiDARandSegment(segmentImages);
//        planeFitEachSegment();
//        cout<<"mvPlanes num "<<mvPlanes.size()<<endl;
//        ORBdetphFromPlane(mvPlanes, mSegmentsInfo, segmentImages);
//        ORBdetphFromSegments(mvLiDARPoints, mSegmentsInfo, segmentImages, imGray);
//        //ComputeStereoFromFusion(mvORBAttributions);
//        //MarkStereoFromFusion(mvORBAttributions);
//        ///---------------------------
//        ComputeStereoFromRGBD(imDepth);
//        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
//        mvbOutlier = vector<bool>(N, false);
//        // This is done only for the first Frame (or after a change in the calibration)
//        if (mbInitialComputations) {
//            ComputeImageBounds(imGray);
//            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
//            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);
//            fx = K.at<float>(0, 0);
//            fy = K.at<float>(1, 1);
//            cx = K.at<float>(0, 2);
//            cy = K.at<float>(1, 2);
//            invfx = 1.0f / fx;
//            invfy = 1.0f / fy;
//            mbInitialComputations = false;
//        }
//        mb = mbf / fx;
//        AssignFeaturesToGrid();
//    }

    void Frame::readSegmentImages(vector<cv::Mat *> &segmentImages, const string &imgFileAddress) {
        int num = mSegmentsInfo.size();
        string imgName;
        for (int i = 1; i < num + 1; i++) {
            if (i < 10)
                imgName = imgFileAddress.substr(0, imgFileAddress.length() - 4) + "-0" + to_string(i) + ".png";
            else
                imgName = imgFileAddress.substr(0, imgFileAddress.length() - 4) + "-" + to_string(i) + ".png";
            cv::Mat *image = new cv::Mat(cv::imread(imgName, cv::IMREAD_UNCHANGED));
            segmentImages[i - 1] = image;
        }
    }
    ///-------------------------------

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
        //提取ORB特征
        ExtractORB(0,imGray);

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
        //在updatenormanddepth中更新
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        //*Step 根据地图点到光心的距离预测一个尺度
        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

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
        //*Step 2 遍历所有cell，寻找匹配上的特征点
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
        if(mBowVec.empty())
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
        //给vector大小
        mvKeysUn.resize(N);
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

    void Frame::ComputeStereoMatches() {
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
        mvuRight = vector<float>(N, -1.0f);
        mvDepth = vector<float>(N, -1.0f);

        //orb相似度阈值
        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

        //第0层金字塔的rows
        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

        //Assign keypoints to row table
        //二维vector存储每一行特征点的列坐标？小六这里写错了吧？似乎只是把特征点索引iR塞进row table
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

        for (int i = 0; i < nRows; i++)
            vRowIndices[i].reserve(200);

        const int Nr = mvKeysRight.size();

        //Step 1. 行特征点统计，因为有金字塔，一个特征点可能存在于多行
        for (int iR = 0; iR < Nr; iR++) {
            //取出特征点row行值
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            //加上一个r浮动值-跟尺度相关
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);

            //似乎只是把特征点索引iR塞进row table
            for (int yi = minr; yi <= maxr; yi++)
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
        const float maxD = mbf / minZ;

        // For each left keypoint search a match in the right image
        //精匹配用来保存相似度和索引
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);

        for (int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            //NOTICE,the row table contains index of right feature
            const vector<size_t> &vCandidates = vRowIndices[vL];

            if (vCandidates.empty())
                continue;

            //极线搜索的范围U range, x range, col range
            const float minU = uL - maxD;
            const float maxU = uL - minD;

            if (maxU < 0)
                continue;

            //初始化orb descriptor distance and INDEX
            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = mDescriptors.row(iL);

            // Compare descriptor to right keypoints
            for (size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];

                //只在金字塔相邻层找
                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;

                const float &uR = kpR.pt.x;
                //只在极线的一定范围内
                if (uR >= minU && uR <= maxU) {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }

            //Step 3. 精确匹配
            // Subpixel match by correlation
            if (bestDist < thOrbDist) {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);

                // sliding window search
                //window width
                const int w = 5;
                //提取左图特征点[scaleduL,scaledvL]为中心半径w范围的图像块patch
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                     scaledvL + w + 1).colRange(
                        scaleduL - w, scaleduL + w + 1);
                IL.convertTo(IL, CV_32F);
                //亮度值归一化
                IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

                int bestDist = INT_MAX;
                //滑动的索引量range(-L,L)
                int bestincR = 0;
                //滑动大小 slide size
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);

                //计算滑动窗口的滑动范围的边界，因为是块匹配，需要算上图像块的尺寸
                //列方向起点 iniu = r0 + 最大窗口滑动范围 - 图像块尺寸
                //列方向终点 endu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
                //似乎L=5 W=5只管了右边是否越界
                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                    continue;

                for (int incR = -L; incR <= +L; incR++) {
                    /* 举个栗子，左边特征点P_l(12,13)，上面取得IL块为（7:18,8:19）。
                    右边特征点为P_r(12,14)，取第一个块为(7:18, 4:15)，最后一个块为(7:18,14:25)。就是以14为中心，x坐标从4到25滑动。
                    */
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                          scaledvL + w + 1).colRange(
                            scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                    IR.convertTo(IR, CV_32F);
                    IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                    //sad 计算
                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestincR = incR;
                    }
                    //存入相似度
                    vDists[L + incR] = dist;
                }

                //?这个越界判断的目的是
                if (bestincR == -L || bestincR == L)
                    continue;

                //Step 4 亚像素插值
                //直接套了opencv sgbm插值公式
                //disparity d_* = d - ( (d+ - d-) / 2*(d+ + d- - 2*d) )
                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];

                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

                //修正量应该在[-1,1]之间
                if (deltaR < -1 || deltaR > 1)
                    continue;

                //修正量加入到scaleduR0中
                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float) scaleduR0 + (float) bestincR + deltaR);

                float disparity = (uL - bestuR);

                //保存深度和视察（不保存匹配结果？）
                if (disparity >= minD && disparity < maxD) {
                    if (disparity <= 0) {
                        disparity = 0.01;
                        bestuR = uL - 0.01;
                    }
                    mvDepth[iL] = mbf / disparity;
                    mvuRight[iL] = bestuR;
                    vDistIdx.push_back(pair<int, int>(bestDist, iL));
                }
            }
        }

        //Step 6. 去除outlier
        //判断条件 norm_sad > 1.5 * 1.4 * median
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        //sort了，第一个<thDist则都小于treshold
        for (int i = vDistIdx.size() - 1; i >= 0; i--) {
            if (vDistIdx[i].first < thDist)
                break;
            else {
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }


    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
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

    void Frame::readSegmentsInfo(vector<SegmentInfo*> &segmentsInfoIn, const string &TXTFileAddress) {
        ifstream infile(TXTFileAddress);
        if (!infile.is_open()) {
            std::cerr << "Error: Unable to open segment info file " << TXTFileAddress << std::endl;
            return;
        }
        std::string line;
        while (std::getline(infile, line)) {
            int id, category_id;
            bool isThing;
            double area;
            std::stringstream ss(line);
            std::string isthing_str;
            string str1,str2,str3,str4,str5,str6,str7;
            ss>>str1>>id>>str2>>str3>>isthing_str>>str5>>category_id>>str6>>str7>>area;
            if (isthing_str == "True,")
                isThing = true;
            else
                isThing = false;
            auto *newseg = new SegmentInfo(id, isThing, category_id, area);
            segmentsInfoIn.push_back(newseg);
        }
        infile.close();
    }

    //from LiDAR raw data to lidar struct types.
    void Frame::processLiDARPts(const vector<vector<float>> & lidarRaw) {
        mvLiDARPoints.resize(lidarRaw.size());
        for (int i = 0; i < lidarRaw.size(); i++) {
            mLiDARPoint *newLidarpt = new mLiDARPoint(lidarRaw[i], i);
            mvLiDARPoints[i] = newLidarpt;
        }
    }

    void Frame::ProjectLiDARtoCamtoImage_KITTI(int cols, int rows) {
        for(int i=0;i<mvLiDARPoints.size();i++){
            //to camera
            cv::Mat P_lidar(4, 1, CV_32F);//3D LiDAR point
            cv::Mat P_cam(4, 1, CV_32F);//3D LiDAR point under Cam coordination
            P_lidar.at<float>(0, 0) = mvLiDARPoints[i]->p3DonLiDAR.x;
            P_lidar.at<float>(1, 0) = mvLiDARPoints[i]->p3DonLiDAR.y;
            P_lidar.at<float>(2, 0) = mvLiDARPoints[i]->p3DonLiDAR.z;
            P_lidar.at<float>(3, 0) = 1;
//            cout<<" mTcamlid "<<endl<<mTcamlid<<endl;
//            cout<<mTcamlid.type()<<endl;
//            cout<<" P_lidar "<<endl<<P_lidar<<endl;
//            cout<<P_lidar.type()<<endl;
//            cout<<"P_cam type "<<P_cam.type()<<endl;
            P_cam = mTcamlid * P_lidar;
            mvLiDARPoints[i]->p3DonCam.x = P_cam.at<float>(0, 0);
            mvLiDARPoints[i]->p3DonCam.y = P_cam.at<float>(1, 0);
            mvLiDARPoints[i]->p3DonCam.z = P_cam.at<float>(2, 0);
            mvLiDARPoints[i]->p3DonCam.intensity = mvLiDARPoints[i]->p3DonLiDAR.intensity;
            //to image
            cv::Mat P_rect_00 = cv::Mat::zeros(3, 4, CV_32F);
            P_rect_00.at<float>(0, 0) = mK.at<float>(0, 0);
            P_rect_00.at<float>(0, 2) = mK.at<float>(0, 2);
            P_rect_00.at<float>(1, 1) = mK.at<float>(1, 1);
            P_rect_00.at<float>(1, 2) = mK.at<float>(1, 2);
            P_rect_00.at<float>(2, 2) = 1;
            cv::Mat P_img(3, 1, CV_32F);//3D LiDAR point under Cam coordination
//            cout<<"P_rect_00"<<endl<<P_rect_00<<endl;
//            cout<<"P_cam "<<endl<<P_cam<<endl;
            P_img = P_rect_00 * P_cam;
            cv::Point2d pt;
            pt.x = P_img.at<float>(0, 0) / P_img.at<float>(2, 0);
            pt.y = P_img.at<float>(1, 0) / P_img.at<float>(2, 0);
            if (pt.x > 0 && pt.x < cols && pt.y > 0 && pt.y < rows) {
                //cout<<X.at<double>(0, 0)<<" "<<X.at<double>(1, 0)<<" "<<X.at<double>(2, 0)<<" -> "<<pt.x<<" "<<pt.y<<endl;
                mvLiDARPoints[i]->pt2D = cv::Point2d(pt.x, pt.y);
            }
        }
    }

    void Frame::groupLiDARandSegment(vector<cv::Mat*> segmentImages){
        for(auto *point : mvLiDARPoints){
            for(int i=0;i<mSegmentsInfo.size();i++){
                if (pointInSegmentMask(point->pt2D, segmentImages[i])) {
                    // Associate the LiDAR point with the corresponding segment
                    mSegmentsInfo[i]->associatedPoints.push_back(point);
                    point->segmentInfo = mSegmentsInfo[i];
                    break; // Break the loop since the point has been associated with a segment
                }
            }
        }
    }

    void Frame::alignORBSegments(vector<cv::Mat*> segmentImages){
        for(int oi = 0; oi<mvORBAttributions.size();oi++){
            for(int i=0;i<mSegmentsInfo.size();i++){
                if(pointInSegmentMask(mvORBAttributions[oi]->keyPt->pt,segmentImages[i])){
                    mvORBAttributions[oi]->segmentIndex = i;
                    mSegmentsInfo[i]->associatedORBs.push_back(oi);
                }
            }
        }
    }



    bool Frame::pointInSegmentMask(const cv::Point2d& point, const cv::Mat* segmentMask) {
        // Check if the point coordinates are within the bounds of the segment mask
        if (point.x < 0 || point.x >= segmentMask->cols || point.y < 0 || point.y >= segmentMask->rows) {
            return false; // Point is outside the mask bounds
        }
        //    imshow("segmentMask", segmentMask);
        //    cv::circle(segmentMask, cv::Point(500,300), 5, cv::Scalar(255, 0, 0), -1);
        //    cv::waitKey(0);
        // Retrieve the pixel value at the point coordinates in the segment mask
        uchar pixelValue = segmentMask->at<uchar>(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
        if (pixelValue > 0)
            return true;
        else
            return false;
    }

    void Frame::planeFitEachSegment(){
        for(int i=0;i<mSegmentsInfo.size();i++){
            if(mSegmentsInfo[i]->category_id !=9 && !mSegmentsInfo[i]->associatedPoints.empty()){
                PlaneFitting(mSegmentsInfo[i]->associatedPoints, mSegmentsInfo[i]->id, mSegmentsInfo[i]->category_id, mvPlanes);
            }
        }
    }

    void Frame::PlaneFitting(vector<mLiDARPoint *> &mLiDARs, int segmentID, int segmentCategory, vector<mPlane*> &mPlanes) {
        ///Step 1 store all inputs into a container
        pcl::PointCloud<pcl::PointXYZ>::Ptr allPoints(new pcl::PointCloud<pcl::PointXYZ>);
        allPoints->resize(mLiDARs.size());
        int lidarCounter = 0;
        for(const auto &lidarPt:mLiDARs){
            allPoints->points[lidarCounter].x = lidarPt->p3DonCam.x;
            allPoints->points[lidarCounter].y = lidarPt->p3DonCam.y;
            allPoints->points[lidarCounter].z = lidarPt->p3DonCam.z;
            lidarCounter++;
        }
        allPoints->resize(lidarCounter);
        allPoints->height=1,allPoints->width=lidarCounter;
        ///Step 2 downsampling --- not working
//        //NOTE lose the index after down-sample
//        pcl::PointCloud<pcl::PointXYZ>::Ptr downSampledPts(new pcl::PointCloud<pcl::PointXYZ>);
//        pcl::VoxelGrid<pcl::PointXYZ> source;
//        source.setInputCloud(allPoints);
//        //KITTI x forward, y left, z up; Camera ref x->right, y->down, z->front
//        source.setLeafSize(0.05f, 0.05f, 0.05f);
//        source.filter(*downSampledPts);
//        //cout << " downsample points " << downSampledPts->points.size();
        ///Step 3 Region Growing
        ///calc norms
        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setInputCloud(allPoints);
        //normal_estimator.setInputCloud(downSampledPts);
        normal_estimator.setKSearch(50);
        normal_estimator.compute(*normals);
        ///calc region growing
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        reg.setMinClusterSize(200);
        reg.setMaxClusterSize(50000);
        reg.setSearchMethod(tree);
        reg.setNumberOfNeighbours(50);
        reg.setInputCloud(allPoints);
        //reg.setInputCloud(downSampledPts);
        reg.setInputNormals(normals);
        reg.setSmoothnessThreshold(10.0/180.0*M_PI);
        reg.setCurvatureThreshold(10.0);
        vector<pcl::PointIndices> clusters;
        reg.extract(clusters);
        ///Step 4 RANSAC for each cluster
        vector<map<int,int>> local2GlobalIndexMaps;
        vector<vector<int>> globalIndexofEachCluster;
        for(auto &cls:clusters){
            vector<int> globalIndiceofThisCluster;
            map<int,int> localGlobalThisCluster;
            pcl::PointCloud<pcl::PointXYZ>::Ptr thisCloud(new pcl::PointCloud<pcl::PointXYZ>);
            thisCloud->points.resize(cls.indices.size());
            thisCloud->width=thisCloud->points.size(),thisCloud->height=1;
            for(size_t j=0; j < cls.indices.size();j++){
                int globalIndex = cls.indices[j];
                thisCloud->points[j] = allPoints->points[globalIndex];
                //thisCloud->points[j] = downSampledPts->points[globalIndex];
                localGlobalThisCluster.insert(make_pair(j,globalIndex));
            }
            mPlane * foundPlane = new mPlane();
            pcl::PointIndices inlierIDX;
            int inPlaneNum = RANSACPlane(thisCloud, foundPlane, inlierIDX);
            if(inPlaneNum>0){
                int planeID = mPlanes.size();
                foundPlane->PlaneID = planeID;
                if(segmentCategory==0){//in road case, don't care about the outlier, all is inliner
                    for(int i=0;i<localGlobalThisCluster.size();i++){
                        globalIndiceofThisCluster.push_back(localGlobalThisCluster[i]);
                        foundPlane->lidarIndexofSegment.push_back(localGlobalThisCluster[i]);
                        foundPlane->segmentCategory = segmentCategory;
                        foundPlane->segmentID = segmentID;
                    }
                }else{
                    for (int indiceI: inlierIDX.indices) {
                        int globalIndex = localGlobalThisCluster[indiceI];
                        globalIndiceofThisCluster.push_back(globalIndex);
                        foundPlane->lidarIndexofSegment.push_back(globalIndex);
                        foundPlane->segmentCategory = segmentCategory;
                        foundPlane->segmentID = segmentID;
                    }
                }
            }
            globalIndexofEachCluster.push_back(globalIndiceofThisCluster);
            mPlanes.push_back(foundPlane);
        }
//        tree.reset();
//        normals.reset();
    }

    //AX+BY+CZ+D=0;
    int Frame::RANSACPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, mPlane *foundPlane,
                           pcl::PointIndices &inliersOutput) {
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = inputCloud.makeShared();
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        //create the segmentation objects
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        //Optional
        seg.setOptimizeCoefficients(true);
        //Mandatory
        seg.setMethodType(pcl::SACMODEL_PLANE);
        seg.setModelType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.1); //kitti 0.2

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0)
            return 0;
        inliersOutput = *inliers;
        foundPlane->A = coefficients->values[0];
        foundPlane->B = coefficients->values[1];
        foundPlane->C = coefficients->values[2];
        foundPlane->D = coefficients->values[3];
        for (int i = 0; i < inliers->indices.size(); i++) {
            double x = cloud->points[inliers->indices[i]].x;
            double y = cloud->points[inliers->indices[i]].y;
            double z = cloud->points[inliers->indices[i]].z;
            //foundPlane.points3D.push_back(cv::Point3d(x,y,z));
            //cout<<"inliner push back "<<x<<" "<<y<<" "<<z<<endl;
        }
        return inliers->indices.size();
    }

    void Frame::ORBdetphFromPlane(vector<mPlane*> allPlane, vector<SegmentInfo *> allSegInfos,
                                  vector<cv::Mat *> allSegImgs){
        for(int i=0;i<mvORBAttributions.size();i++){
            if(mvORBAttributions[i]->depthSource>0)
                continue;
            ///step 1 connect with nearest LiDAR plane
            int minPlaneIndex = -1; float minPlaneDis = 10000;
            float minLiDARZ0 = -1, minLiDARZ1 = -1, minLiDARZ2 = -1;
            for(int pi=0;pi<allPlane.size();pi++){
                if(allPlane[pi]->segmentCategory==0)//skip vegetarian
                    continue;
                int minIndex0 = -1, minIndex1 = -1, minIndex2 = -1;
                float minDist0 = 10000, minDist1 = 10000, minDist2 = 10000;
                int segIndex = allPlane[pi]->segmentID-1;
                for(int li=0;li<allPlane[pi]->lidarIndexofSegment.size();li++){
                    int liDARIndex = allPlane[pi]->lidarIndexofSegment[li];
                    float tempDis = cv::norm(cv::Point2d(mvORBAttributions[i]->keyPt->pt) - allSegInfos[segIndex]->associatedPoints[liDARIndex]->pt2D);
                    if (tempDis < minDist0) {
                        minDist2 = minDist1, minIndex2 = minIndex1;
                        minDist1 = minDist0, minIndex1 = minIndex0;
                        minDist0 = tempDis, minIndex0 = liDARIndex;
                    } else if (tempDis < minDist1 && tempDis >= minDist0) {
                        minDist2 = minDist1, minIndex2 = minIndex1;
                        minDist1 = tempDis, minIndex1 = liDARIndex;
                    } else if (tempDis < minDist2 && tempDis >= minDist1) {
                        minDist2 = tempDis, minIndex2 = liDARIndex;
                    }
                }
                if(minDist0 < minPlaneDis){
                    minPlaneDis = minDist0, minPlaneIndex = pi;
                    minLiDARZ0 = allSegInfos[segIndex]->associatedPoints[minIndex0]->p3DonCam.z,
                    minLiDARZ1 = allSegInfos[segIndex]->associatedPoints[minIndex1]->p3DonCam.z,
                    minLiDARZ2 = allSegInfos[segIndex]->associatedPoints[minIndex2]->p3DonCam.z;
                }
            }
            ///step 2 Get depth from plane
            if (minPlaneDis < 5) {
                //Because the segment comes from ML and pcl, no need more activities for foreground and bakcground
                ///Step 2.1 calc Z
                float A = allPlane[minPlaneIndex]->A, B = allPlane[minPlaneIndex]->B, C = allPlane[minPlaneIndex]->C, D = allPlane[minPlaneIndex]->D;
                double u = mvORBAttributions[i]->keyPt->pt.x, v = mvORBAttributions[i]->keyPt->pt.y;
                double Z = -D / (A * (u - cx) / fx + B * (v - cy) / fy + C); //from Ax+By+Cz + D =0;
                ///Step 2.2 determine if this Depth is good
                bool goodFlag = true;
                if ((Z - minLiDARZ0) > 3 || (Z - minLiDARZ1) > 3 || (Z - minLiDARZ2) > 3)
                    goodFlag = false;
                if (Z > 30)
                    goodFlag = false;
                if (goodFlag) {
                    mvORBAttributions[i]->PlaneID = minPlaneIndex;
                    mvORBAttributions[i]->depthSource = 1;
                    mvORBAttributions[i]->depth = Z;
                    //cout<<"ORB "<<allmORBs[i]->orbID<<" "<<allmORBs[i]->keyPt.pt.x<<" "<<allmORBs[i]->keyPt.pt.y<<" "<<allmORBs[i]->depth<<" source Plane "<<allmORBs[i]->PlaneID<<endl;
                } else {
                    mvORBAttributions[i]->depth = -1;
                }
            }
        }
    }

    void Frame::ORBdetphFromSegments(vector<mLiDARPoint*> allLiDARs, vector<SegmentInfo *> allSegInfos, vector<cv::Mat *> allSegImgs, cv::Mat debugImageIn){
        cv::Mat debugImage_gary = debugImageIn.clone();
        cv::Mat debugImage;
        cv::cvtColor(debugImage_gary,debugImage,cv::COLOR_GRAY2BGR);
        float minX,minY,maxX,maxY;
        for (int i = 0; i < mvORBAttributions.size(); i++) {
            if (mvORBAttributions[i]->depthSource > 0)
                continue;
            ///Step 1 select the LiDAR point in nearby round patch
            //cv::Rect roi(mvORBAttributions[i]->keyPt->pt.x - 10, mvORBAttributions[i]->keyPt->pt.y - 10, 20, 20);
            minX = mvORBAttributions[i]->keyPt->pt.x - 10, minY = mvORBAttributions[i]->keyPt->pt.y - 10;
            maxX = mvORBAttributions[i]->keyPt->pt.x + 10, maxY = mvORBAttributions[i]->keyPt->pt.y + 10;

            // Initialize containers for nearby LiDAR points
            std::vector<mLiDARPoint *> fourCornerPoints(4, nullptr);
            float minDisLeftUp = 10000, minDisLeftDown = 10000, minDisRightUp = 10000, minDisRightDown = 10000;
            for (int liDARindex = 0; liDARindex < allLiDARs.size(); liDARindex++) {
                //if(roi.contains(allLiDARs[liDARindex]->pt2D)){
                if (allLiDARs[liDARindex]->pt2D.x < maxX && allLiDARs[liDARindex]->pt2D.x > minX
                    && allLiDARs[liDARindex]->pt2D.y < maxY && allLiDARs[liDARindex]->pt2D.y > minY) {
                    float dis = cv::norm(cv::Point2d(mvORBAttributions[i]->keyPt->pt) - allLiDARs[liDARindex]->pt2D);
                    if (allLiDARs[liDARindex]->pt2D.x < mvORBAttributions[i]->keyPt->pt.x && allLiDARs[liDARindex]->pt2D.y < mvORBAttributions[i]->keyPt->pt.y) {
                        if (dis < minDisLeftUp) {
                            minDisLeftUp = dis;
                            fourCornerPoints[0] = allLiDARs[liDARindex];
                        }
                    } else if (allLiDARs[liDARindex]->pt2D.x > mvORBAttributions[i]->keyPt->pt.x && allLiDARs[liDARindex]->pt2D.y < mvORBAttributions[i]->keyPt->pt.y) {
                        if (dis < minDisRightUp) {
                            minDisRightUp = dis;
                            fourCornerPoints[1] = allLiDARs[liDARindex];
                        }
                    }else if(allLiDARs[liDARindex]->pt2D.x < mvORBAttributions[i]->keyPt->pt.x && allLiDARs[liDARindex]->pt2D.y > mvORBAttributions[i]->keyPt->pt.y){
                        if(dis<minDisLeftDown){
                            minDisLeftDown = dis;
                            fourCornerPoints[2] = allLiDARs[liDARindex];
                        }
                    }else if(allLiDARs[liDARindex]->pt2D.x > mvORBAttributions[i]->keyPt->pt.x && allLiDARs[liDARindex]->pt2D.y > mvORBAttributions[i]->keyPt->pt.y){
                        if(dis<minDisRightDown){
                            minDisRightDown = dis;
                            fourCornerPoints[3] = allLiDARs[liDARindex];
                        }
                    }
                }
            }
            bool sameSeg = true;

            for(int cornerIndex = 1;cornerIndex<fourCornerPoints.size();cornerIndex++){
                if(fourCornerPoints[cornerIndex] == nullptr || fourCornerPoints[cornerIndex-1] == nullptr) {
                    sameSeg = false;
                    break;
                }
                if(fourCornerPoints[cornerIndex]->segmentInfo != fourCornerPoints[cornerIndex-1]->segmentInfo){
                    sameSeg = false;
                    break;
                }
                if(fourCornerPoints[cornerIndex]->segmentInfo!=NULL&&fourCornerPoints[cornerIndex]->segmentInfo->category_id==8){
                    sameSeg = false;
                    break;
                }
                //TODO add segment catogory to orbpoints
                //if(fourCornerPoints[cornerIndex]->segmentInfo!=NULL && fourCornerPoints[cornerIndex]->segmentInfo->category_id != allmORBs[i]->segmentInfo->cagetgory_id)
            }
            if (sameSeg) {
                interpolateDepth(mvORBAttributions[i], fourCornerPoints);
            }
        }
    }

    void Frame::interpolateDepth(mORBAttribution *mORBPt, const std::vector<mLiDARPoint *> &fourCornerPoints) {
        if (mORBPt->depthSource > 0 || fourCornerPoints.size() != 4)
            return;

        cv::Point2d orbKeypt = mORBPt->keyPt->pt;
        cv::Point2d P1 = fourCornerPoints[0]->pt2D;
        cv::Point2d P2 = fourCornerPoints[1]->pt2D;
        cv::Point2d P3 = fourCornerPoints[2]->pt2D;
        cv::Point2d P4 = fourCornerPoints[3]->pt2D;
        float Z1 = fourCornerPoints[0]->p3DonCam.z, Z2 = fourCornerPoints[1]->p3DonCam.z, Z3 = fourCornerPoints[2]->p3DonCam.z, Z4 = fourCornerPoints[3]->p3DonCam.z;
        //X-axis
        cv::Point2d P12 = (P2.x - orbKeypt.x) / (P2.x - P1.x) * P1 + (orbKeypt.x - P1.x) / (P2.x - P1.x) * P2;
        float Z12 = (P2.x - orbKeypt.x) / (P2.x - P1.x) * Z1 + (orbKeypt.x - P1.x) / (P2.x - P1.x) * Z2;
        cv::Point2d P34 = (P4.x - orbKeypt.x) / (P4.x - P3.x) * P3 + (orbKeypt.x - P3.x) / (P4.x - P3.x) * P4;
        float Z34 = (P4.x - orbKeypt.x) / (P4.x - P3.x) * Z3 + (orbKeypt.x - P3.x) / (P4.x - P3.x) * Z4;
        //Y-axis
        cv::Point2d P1234 = (P34.y - orbKeypt.y) / (P34.y - P12.y) * P12 + (orbKeypt.y - P12.y) / (P34.y - P12.y) * P34;
        float Z1234 = (P34.y - orbKeypt.y) / (P34.y - P12.y) * Z12 + (orbKeypt.y - P12.y) / (P34.y - P12.y) * Z34;

        bool goodDepth = true;
        if(Z1234>30)
            goodDepth = false;
        if((Z1234 - Z1) > 3 || (Z1234 - Z2) > 3 || (Z1234 - Z3) > 3 || (Z1234 - Z4) > 3)
            goodDepth = false;
        if(goodDepth){
            mORBPt->depthSource = 3;
            mORBPt->depth = Z1234;
        }
    }

    /*
    * retrive stereo infor from Fusioned depth.
    * note I set mbf back to 0.
    */
    void Frame::ComputeStereoFromFusion(const vector<mORBAttribution *> ORBAttributions) {
        //cout<<"ComputeStereoFromFusion"<<endl;
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            //const float d = imDepth.at<float>(v,u);
            if (ORBAttributions[i]->depthSource > -1) {
                const float d = ORBAttributions[i]->depth;
                if (d > 0) {
                    mvDepth[i] = d;
                    //mbf = 0;
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
            }
        }
    }

    //Compute rest depth by Stereo
    void Frame::ComputeStereoMatches(vector<mORBAttribution*> &ORBAttributions) {
        //Dont need this. another compute stereo function has done this
//        mvuRight = vector<float>(N, -1.0f);
//        mvDepth = vector<float>(N, -1.0f);

        const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

        const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
        //Assign keypoints to row table
        vector<vector<size_t> > vRowIndices(nRows, vector<size_t>());

        for (int i = 0; i < nRows; i++)
            vRowIndices[i].reserve(200);

        //travel each right feature
        const int Nr = mvKeysRight.size();
        for (int iR = 0; iR < Nr; iR++) {
            const cv::KeyPoint &kp = mvKeysRight[iR];
            const float &kpY = kp.pt.y;
            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = ceil(kpY + r);
            const int minr = floor(kpY - r);
            //push this right feature to potential corresponding row
            for (int yi = minr; yi <= maxr; yi++)
                vRowIndices[yi].push_back(iR);
        }

        // Set limits for search
        //const float minZ = mb;
        const float minZ = 0.53715;
        const float minD = 0;
        //const float maxD = mbf / minZ;
        const float maxD = 0.53715 * 707.0912 / minZ;



        // For each left keypoint search a match in the right image
        vector<pair<int, int> > vDistIdx;
        vDistIdx.reserve(N);

        for (int iL = 0; iL < N; iL++) {
            const cv::KeyPoint &kpL = mvKeys[iL];
            const int &levelL = kpL.octave;
            const float &vL = kpL.pt.y;
            const float &uL = kpL.pt.x;

            const vector<size_t> &vCandidates = vRowIndices[vL];

            if (vCandidates.empty())
                continue;

            const float minU = uL - maxD;
            const float maxU = uL - minD;

            if (maxU < 0)
                continue;

            int bestDist = ORBmatcher::TH_HIGH;
            size_t bestIdxR = 0;

            const cv::Mat &dL = mDescriptors.row(iL);

            // Compare descriptor to right keypoints
            for (size_t iC = 0; iC < vCandidates.size(); iC++) {
                const size_t iR = vCandidates[iC];
                const cv::KeyPoint &kpR = mvKeysRight[iR];

                if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                    continue;

                const float &uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU) {
                    const cv::Mat &dR = mDescriptorsRight.row(iR);
                    const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdxR = iR;
                    }
                }
            }

            // Subpixel match by correlation
            if (bestDist < thOrbDist) {
                // coordinates in image pyramid at keypoint scale
                const float uR0 = mvKeysRight[bestIdxR].pt.x;
                const float scaleFactor = mvInvScaleFactors[kpL.octave];
                const float scaleduL = round(kpL.pt.x * scaleFactor);
                const float scaledvL = round(kpL.pt.y * scaleFactor);
                const float scaleduR0 = round(uR0 * scaleFactor);

                // sliding window search
                const int w = 5;
                cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                     scaledvL + w + 1).colRange(
                        scaleduL - w, scaleduL + w + 1);

                int bestDist = INT_MAX;
                int bestincR = 0;
                const int L = 5;
                vector<float> vDists;
                vDists.resize(2 * L + 1);

                const float iniu = scaleduR0 + L - w;
                const float endu = scaleduR0 + L + w + 1;
                if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                    continue;

                for (int incR = -L; incR <= +L; incR++) {
                    cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w,
                                                                                          scaledvL + w + 1).colRange(
                            scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

                    float dist = cv::norm(IL, IR, cv::NORM_L1);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestincR = incR;
                    }

                    vDists[L + incR] = dist;
                }

                if (bestincR == -L || bestincR == L)
                    continue;

                // Sub-pixel match (Parabola fitting)
                const float dist1 = vDists[L + bestincR - 1];
                const float dist2 = vDists[L + bestincR];
                const float dist3 = vDists[L + bestincR + 1];

                const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

                if (deltaR < -1 || deltaR > 1)
                    continue;

                // Re-scaled coordinate
                float bestuR = mvScaleFactors[kpL.octave] * ((float) scaleduR0 + (float) bestincR + deltaR);

                float disparity = (uL - bestuR);

                if (disparity >= minD && disparity < maxD) {
                    ///Added module
                    if (mvORBAttributions[iL]->depthSource == -1) {
                        if (disparity <= 0) {
                            disparity = 0.01;
                            bestuR = uL - 0.01;
                        }
                        mvDepth[iL] = mbf / disparity;
                        mvuRight[iL] = bestuR;
                        vDistIdx.push_back(pair<int, int>(bestDist, iL));

                        ///added mode mark the ORBattribution
                        //cout << "keypt " << iL << " depthSource " << mvORBAttributions[iL]->depthSource << endl;
                        mvORBAttributions[iL]->depthSource = 4;
                    }
                }
            }
        }

        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--) {
            if (vDistIdx[i].first < thDist)
                break;
            else {
                ///added
                if(mvORBAttributions[i]->depthSource == 4){
                    mvuRight[vDistIdx[i].second] = -1;
                    mvDepth[vDistIdx[i].second] = -1;
                }
            }
        }
    }

    //If a orb point has depth from Fusion, mark it, but not change mvDepth mbf and mvuRight
    //Okay this function if useless
    void Frame::MarkStereoFromFusion(const vector<mORBAttribution *> ORBAttributions) {
        //cout<<"ComputeStereoFromFusion"<<endl;
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);
        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];
            const float &v = kp.pt.y;
            const float &u = kp.pt.x;
            //const float d = imDepth.at<float>(v,u);
            if (ORBAttributions[i]->depthSource > -1) {
                const float d = ORBAttributions[i]->depth;
                if (d > 0) {
                    mvDepth[i] = d;
                    mbf = 0;
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
            }
        }
    }


} //namespace ORB_SLAM

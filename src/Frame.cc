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

///Added module
#include "tic_toc.h"

namespace ORB_SLAM2
{

    long unsigned int Frame::nNextId=0;
    bool Frame::mbInitialComputations=true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame()
    {}

    ///added module--added member copy
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
             ///added LiDAR modules --- copy new added Members
             mLaserPt_cam(frame.mLaserPt_cam),mLaserPoints(frame.mLaserPoints),mLaser16ScansPoints(frame.mLaser16ScansPoints),
             mCornerPointsSharp(frame.mCornerPointsSharp),mCornerPointsLessSharp(frame.mCornerPointsLessSharp),
             mSurfPointsFlat(frame.mSurfPointsFlat),mSurfPointsLessFlat(frame.mSurfPointsLessFlat),
             mLaserCorner_cam(frame.mLaserCorner_cam),mLaserLessCorner_cam(frame.mLaserLessCorner_cam),
             mLaserFlat_cam(frame.mLaserFlat_cam),mLaserLessFlat_cam(frame.mLaserLessFlat_cam),
             mvPlanes(frame.mvPlanes),mvLines(frame.mvLines),mvORBAttributions(frame.mvORBAttributions),//why this keypt wrong?
             givenDepthNum(frame.givenDepthNum),mlsdDescriptors(frame.mlsdDescriptors),
             mvpMapLines(frame.mvpMapLines),N_lines(frame.N_lines),mvbOutlierLines(frame.mvbOutlierLines)
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

///added module --- add some LiDAR feature codes
/**
 * @brief 单目帧构造函数
 *
 * @param[in] imGray //灰度图
 * @param[in] timeStamp //时间戳
 * @param[in] lasers //激光点云
 * @param[in] laserTimes //激光点云的中间时间，开始时间，结束时间
 * @param[in & out] extractor //ORB特征点提取器的句柄
 * @param[in] voc //ORB字典句柄
 * @param[in] K //相机内参矩阵
 * @param[in] bf //baseline*f
 * @param[int]thDepth //区分远近点的深度阈值
 */
    Frame::Frame(const cv::Mat &imGray, const cv::Mat &imGray_right, const double &timeStamp,
                 const vector<vector<double>> &lasers,
                 ORBextractor *extractor, ORBextractor *extractor_right,
                 ORBVocabulary *voc, cv::Mat &K, cv::Mat &Tcamlid,
                 cv::Mat &distCoef, const float &bf, const float &thDepth)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(extractor_right),
              mTimeStamp(timeStamp), mLaserPoints(lasers), mK(K.clone()),
              mTcamlid(Tcamlid.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth) {

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
        thread threadLeft(&Frame::ExtractORB,this,0,imGray);
        thread threadRight(&Frame::ExtractORB,this,1,imGray_right);
        threadLeft.join();
        threadRight.join();

        N = mvKeys.size();

        if(mvKeys.empty())
            return;

        //Step 4 OpenCV的去畸变函数
        UndistortKeyPoints();
        //Step 5 Compute Depth by Stereo
        ComputeStereoMatches();
        mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
        mvbOutlier = vector<bool>(N,false);

        // This is done only for the first Frame (or after a change in the calibration)
        //标志位，只在第一帧或者相机标定参数变化后执行
        if (mbInitialComputations) {
            //计算去畸变图像的边界
            ComputeImageBounds(imGray);

            //一个图像像素相当于多少个图像网格列（grid cols）/ (col length)
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            //一个图像像素相当于多少个图像网格行（grid rows）/ (row height)
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf/fx;

        //把特征点分配到网格中，默认64/48
        AssignFeaturesToGrid();

        ///added modules---------------------------------------------------
        givenDepthNum = 0;
        mvORBAttributions.resize(N);
        for (int i = 0; i < N; i++) {
            mvORBAttributions[i].keyPt = mvKeysUn[i];
            mvORBAttributions[i].ID = i;
        }
        clock_t start = clock();
        ExtractLiDARFeature();//extract LiDAR feature
        clock_t end = clock();
        //cout<<"ExtractLiDARFeature costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        start = clock();
        ProjectLiDARtoCam();        //Project LiDAR point to Cam coordination
        end = clock();
        //cout<<"ProjectLiDARtoCam costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        start = clock();
        ProjectLiDARtoImg(mK, imGray.cols, imGray.rows);
        end = clock();
        //cout<<"ProjectLiDARtoImg costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        start = clock();
        ProjectLiDARFeaturetoImg(mK, imGray.cols, imGray.rows);
        end = clock();
        //cout<<"ProjectLiDARFeaturetoImg costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        ///Pair the LiDAR Features and Img Features
        start = clock();
        PairLaserVisionFeatures(imGray);
        end = clock();
        //cout<<"PairLaserVisionFeatures costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        ///Compare depth with Stereo way
        CompareWithStereo(imGray, imGray_right);
        ///Restore depth to stereo
        ComputeStereoFromFusion(mvORBAttributions);

        ///init lines feature related things
        N_lines = mvLines.size();
        mvpMapLines = vector<MapLine *>(N_lines, static_cast<MapLine *>(NULL));
        mvbOutlierLines = vector<bool>(N_lines, false);
        float wait = 0;
        ///----------------------------------------------------
    }

    //Todo the parameters below should define in some other places
    int N_SCANS = 64;
    const double scanPeriod = 0.1;
    float cloudCurvature[400000];
    int cloudSortInd[400000];
    int cloudNeighborPicked[400000];
    int cloudLabel[400000];
    bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }
    ///Added module
    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                pcl::PointCloud<PointT> &cloud_out, float thres)
    {
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;
        // 把点云距离小于给定阈值的去除掉
        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1; //orignal height is the scan line number
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    ///added module --- add some LiDAR feature codes
/**
 * @brief 单目帧构造函数
 *
 * @param[in] imGray //灰度图
 * @param[in] timeStamp //时间戳
 * @param[in] lasers //激光点云
 * @param[in] laserTimes //激光点云的中间时间，开始时间，结束时间
 * @param[in & out] extractor //ORB特征点提取器的句柄
 * @param[in] voc //ORB字典句柄
 * @param[in] K //相机内参矩阵
 * @param[in] bf //baseline*f
 * @param[int]thDepth //区分远近点的深度阈值
 */
    Frame::Frame(const cv::Mat &imGray, const double &timeStamp,
                 const vector<vector<double>> &lasers,
                 ORBextractor *extractor, ORBextractor *extractor_right,
                 ORBVocabulary *voc, cv::Mat &K, cv::Mat &Tcamlid,
                 cv::Mat &distCoef, const float &bf, const float &thDepth)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(extractor_right),
              mTimeStamp(timeStamp), mLaserPoints(lasers), mK(K.clone()),
              mTcamlid(Tcamlid.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth) {

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
        ExtractORB(0,imGray);

        N = mvKeys.size();

        if(mvKeys.empty())
            return;

        //Step 4 OpenCV的去畸变函数
        UndistortKeyPoints();
        //Step 5 Compute Depth by Stereo
        ComputeStereoMatches();
        mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
        mvbOutlier = vector<bool>(N,false);

        // This is done only for the first Frame (or after a change in the calibration)
        //标志位，只在第一帧或者相机标定参数变化后执行
        if (mbInitialComputations) {
            //计算去畸变图像的边界
            ComputeImageBounds(imGray);

            //一个图像像素相当于多少个图像网格列（grid cols）/ (col length)
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            //一个图像像素相当于多少个图像网格行（grid rows）/ (row height)
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf/fx;

        //把特征点分配到网格中，默认64/48
        AssignFeaturesToGrid();

        ///added modules---------------------------------------------------
        givenDepthNum = 0;
        mvORBAttributions.resize(N);
        for (int i = 0; i < N; i++) {
            mvORBAttributions[i].keyPt = mvKeysUn[i];
            mvORBAttributions[i].ID = i;
        }
        clock_t start = clock();
        ExtractLiDARFeature();//extract LiDAR feature
        clock_t end = clock();
        //cout<<"ExtractLiDARFeature costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        start = clock();
        ProjectLiDARtoCam();        //Project LiDAR point to Cam coordination
        end = clock();
        //cout<<"ProjectLiDARtoCam costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        start = clock();
        ProjectLiDARtoImg(mK, imGray.cols, imGray.rows);
        end = clock();
        //cout<<"ProjectLiDARtoImg costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        start = clock();
        ProjectLiDARFeaturetoImg(mK, imGray.cols, imGray.rows);
        end = clock();
        //cout<<"ProjectLiDARFeaturetoImg costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        ///Pair the LiDAR Features and Img Features
        start = clock();
        PairLaserVisionFeatures(imGray);
        end = clock();
        //cout<<"PairLaserVisionFeatures costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
        ///Compare depth with Stereo way
        //CompareWithStereo(imGray, imGray_right);
        ///Restore depth to stereo
        ComputeStereoFromFusion(mvORBAttributions);

        ///init lines feature related things
        N_lines = mvLines.size();
        mvpMapLines = vector<MapLine *>(N_lines, static_cast<MapLine *>(NULL));
        mvbOutlierLines = vector<bool>(N_lines, false);
        float wait = 0;
        ///----------------------------------------------------
    }

    /*
 * dealing with the case that two keylines overlap.
 * since those two keylines are close,
 * the merged line should be those two endpoints that far-est to each other.
 * Still, merge line2 into line1
 */
    void mergeSelectedKeyLine(cv::line_descriptor::KeyLine &line1, const cv::line_descriptor::KeyLine &line2) {

        float SX1 = line1.startPointX, SY1 = line1.startPointY, SX2 = line2.startPointX, SY2 = line2.startPointY;
        float EX1 = line1.endPointX, EY1 = line1.endPointY, EX2 = line2.endPointX, EY2 = line2.endPointY;
        //to find out the farest points
        float distanceS1E1 = (SX1 - EX1) * (SX1 - EX1) + (SY1 - EY1) * (SY1 - EY1);
        float distanceS1S2 = (SX1 - SX2) * (SX1 - SX2) + (SY1 - SY2) * (SY1 - SY2);
        float distanceS1E2 = (SX1 - EX2) * (SX1 - EX2) + (SY1 - EY2) * (SY1 - EY2);
        float distanceE1S2 = (EX1 - SX2) * (EX1 - SX2) + (EY1 - SY2) * (EY1 - SY2);
        float distanceE1E2 = (EX1 - EX2) * (EX1 - EX2) + (EY1 - EY2) * (EY1 - EY2);
        float distanceS2E2 = (SX2 - EX2) * (SX2 - EX2) + (SY2 - EY2) * (SY2 - EY2);
        float maxNumber = distanceS1E1;
        if (maxNumber < distanceS1S2) maxNumber = distanceS1S2;
        if (maxNumber < distanceS1E2) maxNumber = distanceS1E2;
        if (maxNumber < distanceE1S2) maxNumber = distanceE1S2;
        if (maxNumber < distanceE1E2) maxNumber = distanceE1E2;
        if (maxNumber < distanceS2E2) maxNumber = distanceS2E2;
        if (maxNumber == distanceS1E1) {
            line1.startPointX = SX1;
            line1.startPointY = SY1;
            line1.endPointX = EX1;
            line1.endPointY = EY1;
        }
        if (maxNumber == distanceS1S2) {
            line1.startPointX = SX1;
            line1.startPointY = SY1;
            line1.endPointX = SX2;
            line1.endPointY = SY2;
        }
        if (maxNumber == distanceS1E2) {
            line1.startPointX = SX1;
            line1.startPointY = SY1;
            line1.endPointX = EX2;
            line1.endPointY = EY2;
        }
        if (maxNumber == distanceE1S2) {
            line1.startPointX = SX2;//EX1;
            line1.startPointY = SY2;//EY1;
            line1.endPointX = EX1;//SX2;
            line1.endPointY = EY1;//SY2;
        }
        if (maxNumber == distanceE1E2) {
            line1.startPointX = EX1;
            line1.startPointY = EY1;
            line1.endPointX = EX2;
            line1.endPointY = EY2;
        }
        if (maxNumber == distanceS2E2) {
            line1.startPointX = SX2;
            line1.startPointY = SY2;
            line1.endPointX = EX2;
            line1.endPointY = EY2;
        }
        //merge them
        line1.numOfPixels += line2.numOfPixels; //not reliable because overlap
        //if one angle is negative, +3.14 for it first
        if (line1.angle < 0 && line2.angle > 0)
            line1.angle = ((line2.angle - 3.1415926) + line1.angle) / 2;
        else if (line1.angle > 0 && line2.angle < 0)
            line1.angle = (line1.angle + (line2.angle + 3.1415926)) / 2;
        else
            line1.angle = (line1.angle + line2.angle) / 2;
        line1.lineLength = sqrt((line1.endPointX - line1.startPointX) * (line1.endPointX - line1.startPointX) +
                                (line1.endPointY - line1.startPointY) * (line1.endPointY - line1.startPointY));
        line1.response += line2.response;
        line1.size = (line1.startPointX - line1.endPointX) * (line1.startPointY - line1.endPointY);
    }

    /**
    * merge the primary keylines
    */
    void mergeKeyLines(vector<cv::line_descriptor::KeyLine> &keylines, vector<bool> &keylineMergeFlags, cv::Mat &descriptors, cv::Mat im) {
        ///Search for Keylines to merge
        const float degreeThres = 0.008722222 * 5; //5 degree
        const float distanceThres = 25;
        const float lineLengthThres = 30;
        for (int i = 0; i < keylines.size(); i++) {
            if (keylineMergeFlags[i])//If this key-line has been merged into another
                continue;
            //looking for pairs
            for (int j = 0; j < keylines.size(); j++) {
                if (!keylineMergeFlags[j] && j != i) { //This key-line shouldn't be merged into another | and j!=i
                    float degreeSum = abs(keylines[i].angle) + abs(keylines[j].angle);
                    float degreeDiff = abs(keylines[j].angle - keylines[i].angle);
                    if ((abs(degreeSum - 3.1415926) < degreeThres) || degreeDiff < degreeThres) {//angle < 5 degrees
                        //start to start distance
                        float distanceSS = (keylines[i].startPointX - keylines[j].startPointX) *
                                           (keylines[i].startPointX - keylines[j].startPointX) +
                                           (keylines[i].startPointY - keylines[j].startPointY) *
                                           (keylines[i].startPointY - keylines[j].startPointY);
                        //start to end distance
                        float distanceSE = (keylines[i].startPointX - keylines[j].endPointX) *
                                           (keylines[i].startPointX - keylines[j].endPointX) +
                                           (keylines[i].startPointY - keylines[j].endPointY) *
                                           (keylines[i].startPointY - keylines[j].endPointY);
                        //end to start distance
                        float distanceES = (keylines[i].endPointX - keylines[j].startPointX) *
                                           (keylines[i].endPointX - keylines[j].startPointX) +
                                           (keylines[i].endPointY - keylines[j].startPointY) *
                                           (keylines[i].endPointY - keylines[j].startPointY);
                        //end to end distance
                        float distanceEE = (keylines[i].endPointX - keylines[j].endPointX) *
                                           (keylines[i].endPointX - keylines[j].endPointX) +
                                           (keylines[i].endPointY - keylines[j].endPointY) *
                                           (keylines[i].endPointY - keylines[j].endPointY);
                        float midXi = (keylines[i].startPointX + keylines[i].endPointX) / 2;
                        float midYi = (keylines[i].startPointY + keylines[i].endPointY) / 2;
                        float midXj = (keylines[j].startPointX + keylines[j].endPointX) / 2;
                        float midYj = (keylines[j].startPointY + keylines[j].endPointY) / 2;
                        float distanceMid = (midXi - midXj) * (midXi - midXj) + (midYi - midYj) * (midYi - midYj);
                        bool SS = false, SE = false, ES = false, EE = false, Mid = false;
                        if (distanceSS < distanceThres)SS = true;
                        if (distanceSE < distanceThres)SE = true;
                        if (distanceEE < distanceThres)EE = true;
                        if (distanceES < distanceThres)ES = true;
                        if (distanceMid < distanceThres)Mid = true;
                        if (SS || SE || EE || ES || Mid) {//start points or end points or mid-points are close
                            ///Merge merge merge
                            mergeSelectedKeyLine(keylines[i], keylines[j]);
                            keylineMergeFlags[j] = true;
                        }
                    }
                }
            }
        }
        ///Check and re-Order startPoint and endPoint(startPoint closer to Origin)
        for (int i = 0; i < keylines.size(); i++) {
            if (keylineMergeFlags[i])
                continue;
            if (keylines[i].octave == 0) {
                double distanceStart = keylines[i].startPointX * keylines[i].startPointX +
                                       keylines[i].startPointY * keylines[i].startPointY;
                double distanceEnd = keylines[i].endPointX * keylines[i].endPointX +
                                     keylines[i].endPointY * keylines[i].endPointY;
                if(distanceEnd<distanceStart){//If endpoint is closer to origin, swap it
                    //cout<<"swap start point "<<keylines[i].startPointX<<" "<<keylines[i].startPointY<<" and "<<keylines[i].endPointX<<" "<<keylines[i].endPointY<<endl;
                    float tempX = keylines[i].startPointX, tempY = keylines[i].startPointY;
                    keylines[i].startPointX = keylines[i].endPointX, keylines[i].startPointY = keylines[i].endPointY;
                    keylines[i].endPointX = tempX, keylines[i].endPointY = tempY;
                    //cout<<"rslt start point "<<keylines[i].startPointX<<" "<<keylines[i].startPointY<<" and "<<keylines[i].endPointX<<" "<<keylines[i].endPointY<<endl;
                }
            }
        }
    }

    /**
    * Connect LSDLines with LiDAR point
    * By storing near LSD LiDAR points
    */
    void Frame::connectLSD2LiDAR(vector<mLine> &mLSDLinesIN, vector<PtLsr> &LiDARPtIN, cv::Mat im_in) {
        ///Step 1 store keyLines in terms of Ax+By+C = 0, from two endpoints.
        vector<vector<float>> keyLineABCs;
        for (auto &i: mLSDLinesIN) {
            float x1 = i.LSD.startPointX, y1 = i.LSD.startPointY;
            float x2 = i.LSD.endPointX, y2 = i.LSD.endPointY;
            float A = y2 - y1, B = x1 - x2, C = x2 * y1 - x1 * y2;
            vector<float> thisLine;
            thisLine.push_back(A), thisLine.push_back(B), thisLine.push_back(C);
            keyLineABCs.push_back(thisLine);
        }
        ///Step 2 pair up LSD lines and LiDAR points
        //vector<int> lidarPt2LSD(LiDAR2d.size(), -1);
        for (auto &lidarPt: LiDARPtIN) {
            double x0 = lidarPt.pt2d.x, y0 = lidarPt.pt2d.y;
            ///Step 2.1 First, for each LiDAR point, search for 3 nearest lines first
            float minDistance1 = 2, minDistance2 = 2, minDistance3 = 2;
            int index1 = -1, index2 = -1, index3 = -1;
            for (int j = 0; j < mLSDLinesIN.size(); j++) {
                ///Step 2.1.1 Condition 1 : LiDAR 2d point should be inside of LSD's bounding box
                bool XinBox = false, YinBox = false;
                float x1 = mLSDLinesIN[j].LSD.startPointX, y1 = mLSDLinesIN[j].LSD.startPointY;
                float x2 = mLSDLinesIN[j].LSD.endPointX, y2 = mLSDLinesIN[j].LSD.endPointY;
                if (x1 > x2) {
                    float tmp = x1;
                    x1 = x2, x2 = tmp;
                }
                if (y1 > y2) {
                    float tmp = y1;
                    y1 = y2, y2 = tmp;
                }
                XinBox = (x0 > (x1 - 5) && x0 < (x2 + 5)), YinBox = (y0 > (y1 - 5) && y0 < (y2 + 5));
                if (XinBox && YinBox) {
                    ///Step 2.1.2 Condition 2: point to line distance --- d = abs(Ax0+By0+C) / abs(sqrt(A^2+B^2))
                    float A = keyLineABCs[j][0], B = keyLineABCs[j][1], C = keyLineABCs[j][2];
                    float dis = abs(A * x0 + B * y0 + C) /
                                sqrt(A * A + B * B);
                    if (dis < minDistance1) {
                        minDistance3 = minDistance2, index3 = index2;
                        minDistance2 = minDistance1, index2 = index1;
                        minDistance1 = dis, index1 = j;
                    } else {
                        if (dis >= minDistance1 && dis < minDistance2) {
                            minDistance3 = minDistance2, index3 = index2;
                            minDistance2 = dis, index2 = j;
                        } else {
                            if (dis >= minDistance2 && dis < minDistance3) {
                                minDistance3 = dis, index3 = j;
                            }
                        }
                    }
                }
            }
            //Step 2.2 Check 3 nearest lines
            if (index1 > -1) {
                lidarPt.LSDlineID = mLSDLinesIN[index1].ID;
                lidarPt.LSDline = &mLSDLinesIN[index1];
                //LSD line connect to LiDAR
                int LiDARPtID = lidarPt.ptID;
                mLSDLinesIN[index1].LiDARPtIDs.push_back(lidarPt.ptID);
            } else {
                if (index2 > -1) {
                    lidarPt.LSDlineID = mLSDLinesIN[index2].ID;
                    lidarPt.LSDline = &mLSDLinesIN[index2];
                    //LSD line connect to LiDAR
                    int LiDARPtID = lidarPt.ptID;
                    mLSDLinesIN[index2].LiDARPtIDs.push_back(lidarPt.ptID);
                } else {
                    if (index3 > -1) {
                        lidarPt.LSDlineID = mLSDLinesIN[index3].ID;
                        lidarPt.LSDline = &mLSDLinesIN[index3];
                        int LiDARPtID = lidarPt.ptID;
                        mLSDLinesIN[index3].LiDARPtIDs.push_back(lidarPt.ptID);
                    }
                }
            }
        }

        ///Check
//        cvtColor(im_in, im_in, CV_GRAY2BGR);
//        for (const auto &i: LiDARPtIN) {
//            if (i.LSDlineID > -1) {
//                cv::circle(im_in, cvPoint(i.pt2d.x, i.pt2d.y), 1, cv::Scalar(0, 0, 255), 1);
//                //cout<<"start "<<i.LSDline->LSD.startPointX<<" "<<i.LSDline->LSD.startPointY<<" ends "<<i.LSDline->LSD.endPointX<<" "<<i.LSDline->LSD.endPointY<<endl;
//                cv::line(im_in, cvPoint(i.LSDline->LSD.startPointX, i.LSDline->LSD.startPointY),
//                         cvPoint(i.LSDline->LSD.endPointX, i.LSDline->LSD.endPointY), cv::Scalar(0, 255, 0), 2);
//                imshow("LSD2LiDAR", im_in);
//                //waitKey(10);
//            } else {
//                cv::circle(im_in, cvPoint(i.pt2d.x, i.pt2d.y), 1, cv::Scalar(102, 255, 255), 1);
//                imshow("LSD2LiDAR", im_in);
//                //waitKey(0.1);
//            }
//        }
//        imshow("LSD2LiDAR", im_in);
//        cv::waitKey(1);
//        for (int i = 0; i < mLSDLinesIN.size(); i++) {
//            cout << "mLSDLine " << i << " contains LiDAR point number " << mLSDLinesIN[i].LiDARPtIDs.size() << endl;
//        }
//        int pause = 0;
    }

/**
 * @brief fitting a line for the lsd-lidar intersection
 * @param lineInputs: input mline members
 * @param LiDARInputs: all mLiDARPoint
 * @return
 */
    int LineFitting(vector<mLine> &lineInputs, vector<PtLsr> &LiDARInputs, cv::Mat im) {
//        cv::Mat im_clone = im.clone();
//        cv::cvtColor(im,im_clone,CV_GRAY2BGR);
        for (int lineIndex = 0; lineIndex < lineInputs.size(); lineIndex++) {
            mLine *line = &lineInputs[lineIndex];
            if (!line->LiDARPtIDs.empty()) {
                ///Condition 1: check if this line contains LiDAR points from different scans
                int scanCounter = 0;//count scan number
                map<int, int> ScanAndIndex;//store the pairs of <scanID , scanCounter>
                for (int i = 0; i < line->LiDARPtIDs.size(); i++) {
                    int lidarIndex = line->LiDARPtIDs[i];
                    //int scanID = floor(LiDARInputs[lidarID].intensity);
                    int scanID = LiDARInputs[lidarIndex].scanID;
                    if (ScanAndIndex.find(scanID) == ScanAndIndex.end()) {//if scanID not store before
                        ScanAndIndex.insert({scanID, scanCounter});
                        scanCounter++;
                    }
                }
                if (scanCounter < 2)
                    continue;
                ///NOTE both two ways of pcl function won't work(commented in the end). point number is too less
                ///So simply manually calculate a line.
                vector<vector<PtLsr>> allPoints;
                for (int i = 0; i < scanCounter; i++) {//init a container
                    vector<PtLsr> pointThisScan;
                    allPoints.push_back(pointThisScan);
                }
                //cout<<"line "<<lineIndex<<" LiDARpt num "<<line->LiDARPtIDs.size()<<endl;
                for (int i = 0; i < line->LiDARPtIDs.size(); i++) {
                    int lidarID = line->LiDARPtIDs[i];
                    //int scanID = floor(LiDARInputs[lidarID].intensity);
                    int scanID = LiDARInputs[lidarID].scanID;
                    int scanIndex = (ScanAndIndex.find(scanID))->second;
                    allPoints[scanIndex].push_back(LiDARInputs[lidarID]);
                }
                vector<cv::Point3d> meanPoints;
                for (int i = 0; i < allPoints.size(); i++) {
                    float X_sum = 0, Y_sum = 0, Z_sum = 0;
                    int clusterSize = allPoints[i].size();
                    for (int j = 0; j < clusterSize; j++) {
                        X_sum += allPoints[i][j].pt3d.x;
                        Y_sum += allPoints[i][j].pt3d.y;
                        Z_sum += allPoints[i][j].pt3d.z;
                    }
                    cv::Point3d newMeanPoint((X_sum / clusterSize), (Y_sum / clusterSize), (Z_sum / clusterSize));
                    meanPoints.push_back(newMeanPoint);
                }
                //3d Line should be the longest connect of any two mean points above
                float maxDis = 0;
                int selectedI = -1, selectedJ = -1;
                for (int i = 0; i < meanPoints.size(); i++) {
                    for (int j = 0; j < meanPoints.size(); j++) {
                        if (i == j)
                            continue;
                        double distance = 0;
                        distance = sqrt((meanPoints[i].x - meanPoints[j].x) * (meanPoints[i].x - meanPoints[j].x)
                                        + (meanPoints[i].y - meanPoints[j].y) * (meanPoints[i].y - meanPoints[j].y)
                                        + (meanPoints[i].z - meanPoints[j].z) * (meanPoints[i].z - meanPoints[j].z));
                        if (distance > maxDis) {
                            maxDis = distance;
                            selectedI = i;
                            selectedJ = j;
                        }
                    }
                }
                ///using the 3D line which is perpendicular to two axis
                Eigen::Vector3d line3d;
                line3d << meanPoints[selectedI].x - meanPoints[selectedJ].x,
                        meanPoints[selectedI].y - meanPoints[selectedJ].y,
                        meanPoints[selectedI].z - meanPoints[selectedJ].z;
                Eigen::Vector3d line3dNorm = line3d.normalized();
                Eigen::Vector3d axisX, axisY, axisZ;
                axisX << 1, 0, 0;
                axisY << 0, 1, 0;
                axisZ << 0, 0, 1;
                float angleX = acos((line3dNorm.dot(axisX)) / (line3dNorm.norm() * axisX.norm()));
                float angleY = acos((line3dNorm.dot(axisY)) / (line3dNorm.norm() * axisY.norm()));
                float angleZ = acos((line3dNorm.dot(axisZ)) / (line3dNorm.norm() * axisZ.norm()));
//            cout << "angles to each axis : " << (angleX / 3.141592653 * 180 )<< " " << (angleY / 3.141592653 * 180)  << " " << (angleZ / 3.141592653 * 180 )<< endl;
                double angleThres = (10.0 / 180.0 * 3.141592653);
                bool parallelX = (3.141592653 - angleX) < angleThres || angleX < angleThres;
                bool parallelY = (3.141592653 - angleY) < angleThres || angleY < angleThres;
                bool parallelZ = (3.141592653 - angleZ) < angleThres || angleZ < angleThres;
                if (parallelX || parallelY || parallelZ) {
                    line->fit3DLine = true;
                    line->pt3dStart.x = meanPoints[selectedI].x;
                    line->pt3dStart.y = meanPoints[selectedI].y;
                    line->pt3dStart.z = meanPoints[selectedI].z;
                    line->pt3dEnd.x = meanPoints[selectedJ].x;
                    line->pt3dEnd.y = meanPoints[selectedJ].y;
                    line->pt3dEnd.z = meanPoints[selectedJ].z;
//                cout << "line ID " << line->ID << " found a line " << line->pt3dStart.x << " " << line->pt3dStart.y << " "
//                     << line->pt3dStart.z << " to "
//                     << line->pt3dEnd.x << " " << line->pt3dEnd.y << " " << line->pt3dEnd.z;
                    double l = meanPoints[selectedI].x - meanPoints[selectedJ].x,
                            m = meanPoints[selectedI].y - meanPoints[selectedJ].y,
                            n = meanPoints[selectedI].z - meanPoints[selectedJ].z;
//                cout<<" vector "<<l<<" "<<m<<" "<<n<<endl;
                    ///equation: (x-x1)/l=(y-y1)/m=(z-z1)/n
                    ///If this line is ground line
                    int groundPtNum = 0, nonGroundPtNum = 0;
                    for (int i = 0; i < allPoints.size(); i++) {
                        for (int j = 0; j < allPoints[i].size(); j++) {
                            if (allPoints[i][j].low) {
                                groundPtNum++;
                                //cv::circle(im_clone,cv::Point2d(allPoints[i][j].pt2d.x,allPoints[i][j].pt2d.y),3, cv::Scalar(0,0,255),1);
                            } else {
                                nonGroundPtNum++;
                                //cv::circle(im_clone,cv::Point2d(allPoints[i][j].pt2d.x,allPoints[i][j].pt2d.y),3, cv::Scalar(255,0,0),1);
                            }

                        }
                    }//if 3d line on the ground
                    if (groundPtNum == 0) {
                        line->nonfloorLine = true;
//                    cout << "ground point number 0, non floor line " << endl;
                    } else {
                        if (nonGroundPtNum == 0) {
                            line->nonfloorLine = false;
//                        cout << "non ground point number 0, floor line " << endl;
                        } else {
                            float ratio = float(nonGroundPtNum) / float(groundPtNum + nonGroundPtNum);
                            if (ratio > 0.7) {
                                line->nonfloorLine = true;
//                            cout << "(nonground pt / total pt) ratio " << ratio << " non floor line " << endl;
                            } else {
                                line->nonfloorLine = false;
                            }
                        }
                    }

                    ///NOTE both two function won't work. point is too less
                    ///So simply manually calculate a line.
//            ///FUNCTION1 Step 1 fill up the container
//            pcl::PointCloud<pcl::PointXYZ>::Ptr allPoints(new pcl::PointCloud<pcl::PointXYZ>);
//            int counter = 0;
//            for(int idx : line.LiDARPtIDs){
//                allPoints->points.push_back(pcl::PointXYZ(LiDARInputs[idx].p3DonCam.x,LiDARInputs[idx].p3DonCam.x,LiDARInputs[idx].p3DonCam.x));
//                counter++;
//            }
//            allPoints->resize(counter);
//            allPoints->width = allPoints->size();
//            allPoints->height = 1;
//            allPoints->is_dense=false;
//            ///Step 2
//            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//            pcl::SACSegmentation<pcl::PointXYZ> seg;
//            seg.setOptimizeCoefficients(true);            //Optional
//            seg.setMethodType(pcl::SACMODEL_LINE);
//            seg.setModelType(pcl::SAC_RANSAC);
//            seg.setDistanceThreshold(0.1);
//            seg.setInputCloud(allPoints);
//            seg.segment(*inliers, *coefficients);
////            if (inliers->indices.empty())
////                continue;
//            ///FUNCTION 2 .Step play with PCL functions : RANSAC
//            pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_line(new pcl::SampleConsensusModelLine<pcl::PointXYZ>(allPoints));
//            pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_line);
//            ransac.setDistanceThreshold(0.1);    //内点到模型的最大距离
//            ransac.computeModel();
//            ransac.refineModel();
//            vector<int> inliers2;
//            ransac.getInliers(inliers2);
//            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_line(new pcl::PointCloud<pcl::PointXYZ>);
//            pcl::copyPointCloud<pcl::PointXYZ>(*allPoints, inliers2, *cloud_line);
////            cout<<"cloud_line RANSAC size : "<<cloud_line->points.size()<<endl;
                }
            }
        }
//        cv::imshow("im_clone",im_clone);
//        cv::waitKey(1);
//        return 1;
int pause = 1;
    }


    /**
     * @brief works under LiDAR coordination,
     * this function extract process 64/16 LiDAR scans lidar data
     * this ALOAM function extract edge feature and plane feature from LiDAR source (commented)
     */
    void Frame::ExtractLiDARFeature() {
        ///Step 0: prepare data container
        for (int i = 0; i < 16; i++) {
            pcl::PointCloud<pcl::PointXYZI> newPointCloud;
            mLaser16ScansPoints.push_back(newPointCloud);
        }
        for (int i = 0; i < 64; i++) {
            pcl::PointCloud<pcl::PointXYZI> newPointCloud;
            mlaserScansPoints.push_back(newPointCloud);
        }
        ///Step 1 : fetch 3d lidar points from mLaserPoints (lidar coordination)
        TicToc t_whole;
        TicToc t_prepare;
        std::vector<int> scanStartInd(N_SCANS, 0);
        std::vector<int> scanEndInd(N_SCANS, 0);
        //std::vector<int> scanStartInd(16, 0);
        //std::vector<int> scanEndInd(16, 0);
        pcl::PointCloud<pcl::PointXYZ> laserCloudIn;//all 64 scans points
        std::vector<int> indices;
        int lsrPtNum = mLaserPoints.size();
        laserCloudIn.resize(lsrPtNum);
        for (int i = 0; i < lsrPtNum; i++) {
            laserCloudIn.points[i].x = mLaserPoints[i][0];
            laserCloudIn.points[i].y = mLaserPoints[i][1];
            laserCloudIn.points[i].z = mLaserPoints[i][2];
        }
        pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
        removeClosedPointCloud(laserCloudIn, laserCloudIn, 0.1);//close to sensor
        ///Step2 : calc the angles
        // 计算起始点和结束点的角度，由于激光雷达是顺时针旋转，这里取反就相当于转成了逆时针
        int cloudSize = laserCloudIn.points.size();
        float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
        // atan2范围是[-Pi,PI]，这里加上2PI是为了保证起始到结束相差2PI符合实际
        float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                              laserCloudIn.points[cloudSize - 1].x) +
                       2 * M_PI; //the end point should be 360 degree from the start point

        // 总有一些例外，比如这里大于3PI，和小于PI，就需要做一些调整到合理范围
        if (endOri - startOri > 3 * M_PI) //start -179, ends 179 degree
        {
            endOri -= 2 * M_PI;
        } else if (endOri - startOri < M_PI) //start 179, ends -179 degree
        {
            endOri += 2 * M_PI;
        }
        //printf("end Ori %f\n", endOri);
        bool halfPassed = false;
        int count = cloudSize;
        int count16scan = 0;
        PointType point;
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
        //std::vector<pcl::PointCloud<PointType>> laserCloud16Scans(16);
        ///Added---for pcl alignment issue
        //QUESTION? this is because in some case, when system pushback point to vector<pcl::pointcloud<pointtype>> could lead pcl alignment issue.
        //So init the vector's each pointcloud with same size the resize it.
//        for(int i = 0; i < N_SCANS;i++)
//            laserCloudScans[i].resize(4000);
//        for(int i = 0; i < 16;i++)
//            laserCloud16Scans[i].resize(4000);
//        std::vector<int> eachScanIndexs;//eachScan's size since has been resize to 4000.
//        for(int i = 0; i < N_SCANS;i++)
//            eachScanIndexs.push_back(0);
        // 遍历每一个点
        for (int i = 0; i < cloudSize; i++) {
            point.x = laserCloudIn.points[i].x;
            point.y = laserCloudIn.points[i].y;
            point.z = laserCloudIn.points[i].z;
            // 计算他的俯仰角
            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;
            // 计算是第几根scan
            if (N_SCANS == 16) {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (N_SCANS - 1) || scanID < 0) {
                    count--;
                    continue;
                }
            } else if (N_SCANS == 32) {
                scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
                if (scanID > (N_SCANS - 1) || scanID < 0) {
                    count--;
                    continue;
                }
            } else if (N_SCANS == 64) {
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                // use [0 50]  > 50 remove outlies
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
                    count--;
                    continue;
                }
            } else {
                printf("wrong scan number\n");
                //ROS_BREAK();
            }
            //printf("angle %f scanID %d \n", angle, scanID);
            // 计算水平角
            float ori = -atan2(point.y, point.x);
            if (!halfPassed) {
                // 确保-PI / 2 < ori - startOri < 3 / 2 * PI
                if (ori < startOri - M_PI / 2) {
                    ori += 2 * M_PI;
                } else if (ori > startOri + M_PI * 3 / 2) {
                    ori -= 2 * M_PI;
                }
                // 如果超过180度，就说明过了一半了 //half passed the start point
                if (ori - startOri > M_PI) {
                    halfPassed = true;
                }
            } else {
                // 确保-PI * 3 / 2 < ori - endOri < PI / 2
                ori += 2 * M_PI;    // 先补偿2PI
                if (ori < endOri - M_PI * 3 / 2) {
                    ori += 2 * M_PI;
                } else if (ori > endOri + M_PI / 2) {
                    ori -= 2 * M_PI;
                }
            }
            // 角度的计算是为了计算相对的起始时刻的时间
            float relTime = (ori - startOri) / (endOri - startOri);
            // 整数部分是scan的索引，小数部分是相对起始时刻的时间
            point.intensity = scanID + scanPeriod * relTime;
            ///Added --- for pcl alignment problem explained above
//            int index = eachScanIndexs[scanID];
//            laserCloudScans[scanID].points[index] = point;
//            eachScanIndexs[scanID]++;
            // 根据scan的idx送入各自数组
            laserCloudScans[scanID].push_back(point);//comment this if met alignment issue
            mlaserScansPoints[scanID].push_back(point);
            ///added for 16 scans
//            if (scanID % 4 == 0) {
//                laserCloud16Scans[scanID / 4].push_back(point);
//                mLaser16ScansPoints[scanID / 4].push_back(point);
//                count16scan++;
//            }
        }
        ///Added---for pcl alignment problem since the pointCloud size was resize to 4000 above
//        for(int i=0;i<N_SCANS;i++){
//            //cout<<"reset laserCloudScans["<<i<<"] from "<<laserCloudScans[i].size();
//            laserCloudScans[i].resize(eachScanIndexs[i]);
//            //cout<<" to "<<laserCloudScans[i].size()<<endl;
//        }
        // cloudSize是有效的点云的数目
        cloudSize = count;
        //int cloudSize16 = laserCloud16Scans.size();
//        printf(" valid points size %d ", cloudSize);
//        printf(" 16 scans point size %d \n",count16scan);
        // 全部集合到一个点云里面去，但是使用两个数组标记起始和结果，这里分别+5和-6是为了计算曲率方便
        ///Added--- 16 instead of 64
        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < 16; i++) {//N_SCANS
            scanStartInd[i] = laserCloud->size() + 5; //most left 5 and right 6 didn't count curve value
            *laserCloud += laserCloudScans[i];
            //*laserCloud += laserCloud16Scans[i];
            scanEndInd[i] = laserCloud->size() - 6;
        }
///Calc Curvature and Calc Features
//        //printf("prepare time %f \n", t_prepare.toc());
//        // 开始计算曲率
//        for (int i = 5; i < cloudSize - 5; i++)
//        {
//            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
//            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
//            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
//            // 存储曲率，索引
//            cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
//            cloudSortInd[i] = i;
//            cloudNeighborPicked[i] = 0;
//            cloudLabel[i] = 0;
//        }
//
//        TicToc t_pts;
//
//        pcl::PointCloud<PointType> cornerPointsSharp;
//        pcl::PointCloud<PointType> cornerPointsLessSharp;
//        pcl::PointCloud<PointType> surfPointsFlat;
//        pcl::PointCloud<PointType> surfPointsLessFlat;
//
//        float t_q_sort = 0;
//        // 遍历每个scan
//        for (int i = 0; i < 16; i++)//N_SCANS
//        {
//            // 没有有效的点了，就continue
//            if( scanEndInd[i] - scanStartInd[i] < 6) //this scan soo small
//                continue;
//            // 用来存储不太平整的点
//            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
//            // 将每个scan等分成6等分
//            for (int j = 0; j < 6; j++)
//            {
//                // 每个等分的起始和结束点
//                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
//                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;
//
//                TicToc t_tmp;
//                // 对点云按照曲率进行排序，小的在前，大的在后
//                std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
//                t_q_sort += t_tmp.toc();
//
//                int largestPickedNum = 0;
//                // 挑选曲率比较大的部分
//                for (int k = ep; k >= sp; k--)
//                {
//                    // 排序后顺序就乱了，这个时候索引的作用就体现出来了
//                    int ind = cloudSortInd[k];
//
//                    // 看看这个点是否是有效点，同时曲率是否大于阈值
//                    if (cloudNeighborPicked[ind] == 0 &&
//                        cloudCurvature[ind] > 0.1)
//                    {
//
//                        largestPickedNum++;
//                        // 每段选2个曲率大的点
//                        if (largestPickedNum <= 2)
//                        {
//                            // label为2是曲率大的标记
//                            cloudLabel[ind] = 2;
//                            // cornerPointsSharp存放大曲率的点
//                            cornerPointsSharp.push_back(laserCloud->points[ind]);
//                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
//                        }
//                            // 以及20个曲率稍微大一些的点
//                        else if (largestPickedNum <= 20)
//                        {
//                            // label置1表示曲率稍微大
//                            cloudLabel[ind] = 1;
//                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
//                        }
//                            // 超过20个就算了
//                        else
//                        {
//                            break;
//                        }
//                        // 这个点被选中后 pick标志位置1
//                        cloudNeighborPicked[ind] = 1;
//                        // 为了保证特征点不过度集中，将选中的点周围5个点都置1,避免后续会选到
//                        for (int l = 1; l <= 5; l++)
//                        {
//                            // 查看相邻点距离是否差异过大，如果差异过大说明点云在此不连续，是特征边缘，就会是新的特征，因此就不置位了
//                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
//                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
//                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
//                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
//                            {
//                                break;
//                            }
//
//                            cloudNeighborPicked[ind + l] = 1;
//                        }
//                        // 下面同理
//                        for (int l = -1; l >= -5; l--)
//                        {
//                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
//                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
//                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
//                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
//                            {
//                                break;
//                            }
//
//                            cloudNeighborPicked[ind + l] = 1;
//                        }
//                    }
//                }
//                // 下面开始挑选面点
//                int smallestPickedNum = 0;
//                for (int k = sp; k <= ep; k++)
//                {
//                    int ind = cloudSortInd[k];
//                    // 确保这个点没有被pick且曲率小于阈值
//                    if (cloudNeighborPicked[ind] == 0 &&
//                        cloudCurvature[ind] < 0.1)
//                    {
//                        // -1认为是平坦的点
//                        cloudLabel[ind] = -1;
//                        surfPointsFlat.push_back(laserCloud->points[ind]);
//
//                        smallestPickedNum++;
//                        // 这里不区分平坦和比较平坦，因为剩下的点label默认是0,就是比较平坦
//                        if (smallestPickedNum >= 4)
//                        {
//                            break;
//                        }
//                        // 下面同理
//                        cloudNeighborPicked[ind] = 1;
//                        for (int l = 1; l <= 5; l++)
//                        {
//                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
//                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
//                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
//                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
//                            {
//                                break;
//                            }
//
//                            cloudNeighborPicked[ind + l] = 1;
//                        }
//                        for (int l = -1; l >= -5; l--)
//                        {
//                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
//                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
//                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
//                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
//                            {
//                                break;
//                            }
//
//                            cloudNeighborPicked[ind + l] = 1;
//                        }
//                    }
//                }
//                //Less Flat is default lable 0
//                for (int k = sp; k <= ep; k++)
//                {
//                    // 这里可以看到，剩下来的点都是一般平坦，这个也符合实际
//                    if (cloudLabel[k] <= 0)
//                    {
//                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
//                    }
//                }
//                //In some special case, like corner points number is far more than 20, we just select 20, some of the corner point will be label 0 !!!
//            }
//
//            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
//            surfPointsLessFlatScanDS.resize(100000);
//            pcl::VoxelGrid<PointType> downSizeFilter;
//            // 一般平坦的点比较多，所以这里做一个体素滤波
//            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
//            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
//            downSizeFilter.filter(surfPointsLessFlatScanDS);
//
//            surfPointsLessFlat += surfPointsLessFlatScanDS;
//        }
//        //printf("sort q time %f \n", t_q_sort);
//        //printf("seperate points time %f \n", t_pts.toc());
//
//        ///Step 3 : pass features to frame member
//        mCornerPointsSharp.resize(cornerPointsSharp.size());
//        mCornerPointsSharp=cornerPointsSharp;
//        mCornerPointsLessSharp.resize(cornerPointsLessSharp.size());
//        mCornerPointsLessSharp=cornerPointsLessSharp;
//        mSurfPointsFlat.resize(surfPointsFlat.size());
//        mSurfPointsFlat=surfPointsFlat;
//        mSurfPointsLessFlat.resize(surfPointsLessFlat.size());
//        mSurfPointsLessFlat=surfPointsLessFlat;
//        //printf("Corner Pt: %lu, Less Corner : %lu, Flat Pt: %lu, Less Flat Pt : %lu \n", mCornerPointsSharp.size(),mCornerPointsLessSharp.size(),mSurfPointsFlat.size(),mSurfPointsLessFlat.size());
//        int pause = 1;
    }

    ///added module
    void Frame::PlaneFitting() {
        ///Step 1 store all low ground lidar point into container
        int actualNum = 0;
        pcl::PointCloud<pcl::PointXYZ>::Ptr allPoints(new pcl::PointCloud<pcl::PointXYZ>);
        allPoints->resize(mLaserPt_cam.size());
        for (int i = 0; i < mLaserPt_cam.size(); i++) {
            if (mLaserPt_cam[i].low) {
                //Note LiDAR ref X->front, y->left, z->up. Camera ref x->right, y->down, z->front
                allPoints->points[actualNum].x = float(mLaserPt_cam[i].pt3d.x);
                allPoints->points[actualNum].y = float(mLaserPt_cam[i].pt3d.y);
                allPoints->points[actualNum].z = float(mLaserPt_cam[i].pt3d.z);
                actualNum++;
            }
        }
        allPoints->resize(actualNum);
        cout<<"lower ground point number "<<actualNum<<" / "<<mLaserPt_cam.size();
        ///Step 2 DownSampling and calc norm
        //NOTE lose the index after down-sample
        pcl::PointCloud<pcl::PointXYZ>::Ptr downSampledPts(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(allPoints);
        //KITTI x forward, y left, z up; Camera ref x->right, y->down, z->front
        //sor.setLeafSize(0.02f, 0.02f, 0.02f);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);//for 64 scans
        sor.filter(*downSampledPts);
        cout << " downsample left points " << downSampledPts->points.size() << endl;
        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setInputCloud(downSampledPts);
        normal_estimator.setKSearch(50);
        //normal_estimator.setRadiusSearch(0.10);
        normal_estimator.compute(*normals);
        ///step 3 region growing
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        reg.setMinClusterSize(2000);
        reg.setMaxClusterSize(50000);
        reg.setSearchMethod(tree);
        reg.setNumberOfNeighbours(50);//too little will cause run time error
        //reg.setResidualThreshold(0.10);
        reg.setInputCloud(downSampledPts);
        reg.setInputNormals(normals);
        reg.setSmoothnessThreshold(7.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold(1.0);
        //extract each cluster
        //clock_t startTime = clock();
        std::vector<pcl::PointIndices> clusters;
        reg.extract(clusters);
        //clock_t endTime = clock();
        //double timeUsed = double(endTime - startTime) / CLOCKS_PER_SEC;
        //cout << "Region Growing " << timeUsed << " sec ";
        cout << " region growing clusters total number  " << clusters.size();
        ///step 4  RANSAC plane fitting
        for (auto & thisCluster : clusters) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr thisCloud(new pcl::PointCloud<pcl::PointXYZ>);
            thisCloud->points.resize(thisCluster.indices.size());
            thisCloud->height = 1;
            thisCloud->width = thisCluster.indices.size();
            cout << " | cluster contains " << thisCluster.indices.size() << " "<<endl;
            for (int j = 0; j < thisCluster.indices.size(); j++) {
                int index = thisCluster.indices[j];
                thisCloud->points[j].x = downSampledPts->points[index].x;
                thisCloud->points[j].y = downSampledPts->points[index].y;
                thisCloud->points[j].z = downSampledPts->points[index].z;
                //cout<<"cluster point "<<thisCloud->points[index].x<<" "<<thisCloud->points[index].y<<" "<<thisCloud->points[index].z<<endl;
            }
            mPlane foundPlane;
            //startTime = clock();
            pcl::PointIndices inliersOUT;
            int inPlaneNum = RANSACPlane(thisCloud, foundPlane, inliersOUT);
            //endTime = clock();
            //double timeUsed = double(endTime - startTime) / CLOCKS_PER_SEC;
            //cout << " RANSAC plane " << timeUsed << " sec. Inliners num: " << inPlaneNum;
            if (inPlaneNum > 0) {
                int planeID = this->mvPlanes.size();
                foundPlane.PlaneId = planeID;
                this->mvPlanes.push_back(foundPlane);
                cout<<"frame "<<mnId<<" plane ID "<<foundPlane.PlaneId<<" "<<foundPlane.A<<" "<<foundPlane.B<<" "<<foundPlane.C<<" "<<foundPlane.D<<" | " ;
                float theta = atan2(foundPlane.C,sqrt(foundPlane.A*foundPlane.A+foundPlane.B*foundPlane.B));
                float phi = atan2(foundPlane.B,foundPlane.A);
                cout<<"theta "<<(theta/M_PI*180)<<" phi "<<phi/M_PI*180<<endl;

            }
        }
        ///Step 5 connect all LiDAR points with Found Plane (from down-sampled cluster)
        int PlaneLiDARNum2 = 0;
        for(auto &eachPT:mLaserPt_cam){
            if(eachPT.low){//save half-time
                for(int j=0;j< mvPlanes.size();j++){
                    float A = mvPlanes[j].A, B = mvPlanes[j].B, C = mvPlanes[j].C, D = mvPlanes[j].D;
                    float distance = (abs(eachPT.pt3d.x * A + eachPT.pt3d.y * B + eachPT.pt3d.z * C + D)) /
                                     (sqrt(A * A + B * B + C * C));
                    if (distance < 0.05) {
                        eachPT.planeID = j;
//                        eachPT.A = A, eachPT.B = B, eachPT.C = C, eachPT.D = D;
                        PlaneLiDARNum2++;
                    }
                }
            }
        }
        cout<<" all on Plane LiDAR Num "<<PlaneLiDARNum2<<endl;
    }

    //AX+BY+CZ+D=0;
    int Frame::RANSACPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, mPlane &foundPlane, pcl::PointIndices &inliersOutput)
    {
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = inputCloud.makeShared();
        pcl::ModelCoefficients::Ptr  coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        //create the segmentation objects
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        //Optional
        seg.setOptimizeCoefficients(true);
        //Mandatory
        seg.setMethodType(pcl::SACMODEL_PLANE);
        seg.setModelType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.05);

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        if(inliers->indices.size()==0)
            return 0;
        inliersOutput = *inliers;
        foundPlane.A = coefficients->values[0];foundPlane.B = coefficients->values[1];
        foundPlane.C = coefficients->values[2];foundPlane.D = coefficients->values[3];
        for(int i = 0; i < inliers->indices.size();i++)
        {
            double x = cloud->points[inliers->indices[i]].x;
            double y = cloud->points[inliers->indices[i]].y;
            double z = cloud->points[inliers->indices[i]].z;
            //foundPlane.points3D.push_back(cv::Point3d(x,y,z));
            //cout<<"inliner push back "<<x<<" "<<y<<" "<<z<<endl;
        }
        return inliers->indices.size();
    }


/**
 * Connect ORB point with LiDAR plane Point.
 * The idea is get depth information from LiDAR plane
 * by find the closest lidar point's plane
 * @param LiDARPoints
 * @param ORBFeatures
 * @param extractedPlanes
 * @param threshold
 * @param im
 */
    void Frame::connectORB2Plane(vector<PtLsr> &LiDARPoints, std::vector<cv::KeyPoint> &ORBFeatures,
                                 vector<ORB_SLAM2::mPlane> &extractedPlanes, double threshold, cv::Mat im) {
        int target = -1;
        ///Step 1, extract in-plane lidar points, store in cloud, store in KD tree. keep index map.
        std::map<int, int> indexPlane2indexLiDAR; //prepare a index map <cloud index, inputs index>
        //prepare a pcl kd-tree
        pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
        cloud->width = LiDARPoints.size();
        cloud->height = 1;
        cloud->points.resize(cloud->width * cloud->height);
        int planeLiDARNum = 0;//record cloud size
        for (int i = 0; i < LiDARPoints.size(); i++) {
            if (LiDARPoints[i].planeID > -1) {
                (*cloud)[planeLiDARNum].x = LiDARPoints[i].pt2d.x;
                (*cloud)[planeLiDARNum].y = LiDARPoints[i].pt2d.y;
                indexPlane2indexLiDAR.insert({planeLiDARNum, i});
                planeLiDARNum++;
            }
        }
        cloud->points.resize(planeLiDARNum);
        pcl::KdTreeFLANN<pcl::PointXY> kdtree;
        kdtree.setInputCloud(cloud);
        ///Step 2 search a closest lidar pt for each orb feature
        for (int i = 0; i < ORBFeatures.size(); i++) {
            int minLiDARIndex = -1, minPlaneIndex = -1;
            double lix, liy;
            pcl::PointXY searchPoint;
            searchPoint.x = ORBFeatures[i].pt.x;
            searchPoint.y = ORBFeatures[i].pt.y;
            int searchNum = 5;//how many close point you want?
            std::vector<int> pointIdxKNNSearch(searchNum);
            std::vector<float> pointKNNSquaredDistance(searchNum);
            kdtree.nearestKSearch(searchPoint, searchNum, pointIdxKNNSearch, pointKNNSquaredDistance);
            if (pointKNNSquaredDistance[0] < threshold) //min distance should smaller than threshold
                minPlaneIndex = pointIdxKNNSearch[0];
            else
                continue;
            if (minPlaneIndex > -1)
                minLiDARIndex = indexPlane2indexLiDAR[minPlaneIndex];//get the index of input LiDAR dataset
            else
                continue;
            //cout<<"ORB point ORBFeatures.size(): "<<ORBFeatures.size()<<" each of while loop costs time "<<((double)(end - start) / CLOCKS_PER_SEC)*1000 << " mini sec" << endl;
            if (minLiDARIndex > -1) {//if we find a close lidar point
                if (LiDARPoints[minLiDARIndex].planeID > -1) {// if this lidar point belongs to a plane
                    ///Step 3 get depth from large plane
                    mvORBAttributions[i].floorPlaneID = LiDARPoints[minLiDARIndex].planeID;
                    ///Given Ax+BY+CZ+D=0 and u=fx*X/Z+cx, v = fy*Y/Z + cy
                    ///Z = -D / (A*(u-cx)/fx + B*(v-cy)/fy + C)
                    double A = extractedPlanes[mvORBAttributions[i].floorPlaneID].A,
                            B = extractedPlanes[mvORBAttributions[i].floorPlaneID].B,
                            C = extractedPlanes[mvORBAttributions[i].floorPlaneID].C,
                            D = extractedPlanes[mvORBAttributions[i].floorPlaneID].D;
                    double u = ORBFeatures[i].pt.x, v = ORBFeatures[i].pt.y;
                    double Z = -D / (A * (u - cx) / fx + B * (v - cy) / fy + C); //from Ax+By+Cz + D =0;
                    mvORBAttributions[i].depth = Z;
                    mvORBAttributions[i].depthSource = 1;
                    double X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy;
                    mvORBAttributions[i].p3d_est.x = X, mvORBAttributions[i].p3d_est.y = Y, mvORBAttributions[i].p3d_est.z = Z;
                }
            }
        }
        //draw
//        cv::Mat im_clone = im.clone();
//        cv::cvtColor(im, im_clone, CV_GRAY2BGR);
//        //show floor lidar points
//        for(int i=0;i<LiDARPoints.size();i++){
//            if(LiDARPoints[i].planeID>-1){
//                cv::circle(im_clone, cv::Point(LiDARPoints[i].pt2d.x,LiDARPoints[i].pt2d.y),1,cv::Scalar(0,255,255),1);
//            }
//        }
//        imshow("function connectORB2Plane : on plane ORB", im_clone);
//        cv::waitKey(1);
//        for (int i = 0; i < mvORBAttributions.size(); i++) {
//            if (mvORBAttributions[i].floorPlaneID > -1) {
//                cv::circle(im_clone, cv::Point(ORBFeatures[i].pt.x, ORBFeatures[i].pt.y), 3, cv::Scalar(0, 255, 0), 1);
//                cv::rectangle(im_clone, cv::Point(ORBFeatures[i].pt.x - 5, ORBFeatures[i].pt.y - 10),
//                              cv::Point(ORBFeatures[i].pt.x + 5, ORBFeatures[i].pt.y + 10), cv::Scalar(0, 255, 255));
//            } else {
//                cv::circle(im_clone, cv::Point(ORBFeatures[i].pt.x, ORBFeatures[i].pt.y), 3, cv::Scalar(0, 0, 255), 1);
//            }
//        }
//        imshow("function connectORB2Plane : on plane ORB", im_clone);
//        cv::waitKey(1);
//        int pause = 0;
    }

    /**
    * Connect ORBFeature to LSD lines
    * find 3 nearest LSD lines, by point to lsd distance, and point should withnin lsd bounding box
    */
    void Frame::connectORB2LSD(vector<mLine> &mLSDLinesIN, vector<cv::KeyPoint> &ORBin, cv::Mat im) {
        ///Step 0 transfer lines to Ax+By+C = 0 type
        vector<vector<float>> keyLineABCs;//store key-lines in terms of Ax+By+C = 0;
        for (auto &i: mLSDLinesIN) {
            float x1 = i.LSD.startPointX, y1 = i.LSD.startPointY;
            float x2 = i.LSD.endPointX, y2 = i.LSD.endPointY;
            float A = y2 - y1, B = x1 - x2, C = x2 * y1 - x1 * y2;//https://math.stackexchange.com/questions/422602/convert-two-points-to-line-eq-ax-by-c-0
            vector<float> thisLine;
            thisLine.push_back(A);
            thisLine.push_back(B);
            thisLine.push_back(C);
            keyLineABCs.push_back(thisLine);
        }
        ///Step 1 pair up ORB key-points with 3 nearby LSD lines
        vector<int> keyPt2LSD(ORBin.size(), -1);
        for (int i = 0; i < ORBin.size(); i++) {
            if (mvORBAttributions[i].depthSource != -1)
                continue;

            double x0 = ORBin[i].pt.x, y0 = ORBin[i].pt.y;
            float minDistance1 = 5, minDistance2 = 5, minDistance3 = 5;
            //imgKeyPoint[i].index2line = -1;
            int index1 = -1, index2 = -1, index3 = -1;
            for (int j = 0; j < mLSDLinesIN.size(); j++) {
                ///Step 1.1 search for 3 near by LSD lines
                ///point to line --- d = abs(Ax0+By0+C) / abs(sqrt(A^2+B^2))
                float A = keyLineABCs[j][0], B = keyLineABCs[j][1], C = keyLineABCs[j][2];
                float dis = abs(A * x0 + B * y0 + C) / sqrt(A * A + B * B);
                //check point should be in the bounding box
                float xs = mLSDLinesIN[j].LSD.startPointX, ys = mLSDLinesIN[j].LSD.startPointY;
                float xe = mLSDLinesIN[j].LSD.endPointX, ye = mLSDLinesIN[j].LSD.endPointY;
                //make sure : xstart < x < xend , ystart < y < yend
                if (xe < xs) {
                    float x_tmp = xs;
                    xs = xe;
                    xe = x_tmp;
                }
                if (ye < ys) {
                    float y_tmp = ys;
                    ys = ye;
                    ye = y_tmp;
                }
                if (dis < minDistance1 && (xs - 5) > x0 && x0 < (xe + 5) && (ys - 5) > y0 && y0 < (ye + 5)) {
                    minDistance3 = minDistance2;
                    index3 = index2;
                    minDistance2 = minDistance1;
                    index2 = index1;
                    minDistance1 = dis;
                    index1 = j;
                } else {
                    if (dis >= minDistance1 && dis < minDistance2 && (xs - 5) > x0 && x0 < (xe + 5) &&
                        (ys - 5) > y0 && y0 < (ye + 5)) {
                        minDistance3 = minDistance2;
                        index3 = index2;
                        minDistance2 = dis;
                        index2 = j;
                    } else {
                        if (dis >= minDistance2 && dis < minDistance3 && (xs - 5) > x0 && x0 < (xe + 5) &&
                            (ys - 5) > y0 && y0 < (ye + 5)) {
                            minDistance3 = dis;
                            index3 = j;
                        }
                    }
                }
            }
            ///Step 1.2 if any of the near LSD lines fit a 3D line
            //Because we have make sure the keypoint is within the line bounding box and distance to line is valid.
            //no need to check pt to LSD endpoints and midpoint distance
            if (index1 > -1 && mLSDLinesIN[index1].fit3DLine) {
                mvORBAttributions[i].LSDlineID = mLSDLinesIN[index1].ID;
                mvORBAttributions[i].LSDline = &mLSDLinesIN[index1];
                mvORBAttributions[i].depthSource = 2;
            } else {
                if (index2 > -1 && mLSDLinesIN[index2].fit3DLine) {
                    mvORBAttributions[i].LSDlineID = mLSDLinesIN[index2].ID;
                    mvORBAttributions[i].LSDline = &mLSDLinesIN[index2];
                    mvORBAttributions[i].depthSource = 2;

                } else {
                    if (index3 > -1 && mLSDLinesIN[index3].fit3DLine) {
                        //cout<<"mvORBAttributions size "<<mvORBAttributions.size()<<endl;
                        mvORBAttributions[i].LSDlineID = mLSDLinesIN[index3].ID;
                        mvORBAttributions[i].LSDline = &mLSDLinesIN[index3];
                        mvORBAttributions[i].depthSource = 2;
//                                    cout << "mvORBAttributions i " << i << " " << mvORBAttributions[i].keyPt.pt.x << " "
//                                         << mvORBAttributions[i].keyPt.pt.y << " change depthsource to 2" << endl;

                    }
                }
            }
        }
        ///Check
//        cv::Mat im_clone = im.clone();
//        cv::cvtColor(im,im_clone,CV_GRAY2BGR);
//        for (int i = 0; i < ORBin.size(); i++) {
//            if (mvORBAttributions[i].LSDlineID > 0) {
//                cv::circle(im_clone, cv::Point(ORBin[i].pt.x, ORBin[i].pt.y), 2, cv::Scalar(0, 255, 255));
////            cout<<i.LSDline.LSD.startPointX<<" "<<i.LSDline.LSD.startPointY<<" "
////            <<i.LSDline.LSD.endPointX<<" "<<i.LSDline.LSD.endPointY<<endl;
//
//                cv::line(im_clone, cv::Point(mvORBAttributions[i].LSDline->LSD.startPointX,
//                                       mvORBAttributions[i].LSDline->LSD.startPointY),
//                         cv::Point(mvORBAttributions[i].LSDline->LSD.endPointX,
//                                   mvORBAttributions[i].LSDline->LSD.endPointY), cv::Scalar(102, 102, 255), 2);
////            imshow("ORB2LSD", im);
////            waitKey(1);
//            }
//        }
//        imshow("ORB to LSD", im_clone);
//        cv::waitKey(1);
//        int pause = 0;
    }

    void Frame::ORBdepthFromLine(vector<mLine> &lineInputs, vector<mORBAttribution> &ORBinputs, cv::Mat im) {
        for (int i = 0; i < ORBinputs.size(); i++) {
            if (ORBinputs[i].depthSource == 1)
                continue;
            if (ORBinputs[i].LSDlineID <= -1)
                continue;
            mLine *thisLine = &lineInputs[mvORBAttributions[i].LSDlineID];
            if (thisLine->fit3DLine) {
                ///Complicate formula below, be careful
                ///Given X=(u-Cx)*Z/fx | Y=(v-Cy)*Z/fy | Z=(Fx*X)/(u-Cx)
                ///Given (X-x1)/l=(Y-y1)/m=(Z-z1)/n
                ///We have:
                ///X=(Cx*x1*n-u*x1*n+u*z1*l-Cx*z1*l)/(fx*l-u*n+Cx*n)
                ///Y=(Cy*y1*n-v*y1*n+v*z1*m-Cy*z1*m)/(fy*m+Cy*n-v*n)
                ///Z=(fx*X)/(u-Cx)

                ///Step1 get the (X-x1)/l=(Y-y1)/m=(Z-z1)/n
                double u = ORBinputs[i].keyPt.pt.x, v = ORBinputs[i].keyPt.pt.y;
                double l = (thisLine->pt3dStart.x - thisLine->pt3dEnd.x),
                        m = (thisLine->pt3dStart.y - thisLine->pt3dEnd.y),
                        n = (thisLine->pt3dStart.z - thisLine->pt3dEnd.z);
                double x1 = thisLine->pt3dStart.x, y1 = thisLine->pt3dStart.y, z1 = thisLine->pt3dStart.z;
                double X = (cx * x1 * n - u * x1 * n + u * z1 * l - cx * z1 * l) / (fx * l - u * n + cx * n);
                double Y = (cy * y1 * n - v * y1 * n + v * z1 * m - cy * z1 * m) / (fy * m + cy * n - v * n);
                double Z = (fx * X) / (u - cx);
                mvORBAttributions[i].depth = Z;
                mvORBAttributions[i].p3d_est.x = X, mvORBAttributions[i].p3d_est.y = Y, mvORBAttributions[i].p3d_est.z = Z;
                mvORBAttributions[i].depthSource = 2;
            }
        }
    }

    /**
     * get depth from nearby LiDAR point. only for those keypoints that away from Other LiDAR features(plane and line)
     * @param lineInputs
     * @param ORBinputs
     * @param im
     */
    void Frame::ORBdepthFromPoint(vector<PtLsr> &LiDARInputs, vector<mORBAttribution> &ORBinputs, double threshold,
                                  cv::Mat im) {

        double thresholdSquare = threshold * threshold;
        //prepare a pcl kdtree
        pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
        cloud->width = LiDARInputs.size();
        cloud->height = 1;
        cloud->points.resize(cloud->width * cloud->height);
        for (int i = 0; i < LiDARInputs.size(); i++) {
            (*cloud)[i].x = LiDARInputs[i].pt2d.x;
            (*cloud)[i].y = LiDARInputs[i].pt2d.y;
        }
        pcl::KdTreeFLANN<pcl::PointXY> kdtree;
        kdtree.setInputCloud(cloud);

        //Search !
        for (int i = 0; i < ORBinputs.size(); i++) {
            if (ORBinputs[i].depthSource > 0)
                continue;
            //Option 1 naive way
            int minID = -1;
//            double minDistance = 999;
//            for (auto liPT: LiDARInputs) {
//                double distance = (liPT.pt2d.x - mvKeysUn[i].pt.x) * (liPT.pt2d.x - mvKeysUn[i].pt.x)
//                                  + (liPT.pt2d.y - mvKeysUn[i].pt.y) * (liPT.pt2d.y - mvKeysUn[i].pt.y);
//                if (distance < minDistance) {
//                    minDistance = distance;
//                    minID = liPT.ptID;
//                }
//            }

            //Option 2 kd-tree search
            pcl::PointXY searchPoint;
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            searchPoint.x = mvKeysUn[i].pt.x;
            searchPoint.y = mvKeysUn[i].pt.y;
            kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
            if (pointNKNSquaredDistance[0] < thresholdSquare) {
                //cout<<"minID "<<minID<<" dis "<<minDistance<<" pointIdxNKNSearch "<<pointIdxNKNSearch[0]<<" dis "<<pointNKNSquaredDistance[0]<<endl;
                minID = pointIdxNKNSearch[0];
                ORBinputs[i].depthSource = 3;
                ORBinputs[i].depth = LiDARInputs[minID].pt3d.z;
                ORBinputs[i].p3d_est.x = (mvKeysUn[i].pt.x - cx) * ORBinputs[i].depth / fx;
                ORBinputs[i].p3d_est.y = (mvKeysUn[i].pt.y - cy) * ORBinputs[i].depth / fy;
                ORBinputs[i].p3d_est.z = ORBinputs[i].depth;
                ORBinputs[i].LiDARPtID = minID;
                ORBinputs[i].LiDARPt = &LiDARInputs[minID];
                //cout<<pt.keyPt<<" id "<<pt.ID<<" min distance "<<minDistance<<endl;
            }
        }
//        cv::Mat im_clone = im.clone();
//        cv::cvtColor(im,im_clone,CV_GRAY2BGR);
//        for (int i = 0; i < ORBinputs.size(); i++) {
//            if (ORBinputs[i].depthSource != 3)
//                continue;
//            cv::circle(im_clone,cv::Point(ORBinputs[i].keyPt->pt.x,ORBinputs[i].keyPt->pt.y),1,cv::Scalar(0,255,0));
//            cv::line(im_clone,cv::Point(ORBinputs[i].keyPt->pt.x,ORBinputs[i].keyPt->pt.y),
//                     cv::Point(ORBinputs[i].LiDARPt->pt2d.x,ORBinputs[i].LiDARPt->pt2d.y),cv::Scalar(255,0,0) );
//        }
//        cv::imshow("ORB depth from LiDAR Point ", im_clone);
//        cv::waitKey(0);
//        int pause = 1;
    }

    /**
 * get depth from nearby LiDAR patch plane.
 * @param LiDARInputs
 * @param ORBinputs
 * @param threshold
 * @param im
 */
    void Frame::ORBdepthFromPointPatch(vector<PtLsr> &LiDARInputs, vector<mORBAttribution> &ORBinputs, double threshold, cv::Mat im){
        for(auto &pt:ORBinputs){
            if(pt.depthSource>0)
                continue;

            //Step 1 find a nearest bin
            //Find the nearest LiDAR points
            pcl::PointCloud<pcl::PointXYZ>::Ptr localPatch
            = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
            double minDistance = 999;
            int minID = -1;
            for(const auto& lidPT:LiDARInputs){
                bool xok, yok;
                xok = abs(lidPT.pt2d.x - pt.keyPt.pt.x) <= threshold;
                yok = abs(lidPT.pt2d.y - pt.keyPt.pt.y) <= threshold;
                if(xok&&yok)
                    localPatch->points.emplace_back(lidPT.pt3d.x,lidPT.pt3d.y,lidPT.pt3d.z);
            }
            if(localPatch->points.size()>=3){
                //Divide into frontend, backend by histogram
                pcl::PointXYZ min_pt, max_pt;
                pcl::getMinMax3D(*localPatch, min_pt, max_pt);
                // Compute the bin number
                int num_bins = ceil((max_pt.z - min_pt.z)/0.3);
                // Create a histogram with num_bins bins
                std::vector<std::vector<int>> histogram;
                for (int i = 0; i < num_bins; i++) {
                    std::vector<int> localBin;
                    histogram.push_back(localBin);
                }
                // Iterate over the points in the point cloud
                for (int i = 0; i < localPatch->points.size(); i++) {
                    // Compute the bin index for the point based on its distance from the middle of the bounding box
                    int bin_index = floor(num_bins * (localPatch->points[i].z - min_pt.z) / (max_pt.z - min_pt.z));
                    if(localPatch->points[i].z==max_pt.z)
                        bin_index = num_bins -1;
                    // Add the point to the appropriate histogram bin
                    histogram[bin_index].push_back(i);
                }
                // Define the significant bin size and find nearest significant bin index
                int numThreshold = localPatch->points.size() * 0.2;
                int selectedBinIndex = -1;
                for(int i=0;i<num_bins;i++){
//                    if(histogram[i].size()>0){
//                        cout<<"histogram "<<i<<" has z ";
//                        for(int j=0;j<histogram[i].size();j++){
//                            int index = histogram[i][j];
//                            cout<<localPatch->points[index].z<<" ";
//                        }
//                        cout<<endl;
//                    }
                    //find the nearest bin
                    if(histogram[i].size()>=numThreshold){
                        selectedBinIndex = i;
                        break;
                    }
                }
//                cout<<"selectedBinIndex "<<selectedBinIndex<<endl;
                if(selectedBinIndex>-1&&histogram[selectedBinIndex].size()>=3){
                    //Step 2 fit a plane by those points
                    pcl::PointCloud<pcl::PointXYZ>::Ptr localPointCloud =
                            pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
                    for(int i=0;i<histogram[selectedBinIndex].size();i++){
                        int ptIndex = histogram[selectedBinIndex][i];
                        localPointCloud->points.push_back(localPatch->points[ptIndex]);
                        //cout<<localPatch->points[ptIndex]<<endl;
                    }
                    //RANSAC
                    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(localPointCloud));
                    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
                    ransac.setDistanceThreshold(0.1);
                    ransac.computeModel();
                    Eigen::VectorXf coefficients;
                    ransac.getModelCoefficients(coefficients);
                    //cout<<coefficients[0]<<" "<<coefficients[1]<<" "<<coefficients[2]<<" "<<coefficients[3]<<endl;
                    //calc Z
                    double A = coefficients[0], B = coefficients[1], C = coefficients[2], D = coefficients[3];
                    double u = pt.keyPt.pt.x, v = pt.keyPt.pt.y;
                    //cout<<"option 2 "<<endl;
                    //cout<<"u "<<u<<" v "<<v<<" A "<<A<<" B "<<B<<" C "<<C<<" D "<<D<<" fx "<<fx<<" fy "<<fy<<" cx "<<cx<<" cy "<<cy<<" ---------------------------------------------------------------------"<<endl;
                    double Z = -D / (A*(u-cx)/fx + B*(v-cy)/fy + C); //from Ax+By+Cz + D =0;
//                    if(abs(Z-localPointCloud->points[0].z)<=1){
                        pt.depth = Z;
                        pt.depthSource = 3;
                        double X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy;
                        pt.p3d_est.x = X, pt.p3d_est.y = Y, pt.p3d_est.z = Z;
//                    }else{
//                        //cout<<"estimated Z "<<Z<<" far away from candidates Z "<<localPointCloud->points[0].z<<endl;
//                    }

                }
            }
        }
    }

    ///Added module
    /**
     * @brief search the image feature with nearby Laser depth
     */
    void Frame::PairLaserVisionFeatures(const cv::Mat &im) {
        std::clock_t start = clock();
        ///Step 1 Search for LiDAR Plane
        PlaneFitting();
        std::clock_t end = clock();
        //cout << "PlaneFitting costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;

        start = clock();
        ///Step 2 Search for LSD
        cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        vector<cv::line_descriptor::KeyLine> priKeylines;
        cv::Mat mask = cv::Mat::ones(im.size(), CV_8UC1);
        bd->detect(im, priKeylines, mask);
        cv::Mat descriptors;
        std::vector<bool> keylineMergeFlags(priKeylines.size(), false);
        mergeKeyLines(priKeylines, keylineMergeFlags, descriptors, im.clone());
        vector<cv::line_descriptor::KeyLine> selectedKeyLines;
        int lineCounter = 0;
        const float lineLengthThres = 50;
        for (int i = 0; i < priKeylines.size(); i++) {
            if (keylineMergeFlags[i])
                continue;
            if (priKeylines[i].lineLength > lineLengthThres) {
                selectedKeyLines.push_back(priKeylines[i]);
                //Store to mLines
                mLine newLine(lineCounter, priKeylines[i]);
                mvLines.push_back(newLine);
                lineCounter++;
            }
        }
        //calc descriptors
        bd->compute(im, selectedKeyLines, descriptors);
        mlsdDescriptors = descriptors.clone();
        end = clock();
        //cout << "Search for LSD and LiDAR line costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;

        ///Step 3 connect LSD and LiDAR, fitting a 3D Line
        start = clock();
        connectLSD2LiDAR(mvLines, mLaserPt_cam, im.clone());
        end = clock();
        //cout << "connectLSD2LiDAR costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;
        start = clock();
        LineFitting(mvLines, mLaserPt_cam, im.clone());
        end = clock();
        //cout << "LineFitting costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;

        ///Step 4 pair ORB point with LiDAR Plane
        start = clock();
        connectORB2Plane(mLaserPt_cam, mvKeysUn, mvPlanes, 100, im.clone());
        end = clock();

        ///Step 5 connect ORB point and LiDAR line
        start = clock();
        connectORB2LSD(mvLines, mvKeysUn, im.clone());
        end = clock();
        //cout << "mvLines costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;

        //check mvvkeysun and mvorbattributions, a debug check. should not print anything
        for (int i = 0; i < mvKeysUn.size(); i++)
            if (mvKeysUn[i].pt.x != mvORBAttributions[i].keyPt.pt.x
                || mvKeysUn[i].pt.y != mvORBAttributions[i].keyPt.pt.y)
                cout << "mvkeysun unmatch i to orbattribution i " << i << " : " << mvKeysUn[i].pt.x << " "
                     << mvKeysUn[i].pt.y
                     << "  | " << mvORBAttributions[i].keyPt.pt.x << " " << mvORBAttributions[i].keyPt.pt.y << endl;

        ///! NOTE there is a big issue that ORB member's line is not the same pointer with mline vector.
        /// For safety, reach Line by Line ID and mLines
        start = clock();
        ORBdepthFromLine(mvLines, mvORBAttributions, im.clone());
        end = clock();
        //cout << "ORBdepthFromLine costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;
//        counter = 0;
//        for(int i=0;i<mvORBAttributions.size();i++){
//            if (mvORBAttributions[i].depthSource == 2) {
//                cout<<"ID"<<mvORBAttributions[i].ID<<" 2d "<<mvORBAttributions[i].keyPt.pt.x<<" "<<mvORBAttributions[i].keyPt.pt.y
//                    <<" LineID "<<mvORBAttributions[i].LSDlineID<<" depthsource "<<mvORBAttributions[i].depthSource<<" depth "<<mvORBAttributions[i].depth<<endl;
//                counter++;
//            }
//        }
//        cout << counter << " ORBS points on line" << endl;

        ///Step 6 connect ORB point and LiDAR point / LiDAR point patch
        start = clock();
        ORBdepthFromPoint(mLaserPt_cam, mvORBAttributions, 2, im.clone());
        //ORBdepthFromPointPatch(mLaserPt_cam, mvORBAttributions, 5, im.clone());
        end = clock();
        //cout << "ORBdepthFromPoint costs " << ((double)(end - start) / CLOCKS_PER_SEC)*1000 << "mini second" << endl;

//        counter = 0;
//        for(int i=0;i<mvORBAttributions.size();i++){
//            if (mvORBAttributions[i].depthSource == 3) {
//                cout<<"ID"<<mvORBAttributions[i].ID<<" 2d "<<mvORBAttributions[i].keyPt.pt.x<<" "<<mvORBAttributions[i].keyPt.pt.y
//                    <<" depthsource "<<mvORBAttributions[i].depthSource<<" depth "<<mvORBAttributions[i].depth<<endl;
//                counter++;
//            }
//        }
//        cout << counter << " ORBS points on point" << endl;

        //show-all
//        cv::Mat im_clone=im.clone();
//        cv::cvtColor(im, im_clone, CV_GRAY2BGR);
//        for(int i=0;i<mLaserPt_cam.size();i++){
//            if(mLaserPt_cam[i].planeID==-1)
//                cv::circle(im_clone,cv::Point(mLaserPt_cam[i].pt2d.x,mLaserPt_cam[i].pt2d.y),1,cv::Scalar(102,255,255));
//            if(mLaserPt_cam[i].planeID>-1){
//                cv::circle(im_clone,cv::Point(mLaserPt_cam[i].pt2d.x,mLaserPt_cam[i].pt2d.y),1,cv::Scalar(51,153,255));//orange
//            }
//        }
//        for(int i=0;i<mvORBAttributions.size();i++){
//            if (mvORBAttributions[i].depthSource == -1) {
//                cv::circle(im_clone,cv::Point(mvORBAttributions[i].keyPt.pt.x,mvORBAttributions[i].keyPt.pt.y)
//                        ,2,cv::Scalar(0,0,255),-1);//red
//            }
//            if (mvORBAttributions[i].depthSource == 1) {
//                cv::circle(im_clone,cv::Point(mvORBAttributions[i].keyPt.pt.x,mvORBAttributions[i].keyPt.pt.y)
//                           ,3,cv::Scalar(0,255,0),-1);//green
//            }
//            if (mvORBAttributions[i].depthSource == 2) {
//                cv::circle(im_clone,cv::Point(mvORBAttributions[i].keyPt.pt.x,mvORBAttributions[i].keyPt.pt.y)
//                           ,3,cv::Scalar(255,0,127),-1);//purple
//            }
//            if (mvORBAttributions[i].depthSource == 3) {
//                cv::circle(im_clone,cv::Point(mvORBAttributions[i].keyPt.pt.x,mvORBAttributions[i].keyPt.pt.y)
//                           ,3,cv::Scalar(255,255,0),-1);//light blue
//            }
//        }
//        string windowName = "fusioned features " + to_string(mnId);
//        //cout<<windowName<<endl;
//        cv::imshow(windowName, im_clone);
//        cv::waitKey(0);

        ///Step 6 check if est depth larger than 20 meter
        for (int i = 0; i < mvORBAttributions.size(); i++) {
            if (mvORBAttributions[i].depthSource > -1) {
                givenDepthNum++;
                if (mvORBAttributions[i].depth > 20) {
                    mvORBAttributions[i].depthSource = -1;
                    mvORBAttributions[i].depth = -1;
                    mvORBAttributions[i].p3d_est = cv::Point3d(-1, -1, -1);
                    givenDepthNum--;
                }
            }
        }
        //cout<<"givenDepthNum "<<givenDepthNum<<endl;
    }

/** Added module
 * @brief Project LiDAR Points to Image. src: https://github.com/williamhyin/lidar_to_camera/blob/master/src/project_lidar_to_camera.cpp
 * @param[in] mK : camera distortion and project parameters
 * @param[in] cols : image cols boundary
 * @param[in] rows : image rows boundary
 */
    void Frame::ProjectLiDARtoImg(cv::Mat mK, int cols, int rows) {
        //it seems like OpenCV upgraded, then the func changed?
        //cv::Mat P_rect_00 = cv::Mat::zeros(CvSize(4, 3), CV_64F);
        cv::Mat P_rect_00 = cv::Mat::zeros(3, 4, CV_64F);
        P_rect_00.at<double>(0, 0) = (double) mK.at<float>(0, 0);
        P_rect_00.at<double>(0, 2) = (double) mK.at<float>(0, 2);
        P_rect_00.at<double>(1, 1) = (double) mK.at<float>(1, 1);
        P_rect_00.at<double>(1, 2) = (double) mK.at<float>(1, 2);
        P_rect_00.at<double>(2, 2) = 1;
        //cv::Mat R_rect_00 = cv::Mat::eye(CvSize(4, 4), CV_64F);
        cv::Mat R_rect_00 = cv::Mat::eye(4, 4, CV_64F);
        int ptNum = mLaserPt_cam.size();
        cv::Mat X(4, 1, CV_64F);//3D LiDAR point
        cv::Mat Y(3, 1, CV_64F);//2D LiDAR projection
        int counter = 0;
        for (int pi = 0; pi < ptNum; pi++) {
            cv::Point2d pt;
            X.at<double>(0, 0) = mLaserPt_cam[pi].pt3d.x;
            X.at<double>(1, 0) = mLaserPt_cam[pi].pt3d.y;
            X.at<double>(2, 0) = mLaserPt_cam[pi].pt3d.z;
            X.at<double>(3, 0) = 1;
            Y = P_rect_00 * R_rect_00 * X;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
            if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) {
                //cout<<X.at<double>(0, 0)<<" "<<X.at<double>(1, 0)<<" "<<X.at<double>(2, 0)<<" -> "<<pt.x<<" "<<pt.y<<endl;
                mLaserPt_cam[pi].index2d = -1;
                continue;
            }
            //cout<<X.at<double>(0, 0)<<" "<<X.at<double>(1, 0)<<" "<<X.at<double>(2, 0)<<" -> "<<pt.x<<" "<<pt.y<<endl;
            mLaserPt_cam[pi].pt2d = pt;
            mLaserPt_cam[pi].index2d = counter;
            counter++;
        }
        //cout << "Lidar points total : " << mLaserPt_cam.size() << " in image frame : " << counter << endl;
    }


    /**
     * @brief Project LiDAR feature (camera coordination) to image
     * @param mK camera intrinsic parameter
     * @param cols
     * @param rows
     */
    void Frame::ProjectLiDARFeaturetoImg(cv::Mat mK, int cols, int rows) {
        //it seems like OpenCV upgraded, then the func changed?
        //cv::Mat P_rect_00 = cv::Mat::zeros(CvSize(4, 3), CV_64F);
        cv::Mat P_rect_00 = cv::Mat::zeros(3,4,CV_64F);
        P_rect_00.at<double>(0, 0) = (double) mK.at<float>(0, 0);
        P_rect_00.at<double>(0, 2) = (double) mK.at<float>(0, 2);
        P_rect_00.at<double>(1, 1) = (double) mK.at<float>(1, 1);
        P_rect_00.at<double>(1, 2) = (double) mK.at<float>(1, 2);
        P_rect_00.at<double>(2, 2) = 1;
        //cv::Mat R_rect_00 = cv::Mat::eye(CvSize(4, 4), CV_64F);
        cv::Mat R_rect_00 = cv::Mat::eye(4, 4, CV_64F);
        ///Step 1. Project Corner Point First
        int ptNum = mLaserCorner_cam.size();
        cv::Mat X(4, 1, CV_64F);//3D LiDAR point
        cv::Mat Y(3, 1, CV_64F);//2D LiDAR projection
        int counter = 0;
        for (int pi = 0; pi < ptNum; pi++) {
            cv::Point pt;
            X.at<double>(0, 0) = mLaserCorner_cam[pi].pt3d.x;
            X.at<double>(1, 0) = mLaserCorner_cam[pi].pt3d.y;
            X.at<double>(2, 0) = mLaserCorner_cam[pi].pt3d.z;
            X.at<double>(3, 0) = 1;
            Y = P_rect_00 * R_rect_00 * X;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
            //cout<<mCornerPointsSharp.points[pi].x<<" "<<mCornerPointsSharp.points[pi].y<<" "<<mCornerPointsSharp.points[pi].z<<" "<<pt.x<<" "<<pt.y<<endl;
            if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) {
                mLaserCorner_cam[pi].index2d = -1;
                continue;
            }
            mLaserCorner_cam[pi].pt2d = pt;
            mLaserCorner_cam[pi].index2d = counter;
            counter++;
        }

        //cout <<" Lidar in-frame corner points "<<counter<<"/"<<mLaserCorner_cam.size();
        ///Step 2. Project Less Corner Point First
        ptNum = mLaserLessCorner_cam.size();
        counter = 0;
        for (int pi = 0; pi < ptNum; pi++) {
            cv::Point pt;
            X.at<double>(0, 0) = mLaserLessCorner_cam[pi].pt3d.x;
            X.at<double>(1, 0) = mLaserLessCorner_cam[pi].pt3d.y;
            X.at<double>(2, 0) = mLaserLessCorner_cam[pi].pt3d.z;
            X.at<double>(3, 0) = 1;
            Y = P_rect_00 * R_rect_00 * X;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
            if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) {
                mLaserLessCorner_cam[pi].index2d = -1;
                continue;
            }
            mLaserLessCorner_cam[pi].pt2d = pt;
            mLaserLessCorner_cam[pi].index2d = counter;
            counter++;
        }
        //cout <<" Lidar less corner points total : " << mCornerPointsLessSharp.size() << " in image frame : " << counter;
        //cout <<" Lidar in-frame less corner points "<<counter<<"/"<<mLaserLessCorner_cam.size();
        ///Step 3. Project Flat Point First
        ptNum = mLaserFlat_cam.size();
        counter = 0;
        for (int pi = 0; pi < ptNum; pi++) {
            cv::Point pt;
            X.at<double>(0, 0) = mLaserFlat_cam[pi].pt3d.x;
            X.at<double>(1, 0) = mLaserFlat_cam[pi].pt3d.y;
            X.at<double>(2, 0) = mLaserFlat_cam[pi].pt3d.z;
            X.at<double>(3, 0) = 1;
            Y = P_rect_00 * R_rect_00 * X;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
            if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) {
                mLaserFlat_cam[pi].index2d = -1;
                continue;
            }
            mLaserFlat_cam[pi].pt2d = pt;
            mLaserFlat_cam[pi].index2d = counter;
            counter++;
        }
        //cout <<" Lidar surface points total : " << mSurfPointsFlat.size() << " in image frame : " << counter;
        //cout <<" Lidar in-frame surface points "<<counter<<"/"<<mLaserFlat_cam.size();
        ///Step 4. Project Less Flat Point First
        ptNum = mLaserLessFlat_cam.size();
        counter = 0;
        for (int pi = 0; pi < ptNum; pi++) {
            cv::Point pt;
            X.at<double>(0, 0) = mLaserLessFlat_cam[pi].pt3d.x;
            X.at<double>(1, 0) = mLaserLessFlat_cam[pi].pt3d.y;
            X.at<double>(2, 0) = mLaserLessFlat_cam[pi].pt3d.z;
            X.at<double>(3, 0) = 1;
            Y = P_rect_00 * R_rect_00 * X;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
            if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) {
                mLaserLessFlat_cam[pi].index2d = -1;
                continue;
            }
            mLaserLessFlat_cam[pi].pt2d = pt;
            mLaserLessFlat_cam[pi].index2d = counter;
            counter++;
        }
        //cout <<" Lidar less surface points total : " << mSurfPointsLessFlat.size() << " in image frame : " << counter <<endl;
//        cout <<"Lidar in-frame less surface points "<<counter<<"/"<<mLaserLessFlat_cam.size();
    }

    /** Added Module
     * @brief Project LiDAR Points and Features (PCL Pointset) to Camera coordination system
     * Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
     * Velodyne System : X front, Y left, Z up
     */
    void Frame::ProjectLiDARtoCam() {
        double maxX = 50.0, maxY = 50.0, minZ = 20.0;
        ///Step1 Project common LiDAR points
        int IDCounter = 0;
        //chose mlaserScansPoints or mLaser16ScansPoints
        for (int i = 0; i < mlaserScansPoints.size(); i++) {
            for (int j = 0; j < mlaserScansPoints[i].points.size(); j++) {
                cv::Mat P_lidar(4, 1, CV_64F);//3D LiDAR point
                cv::Mat P_cam(4, 1, CV_64F);//3D LiDAR point under Cam coordination
                //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
                //X front, Y left, Z up
                if (mlaserScansPoints[i].points[j].x > maxX || mlaserScansPoints[i].points[j].x < 0.0
                    || mlaserScansPoints[i].points[j].y > maxY || mlaserScansPoints[i].points[j].y < -maxY
                    || mlaserScansPoints[i].points[j].z > minZ || mlaserScansPoints[i].points[j].z < -minZ) {
                    continue;
                }
                P_lidar.at<double>(0, 0) = mlaserScansPoints[i].points[j].x;
                P_lidar.at<double>(1, 0) = mlaserScansPoints[i].points[j].y;
                P_lidar.at<double>(2, 0) = mlaserScansPoints[i].points[j].z;
                P_lidar.at<double>(3, 0) = 1;
                P_cam = mTcamlid * P_lidar;
                cv::Point3d newP;
                newP.x = P_cam.at<double>(0, 0);
                newP.y = P_cam.at<double>(1, 0);
                newP.z = P_cam.at<double>(2, 0);
                PtLsr newPtLsr;
                newPtLsr.pt3d = newP;
                newPtLsr.intensity = mlaserScansPoints[i].points[j].intensity;
                newPtLsr.scanID = i;
                newPtLsr.pointID = j;
                newPtLsr.low = false;
                newPtLsr.planeID = -1;
                newPtLsr.ptID = IDCounter;
                if (mlaserScansPoints[i].points[j].z < -1.6)
                    newPtLsr.low = true;
                IDCounter++;
                mLaserPt_cam.push_back(newPtLsr);
            }
        }
//        for (int i = 0; i < mLaser16ScansPoints.size(); i++) {
//            for (int j = 0; j < mLaser16ScansPoints[i].points.size(); j++) {
//                cv::Mat P_lidar(4, 1, CV_64F);//3D LiDAR point
//                cv::Mat P_cam(4, 1, CV_64F);//3D LiDAR point under Cam coordination
//                //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
//                //X front, Y left, Z up
//                if (mLaser16ScansPoints[i].points[j].x > maxX || mLaser16ScansPoints[i].points[j].x < 0.0
//                    || mLaser16ScansPoints[i].points[j].y > maxY || mLaser16ScansPoints[i].points[j].y < -maxY
//                    || mLaser16ScansPoints[i].points[j].z > minZ || mLaser16ScansPoints[i].points[j].z  < -minZ) {
//                    continue;
//                }
//                P_lidar.at<double>(0, 0) = mLaser16ScansPoints[i].points[j].x;
//                P_lidar.at<double>(1, 0) = mLaser16ScansPoints[i].points[j].y;
//                P_lidar.at<double>(2, 0) = mLaser16ScansPoints[i].points[j].z;
//                P_lidar.at<double>(3, 0) = 1;
//                P_cam = mTcamlid * P_lidar;
//                cv::Point3d newP;
//                newP.x = P_cam.at<double>(0, 0);
//                newP.y = P_cam.at<double>(1, 0);
//                newP.z = P_cam.at<double>(2, 0);
//                PtLsr newPtLsr;
//                newPtLsr.pt3d = newP;
//                newPtLsr.intensity =  mLaser16ScansPoints[i].points[j].intensity;
//                newPtLsr.scanID = i;
//                newPtLsr.pointID = j;
//                newPtLsr.low = false;
//                newPtLsr.planeID = -1;
//                newPtLsr.ptID = IDCounter;
//                if (mLaser16ScansPoints[i].points[j].z < -1.6)
//                    newPtLsr.low = true;
//                IDCounter++;
//                mLaserPt_cam.push_back(newPtLsr);
//            }
//        }
        ///Step 2 Project LiDAR feature points
        //int lsrCornerNum = mLsrKeyCorner.size();
        int lsrCornerNum = mCornerPointsSharp.points.size();
        if (lsrCornerNum > 0) {
            cv::Mat P_lidar(4, 1, CV_64F);//3D LiDAR point
            cv::Mat P_cam(4, 1, CV_64F);//3D LiDAR point under Cam coordination
            int counter = 0;
            for (int li = 0; li < lsrCornerNum; li++) {
                //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
                //X front, Y left, Z up
                //Todo Test a valid threshold?
                double maxX = 25.0, maxY = 25.0, minZ = 1.8;
                if (mCornerPointsSharp.points[li].x > maxX || mCornerPointsSharp.points[li].x < 0.0
                    || mCornerPointsSharp.points[li].y > maxY || mCornerPointsSharp.points[li].y < -maxY
                    || mCornerPointsSharp.points[li].z > 10 || mCornerPointsSharp.points[li].z < -minZ) {
                    continue;
                }
                P_lidar.at<double>(0, 0) = mCornerPointsSharp.points[li].x;
                P_lidar.at<double>(1, 0) = mCornerPointsSharp.points[li].y;
                P_lidar.at<double>(2, 0) = mCornerPointsSharp.points[li].z;
                P_lidar.at<double>(3, 0) = 1;
                P_cam = mTcamlid * P_lidar;
                cv::Point3d newP;
                newP.x = P_cam.at<double>(0, 0);
                newP.y = P_cam.at<double>(1, 0);
                newP.z = P_cam.at<double>(2, 0);
                PtLsr newPtLsr;
                newPtLsr.pt3d = newP;
                newPtLsr.index3d = counter;
                newPtLsr.intensity = mCornerPointsSharp.points[li].intensity;
                //if(counter<5)
                //    cout<<"pass mCornerPointsSharp.points[li].intensity; "<<mCornerPointsSharp.points[li].intensity<<" to newPtLsr.intensity "<<newPtLsr.intensity<<endl;
                counter++;
                mLaserCorner_cam.push_back(newPtLsr);
                //cout<<mLaserCorner_cam[counter-1].intensity<<endl;
            }
        }
        int lsrLessCornerNum = mCornerPointsLessSharp.size();
        if (lsrLessCornerNum > 0) {
            cv::Mat P_lidar(4, 1, CV_64F);//3D LiDAR point
            cv::Mat P_cam(4, 1, CV_64F);//3D LiDAR point under Cam coordination
            int counter = 0;
            for (int li = 0; li < lsrLessCornerNum; li++) {
                //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
                //X front, Y left, Z up
                //Todo Test a valid threshold?
                double maxX = 25.0, maxY = 25.0, minZ = 1.8;
                if (mCornerPointsLessSharp.points[li].x > maxX || mCornerPointsLessSharp.points[li].x < 0.0
                    || mCornerPointsLessSharp.points[li].y > maxY || mCornerPointsLessSharp.points[li].y < -maxY
                    || mCornerPointsLessSharp.points[li].z > 10 || mCornerPointsLessSharp.points[li].z < -minZ) {
                    continue;
                }
                P_lidar.at<double>(0, 0) = mCornerPointsLessSharp.points[li].x;
                P_lidar.at<double>(1, 0) = mCornerPointsLessSharp.points[li].y;
                P_lidar.at<double>(2, 0) = mCornerPointsLessSharp.points[li].z;
                P_lidar.at<double>(3, 0) = 1;
                P_cam = mTcamlid * P_lidar;
                cv::Point3d newP;
                newP.x = P_cam.at<double>(0, 0);
                newP.y = P_cam.at<double>(1, 0);
                newP.z = P_cam.at<double>(2, 0);
                PtLsr newPtLsr;
                newPtLsr.pt3d = newP;
                newPtLsr.index3d = counter;
                newPtLsr.intensity = mCornerPointsLessSharp.points[li].intensity;
                //if(counter<5)
                //    cout<<"pass mCornerPointsLessSharp.points[li].intensity; "<<mCornerPointsLessSharp.points[li].intensity<<" to newPtLsr.intensity "<<newPtLsr.intensity<<endl;
                counter++;
                mLaserLessCorner_cam.push_back(newPtLsr);
            }
        }
        int lsrFlatNum = mSurfPointsFlat.size();
        if (lsrFlatNum > 0) {
            cv::Mat P_lidar(4, 1, CV_64F);//3D LiDAR point
            cv::Mat P_cam(4, 1, CV_64F);//3D LiDAR point under Cam coordination
            int counter = 0;
            for (int li = 0; li < lsrFlatNum; li++) {
                //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
                //X front, Y left, Z up
                //Todo Test a valid threshold?
                double maxX = 25.0, maxY = 25.0, minZ = 1.8;
                if (mSurfPointsFlat.points[li].x > maxX || mSurfPointsFlat.points[li].x < 0.0
                    || mSurfPointsFlat.points[li].y > maxY || mSurfPointsFlat.points[li].y < -maxY
                    || mSurfPointsFlat.points[li].z > 10 || mSurfPointsFlat.points[li].z < -minZ) {
                    continue;
                }
                P_lidar.at<double>(0, 0) = mSurfPointsFlat.points[li].x;
                P_lidar.at<double>(1, 0) = mSurfPointsFlat.points[li].y;
                P_lidar.at<double>(2, 0) = mSurfPointsFlat.points[li].z;
                P_lidar.at<double>(3, 0) = 1;
                P_cam = mTcamlid * P_lidar;
                cv::Point3d newP;
                newP.x = P_cam.at<double>(0, 0);
                newP.y = P_cam.at<double>(1, 0);
                newP.z = P_cam.at<double>(2, 0);
                PtLsr newPtLsr;
                newPtLsr.pt3d = newP;
                newPtLsr.index3d = counter;
                newPtLsr.intensity = mSurfPointsFlat.points[li].intensity;
                //if(counter < 5)
                //    cout<<"pass mSurfPointsFlat.points[li].intensity; "<<mSurfPointsFlat.points[li].intensity<<" to newPtLsr.intensity "<<newPtLsr.intensity<<endl;
                counter++;
                mLaserFlat_cam.push_back(newPtLsr);
            }
        }
        int lsrLessFlatNum = mSurfPointsLessFlat.size();
        if (lsrLessFlatNum > 0) {
            cv::Mat P_lidar(4, 1, CV_64F);//3D LiDAR point
            cv::Mat P_cam(4, 1, CV_64F);//3D LiDAR point under Cam coordination
            int counter = 0;
            for (int li = 0; li < lsrLessFlatNum; li++) {
                //Velodyne Vertical FOV 26.9 mounted on 1.73. At 6 meter-distance, tan(26.9/2). it can only detect ~ 1.52+1.73 height
                //X front, Y left, Z up
                //Todo Test a valid threshold?
                double maxX = 25.0, maxY = 25.0, minZ = 1.8;
                if (mSurfPointsLessFlat.points[li].x > maxX || mSurfPointsLessFlat.points[li].x < 0.0
                    || mSurfPointsLessFlat.points[li].y > maxY || mSurfPointsLessFlat.points[li].y < -maxY
                    || mSurfPointsLessFlat.points[li].z > 10 || mSurfPointsLessFlat.points[li].z < -minZ) {
                    continue;
                }
                P_lidar.at<double>(0, 0) = mSurfPointsLessFlat.points[li].x;
                P_lidar.at<double>(1, 0) = mSurfPointsLessFlat.points[li].y;
                P_lidar.at<double>(2, 0) = mSurfPointsLessFlat.points[li].z;
                P_lidar.at<double>(3, 0) = 1;
                P_cam = mTcamlid * P_lidar;
                cv::Point3d newP;
                newP.x = P_cam.at<double>(0, 0);
                newP.y = P_cam.at<double>(1, 0);
                newP.z = P_cam.at<double>(2, 0);
                PtLsr newPtLsr;
                newPtLsr.pt3d = newP;
                newPtLsr.index3d = counter;
                newPtLsr.intensity = mSurfPointsLessFlat.points[li].intensity;
                //if(counter < 5)
                //    cout<<"pass mSurfPointsLessFlat.points[li].intensity; "<<mSurfPointsLessFlat.points[li].intensity<<" to newPtLsr.intensity "<<newPtLsr.intensity<<endl;
                counter++;
                mLaserLessFlat_cam.push_back(newPtLsr);
            }
        }
//        cout << "frame " << mnId << " mLaserPt_cam " << mLaserPt_cam.size() << " mLaserCorner_cam "
//             << mLaserCorner_cam.size() << " mLaserLessCorner_cam " << mLaserLessCorner_cam.size() << " mLaserFlat_cam "
//             << mLaserFlat_cam.size() << " mLaserLessFlat_cam " << mLaserLessFlat_cam.size() << endl;
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

    /*
     * update mRcw, mRwc, mtcw, mOw
     */
    void Frame::UpdatePoseMatrices()
    {
        mRcw = mTcw.rowRange(0,3).colRange(0,3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0,3).col(3);
        mOw = -mRcw.t()*mtcw;
    }

/**
 * @brief 地图点是否在视野内
 * 0 计算该地图点的世界坐标
 * 1 该点在当前相机的深度>0
 * 2 投影到像素平面，要在范围内
 * 3 mappoint到相机中心的距离，是否在尺度变化的距离内
 * 4 当前视角和平均观测方向的夹角的余弦值，要大于阈值
 * 5 地图点到光心的距离来预测一个尺度（仿造特征点金字塔层级）
 * 6 记录一些参数
 * @param[in] pMP 当前地图点
 * @param[in] viewingCosLimit 余弦阈值
 * @return true false
*/
    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        //mbTrackInView decide if a mappoint is going to be projected into some frame
        pMP->mbTrackInView = false;

        ///*Step0 获得地图点世界坐标
        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        ///*Step 1 深度判断 (局部地图点来自于局部关键帧，它有一些可能在当前视角的背面。
        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        ///*Step 2 成像平面内判断
        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        ///*Step 3 地图点到相机中心的距离 在尺度变化内
        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        //mOW 当前相机光心在世界坐标系下的坐标
        const cv::Mat PO = P - mOw;//a vector (from camera centre) points to P
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        ///*Step 4 相机视角 和 地图点平均观测方向 的夹角的余弦
        // Check viewing angle
        //在UpdateNormAndDepth中更新
        cv::Mat Pn = pMP->GetNormal();//the average direction of this mappoint(a mappoint could be observed by may keyframes)

        //the cos of current view vector and normal observe vector.
        const float viewCos = PO.dot(Pn) / dist; //cos = a.dot(b)/(|a||b|)
        //question is the formula above right?

        if (viewCos < viewingCosLimit)
            return false;

        ///*Step 根据地图点到光心的距离预测一个尺度
        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        //像素横坐标 左图
        pMP->mTrackProjX = u;
        //bf/z视察 即得到像素右视图坐标
        pMP->mTrackProjXR = u - mbf * invz;
        //像素纵坐标 左图
        pMP->mTrackProjY = v;
        //预测到的层级
        pMP->mnTrackScaleLevel = nPredictedLevel;
        //包留视角和平均观测方向的夹角
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    /**
    * @brief 地图Line是否在视野内
    * 0 计算该地图Line的世界坐标
    * 1 该点在当前相机的深度>0
    * 2 投影到像素平面，要在范围内
    * 3 endpoints到相机中心的距离，是否在尺度变化的距离内
    * 4 当前视角和平均观测方向的夹角的余弦值，要大于阈值
    * 5 地图点到光心的距离来预测一个尺度（仿造特征点金字塔层级）
    * 6 记录一些参数
    * @param[in] pML 当前地图line
    * @param[in] viewingCosLimit 余弦阈值
    * @return true false
    */
    bool Frame::isInFrustumLine(MapLine *pML, float viewingCosLimit) {
        //mbTrackInView decide if a mapline is going to be projected into some frame
        pML->mbTrackInView = false;
        ///*Step 0 获得地图点世界坐标
        // 3D in absolute coordinates
        cv::Mat P = pML->GetWorldPos();
        //cout<<"P"<<endl<<P<<endl;
        cv::Mat P0 = P.rowRange(0, 3).colRange(0, 1);
        //cout<<"P0"<<endl<<P0<<endl;
        cv::Mat P1 = P.rowRange(0, 3).colRange(1, 2);
        //cout<<"P1"<<endl<<P1<<endl;
        // 3D in camera coordinates
        //cout<<"mRcw tyoe"<<endl<<mRcw.type()<<endl;
        //cout<<"P1 tyoe"<<endl<<P1.type()<<endl;
        const cv::Mat Pc0 = mRcw * P0 + mtcw;
        const float &PcX0 = Pc0.at<float>(0);
        const float &PcY0 = Pc0.at<float>(1);
        const float &PcZ0 = Pc0.at<float>(2);
        const cv::Mat Pc1 = mRcw * P1 + mtcw;
        const float &PcX1 = Pc1.at<float>(0);
        const float &PcY1 = Pc1.at<float>(1);
        const float &PcZ1 = Pc1.at<float>(2);
        ///*Step 1 深度判断 (局部地图点来自于局部关键帧，它有一些可能在当前视角的背面。
        // Check positive depth
        if (PcZ0 < 0.0f || PcZ1 < 0.0f)
            return false;

        ///*Step 2 成像平面内判断
        // Project in image and check it is not outside
        const float invz0 = 1.0f / PcZ0;
        const float u0 = fx * PcX0 * invz0 + cx;
        const float v0 = fy * PcY0 * invz0 + cy;
        if (u0 < mnMinX || u0 > mnMaxX)
            return false;
        if (v0 < mnMinY || v0 > mnMaxY)
            return false;
        const float invz1 = 1.0f / PcZ1;
        const float u1 = fx * PcX1 * invz1 + cx;
        const float v1 = fy * PcY1 * invz1 + cy;
        if (u1 < mnMinX || u1 > mnMaxX)
            return false;
        if (v1 < mnMinY || v1 > mnMaxY)
            return false;

//        ///*Step 3 地图点到相机中心的距离 在尺度变化内
//        // Check distance is in the scale invariance region of the MapPoint
//        const float maxDistance = pMP->GetMaxDistanceInvariance();
//        const float minDistance = pMP->GetMinDistanceInvariance();
//        //mOW 当前相机光心在世界坐标系下的坐标
//        const cv::Mat PO = P - mOw;//a vector (from camera centre) points to P
//        const float dist = cv::norm(PO);
//        if (dist < minDistance || dist > maxDistance)
//            return false;
//
//        ///*Step 4 相机视角 和 地图点平均观测方向 的夹角的余弦
//        // Check viewing angle
//        //在UpdateNormAndDepth中更新
//        cv::Mat Pn = pMP->GetNormal();//the average direction of this mappoint(a mappoint could be observed by may keyframes)
//        //the cos of current view vector and normal observe vector.
//        const float viewCos = PO.dot(Pn) / dist; //cos = a.dot(b)/(|a||b|)
//        //question is the formula above right?
//        if (viewCos < viewingCosLimit)
//            return false;
//
//        ///*Step 根据地图点到光心的距离预测一个尺度
//        // Predict scale in the image
//        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pML->mbTrackInView = true;
//        //像素横坐标 左图
//        pMP->mTrackProjX = u;
//        //bf/z视察 即得到像素右视图坐标
//        pMP->mTrackProjXR = u - mbf * invz;
//        //像素纵坐标 左图
//        pMP->mTrackProjY = v;
//        //预测到的层级
//        pMP->mnTrackScaleLevel = nPredictedLevel;
//        //包留视角和平均观测方向的夹角
//        pMP->mTrackViewCos = viewCos;

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
        //Set limits for search
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

                //?这个越界判断的目的是
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

    /*
     * retrive stereo infor from Fusioned depth.
     * note I set mbf back to 0.
     */
    void Frame::ComputeStereoFromFusion(const vector<mORBAttribution> ORBAttributions) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            //const float d = imDepth.at<float>(v,u);
            if (ORBAttributions[i].depthSource > -1) {
                const float d = ORBAttributions[i].depth;
                if (d > 0) {
                    mvDepth[i] = d;
                    mbf = 0;
                    mvuRight[i] = kpU.pt.x - mbf / d;
                }
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

    /*
     * A debug function, Compare my aid-depth with stereo depth
     */
    void Frame::CompareWithStereo(cv::Mat im0, cv::Mat im1) {
        float minDiffer = 999, maxDiffer = -999;
        float binGap = 0.3;
        //depth from each kind of LiDAR source
        int planeOutlier= 0, planeTotal = 0, lineOutlier = 0, lineTotal = 0, pointOutlier = 0, pointTotal = 0;
        for (int i = 0; i < N; i++) {
            float stereoDepth = mvDepth[i];
            float fusionDepth = mvORBAttributions[i].depth;
            if (stereoDepth > -1 && fusionDepth > -1) {
                int fusionDepthSource = mvORBAttributions[i].depthSource;
                float depthDifferent = stereoDepth - fusionDepth;
                if (abs(depthDifferent > 1.0)) {
//                    cout << mvKeysUn[i].pt.x << " , " << mvKeysUn[i].pt.y << " : " << stereoDepth
//                         << " - " << fusionDepth << " = " << depthDifferent << " src "
//                         << mvORBAttributions[i].depthSource << endl;
                    if(mvORBAttributions[i].depthSource==1)
                        planeOutlier++;
                    if(mvORBAttributions[i].depthSource==2)
                        lineOutlier++;
                    if(mvORBAttributions[i].depthSource==3)
                        pointOutlier++;
                }
                if(mvORBAttributions[i].depthSource==1)
                    planeTotal++;
                if(mvORBAttributions[i].depthSource==2)
                    lineTotal++;
                if(mvORBAttributions[i].depthSource==3)
                    pointTotal++;
                if (abs(depthDifferent) < minDiffer)
                    minDiffer = abs(depthDifferent);
                if (abs(depthDifferent) > maxDiffer)
                    maxDiffer = abs(depthDifferent);
            }
        }
        cout<<"plane outlier "<<planeOutlier<<" / "<<planeTotal
        <<" line outlier "<<lineOutlier<<" / "<<lineTotal
        <<" source3 outlier "<<pointOutlier<<" / "<<pointTotal<<endl;
        int binNum = ceil((maxDiffer-minDiffer)/binGap);
        //cout<<"max "<<maxDiffer<<" min "<<minDiffer<<" bin number "<<binNum<<endl;
        vector<vector<int>> binContainer;
        for (int i = 0; i < binNum; i++) {
            vector<int> thisBin;
            binContainer.push_back(thisBin);
        }
        for (int i = 0; i < N; i++) {
            float stereoDepth = mvDepth[i];
            float fusionDepth = mvORBAttributions[i].depth;
            if (stereoDepth > -1 && fusionDepth > -1) {
                int fusionDepthSource = mvORBAttributions[i].depthSource;
                float depthDifferent = abs(stereoDepth - fusionDepth);
                int binIndex = int(depthDifferent - minDiffer) / binGap;
                binContainer[binIndex].push_back(i);
            }
        }
//        for (int i = 0; i < binNum; i++) {
//            if(binContainer[i].size()>0)
//                cout<<"from "<<minDiffer+binGap*i<<" to "<<minDiffer+binGap*(i+1)<<" has "<<binContainer[i].size()<<endl;
//        }
        for (int i = 0; i < N; i++) {
            float stereoDepth = mvDepth[i];
            float fusionDepth = mvORBAttributions[i].depth;
            if (stereoDepth > -1 && fusionDepth > -1) {
                int fusionDepthSource = mvORBAttributions[i].depthSource;
                float depthDifferent = abs(stereoDepth - fusionDepth);
                if (abs(depthDifferent) > 1.0) { //remove fusion depth if it is differ with stereo depth
                    mvORBAttributions[i].depthSource = -1;
                    mvORBAttributions[i].depth = -1;
                }
            }
            if (stereoDepth == -1 && fusionDepth > -1) {//remove fusion depth if stereo has no depth
                    mvORBAttributions[i].depthSource = -1;
                    mvORBAttributions[i].depth = -1;
            }
        }
    }

} //namespace ORB_SLAM

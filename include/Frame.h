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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

///Added
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "pcl/sample_consensus/sac_model_line.h"
#include <pcl/sample_consensus/ransac.h>
#include <pcl/features/pfh.h>
#include "pcl/common/common.h"
#include <pcl/sample_consensus/sac_model_plane.h>
///-----------

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class MapPoint;
    class KeyFrame;
    ///Added
    struct mORBAttribution;
    struct mLiDARPoint;
    class SegmentInfo;
    class mPlane;
    ///------------------
    class Frame {
    public:
        Frame();

        // Copy constructor.
        Frame(const Frame &frame);

        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
              ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
              const float &thDepth);// Constructor for stereo cameras.

        ///Added module
        Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft,
              ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
              const vector<vector<float>> &LiDARRaw, const string &ImageFileNAme, const string &SegFileAddress,
              const cv::Mat &mTcamlid);
        ///-------------------------

        // Constructor for RGB-D cameras.
        Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor,
              ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

//        ///Added --- for RGBD
//        Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor,
//              ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef,
//              const float &bf, //baseline * f
//              const float &thDepth,//区分远近点的深度阈值
//              const vector<vector<float>> &LiDARRaw, const string &ImageFileNAme, const string &SegInfoFileName,
//              const cv::Mat &TCamLid);

        void readSegmentsInfo(vector<SegmentInfo*> &segmentsInfoIn, const string &TXTFileAddress);
        void readSegmentImages(vector<cv::Mat *> &segmentImages, const string &imgFileAddress);
        void processLiDARPts(const vector<vector<float>> &);
        void ProjectLiDARtoCamtoImage_KITTI(int cols, int rows);
        void groupLiDARandSegment(vector<cv::Mat*> segmentImages);
        void alignORBSegments(vector<cv::Mat*> segmentImages);
        bool pointInSegmentMask(const cv::Point2d& point, const cv::Mat* segmentMask);
        void planeFitEachSegment();
        void PlaneFitting(vector<mLiDARPoint *> &mLiDARs, int segmentID, int segmentCategory, vector<mPlane*> &mPlanes);
        int RANSACPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, mPlane *foundPlane,
                               pcl::PointIndices &inliersOutput);
        void ORBdetphFromPlane(vector<mPlane*> allPlane, vector<SegmentInfo *> allSegInfos, vector<cv::Mat *> allSegImgs);
        void ORBdetphFromSegments(vector<mLiDARPoint*> allLiDARs, vector<SegmentInfo *> allSegInfos, vector<cv::Mat *> allSegImgs, cv::Mat debugImageIn);
        void interpolateDepth(mORBAttribution *mORBPt, const std::vector<mLiDARPoint *> &fourCornerPoints);
        void ComputeStereoFromFusion(vector<mORBAttribution *> ORBAttributions);
        void MarkStereoFromFusion(const vector<mORBAttribution *> ORBAttributions);
        void ComputeStereoMatches(vector<mORBAttribution*> &ORBAttributions);
        ///---------------------------------------------------

        // Constructor for Monocular cameras.
        Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K,
              cv::Mat &distCoef, const float &bf, const float &thDepth);

        // Extract ORB on the image. 0 for left image and 1 for right image.
        void ExtractORB(int flag, const cv::Mat &im);

        // Compute Bag of Words representation.
        void ComputeBoW();

        // Set the camera pose.
        void SetPose(cv::Mat Tcw);

        // Computes rotation, translation and camera center matrices from the camera pose.
        void UpdatePoseMatrices();

        // Returns the camera center.
        inline cv::Mat GetCameraCenter() {
            return mOw.clone();
        }

        // Returns inverse of rotation
        inline cv::Mat GetRotationInverse() {
            return mRwc.clone();
        }

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

        // Compute the cell of a keypoint (return false if outside the grid)
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1,
                                         const int maxLevel = -1) const;

        // Search a match for each keypoint in the left image to a keypoint in the right image.
        // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
        void ComputeStereoMatches();

        // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
        void ComputeStereoFromRGBD(const cv::Mat &imDepth);

        // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
        cv::Mat UnprojectStereo(const int &i);

    public:
        // Vocabulary used for relocalization.
        ORBVocabulary *mpORBvocabulary;

        // Feature extractor. The right is used only in the stereo case.
        ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

        // Frame timestamp.
        double mTimeStamp;

        ///added module
        cv::Mat mTcamlid;//Transform from LiDAR to Camera
        vector<mORBAttribution*> mvORBAttributions; //store ORB-LiDAR related attributions
        vector<SegmentInfo*> mSegmentsInfo;
        vector<mLiDARPoint*> mvLiDARPoints;
        vector<mPlane*> mvPlanes;
        ///------------------------

        // Calibration matrix and OpenCV distortion parameters.
        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;
        cv::Mat mDistCoef;

        // Stereo baseline multiplied by fx.
        float mbf;

        // Stereo baseline in meters.
        float mb;

        // Threshold close/far points. Close points are inserted from 1 view.
        // Far points are inserted as in the monocular case from 2 views.
        float mThDepth;

        // Number of KeyPoints.
        int N;

        // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
        // In the stereo case, mvKeysUn is redundant as images must be rectified.
        // In the RGB-D case, RGB images can be distorted.
        std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
        std::vector<cv::KeyPoint> mvKeysUn;

        // Corresponding stereo coordinate and depth for each keypoint.
        // "Monocular" keypoints have a negative value.
        std::vector<float> mvuRight;
        std::vector<float> mvDepth;

        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // ORB descriptor, each row associated to a keypoint.
        cv::Mat mDescriptors, mDescriptorsRight;

        // MapPoints associated to keypoints, NULL pointer if no association.
        std::vector<MapPoint *> mvpMapPoints;

        // Flag to identify outlier associations.
        std::vector<bool> mvbOutlier;

        // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        // Camera pose.
        cv::Mat mTcw;

        // Current and Next Frame id.
        static long unsigned int nNextId; //frame类的静态成员
        long unsigned int mnId;

        // Reference Keyframe.
        KeyFrame *mpReferenceKF{};

        // Scale pyramid info.
        int mnScaleLevels;
        float mfScaleFactor;
        float mfLogScaleFactor;
        vector<float> mvScaleFactors;
        vector<float> mvInvScaleFactors;
        vector<float> mvLevelSigma2;
        vector<float> mvInvLevelSigma2;

        // Undistorted Image Bounds (computed once).
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;

        static bool mbInitialComputations;


    private:

        // Undistort keypoints given OpenCV distortion parameters.
        // Only for the RGB-D case. Stereo must be already rectified!
        // (called in the constructor).
        void UndistortKeyPoints();

        // Computes image bounds for the undistorted image (called in the constructor).
        void ComputeImageBounds(const cv::Mat &imLeft);

        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        void AssignFeaturesToGrid();

        // Rotation, translation and camera center
        cv::Mat mRcw;
        cv::Mat mtcw;
        cv::Mat mRwc;
        cv::Mat mOw; //==mtwc
    };


    ///Added modules

    struct mORBAttribution{
        int orbID; //orb unKey[] index in this frame
        cv::KeyPoint *keyPt;
        int depthSource;//1 for plane 2 for line 3 for point/patch
        float depth;
        int PlaneID;
        int segmentIndex;
        //PtLsr * LiDARPt;
        mORBAttribution() : orbID(-1), depthSource(-1), depth(-1), PlaneID(-1), keyPt{nullptr}{};
    };

    struct mLiDARPoint {
        int ID;
        int scanID;
        pcl::PointXYZI p3DonCam;
        pcl::PointXYZI p3DonLiDAR;
        cv::Point2d pt2D;
        int LSDlineID;
        int PlaneID;
        float A,B,C,D;
        SegmentInfo* segmentInfo;

        //Constrcutor
        mLiDARPoint(vector<float> rawLiDARKITTI, int IDin) :
                ID(IDin), scanID(-1), LSDlineID(-1),
                PlaneID(-1), A(-1), B(-1), C(-1), D(-1),
                segmentInfo(nullptr) {
            p3DonCam = pcl::PointXYZI(-1, -1, -1, -1);
            p3DonLiDAR = pcl::PointXYZI(rawLiDARKITTI[0], rawLiDARKITTI[1], rawLiDARKITTI[2], -1);
            pt2D = cv::Point2d(-1, -1);
        }

        // Constructor
        mLiDARPoint() : ID(-1), scanID(-1), LSDlineID(-1), PlaneID(-1),
                        segmentInfo(nullptr),
                        A(0), B(0), C(0), D(0) {
            p3DonLiDAR = pcl::PointXYZI(0, 0, 0, 0);
            p3DonCam = pcl::PointXYZI(0, 0, 0, 0);
            pt2D = cv::Point2d(0, 0);
        }

        // Constructor with parameters
        mLiDARPoint(int id, int scanId, bool isLow,
                    const pcl::PointXYZI &camPoint, const pcl::PointXYZI &lidarPoint,
                    const cv::Point2d &imagePoint, int lineId, int planeId, float coeffA, float coeffB,
                    float coeffC, float coeffD, SegmentInfo *segInfo)
                : ID(id), scanID(scanId),
                  p3DonCam(camPoint), p3DonLiDAR(lidarPoint),
                  pt2D(imagePoint), LSDlineID(lineId),
                  PlaneID(planeId), A(coeffA), B(coeffB), C(coeffC),
                  D(coeffD), segmentInfo(segInfo) {}

        mLiDARPoint(const pcl::PointXYZI lidarPoint) {
            ID = -1, scanID = -1, LSDlineID = -1, PlaneID = -1, segmentInfo = nullptr;
            p3DonLiDAR.x = lidarPoint.x, p3DonLiDAR.y = lidarPoint.y, p3DonLiDAR.z = lidarPoint.z, p3DonLiDAR.intensity = lidarPoint.intensity;
            p3DonCam.x = -1, p3DonCam.y = -1, p3DonCam.z = -1, p3DonCam.intensity = -1;
            pt2D.x = -1, pt2D.y = -1;
            A = 0, B = 0, C = 0, D = 0;
        }
    };

    class SegmentInfo{
    public:
        int id;
        bool isThing;
        int category_id;
        double area;
        std::vector<mLiDARPoint*> associatedPoints;
        std::vector<int> associatedORBs;
        SegmentInfo(int id, bool isThing, int category_id, double area):
                id(id), isThing(isThing), category_id(category_id), area(area){}
    };

    class mPlane {
    public:
        int PlaneID;//Id in frame
        int planeIdGlobal;//In in map/global
        vector<cv::Point2d> point2D;
        vector<cv::Point3d> point3D;
        int segmentID;//Caution ! starts from 1
        int segmentCategory;
        vector<int> lidarIndexofSegment;
        float A, B, C, D;
        vector<mLiDARPoint> vPointsLiDAR;
        vector<int> vIndexKeyPt; //TODO index of ORB keyPt in the plane
        mPlane(double Ain, double Bin, double Cin, double Din) : A(Ain), B(Bin), C(Cin), D(Din) { PlaneID = -1; }
        mPlane() {
            PlaneID = -1, A = -1, B = -1, C = -1, D = -1;
        }
    };
    ///-------------------------------------

}// namespace ORB_SLAM

#endif // FRAME_H

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
///Added module
#include "MapLine.h"
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
#include "opencv2/line_descriptor/descriptor.hpp"
#include <cmath>

typedef pcl::PointXYZI PointType;

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
///Added module
class MapLine;

///added module
    struct mLine {
        int ID;
        cv::line_descriptor::KeyLine LSD;
        bool nonfloorLine;
        bool fit3DLine;
        std::vector<int> LiDARPtIDs;
        cv::Point3d pt3dStart;
        cv::Point3d pt3dEnd;

        mLine() {
            ID = -1;
            fit3DLine = false;
            nonfloorLine = false;
            pt3dStart = cv::Point3d(0, 0, 0);
            pt3dEnd = cv::Point3d(0, 0, 0);
        };

        mLine(int IDin, cv::line_descriptor::KeyLine LSDin) :
                ID(IDin), LSD(LSDin), fit3DLine(false), nonfloorLine(false), pt3dStart(cv::Point3d(0, 0, 0)),
                pt3dEnd(cv::Point3d(0, 0, 0)) {};
    };

    class PtLsr {
    public:
        int ptID;
        cv::Point2d pt2d;
        cv::Point3d pt3d;
        int index2d;//why I have this?
        int index3d;
        float intensity;
        int scanID;// i-th scan
        int pointID;// j-th point
        bool low;
        int planeID;
        int LSDlineID;
        mLine *LSDline;//not tested

        PtLsr() {
            ptID = -1;
            scanID = -1;
            pointID = -1;
            low = false;
            LSDlineID = -1;
            planeID = -1;
        }
    };

    class mPlane {
    public:
        double A, B, C, D;
        double phi, theta, dis;
        int count;
        int PlaneId;//Id in frame
        int planeIdGlobal;//In in map/global
        mPlane(double Ain, double Bin, double Cin, double Din) : A(Ain), B(Bin), C(Cin), D(Din) { PlaneId = -1; }

        mPlane(double phiin, double thetain, double disin) : phi(phiin), theta(thetain), dis(disin) { PlaneId = -1; }

        mPlane() {
            PlaneId = -1;
            A = -1;
            B = -1;
            C = -1;
            phi = -1;
            theta = -1;
            dis = -1;
        }

        vector<PtLsr> vPointsLiDAR;
        vector<int> vIndexKeyPt; //TODO index of ORB keyPt in the plane
    };

    struct mORBAttribution{
        int ID;
        cv::KeyPoint keyPt;
        int depthSource;//1 for plane 2 for line
        float depth;//current using closest PT3d. but should use intersection depth in the future
        int LSDlineID;
        mLine *LSDline;
        int floorPlaneID;
        //mPlane *Plane; We visit the plane by planeID.
        int LiDARPtID;
        PtLsr * LiDARPt;
        cv::Point3d p3d_est;
        cv::Point3d p3d_tri;//p3d from triangulation comes from ORBSLAM2
        cv::Point3d p3d_tri_scaled;//p3d_tri after scaled with LiDAR
        mORBAttribution() : ID(-1), depthSource(-1), depth(-1),
                            LSDlineID(-1), LSDline(nullptr), floorPlaneID(-1), LiDARPtID(-1),LiDARPt(nullptr),
                            p3d_est(NULL), p3d_tri(NULL), p3d_tri_scaled(NULL) {};
    };



class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);
    ///added module
    Frame(const cv::Mat &imGray, const double &timeStamp, const vector<vector<double>> &lasers, ORBextractor *extractor, ORBextractor *extractor1, ORBVocabulary *voc, cv::Mat &K, cv::Mat &Tcamlid,
          cv::Mat &distCoef, const float &bf, const float &thDepth);
    Frame(const cv::Mat &imGray, const cv::Mat &imGray_r, const double &timeStamp, const vector<vector<double>> &lasers, ORBextractor *extractor, ORBextractor *extractor1, ORBVocabulary *voc, cv::Mat &K, cv::Mat &Tcamlid,
          cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    // Computes mRcw, mRwc, mtcw, mOw from mTcw
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    ///Added Module
    bool isInFrustumLine(MapLine *pML, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);
    ///Added Module
    void ComputeStereoFromFusion(const vector<mORBAttribution> ORBAttributions);
    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    ///added modules
    cv::Mat mTcamlid;//Transform from LiDAR to Camera
    int givenDepthNum;
    void PlaneFitting();
    int RANSACPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, mPlane &foundPlane, pcl::PointIndices &inliersOutput);
    void ProjectLiDARtoCam();
    void ProjectLiDARtoImg(cv::Mat, int cols, int rows);
    void ExtractLiDARFeature();
    void ProjectLiDARFeaturetoImg(cv::Mat, int cols, int rows);
    void PairLaserVisionFeatures(const cv::Mat &im);
    void CompareWithStereo(cv::Mat im0, cv::Mat im1);
    void connectLSD2LiDAR(vector<mLine> &mLSDLinesIN, vector<PtLsr> &LiDARPtIN, cv::Mat im);
    void connectORB2Plane(vector<PtLsr> &LiDARPoints, std::vector<cv::KeyPoint> &ORBFeatures, vector<ORB_SLAM2::mPlane> &extractedPlanes, double threshold, cv::Mat im);
    void connectORB2LSD(vector<mLine> &mLSDLinesIN, vector<cv::KeyPoint> &ORBin, cv::Mat im);
    void ORBdepthFromLine(vector<mLine> &lineInputs, vector<mORBAttribution> &ORBinputs, cv::Mat im);
    void ORBdepthFromPoint( vector<PtLsr> &LiDARInputs,  vector<mORBAttribution> &ORBinputs, double threshold, cv::Mat im);
    void ORBdepthFromPointPatch(vector<PtLsr> &LiDARInputs, vector<mORBAttribution> &ORBinputs, double threshold, cv::Mat im);
    vector<std::vector<double>> mLaserPoints; //Raw LiDAR point under LiDAR coordination System
    vector<pcl::PointCloud<pcl::PointXYZI>> mLaser16ScansPoints; //Raw LiDAR point under LiDAR coordination System
    vector<pcl::PointCloud<pcl::PointXYZI>> mlaserScansPoints;
    pcl::PointCloud<PointType> mCornerPointsSharp; //LiDAR feature under LiDAR system
    pcl::PointCloud<PointType> mCornerPointsLessSharp;//LiDAR feature under LiDAR system
    pcl::PointCloud<PointType> mSurfPointsFlat;//LiDAR feature under LiDAR system
    pcl::PointCloud<PointType> mSurfPointsLessFlat;//LiDAR feature under LiDAR system
    vector<PtLsr> mLaserPt_cam;//Projected LiDAR under Camera Coordination System
    vector<PtLsr> mLaserCorner_cam;//Projected LiDAR Corner point under Camera Coordination System
    vector<PtLsr> mLaserLessCorner_cam;//Projected LiDAR less Corner point under Camera Coordination System
    vector<PtLsr> mLaserFlat_cam;//Projected LiDAR flat  point under Camera Coordination System
    vector<PtLsr> mLaserLessFlat_cam;//Projected LiDAR less flat point under Camera Coordination System
    vector<double> mLaserTimes; //{middle time, start, end}
    vector<ORB_SLAM2::mPlane> mvPlanes;
    vector<mLine> mvLines;
    vector<mORBAttribution> mvORBAttributions; //store ORB-LiDAR related attributions
    cv::Mat mlsdDescriptors;
    ///-------------------------------------

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
    ///Added Module
    int N_lines;

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
    std::vector<MapPoint*> mvpMapPoints;
    ///Added Module
    std::vector<MapLine*> mvpMapLines;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;
    ///Added Module
    std::vector<bool> mvbOutlierLines;

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
    KeyFrame* mpReferenceKF{};

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
    cv::Mat mOw; //==mtwc. Camera centre in World
};

}// namespace ORB_SLAM

#endif // FRAME_H

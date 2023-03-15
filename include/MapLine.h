//
// Created by xin on 01/03/23.
//

#ifndef MAPLINE_H
#define MAPLINE_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>
namespace ORB_SLAM2{
    class KeyFrame;

    class Map;

    class Frame;

    class MapLine{
    public:
        //regular initializer
        MapLine(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);

        MapLine(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

        void SetWorldPos(const cv::Mat &Pos);

        cv::Mat GetWorldPos();

        cv::Mat GetNormal();

        KeyFrame *GetReferenceKeyFrame();

        std::map<KeyFrame *, size_t> GetObservations();

        int Observations();

        void AddObservation(KeyFrame *pKF, size_t idx);

        void EraseObservation(KeyFrame *pKF);

        int GetIndexInKeyFrame(KeyFrame *pKF);

        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();

        bool isBad();

        void Replace(MapLine *pMP);

        MapLine *GetReplaced();

        void IncreaseVisible(int n = 1);

        void IncreaseFound(int n = 1);

        float GetFoundRatio();

        inline int GetFound() {
            return mnFound;
        }

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateNormalAndDepth();

        float GetMinDistanceInvariance();

        float GetMaxDistanceInvariance();

        int PredictScale(const float &currentDist, KeyFrame *pKF);

        int PredictScale(const float &currentDist, Frame *pF);
    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;
        // Variables used by the tracking
        float mTrackProjX0;
        float mTrackProjY0;
        float mTrackProjX1;
        float mTrackProjY1;
        //Question : not used in line?
//        float mTrackProjXR;//?
        bool mbTrackInView;
//        int mnTrackScaleLevel;
//        float mTrackViewCos;
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;
        // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;
        // Variables used by loop closing
        long unsigned int mnLoopPointForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;//Need change to Line version? 3 row 2 col cv_32f
        long unsigned int mnBAGlobalForKF;
        static std::mutex mGlobalMutex;
    protected:
        // Position in absolute coordinates
        cv::Mat mWorldPos;//3 rows 2 cols CV_64F double -> change to CV_32F Float
        // Keyframes observing the line and associated index in keyframe
        std::map<KeyFrame *, size_t> mObservations;
        // Mean viewing direction
        cv::Mat mNormalVector;//useful in line case?
        // Best descriptor to fast matching
        cv::Mat mDescriptor;
        // Reference KeyFrame
        KeyFrame *mpRefKF;
        // Tracking counters
        int mnVisible;
        int mnFound;
        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        MapLine *mpReplaced;
        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;
        Map *mpMap;
        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };
}

#endif //MAPLINE_H

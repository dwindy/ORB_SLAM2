//
// Created by xin on 17/01/24.
//

#ifndef ORB_SLAM2_KALMANFILTER_H
#define ORB_SLAM2_KALMANFILTER_H

#include<opencv2/core/core.hpp>
#include "Frame.h"

namespace ORB_SLAM2 {
    class Frame;
/*
 * class used for tracking objects in 2d image
 */
    class KalmanFilter {
    public:
        int label;
        int globalID;
        //Frame *frame;
        //vector<Frame*> allFrames;//No need to record this. CurrentFrame pointer will be updated to the newest frame.
        float confidence_Pre;
        vector<int> dynamicsHistory;
        float dynamic_status;
        int lostCounter;
        bool lost;
        double dt;
        cv::Mat X;
        cv::Mat X_pre;
        cv::Mat U;
        cv::Mat A;
        cv::Mat B;
        cv::Mat H;
        cv::Mat Q;
        cv::Mat R;
        cv::Mat P;
        cv::Mat K;
        cv::Mat S;
        double std_mear;
        std::vector<cv::Point2f> centres;

        KalmanFilter();

        KalmanFilter(int globalID, int label, double U, double V);

        void predictStatus();

        void updateStatus(cv::Mat measurement);

        void updateVelocity();

        void updateDynamics();
    };
}
#endif //ORB_SLAM2_KALMANFILTER_H

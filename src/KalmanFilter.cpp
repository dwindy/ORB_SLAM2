//
// Created by xin on 17/01/24.
//

#include "KalmanFilter.h"

namespace ORB_SLAM2 {
    KalmanFilter::KalmanFilter() {}

    KalmanFilter::KalmanFilter(int globalIDin, int labelin, double Uin, double Vin)
    {
            confidence_Pre = 1;
            lostCounter = 0;
            lost = false;
            globalID = globalIDin;
            label = labelin;
            X = (cv::Mat_<double>(4, 1) << Uin, Vin, 0, 0);
            U = (cv::Mat_<double>(2, 1) << 0, 0);
            dt = 1;
            std_mear = 0.1;
            A = (cv::Mat_<double>(4, 4) << 1, 0, dt, 0,
                    0, 1, 0, dt,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
            B = (cv::Mat_<double>(4, 2) << dt * dt / 2, 0,
                    0, dt * dt / 2,
                    dt, 0,
                    0, dt);
            H = (cv::Mat_<double>(2, 4) << 1, 0, 0, 0,
                    0, 1, 0, 0);
            Q = (cv::Mat_<double>(4, 4) << dt * dt * dt * dt / 4, 0, dt * dt * dt / 2, 0,
                    0, dt * dt * dt * dt / 4, 0, dt * dt * dt / 2,
                    dt * dt * dt / 2, 0, dt * dt, 0,
                    0, dt * dt * dt / 2, 0, dt * dt);
            R = (cv::Mat_<double>(2, 2) << std_mear * std_mear, 0
                    , 0, std_mear * std_mear);
            P = cv::Mat_<double>::eye(4, 4);
            //    std::cout<<"Global ID "<<globalID<<" is created."<<std::endl;
            //    std::cout<<"X: "<<X<<std::endl;
            //    std::cout<<"U: "<<U<<std::endl;
            //    std::cout<<"P: "<<P<<std::endl;
            //    std::cout<<"Q: "<<Q<<std::endl;
            //    std::cout<<"R: "<<R<<std::endl;
            //    std::cout<<"H: "<<H<<std::endl;
            //    std::cout<<"B: "<<B<<std::endl;
            //    std::cout<<"A: "<<A<<std::endl;
            //    std::cout<<"dt: "<<dt<<std::endl;
            //    std::cout<<"std_mear: "<<std_mear<<std::endl;
            //    std::cout<<"lostCounter: "<<lostCounter<<std::endl;
            //    std::cout<<"label: "<<label<<std::endl;
            //    std::cout<<"========================"<<std::endl;
            //    std::cout<<std::endl;
    }


    void KalmanFilter::predictStatus() {
//    if (globalID == 0)
//        std::cout << "GlobalID " << globalID << " is predicted." << std::endl;
        X_pre = X.clone();
        centres.push_back(cv::Point2f(X_pre.at<double>(0, 0), X_pre.at<double>(1, 0)));
//    if (globalID == 0)
//        std::cout<<"X_pre: "<<X_pre<<std::endl;
        X = A * X + B * U;
//    if (globalID == 0)
//        std::cout<<"predict X: "<<X<<std::endl;
        P = A * P * A.t() + Q;
    }

    void KalmanFilter::updateStatus(cv::Mat measurement) {
//    if (globalID == 0) {
//        std::cout << "GlobalID " << globalID << " is updated." << std::endl;
//        std::cout << "X: " << X << std::endl;
//        std::cout << "measurement:  " << measurement << std::endl;
//        std::cout << "R: " << R << std::endl;
//        std::cout << "P: " << P << std::endl;
//    }
        S = H * P * H.t() + R;
        K = P * H.t() * S.inv();
//    if(globalID==0)
//        std::cout<<"K: "<<K<<std::endl;
        X = X + K * (measurement - H * X);
//    if(globalID==0)
//        std::cout<<"update X: "<<X<<std::endl;
        P = (cv::Mat_<double>::eye(4, 4) - K * H) * P;
    }

    void KalmanFilter::updateVelocity() {
//    if(globalID==0){
//        std::cout<<"GlobalID "<<globalID<<" is updated velocity."<<std::endl;
//        std::cout<<"X: "<<X<<std::endl;
//        std::cout<<"X_pre: "<<X_pre<<std::endl;
//    }
        cv::Mat diff = X - X_pre;
//    if(globalID==0)
//        std::cout<<"diff: "<<diff<<std::endl;
        X.at<double>(2, 0) = diff.at<double>(0, 0) / dt;
        X.at<double>(3, 0) = diff.at<double>(1, 0) / dt;
//    if(globalID==0)
//        std::cout<<"updated velocity "<<X.t()<<std::endl;
    }

    void KalmanFilter::updateDynamics() {
        float sum = 0;
        for (int i = 0; i < dynamicsHistory.size(); i++) {
            sum += dynamicsHistory[i];
        }
        //todo confirm why in some case dynamics.size is 0? which lead nan problem.
        if (dynamicsHistory.size() > 0)
            dynamic_status = sum / dynamicsHistory.size();
        else
            dynamic_status = 0;
    }
}

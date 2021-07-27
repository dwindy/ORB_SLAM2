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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

///Added module
#include <cmath>
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
namespace ORB_SLAM2
{


void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
    vector<MapPlane *> vpMPln = pMap->GetAllMapPlanes();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

///Added Module
//Define the Plane vertex
    class VertexPlane : public g2o::BaseVertex<3, Eigen::Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        VertexPlane() {}

        virtual void setToOriginImpl() override {
            _estimate = Eigen::Vector3d( 0, 0, 0);
        };

        ///todo Are we going to update the Plane?
        virtual void oplusImpl(const double *update) override {}

        virtual bool read(istream &in) {}

        virtual bool write(ostream &out) const {}
    };

//Define unary edge of Plane
    class UnaryEdgePlane : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void computeError() override {
            g2o::VertexSE3Expmap *pose = (g2o::VertexSE3Expmap *) _vertices[0];
            Eigen::Vector3d n_map = PI_map / PI_map.norm();
            double d_map = PI_map.norm();
            auto rotation = pose->estimate().rotation();
            auto translation = pose->estimate().translation();
            Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
            Eigen::Matrix<double, 3, 1> translation_matrix = translation;
            Eigen::Matrix4d T_lm = Eigen::Matrix4d::Identity();
            T_lm.block<3, 3>(0, 0) = rotation_matrix;
            T_lm.block<3, 1>(0, 3) = translation_matrix;
            Eigen::Matrix4d T_ml = T_lm.inverse();
            Eigen::Matrix<double, 3, 1> tranlate_ml = T_ml.block<3, 1>(0, 3);
            ///PI' = (R^L_A * n^A) * (d^A - P^A_L.translate * n^A)
            Eigen::Vector3d PI_proj = (rotation_matrix * n_map) * (d_map - tranlate_ml.transpose() * n_map);
            _error = _measurement - PI_proj;
        }

        void printError() {
            g2o::VertexSE3Expmap *pose = (g2o::VertexSE3Expmap *) _vertices[0];
            Eigen::Vector3d n_map = PI_map / PI_map.norm();
            double d_map = PI_map.norm();
            auto rotation = pose->estimate().rotation();
            auto translation = pose->estimate().translation();
            Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
            Eigen::Matrix<double, 3, 1> translation_matrix = translation;
            Eigen::Matrix4d T_lm = Eigen::Matrix4d::Identity();
            T_lm.block<3, 3>(0, 0) = rotation_matrix;
            T_lm.block<3, 1>(0, 3) = translation_matrix;
            Eigen::Matrix4d T_ml = T_lm.inverse();
            Eigen::Matrix<double, 3, 1> tranlate_ml = T_ml.block<3, 1>(0, 3);
            ///PI' = (R^L_A * n^A) * (d^A - P^A_L.translate * n^A)
            Eigen::Vector3d PI_proj = (rotation_matrix * n_map) * (d_map - tranlate_ml.transpose() * n_map);
            _error = _measurement - PI_proj;
            cout << " PI world " << PI_map[0] << " " << PI_map[1] << " " << PI_map[2] << " | ";
            cout << " PI proj " << PI_proj.transpose() << " | ";
            cout << " PI ober " << _measurement[0] << " " << _measurement[1] << " " << _measurement[2] << " | ";
            //_error = PI_proj - _measurement; //paper is project - measurement
            _error = _measurement - PI_proj;
            cout << "_error " << _error.transpose() << " chi2 = " << chi2() << endl;
        }


    public:
        //Members
        Eigen::Vector3d PI_map; //PI_local is observation

        virtual bool read(istream &in) {}

        virtual bool write(ostream &out) const {}
    };

//Define the VertexSE3Expmap-Plane(PI form) edge (plane as point)
class Edge3D3D : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap,g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
///Way of LIPS paper
        virtual void computeError() override {
            auto pose = (g2o::VertexSE3Expmap *) _vertices[0];
            auto vPlane = (VertexPlane *) _vertices[1];//PI
            Eigen::Vector3d PI_world = vPlane->estimate();
            Eigen::Vector3d n_world = PI_world / PI_world.norm(); // norm = PI/|PI|
            double d_world = PI_world.norm();// d = |PI|
            auto rotation = pose->estimate().rotation();
            auto translation = pose->estimate().translation();
            Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
            Eigen::Matrix<double, 3, 1> translation_matrix = translation;
            Eigen::Matrix4d TLA = Eigen::Matrix4d::Identity();
            TLA.block<3,3>(0,0) = rotation_matrix;
            TLA.block<3,1>(0,3) = translation_matrix;
//            cout<<" T^L_A "<<endl<<TLA<<endl;
            Eigen::Matrix4d TAL = TLA.inverse();
//            cout<<" T^A_L "<<endl<<TAL<<endl;
            Eigen::Matrix<double,3,1> tranlate_AL = TAL.block<3,1>(0,3);
            ///PI' = (R^L_A * n^A) * (d^A - P^A_L.translate * n^A)
            Eigen::Vector3d PI_proj = (rotation_matrix * n_world)*(d_world - tranlate_AL.transpose()*n_world );
//            cout <<" PI world "<<PI_world[0]<<" "<<PI_world[1]<<" "<<PI_world[2]<<" | ";
//            cout << "PI proj " << PI_proj.transpose() <<" | ";
//            cout <<" PI ober "<<_measurement[0]<<" "<<_measurement[1]<<" "<<_measurement[2]<<" | ";
            //_error = PI_proj - _measurement; //paper is project - measurement
            _error = _measurement - PI_proj;
//            cout << "_error " << _error.transpose() << endl;
        }


        void printError() {
            auto pose = (g2o::VertexSE3Expmap *) _vertices[0];
            auto vPlane = (VertexPlane *) _vertices[1];//PI
            Eigen::Vector3d PI_world = vPlane->estimate();
            Eigen::Vector3d n_world = PI_world / PI_world.norm(); // norm = PI/|PI|
            double d_world = PI_world.norm();// d = |PI|
            auto rotation = pose->estimate().rotation();
            auto translation = pose->estimate().translation();
            Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
            Eigen::Matrix<double, 3, 1> translation_matrix = translation;
            Eigen::Matrix4d TLA = Eigen::Matrix4d::Identity();
            TLA.block<3, 3>(0, 0) = rotation_matrix;
            TLA.block<3, 1>(0, 3) = translation_matrix;
//            cout<<" T^L_A "<<endl<<TLA<<endl;
            Eigen::Matrix4d TAL = TLA.inverse();
//            cout<<" T^A_L "<<endl<<TAL<<endl;
            Eigen::Matrix<double, 3, 1> tranlate_AL = TAL.block<3, 1>(0, 3);
            ///PI' = (R^L_A * n^A) * (d^A - P^A_L.translate * n^A)
            Eigen::Vector3d PI_proj = (rotation_matrix * n_world) * (d_world - tranlate_AL.transpose() * n_world);
            cout << " PI world " << PI_world[0] << " " << PI_world[1] << " " << PI_world[2] << " | ";
            cout << " PI proj " << PI_proj.transpose() << " | ";
            cout << " PI ober " << _measurement[0] << " " << _measurement[1] << " " << _measurement[2] << " | ";
            //_error = PI_proj - _measurement; //paper is project - measurement
            _error = _measurement - PI_proj;
            cout << "_error " << _error.transpose() << endl;
        }

    ///Way of Point transform
//    virtual void computeError()override{
//        auto pose = (g2o::VertexSE3Expmap *) _vertices[0];
//        auto Pw = (g2o::VertexSBAPointXYZ *) _vertices[1];
//        Eigen::Vector3d Pw_pjt = pose->estimate().map(Pw->estimate());
//        _error = _measurement - Pw_pjt;
////        cout<<" Pose "<<endl<<pose->estimate()<<endl;
////        cout<<" Pw "<<Pw->estimate().transpose()<<endl;
////        cout<<" Pw_pjt "<<Pw_pjt.transpose()<<endl;
////        cout<<" measre "<<_measurement.transpose()<<endl;
////        cout<<" error  "<<_error.transpose()<<endl;
//    }

//    virtual void linearizeOplus() override{
//        auto pose = (g2o::VertexSE3Expmap *) _vertices[0];
//        auto Pw = (g2o::VertexSBAPointXYZ *) _vertices[1];
//        Eigen::Vector3d Pw_pjt = pose->estimate().map(Pw->estimate());
//        //In g2o VertexSE3Expmap, R first then t
//        _jacobianOplusXi << 0, -Pw_pjt[2], Pw_pjt[1], -1, 0, 0,
//                            Pw_pjt[2], 0, -Pw_pjt[0], 0, -1, 0,
//                            -Pw_pjt[1], Pw_pjt[0], 0, 0, 0, -1;
//        //_jacobianOplusXi = _jacobianOplusXi * -1;
//    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

//Define the VertexSE3Expmap-Plane edge
//Observation 3 dimension -> PI
    class EdgePlanePlane : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap, VertexPlane> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void computeError() override {
//            cout << "call compute Error--------------------------" << endl;
            auto pose = (g2o::VertexSE3Expmap *) _vertices[0];
            auto vPlane = (VertexPlane *) _vertices[1];//PI
            Eigen::Vector3d n_world = vPlane->estimate() / vPlane->estimate().norm(); // norm = PI/|PI|
            double d_world = vPlane->estimate().norm();// d = |PI|
            auto rotation = pose->estimate().rotation();
            auto translation = pose->estimate().translation();
            Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
            Eigen::Matrix<double, 3, 1> translation_matrix = translation;
            ///way 1
            cout << "rotation " << endl << rotation_matrix << endl << " translation " << endl << translation_matrix
                 << endl;
            cout << "world PI " << vPlane->estimate()[0] << " " << vPlane->estimate()[1] << " "
                 << vPlane->estimate()[2] <<" norm "<<n_world.transpose()<<" d "<<d_world<<endl;
            cout << "obser PI " << _measurement[0] << " " << _measurement[1] << " " << _measurement[2] << endl;
            //project from map to local
            ///[xl,yl,zl]^T = R^L_A * nA
            auto n_proj = rotation * n_world; //todo rotation is quaterion or matrix?
            cout << "n_proj " << n_proj.transpose() << endl;
            ///dl = - (nA * P^A_L) + dA
            auto t_A_L = pose->estimate().inverse().translation();
            auto d_proj = d_world - (t_A_L).dot(n_world);
            cout << "d_proj " << d_proj << endl;
            ///PI' = nL * dL;
            Eigen::Vector3d PI_proj = (n_proj) * (d_proj);
            cout << "PI proj " << PI_proj.transpose()<<endl;
            //_error = PI_proj - _measurement; //paper is project - measurement
            _error = _measurement - PI_proj;
            cout << "_error " << _error.transpose() << endl;
        }
        void printError(){
            computeError();
//            cout <<" error "<<_error.transpose()<<" chi2 "<<chi2()<<endl;
        }

        virtual bool read(istream &in) {}

        virtual bool write(ostream &out) const {}
    };

void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    //bool label of not include points
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    //declare a Optimizer and Solver
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    //and Solver ptr
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    //declare optimization algorithm
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        //declare a LEE algebra vertex
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        //first frame need fixed
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        //declare a 3d Point vertex
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        //yes point can be mariginalized
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        //get observation for each 3D point
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
            //Mono
            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;
                //declare a Edge SE3ProjectXYZ type
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//id: g2o id of cur 3d point
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));//id: corresponding KF id (also g2o id)
                e->setMeasurement(obs);
                //set information matrix based on the octave level
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }
                //set edge attribution
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }
        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }
    ///print points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        cout<<pMP->GetWorldPos().at<float>(0)<<" "<<pMP->GetWorldPos().at<float>(1)<<" "<<pMP->GetWorldPos().at<float>(2)<<endl;
    }
    cout << "after BA" << endl;
    cout<<"-----------------------------------"<<endl;
    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
            cout<<pMP->GetWorldPos().at<float>(0)<<" "<<pMP->GetWorldPos().at<float>(1)<<" "<<pMP->GetWorldPos().at<float>(2)<<endl;
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        cout << pKF->mnId << endl;
        cout << pKF->GetPose() << endl;
    }
}

/**
 * @brief Plane optimization
 * @param mpMap : map
 * @param pFrame : current frame
 * @param matchPlanes : current plane to map plane pair relations
 */
    void Optimizer::PlaneOptimization(Map *mpMap, Frame *pFrame, vector<int> matchPlanes) {
        //Declare g2p optimizer
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto *solverinstance = new LinearSolverType();
        auto *blockersolverinstance = new BlockSolverType(solverinstance);
        auto solver = new g2o::OptimizationAlgorithmLevenberg(blockersolverinstance);
        //auto solver = new g2o::OptimizationAlgorithmGaussNewton(blockersolverinstance);
        g2o::SparseOptimizer optimizer;//sparse?
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);

        //Set Frame Pose Vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
//        cv::Mat play = cv::Mat::eye(4,4,CV_32F);
//        play.at<float>(0,3) = 0.005;
//        play.at<float>(1,3) = 0.005;
//        play.at<float>(2,3) = 0.005;
//        vSE3->setEstimate(Converter::toSE3Quat(play));
//        cout<<"vSE3 estimeta "<<vSE3->estimate()<<endl;
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
//        if(pFrame->mnId==120)
//        {
//            ofstream writer;
//            writer.open("frame150localplanes.txt");

            cout << "pose before plane optimization " << endl;
            cout << pFrame->mTcw << endl;
            vector<float> q1 = Converter::toQuaternion(pFrame->mTcw(cv::Rect(0, 0, 3, 3)));
            cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
                 << " " << pFrame->mTcw.at<float>(0, 3) << " " << pFrame->mTcw.at<float>(1, 3) << " " << pFrame->mTcw.at<float>(2, 3)
                 << " " << q1[0] << " " << q1[1] << " " << q1[2] << " "<< q1[3] << endl;

//            for (size_t i = 0; i < pFrame->mvPlanes.size(); i++) {
//                if (matchPlanes[i] >= 0) {
//                    writer << pFrame->mvPlanes[i].PI[0] << " " << pFrame->mvPlanes[i].PI[1] << " "<< pFrame->mvPlanes[i].PI[2] << endl;
//                }
//            }
//            writer.close();
//
//            writer.open("frame150mapplanes.txt");
//            vector<MapPlane *> mapPlanes = mpMap->GetAllMapPlanes();
//            for (size_t i = 0; i < pFrame->mvPlanes.size(); i++) {
//                if (matchPlanes[i] >= 0) {
//                    int mapIndex = matchPlanes[i];
//                    for (int j = 0; j < mapPlanes.size(); j++) {
//                        if (mapPlanes[j]->mnId == mapIndex) {
//                            writer << mapPlanes[j]->PI0 << " " << mapPlanes[j]->PI1 << " " << mapPlanes[j]->PI2 << endl;
//                        }
//                    }
//                }
//            }
//            writer.close();
//        }

        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        int PlaneVertexNum = 0;
        vector<int> planeVertexID;
        //std::vector<EdgePlanePlane *> allEdges;
        std::vector<Edge3D3D *>all3d3dEdges;
        int edgeNum = 0;
        //Set Plane Vertex and Pose-Plane Edge
        vector<MapPlane *> mapPlanes = mpMap->GetAllMapPlanes();
        for (size_t i = 0; i < pFrame->mvPlanes.size(); i++) {
            if (matchPlanes[i] >= 0) {
                //set map plane vertex
                int mapPlaneID = matchPlanes[i];
                int mapIndex = -1;
                if (pFrame->mnId == 418) {
                    cout << "looking for " << mapPlaneID << " : ";
                }
                for (size_t temp = 0; temp < mapPlanes.size(); temp++) {
                    if (mapPlanes[temp]->mnId == mapPlaneID)
                    {
                        mapIndex = temp;
                        //cout<<mapPlanes[temp]->mnId<<" ";
                    }
                }
                //cout<<endl;

                ///Set PlaneVertex
                MapPlane *thisMapPlane = mapPlanes[mapIndex];
                if (pFrame->mnId == 418) {cout<<" found "<<thisMapPlane->mnId<<endl;}
                g2o::VertexSBAPointXYZ *newVertex = new g2o::VertexSBAPointXYZ();
                newVertex->setEstimate(Eigen::Vector3d(thisMapPlane->PI0,thisMapPlane->PI1,thisMapPlane->PI2));
                newVertex->setId(1 + PlaneVertexNum);
                newVertex->setFixed(true);
                optimizer.addVertex(newVertex);
                planeVertexID.push_back(1 + PlaneVertexNum);
                //cout << "add new world plane vertex " << 1 + PlaneVertexNum << " : PI : " << thisMapPlane->PI0<<" "<<thisMapPlane->PI1<<" "<<thisMapPlane->PI2<<" "<< " map plane id "
                //     << thisMapPlane->mnId << endl;
//                cout << "  edge observe " << pFrame->mvPlanes[i].PI.transpose()<< " ";
//                cout<<"map plane vertex "<<mapPlanes[mapIndex]->PI0<<" "<< mapPlanes[mapIndex]->PI1<<" "<< mapPlanes[mapIndex]->PI2<<endl;
                //Set 3d3d plane Edge
                Edge3D3D *newEdge = new Edge3D3D;
                newEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                newEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(1 + PlaneVertexNum)));
                newEdge->setMeasurement(pFrame->mvPlanes[i].PI);
                newEdge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());//
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                newEdge->setRobustKernel(rk);
                const float thHuber2D = sqrt(5.99);
                const float thHuber3D = sqrt(7.815);
                rk->setDelta(thHuber3D);
                //newEdge->setId();
                optimizer.addEdge(newEdge);
                //store edge
                all3d3dEdges.push_back(newEdge);
                //vertex index ++
                PlaneVertexNum++;
                edgeNum++;
                //newEdge->printError();
                if (pFrame->mnId == 418) {
                    cout <<setprecision(6)<< "add new world plane vertex " << 1 + PlaneVertexNum << " : PI : " << thisMapPlane->PI0
                         << " " << thisMapPlane->PI1 << " " << thisMapPlane->PI2 << " " << " map plane id "
                         << thisMapPlane->mnId << endl;
                    cout <<setprecision(6)<< "  edge observe " << pFrame->mvPlanes[i].PI.transpose() <<endl;
//                    newEdge->printError();
//                    cout<<" | chi2 "<<newEdge->chi2()<<endl;
                }

            } else
                planeVertexID.push_back(-1);
        }
        if (pFrame->mnId == 418) {
            cout << "Edge error : " << endl;
            for (size_t i = 0; i < all3d3dEdges.size(); i++) {
//                all3d3dEdges[i]->computeError();
                all3d3dEdges[i]->printError();
                cout << "chi2 " << all3d3dEdges[i]->chi2() << endl;
            }
        }

        //run optimizer
        optimizer.initializeOptimization();
        optimizer.optimize(10);


        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
//        if (pFrame->mnId == 120) {
//
//            cout<<"Edge error : "<<endl;
//            for (size_t i = 0; i < all3d3dEdges.size(); i++) {
////                all3d3dEdges[i]->computeError();
//                all3d3dEdges[i]->printError();
//                cout<<"chi2 "<<all3d3dEdges[i]->chi2()<<endl;
//            }

        cout << "after plane optimize" << endl;
        cout << pose << endl;
        vector<float> qcw2 = Converter::toQuaternion(pose(cv::Rect(0, 0, 3, 3)));
        cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
             << " " << pose.at<float>(0, 3) << " " << pose.at<float>(1, 3) << " " << pose.at<float>(2, 3)
             << " " << qcw2[0] << " " << qcw2[1] << " " << qcw2[2] << " " << qcw2[3] << endl;
        ofstream writer;
        writer.open("150frameRGBD.txt");
        for (int i = 0; i < pFrame->mvPtRGBD.size(); i++) {
            writer << pFrame->mvPtRGBD[i].x << " " << pFrame->mvPtRGBD[i].y << " " << pFrame->mvPtRGBD[i].z << endl;
        }
        writer.close();
        int pause = 0;
//        }
        //pFrame->SetPose(pose);
    }

/**
 * Added Module. optimization pose by 3d-3d matchs | for testing my g2o is working or not
 * @param pFrame
 * @return
 */
    void Optimizer::Point3dOptimization(Map *mpMap, Frame *pFrame, vector<int> matchPlanes){
        //Declare g2p optimizer
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto *solverinstance = new LinearSolverType();
        auto *blockersolverinstance = new BlockSolverType(solverinstance);
        //auto solver = new g2o::OptimizationAlgorithmLevenberg(blockersolverinstance);
        auto solver = new g2o::OptimizationAlgorithmGaussNewton(blockersolverinstance);
        g2o::SparseOptimizer optimizer;//sparse?
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);

        int VertexNum = 0;
        //Set Frame Pose Vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
//        cv::Mat play = cv::Mat::eye(4,4,CV_32F);
//        play.at<float>(0,3) = 0.005;
//        play.at<float>(1,3) = 0.005;
//        play.at<float>(2,3) = 0.005;
//        vSE3->setEstimate(Converter::toSE3Quat(play));
//        cout<<"vSE3 estimeta "<<vSE3->estimate()<<endl;
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        if(pFrame->mnId==120)
        {
            ofstream writer;
            writer.open("frame150local3Dpoint.txt");
            cout << "pose before point3d optimization " << endl;
            cout << pFrame->mTcw << endl;
            vector<float> q1 = Converter::toQuaternion(pFrame->mTcw(cv::Rect(0, 0, 3, 3)));
            cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
                 << " " << pFrame->mTcw.at<float>(0, 3) << " " << pFrame->mTcw.at<float>(1, 3) << " " << pFrame->mTcw.at<float>(2, 3)
                 << " " << q1[0] << " " << q1[1] << " " << q1[2] << " "<< q1[3] << endl;

            for (size_t i = 0; i < pFrame->mvKeyPt3D.size(); i++) {
                if (pFrame->mvpMapPoints[i]) {
                    writer << pFrame->mvKeyPt3D[i].x << " " << pFrame->mvKeyPt3D[i].y << " " << pFrame->mvKeyPt3D[i].z
                           << endl;
                }
            }
            writer.close();

            writer.open("frame150map3Dpoint.txt");
            vector<MapPlane *> mapPlanes = mpMap->GetAllMapPlanes();
            for (size_t i = 0; i < pFrame->mvpMapPoints.size(); i++) {
                if (pFrame->mvpMapPoints[i]) {
                    writer << pFrame->mvpMapPoints[i]->GetWorldPos().at<float>(0, 0) << " "
                           << pFrame->mvpMapPoints[i]->GetWorldPos().at<float>(1, 0) << " "
                           << pFrame->mvpMapPoints[i]->GetWorldPos().at<float>(2, 0) << endl;
                }
            }
            writer.close();
        }


        vSE3->setId(VertexNum);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        VertexNum++;

        vector<int> planeVertexID;
        planeVertexID.resize(pFrame->mvpMapPoints.size());
//        ///Set map 3dPoint vertex
//        for (size_t i = 0; i < pFrame->mvpMapPoints.size(); i++) {
//            if (pFrame->mvpMapPoints[i]) {
//                cv::Mat mappoint = pFrame->mvpMapPoints[i]->GetWorldPos();
//                g2o::VertexSBAPointXYZ *newVertex = new g2o::VertexSBAPointXYZ();
//                newVertex->setEstimate(
//                        Eigen::Vector3d(mappoint.at<float>(0, 0), mappoint.at<float>(1, 0), mappoint.at<float>(2, 0)));
//                newVertex->setId(VertexNum + i);
//                newVertex->setFixed(true);
//                optimizer.addVertex(newVertex);
//                planeVertexID[i] = (VertexNum + i);
//            }
//        }
        ///Set map plane vertex
        int edgeID = 0;
        vector<int> edgeIDs;
        std::vector<Edge3D3D *> all3d3dEdges;
        vector<MapPlane *> mapPlanes = mpMap->GetAllMapPlanes();
        vector<int> planeVertexIDs;
        for (size_t i = 0; i < pFrame->mvPlanes.size(); i++) {
            if (matchPlanes[i] >= 0) {
                int matchID = matchPlanes[i];
                for (size_t j = 0; j < mapPlanes.size(); j++) {
                    if (mapPlanes[j]->mnId == matchID) {
                        g2o::VertexSBAPointXYZ *newVertex = new g2o::VertexSBAPointXYZ();
                        newVertex->setEstimate(
                                Eigen::Vector3d(mapPlanes[j]->PI0, mapPlanes[j]->PI1, mapPlanes[j]->PI2));
                        newVertex->setId(VertexNum + 1);

                        newVertex->setFixed(true);
                        optimizer.addVertex(newVertex);
                        planeVertexIDs.push_back(VertexNum + 1);
                        ///Add mapplane-pose edge

                        //Set 3d3d plane Edge
                        Edge3D3D *newEdge = new Edge3D3D;
                        newEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        newEdge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(VertexNum + 1)));
                        newEdge->setMeasurement(
                                Eigen::Vector3d(pFrame->mvPlanes[i].PI[0], pFrame->mvPlanes[i].PI[1],
                                                pFrame->mvPlanes[i].PI[2]));
                        newEdge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());//
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        newEdge->setRobustKernel(rk);
                        const float thHuber2D = sqrt(5.99);
                        const float thHuber3D = sqrt(7.815);
                        rk->setDelta(thHuber3D);
                        newEdge->setId(edgeID);
                        optimizer.addEdge(newEdge);
                        //store edge
                        all3d3dEdges.push_back(newEdge);
                        edgeIDs.push_back(edgeID);
                        edgeID++;
                        //newEdge->printError();
                        VertexNum++;
                        cout<<"edge observe "<<pFrame->mvPlanes[i].PI[0]<<" "<<pFrame->mvPlanes[i].PI[1]<<" "<<pFrame->mvPlanes[i].PI[2]<<" ";
                        cout<<"map plane vertex "<<mapPlanes[j]->PI0<<" "<< mapPlanes[j]->PI1<<" "<< mapPlanes[j]->PI2<<endl;
                    }
                }
            }
        }

//        ///Set mapPoint - Pose Edge
//        int edgeID = 0;
//        vector<int> edgeIDs;
//        std::vector<Edge3D3D *> all3d3dEdges;
//        for (size_t i = 0; i < pFrame->mvpMapPoints.size(); i++) {
//            if (pFrame->mvpMapPoints[i]) {
////                cout << "mappoint " << pFrame->mvpMapPoints[i]->GetWorldPos().at<float>(0, 0) << " " <<
////                     pFrame->mvpMapPoints[i]->GetWorldPos().at<float>(1, 0) << " " <<
////                     pFrame->mvpMapPoints[i]->GetWorldPos().at<float>(2, 0) << " ";
////                cout<<" local oberve "<<pFrame->mvKeyPt3D[i].x<<" "<< pFrame->mvKeyPt3D[i].y<<" "<< pFrame->mvKeyPt3D[i].z<<endl;
//                //Set 3d3d plane Edge
//                Edge3D3D *newEdge = new Edge3D3D;
//                newEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
////                cout<<" planeVertexID[i] "<<planeVertexID[i]<<endl;
//                newEdge->setVertex(1,
//                                   dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(planeVertexID[i])));
//                newEdge->setMeasurement(
//                        Eigen::Vector3d(pFrame->mvKeyPt3D[i].x, pFrame->mvKeyPt3D[i].y, pFrame->mvKeyPt3D[i].z));
//                newEdge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());//
//                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                newEdge->setRobustKernel(rk);
//                const float thHuber2D = sqrt(5.99);
//                const float thHuber3D = sqrt(7.815);
//                rk->setDelta(thHuber3D);
//                newEdge->setId(edgeID);
//                optimizer.addEdge(newEdge);
//                //store edge
//                all3d3dEdges.push_back(newEdge);
//                edgeIDs.push_back(edgeID);
//                edgeID++;
//                //newEdge->printError();
//            }
//        }

//        ///play, only use 5 points
//        int rand1 = rand() % (all3d3dEdges.size()-1);
//        int rand2 = rand() % (all3d3dEdges.size()-1);
//        int rand3 = rand() % (all3d3dEdges.size()-1);
//        int rand4 = rand() % (all3d3dEdges.size()-1);
//        int rand5 = rand() % (all3d3dEdges.size()-1);
//        for (size_t i = 0; i < all3d3dEdges.size(); i++) {
//            all3d3dEdges[i]->setLevel(1);
//            if (i == rand1)
//                all3d3dEdges[i]->setLevel(0);
//            if (i == rand2)
//                all3d3dEdges[i]->setLevel(0);
//            if (i == rand3)
//                all3d3dEdges[i]->setLevel(0);
//            if (i == rand4)
//                all3d3dEdges[i]->setLevel(0);
//            if (i == rand5)
//                all3d3dEdges[i]->setLevel(0);
//        }

        cout << "Edge errors : " << endl;
        for (size_t i = 0; i < all3d3dEdges.size(); i++) {
            all3d3dEdges[i]->computeError();
            if(all3d3dEdges[i]->level()==0)
                cout << all3d3dEdges[i]->chi2() << " ";
        }
        cout << endl;

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        //初始化优化器，默认为0，0指的是只对level为0的边进行优化
        optimizer.initializeOptimization(0);
        //优化10次
        optimizer.optimize(10);

//        ///remove outlier then optimization, 4 times total
//        bool outlierFlags[pFrame->mvKeyPt3D.size()];
//        for(size_t i =0;i<pFrame->mvKeyPt3D.size();i++){
//            outlierFlags[i] = false;
//        }
//        const int its[4]={10,10,10,10};
//        const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
//        int nBad=0;
//        for(size_t it=0; it<4; it++)
//        {
//            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
//
//            //初始化优化器，默认为0，0指的是只对level为0的边进行优化
//            optimizer.initializeOptimization(0);
//            //优化10次
//            optimizer.optimize(its[it]);
//
//            nBad = 0;
//            //优化结束后，遍历边找outlier
//            for (size_t i = 0, iend = all3d3dEdges.size(); i < iend; i++) {
//                Edge3D3D *e = all3d3dEdges[i];
//
//                const size_t idx = edgeIDs[i];
//
//                if (outlierFlags[idx]) {
//                    e->computeError();
//                }
//
//                const float chi2 = e->chi2();
//
//                if (chi2 > chi2Stereo[it]) {
//                    pFrame->mvbOutlier[idx] = true;
//                    e->setLevel(1);
//                    nBad++;
//                } else {
//                    e->setLevel(0);
//                    pFrame->mvbOutlier[idx] = false;
//                }
//                //Only first two optimization need kernel
//                if (it == 2)
//                    e->setRobustKernel(0);
//            }
//
//            if(optimizer.edges().size()<10)
//                break;
//        }

        cout<<"Edge errors : "<<endl;
        for (size_t i = 0; i < all3d3dEdges.size(); i++) {
            all3d3dEdges[i]->computeError();
            if(all3d3dEdges[i]->level()==0)
                cout<<all3d3dEdges[i]->chi2()<<" ";
        }
        cout<<endl;

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        if (pFrame->mnId == 120) {
            cout << "after point3d optimize" << endl;
            cout << pose << endl;
            vector<float> qcw2 = Converter::toQuaternion(pose(cv::Rect(0, 0, 3, 3)));
            cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
                 << " " << pose.at<float>(0, 3) << " " << pose.at<float>(1, 3) << " " << pose.at<float>(2, 3)
                 << " " << qcw2[0] << " " << qcw2[1] << " " << qcw2[2] << " " << qcw2[3] << endl;
            ofstream writer;
            writer.open("150frameRGBD.txt");
            for (int i = 0; i < pFrame->mvPtRGBD.size(); i++) {
                writer << pFrame->mvPtRGBD[i].x << " " << pFrame->mvPtRGBD[i].y << " " << pFrame->mvPtRGBD[i].z << endl;
            }
            writer.close();
            int pause = 0;
        }
        //pFrame->SetPose(pose);

    }

    /**
 * If matched plane number reach threshold, run this function instead of the original Poseoptimization
 * the plane feature is represented as unaryEdge
 */
    int Optimizer::JointOptimization(Map *mpMap, Frame *pFrame, vector<int> matchPlanes){
        ///Step 1 declar g2o optimizer, blocksolver_6_3, Pose 6 dimension, LandMark 3 dimension
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3 ::LinearSolverType *linearSolver;
        linearSolver = new  g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        int nInitialCorrespondences = 0;

        ///Print
//        cout << "pose before plane optimization " << endl;
//        cout << pFrame->mTcw << endl;
//        vector<float> q1 = Converter::toQuaternion(pFrame->mTcw(cv::Rect(0, 0, 3, 3)));
//        cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
//             << " " << pFrame->mTcw.at<float>(0, 3) << " " << pFrame->mTcw.at<float>(1, 3) << " " << pFrame->mTcw.at<float>(2, 3)
//             << " " << q1[0] << " " << q1[1] << " " << q1[2] << " "<< q1[3] << endl;

        ///Step2 add pose vertex --- the pose to be optimized
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        //set vertex id
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        //Store MapPoint vertices
        const int N = pFrame->N;
        //Stereo
        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
        vector<size_t> vpIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vpIndexEdgeStereo.reserve(N);

        //2 Dof 卡方檢驗
        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);

        ///Added module. weighted error
        double pointErrorSum=0;
        double pointWeightSum = 0;
        int pointVertexNum = 0;
        ///Step 3 Add 3D2D unary edge
        {
            //while adding edge, we dont want the mappoint be modified
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);
            //traverse all mappoint
            for(size_t i =0;i<N;i++){
                MapPoint * pMP = pFrame->mvpMapPoints[i];
                if(pMP){
                    //Stereo Observation
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i]=false;
                    //Set Edge
                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs<<kpUn.pt.x,kpUn.pt.y,kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    pointWeightSum += invSigma2; //record weight
                    pointVertexNum++; //record point vertex number
                    e->setInformation(Info);
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);
                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpIndexEdgeStereo.push_back(i);

                    e->computeError();
                    pointErrorSum += abs(e->chi2());//record sum error
                }
            }
        }
        cout<<"add "<<pointVertexNum<<" point Vertex with AVG weight "<<pointWeightSum/pointVertexNum;

        if(nInitialCorrespondences<3)
            return 0;

        //Plane
        vector<UnaryEdgePlane *> vpEdgesPlane;
        vector<bool> planeFlags;
        double planeErrorSum = 0;
        double planeWeightSum = 0;
        int planeVertexNum = 0;
        ///Step 4 Add Plane Edge
        {
            //while adding edge, we dont want the mappoint be modified
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);
            vector<MapPlane *> mapPlanes = mpMap->GetAllMapPlanes();
            //traverse all matchPlnaes
            for (size_t i = 0; i < matchPlanes.size(); i++) {
                if (matchPlanes[i] >= 0) {
                    int mapPlaneID = matchPlanes[i];
                    int mapPlaneIndex = -1;
                    //found map plane index
                    for (size_t j = 0; j < mapPlanes.size(); j++) {
                        if (mapPlanes[j]->mnId == mapPlaneID) {
                            mapPlaneIndex = j;
                            break;
                        }
                    }
                    //Set plane Vertex
                    UnaryEdgePlane *newEdge = new UnaryEdgePlane;
                    newEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    newEdge->setMeasurement(pFrame->mvPlanes[i].PI);
                    newEdge->PI_map = Eigen::Vector3d(mapPlanes[mapPlaneIndex]->PI0, mapPlanes[mapPlaneIndex]->PI1, mapPlanes[mapPlaneIndex]->PI2);
                    newEdge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());//
                    planeWeightSum+=1;// record weight
                    planeVertexNum++;// record plane vertex number
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    newEdge->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);
                    optimizer.addEdge(newEdge);
                    vpEdgesPlane.push_back(newEdge);
                    planeFlags.push_back(false);
                    //newEdge->printError();

                    newEdge->computeError();// record plane vertex error
                    planeErrorSum += newEdge->chi2();
                }
            }
        }
        cout<<"add "<<planeVertexNum<<" plane Vertex with AVG Weight "<<planeWeightSum/planeVertexNum<<endl;


        ///run optimizer
//        optimizer.initializeOptimization();
//        optimizer.optimize(10);
        ///Step 5 start optimization, 4 times, filter outlier
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};
        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            //vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));//? why, this is reset
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];
                const size_t idx = vpIndexEdgeStereo[i];
                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }
                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }
                //Only first two optimization need kernel
                if (it == 2)
                    e->setRobustKernel(0);
            }
            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                UnaryEdgePlane *e = vpEdgesPlane[i];
                if (planeFlags[i]) {
                    e->computeError();
                }
                const float chi2 = e->chi2();
                if (chi2 > chi2Stereo[it]) {
                    planeFlags[i] = true;
                    e->setLevel(1);
                } else {
                    e->setLevel(0);
                    planeFlags[i] = false;
                }
                //Only first two optimization need kernel
                if (it == 2)
                    e->setRobustKernel(0);
            }
            if(optimizer.edges().size()<10)
                break;
        }

        /// Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);
        ///Print
//        cout << "after plane optimize" << endl;
//        cout << pose << endl;
//        vector<float> qcw2 = Converter::toQuaternion(pose(cv::Rect(0, 0, 3, 3)));
//        cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
//             << " " << pose.at<float>(0, 3) << " " << pose.at<float>(1, 3) << " " << pose.at<float>(2, 3)
//             << " " << qcw2[0] << " " << qcw2[1] << " " << qcw2[2] << " " << qcw2[3] << endl;

        int pause = 1;
    }

/**
 * @brief Pose only optimization
 * 3D-2D 最小化重投影误差 e=(u,v)-project(Tcw*Pw)
 * 
 * 1. Vertex: g2o::VertexSE3Expmap(),当前帧Tcw
 * 2. Edge：
 *      -g2o:EdgeSE3ProjectXYZOnlyPose(), BaseUnaryEdge
 *          +Vertex: 待优化当前帧的Tcw
 *          +measurement: MapPoint在当前帧中的二维位置（u，v）
 *          +InfoMatrix：invSigma2（与特征点所在的尺度有关）
 *      -g2o::EdgeStereoSE3ProjectXYZOnlyPose(), BaseUnaryEdge
 *          +Vertex: 待优化当前帧的Tcw
 *          +measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *          +InfoMatrix:invSigma2(与特征点所在的尺度有关
 * @param pFrame Frame
 * @return inliers数量
*/
int Optimizer::PoseOptimization(Frame *pFrame)
{
    //*Step 1: 构造g2o优化器,BlockSolver_6_3：位姿_PoseDim 6维 路标 _LandmarkDim 3维
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    //*Step 2: 添加顶点,待优化帧的Tcw
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));

//    if (pFrame->mnId == 120) {
//        ofstream writer("frame150points.txt");
//        cout << "pose before pose optimization " << endl;
//        cout << pFrame->mTcw << endl;
//        vector<float> qcw1 = Converter::toQuaternion(pFrame->mTcw(cv::Rect(0, 0, 3, 3)));
//        cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7)
//             << " | " << pFrame->mTcw.at<float>(0, 3) << " " << pFrame->mTcw.at<float>(1, 3) << " " << pFrame->mTcw.at<float>(2, 3)
//             << " | " << qcw1[0] << " " << qcw1[1] << " " << qcw1[2] << " " << qcw1[3] << endl;
//        for (int i = 0; i < pFrame->N; i++) {
//            MapPoint *pMP = pFrame->mvpMapPoints[i];
//            if (pMP) {
//                writer << pMP->GetWorldPos().at<float>(0) << " " << pMP->GetWorldPos().at<float>(1) << " "
//                       << pMP->GetWorldPos().at<float>(2) << endl;
//            }
//        }
//        writer.close();
//    }

    //设置ID
    vSE3->setId(0);
    //要优化所以不能fixed
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    //Monocular
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);
    //Stereo
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    //自由度为2的卡方分布，显著性水平为0.05,对应的临界值是5.991
    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    //*Step 3: 添加一元边
    {
    //使用地图点构建图的时候，不希望地图点被修改。
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    //遍历（当前帧）地图点
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;
                //新建节点
                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                //填充
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                //这个点的可信度和金字塔层级有关
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                //?跟其他地方信息矩阵用方差的逆有何区别？
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                //鲁棒核函数
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }


    if(nInitialCorrespondences<3)
        return 0;

    //*Step 4: 开始优化，总共优化4次，每次优化迭代10次，每次优化后将观测点区分为outlier和inlier
    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        //初始化优化器，默认为0，0指的是只对level为0的边进行优化
        optimizer.initializeOptimization(0);
        //优化10次
        optimizer.optimize(its[it]);

        nBad=0;
        //优化结束后，遍历边找outlier
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            //? 优化结束后compute error的目的？
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            //error * informatrix * error
            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                //如果误差大于阈值，离群标记true
                //点level设置为1
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            //只有前两次需要鲁棒核函数，之后重投影误差显著下降，不再需要。
            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }
            //Only first two optimization need kernel
            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    //* Step 5 得到优化后的当前帧的位姿
    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);
//    if (pFrame->mnId ==120) {
//        cout << "pose after pose optimization " << endl;
//        cout << pose << endl;
//        vector<float> qcw2 = Converter::toQuaternion(pose(cv::Rect(0, 0, 3, 3)));
//        cout << setprecision(6) << pFrame->mTimeStamp << setprecision(7) << " | "
//             << pose.at<float>(0, 3) << " " << pose.at<float>(1, 3) << " " << pose.at<float>(2, 3) << " | "
//             << qcw2[0] << " " << qcw2[1] << " " << qcw2[2] << " " << qcw2[3] << endl;
//    }
    return nInitialCorrespondences-nBad;
}

/**
 * @brief local Bundle Adjustment
 * 1. Vertex:
 *  -g2o::VertexSE3Expmap(), LocalKeyFrames, 即当前关键帧的位姿，与当前关键帧相连的关键帧的位姿
 *  -g2o::VertexSE3Expmap(), FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocakKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *  -g2o::VertexS/baPointXYZ(), LocalMapPoints, 即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *  -g2o::EdgeSE3ProjectXYZ(), BaseBinaryEdge
 *      +vertex: 关键帧的Tcw，Mappoint的Pw
 *      +measurement：Mappoint在关键帧中的二维位置uv
 *      +Infomatrix：invSigma2（于特征点所在的尺度有关）
 *  -g2o：：EdgeStereoSEProjectXYZ（），Base Binary Edge
 *      +Vertex：关键帧的TWC，Mappoint的PW
 *      +measurement：MapPoint在关键帧中的二维位置ul，v，ur
 *      +InfoMatrix：invsigma2（于特征点所在的尺度有关）
 * 
 * @param pKF KeyFrame
 * @param pbStopFlag 是否停止优化标志
 * @param pMap 在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * @note 由局部建图线程调用，对局部地图进行优化的函数
*/
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    ///Added Module --- to record points and planes
    ofstream writer1,writer2,writer3;
    writer1.open("localBundlePoints.txt");
    writer2.open("localBundlePlanes.txt");
    writer3.open("localBundlePose.txt");

    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    //*Step 1 当前关键帧 和 共视关键帧 加入到localkeyframe
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        //参与局部BA的每一个关键帧的mnBALocalForKF设置为当前关键帧的mnID，防止重复添加
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    //*Step 2 遍历共视关键帧，把观测到的地图点加入localmappoints
    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        //取出地图点集合
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        //遍历每一个地图点
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                //地图点的localforkf标签设置
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    //*Step 3 找到能被局部MapPoints观测到，但是不属于上述局部关键帧集合的关键帧（二级关键帧），这些关键帧在局部BA不优化
    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        //观测到该Mappoint的kf和该mappoint在kf中的索引
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        //遍历所有观测到该maopoint的关键帧
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            //遍历到的关键帧的mnBALocalForKF 还没有被分配到当前关键帧（上面的步骤）
            //还没有被标记为FIX关键帧
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                //将遍历到的关键帧fix
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    //*Step 4 构造g2o优化器
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    //外界设置的停止优化标志
    //可能在Tracking::neednewkeyframe里面置位
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    //参与局部BA的最大关键帧mnID
    unsigned long maxKFid = 0;

    //*Step 5 添加 待优化的位姿顶点 pose of local keyframe
    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        ///Added Module --- store data
        writer3>>Converter::toSE3Quat(pKFi->GetPose())<<endl;
        writer3>>pKFi->mnId<<endl;
        ///-----
        //第0帧不优化
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    //*Step 6 添加不优化的位姿顶点 pose of fixed keyframe
    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        //fix
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }
    //*Step 7 添加待优化的3D地图点顶点
    //边的数目=pose数目*地图点数目
    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    //遍历局部地图中的地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        //用到了前面的最大关键帧ID
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        //因为使用了linearsolvertype，所有的三维点可以边缘化
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        //观测到该地图点的kf和该地图点在kf中的索引
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //*Step 8 添加完一个地图点之后，对每一对关联的mappoint和keyframe构建边
        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    //*Step 9 开始优化
    optimizer.initializeOptimization();
    //先优化五次
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        //*Step 10 outlier不再参与优化
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        //误差超过5.991或者深度为负不再优化
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }
        //第二阶段不再需要鲁棒和函数
        e->setRobustKernel(0);
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    //*Step 11 排除outlier之后再10次优化
    // Optimize again without the outliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    //*Step 12 重新计算误差，剔除误差大的关键帧和mappoint
    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            //加入到删除vector
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    //锁住地图点
    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            //帧删除点，点删除帧
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //*Step 13 优化后更新关键帧位姿 地图点位置 和平均观测方向等属性
    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
//        ///print
//        cout << "pose before localBundle optimization " << endl;
//        //cout << pKF->mTcw << endl;
//        vector<float> q1 = Converter::toQuaternion(pKF->GetPose()(cv::Rect(0, 0, 3, 3)));
//        cout << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " | " << pKF->GetPose().at<float>(0, 3) << " "
//             << pKF->GetPose().at<float>(1, 3) << " " << pKF->GetPose().at<float>(2, 3) << " | " << q1[0] << " " << q1[1] << " "
//             << q1[2] << " "
//             << q1[3] << endl;

        pKF->SetPose(Converter::toCvMat(SE3quat));

//        ///print
//        auto pose = Converter::toCvMat(SE3quat);
//        cout << "pose after localBundle optimization " << endl;
//        //cout << Converter::toCvMat(SE3quat) << endl;
//        vector<float> q2 = Converter::toQuaternion(pose(cv::Rect(0, 0, 3, 3)));
//        cout << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " | "
//             << pose.at<float>(0, 3) << " "<< pose.at<float>(1, 3) << " " << pose.at<float>(2, 3) << " | "
//             << q2[0] << " " << q2[1] << " "<< q2[2] << " "<< q2[3] << endl;
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    //*Step 1 初始化g2o优化器
    //构造求解器
    g2o::SparseOptimizer optimizer;
    //构造线性方程求解器 Hx=-b的求解器
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    //使用dense求解器(常见非dense求解器有cholmod先行求解器喝shur补线性求解器)
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    //使用LM迭代
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    //内参矩阵
    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    //*Step 2 设置Sim3 作为顶点
    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    //根据传感器类型决定是否固定尺度
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    //Sim3不fix 需要优化
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2); //光心横坐标cx
    vSim3->_principle_point1[1] = K1.at<float>(1,2); //光新纵坐标cy
    vSim3->_focal_length1[0] = K1.at<float>(0,0); //焦距fx
    vSim3->_focal_length1[1] = K1.at<float>(1,1); //焦距fy
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    //*Step 3 设置地图点作为顶点
    // Set MapPoint vertices
    const int N = vpMatches1.size();
    //获取pKF1的地图点
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12; //pKF2对应的地图点到pKF1的投影边
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21; //pKF1对应的地图点到pKF2的投影边
    vector<size_t> vnIndexEdge; //边的索引

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    //核函数的阈值
    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    //遍历每对匹配点
    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        //i2是pMP2在pKF2中的索引
        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                //添加PointXYZ顶点
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                //地图点 转换到 各自相机坐标系下的三维点
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                //地图点不优化
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        //对匹配关系计数
        nCorrespondences++;

        //*Step 4 添加边（地图点投影到特征点为边）
        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;
        //*Step 4.1 闭环候选帧 投影到 关键帧 的边 正向投影 T12
        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        //id2 说明是以pKF2为索引在添加边（正向投影
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        //*另一个顶点是待优化的sim3相似变换。
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        //*Step 4.2 反向投影
        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }
    //*Step 5 优化5次
    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    //*Step 剔除outlier
    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            //正反投影任意一个误差超过阈值都要删除该边
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    //有删除误差边 说明质量不好 增加迭代次数
    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    //*Step 7 再次迭代 
    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    //统计inlier
    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            //因为不再优化了 就不删除边了 只是匹配关系置为null
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    //*Step 8 优化后的结果
    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM

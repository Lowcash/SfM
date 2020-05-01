#include "reconstruction.h"

Reconstruction::Reconstruction(const std::string triangulateMethod, const std::string baMethod, const std::string cSolverType, const float minDistance, const float maxDistance, const float maxProjectionError, const bool useNormalizePts)
    : m_triangulateMethod(triangulateMethod), m_baMethod(baMethod), m_cSolverType(cSolverType), m_minDistance(minDistance), m_maxDistance(maxDistance), m_maxProjectionError(maxProjectionError), m_useNormalizePts(useNormalizePts) {}

void Reconstruction::pointsToRGBCloud(Camera camera, cv::Mat imgColor, cv::Mat R, cv::Mat t, cv::Mat points3D, cv::Mat inputPts2D, std::vector<cv::Vec3d>& cloud3D, std::vector<cv::Vec3b>& cloudRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask) {
    cv::Mat _pts2D; cv::projectPoints(points3D, R, t, camera.K, cv::Mat(), _pts2D);

    cloud3D.clear();
    cloudRGB.clear();
    
    for(size_t i = 0; i < points3D.rows; ++i) {
        const cv::Vec3d point3D = m_useNormalizePts ? 
            points3D.at<cv::Vec3d>(i) : (cv::Vec3d)points3D.at<cv::Vec3f>(i);
        const cv::Vec2d _point2D = m_useNormalizePts ? 
            _pts2D.at<cv::Vec2d>(i) : (cv::Vec2d)_pts2D.at<cv::Vec2f>(i);
        
        const cv::Vec2d point2D = inputPts2D.at<cv::Vec2d>(i);
        const cv::Vec3b imPoint2D = imgColor.at<cv::Vec3b>(cv::Point(point2D));

        const double err = cv::norm(_point2D - point2D);

        cloud3D.push_back( point3D );
        cloudRGB.push_back( imPoint2D );

        mask.push_back( err <  maxProjErr && point3D[2] > minDist && point3D[2] < maxDist );
    }
}

void Reconstruction::triangulateCloud(Camera camera, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const cv::Mat colorImage, std::vector<cv::Vec3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, std::vector<bool>& mask, const cv::Matx34d prevPose, const cv::Matx34d currPose, RecoveryPose& recPose) {
    cv::Mat _prevPtsN; cv::undistort(prevPts, _prevPtsN, camera.K33d, cv::Mat());
    cv::Mat _currPtsN; cv::undistort(currPts, _currPtsN, camera.K33d, cv::Mat());

    cv::Mat _prevPtsMat, _currPtsMat; 

    if (m_useNormalizePts) {
        pointsToMat(_prevPtsN, _prevPtsMat);
        pointsToMat(_currPtsN, _currPtsMat);
    } else {
        pointsToMat(prevPts, _prevPtsMat);
        pointsToMat(currPts, _currPtsMat);
    }

    cv::Mat _homogPts, _pts3D;

    if (m_triangulateMethod == "DLT") {
        std::vector<cv::Mat> _pts, _projMats;
        _pts.push_back(_prevPtsMat);
        _pts.push_back(_currPtsMat);
        _projMats.push_back(cv::Mat(camera.K33d * prevPose));
        _projMats.push_back(cv::Mat(camera.K33d * currPose));

        cv::sfm::triangulatePoints(_pts, _projMats, _pts3D); _pts3D = _pts3D.t();
    } else {
        cv::triangulatePoints(camera.K33d * prevPose, camera.K33d * currPose, _prevPtsMat, _currPtsMat, _homogPts);
        cv::convertPointsFromHomogeneous(_homogPts.t(), _pts3D);
    }

    pointsToRGBCloud(camera, colorImage, cv::Mat(recPose.R), cv::Mat(recPose.t), _pts3D, _currPtsMat.t(), points3D, pointsRGB, m_minDistance, m_maxDistance, m_maxProjectionError, mask);
}

void Reconstruction::adjustBundle(Camera& camera, std::vector<TrackView>& tracks, std::vector<cv::Matx34d>& camPoses, uint maxIter) {
    std::cout << "Bundle adjustment...\n" << std::flush;

    if (m_baMethod == "CERES") {
        cv::Matx14d intrinsics4d (
            camera.focal.x,
            camera.focal.y,
            camera.pp.x,
            camera.pp.y
        );

        std::vector<cv::Matx16d> extrinsics6d;
    
        for (auto [c, cEnd, it] = std::tuple{camPoses.cbegin(), camPoses.cend(), 0}; c != cEnd && it < maxIter; ++c, ++it) {
            cv::Matx34d cam = (cv::Matx34d)*c;

            if (cam(0, 0) == 0 && cam(1, 1) == 0 && cam(2, 2) == 0) { 
                extrinsics6d.push_back(cv::Matx16d());
                continue; 
            }

            cv::Vec3f t(cam(0, 3), cam(1, 3), cam(2, 3));
            cv::Matx33f R = cam.get_minor<3, 3>(0, 0);
            float angleAxis[3]; ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);

            extrinsics6d.push_back(cv::Matx16d(
                angleAxis[0],
                angleAxis[1],
                angleAxis[2],
                t(0),
                t(1),
                t(2)
            ));
        }

        ceres::Problem problem;

        bool isCameraLocked = false;
        for (auto [t, tEnd, c, cEnd, it] = std::tuple{tracks.begin(), tracks.end(), extrinsics6d.begin(), extrinsics6d.end(), 0}; t != tEnd && c != cEnd && it < maxIter; ++t, ++c, ++it) {
            for (size_t i = 0; i < t->numTracks; ++i) {
                cv::Point2f p2d = t->points2D[i];
                p2d.x -= camera.K33d(0, 2);
                p2d.y -= camera.K33d(1, 2);
                
                ceres::CostFunction* costFunc = SnavelyReprojectionError::Create(p2d.x, p2d.y);

                problem.AddResidualBlock(costFunc, NULL, c->val, &intrinsics4d, t->points3D[i].val);

                if (!isCameraLocked) {
                    problem.SetParameterBlockConstant(&extrinsics6d[0](0));

                    isCameraLocked = true;
                }
            }
        }

        problem.SetParameterBlockConstant(&intrinsics4d(0));
        problem.SetParameterBlockConstant(&intrinsics4d(1));
        problem.SetParameterBlockConstant(&intrinsics4d(2));
        problem.SetParameterBlockConstant(&intrinsics4d(3));

        ceres::Solver::Options options;

        options.linear_solver_type = m_cSolverType == "DENSE_SCHUR" ? 
            ceres::LinearSolverType::DENSE_SCHUR : ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;

        options.minimizer_progress_to_stdout = true;
        options.eta = 1e-2;
        options.num_threads = std::thread::hardware_concurrency();
        options.max_num_iterations = 100;

        // options.use_nonmonotonic_steps = true;
        // options.preconditioner_type = ceres::SCHUR_JACOBI;
        // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        // options.use_inner_iterations = true;
        // options.max_num_iterations = 100;
        // options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;

        cv::Mat K; camera.K.copyTo(K);

        K.at<double>(0,0) = intrinsics4d(0);
        K.at<double>(1,1) = intrinsics4d(1);
        K.at<double>(0,2) = intrinsics4d(2);
        K.at<double>(1,2) = intrinsics4d(3);

        camera.updateCameraParameters(K, camera.distCoeffs);

        for (auto [c, cEnd, c6, c6End, it] = std::tuple{camPoses.begin(), camPoses.end(), extrinsics6d.begin(), extrinsics6d.end(), 0}; c != cEnd && c6 != c6End && it < maxIter; ++c, ++c6, ++it) {
            cv::Matx34d& cam = (cv::Matx34d&)*c;
            cv::Matx16d& cam6 = (cv::Matx16d&)*c6;

            if (cam(0, 0) == 0 && cam(1, 1) == 0 && cam(2, 2) == 0) { continue; }

            double rotationMat[9] = { 0 };
            ceres::AngleAxisToRotationMatrix(cam6.val, rotationMat);

            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    cam(column, row) = rotationMat[row * 3 + column];
                }
            }

            cam(0, 3) = cam6(3); 
            cam(1, 3) = cam6(4); 
            cam(2, 3) = cam6(5);
        }
    } else {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;

        gtsam::Cal3_S2 K = gtsam::Cal3_S2(camera.focal.x, camera.focal.y, 0 /*skew*/, camera.pp.x, camera.pp.y);

        gtsam::noiseModel::Diagonal::shared_ptr camNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(5) << 0, 0, 0 /*skew*/, 0, 0).finished());
        graph.emplace_shared<gtsam::PriorFactor<gtsam::Cal3_S2>>(gtsam::Symbol('K', 0), K, camNoise);

        initial.insert(gtsam::Symbol('K', 0), K);

        gtsam::noiseModel::Isotropic::shared_ptr measurementNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);

        int startingIter = MAX((int)(camPoses.size() - maxIter), 0);

        for (size_t i = startingIter, points = 0; i < tracks.size(); ++i) {
            cv::Matx34d& _c = (cv::Matx34d&)camPoses[i];

            gtsam::Rot3 R(
                _c(0,0), _c(0,1), _c(0,2),
                _c(1,0), _c(1,1), _c(1,2),
                _c(2,0), _c(2,1), _c(2,2)  
            );

            gtsam::Point3 t(
                _c(0,3), _c(1,3), _c(2,3)
            );
            
            gtsam::Pose3 pose(R, t);

            // Add prior for the first image
            if (i == startingIter) {
                gtsam::noiseModel::Diagonal::shared_ptr poseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.1)).finished());
                graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3> >(gtsam::Symbol('x', startingIter), pose, poseNoise); // add directly to graph
            }

            initial.insert(gtsam::Symbol('x', i), pose);

            TrackView& _t = (TrackView&)tracks[i];
            
            for (size_t m = 0; m < _t.points2D.size() && m < _t.points3D.size(); ++m) {
                gtsam::Point2 gP2d(_t.points2D[m].x, _t.points2D[m].y);

                graph.emplace_shared<gtsam::GeneralSFMFactor2<gtsam::Cal3_S2>>(gP2d, measurementNoise, gtsam::Symbol('x', i), gtsam::Symbol('l', points), gtsam::Symbol('K', 0));
                
                cv::Vec3d &p3dRef = (cv::Vec3d&)_t.points3D[m];

                initial.insert<gtsam::Point3>(gtsam::Symbol('l', points), gtsam::Point3(p3dRef[0], p3dRef[1], p3dRef[2]));

                if (i == startingIter) {
                    gtsam::noiseModel::Isotropic::shared_ptr pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
                    gtsam::Point3 p(p3dRef[0], p3dRef[1], p3dRef[2]);
                    graph.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(gtsam::Symbol('l', points), p, pointNoise);
                }

                points++;
            }
        }

        gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
        
        graph.print();

        std::cout << "\n";
        std::cout << "initial graph error = " << graph.error(initial) << "\n";
        std::cout << "final graph error = " << graph.error(result) << "\n";

        gtsam::Matrix3 K_refined = result.at<gtsam::Cal3_S2>(gtsam::Symbol('K', 0)).K();

        cv::Mat _K; camera.K.copyTo(_K);

        _K.at<double>(0,0) = K_refined(0,0);
        _K.at<double>(1,1) = K_refined(1,1);
        _K.at<double>(0,2) = K_refined(0,2);
        _K.at<double>(1,2) = K_refined(1,2);

        camera.updateCameraParameters(_K, camera.distCoeffs);
        
        for (size_t i = startingIter, points = 0; i < tracks.size(); ++i) {
            gtsam::Point3 gT = result.at<gtsam::Pose3>(gtsam::Symbol('x', i)).translation();

            gtsam::Rot3 gR = result.at<gtsam::Pose3>(gtsam::Symbol('x', i)).rotation();
            gtsam::Point3 gR1 = gR.r1();
            gtsam::Point3 gR2 = gR.r2();
            gtsam::Point3 gR3 = gR.r3();

            camPoses[i] = cv::Matx34d(
                gR1[0], gR1[1], gR1[2], gT[0],
                gR2[0], gR2[1], gR2[2], gT[1],
                gR3[0], gR3[1], gR3[2], gT[2]
            );

            TrackView& t = (TrackView&)tracks[i];

            for (size_t m = 0; m < t.points2D.size() && m < t.points3D.size(); ++m) {
                gtsam::Point3 gP3d = result.at<gtsam::Point3>(gtsam::Symbol('l', points));

                cv::Vec3d &p3dRef = (cv::Vec3d&)t.points3D[m];
                
                p3dRef[0] = gP3d[0];
                p3dRef[1] = gP3d[1];
                p3dRef[2] = gP3d[2];

                points++;
            }
        }
    }

    std::cout << "[DONE]\n";
}
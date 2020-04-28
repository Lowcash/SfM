#include "reconstruction.h"

Reconstruction::Reconstruction(const std::string method, const float minDistance, const float maxDistance, const float maxProjectionError, const bool useNormalizePts)
    : m_method(method), m_minDistance(minDistance), m_maxDistance(maxDistance), m_maxProjectionError(maxProjectionError), m_useNormalizePts(useNormalizePts) {}

void Reconstruction::pointsToRGBCloud(cv::Mat imgColor, Camera camera, cv::Mat R, cv::Mat t, cv::Mat points3D, cv::Mat inputPts2D, std::vector<cv::Vec3d>& cloud3D, std::vector<cv::Vec3b>& cloudRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask) {
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

void Reconstruction::triangulateCloud(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const cv::Mat colorImage, std::vector<cv::Vec3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, std::vector<bool>& mask, Camera camera, const cv::Matx34d prevPose, const cv::Matx34d currPose, RecoveryPose& recPose) {
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

    if (m_method == "DLT") {
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

    pointsToRGBCloud(colorImage, camera, cv::Mat(recPose.R), cv::Mat(recPose.t), _pts3D, _currPtsMat.t(), points3D, pointsRGB, m_minDistance, m_maxDistance, m_maxProjectionError, mask);
}

void Reconstruction::adjustBundle(std::vector<TrackView>& tracks, Camera& camera, std::vector<cv::Matx34f>& camPoses, std::string solverType, std::string lossType, double lossFunctionScale, uint maxIter) {
    std::cout << "Bundle adjustment...\n" << std::flush;

    std::vector<cv::Matx16d> camPoses6d;
    
    for (auto [c, cEnd, it] = std::tuple{camPoses.cbegin(), camPoses.cend(), 0}; c != cEnd && it < maxIter; ++c, ++it) {
        cv::Matx34f cam = (cv::Matx34f)*c;

        if (cam(0, 0) == 0 && cam(1, 1) == 0 && cam(2, 2) == 0) { 
            camPoses6d.push_back(cv::Matx16d());
            continue; 
        }

        cv::Vec3f t(cam(0, 3), cam(1, 3), cam(2, 3));
        cv::Matx33f R = cam.get_minor<3, 3>(0, 0);
        float angleAxis[3]; ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);

        camPoses6d.push_back(cv::Matx16d(
            angleAxis[0],
            angleAxis[1],
            angleAxis[2],
            t(0),
            t(1),
            t(2)
        ));
    }

    ceres::Problem problem;
    ceres::LossFunction* lossFunction;

    if (lossType == "HUBER")
        lossFunction = new ceres::HuberLoss(lossFunctionScale);
    else
        lossFunction = new ceres::CauchyLoss(lossFunctionScale);

    double focalLength = camera.focalLength;

    bool isCameraLocked = false;
    for (auto [t, tEnd, c, cEnd, it] = std::tuple{tracks.begin(), tracks.end(), camPoses6d.begin(), camPoses6d.end(), 0}; t != tEnd && c != cEnd && it < maxIter; ++t, ++c, ++it) {
        for (size_t i = 0; i < t->numTracks; ++i) {
            cv::Point2f p2d = t->points2D[i];
            p2d.x -= camera.K33d(0, 2);
            p2d.y -= camera.K33d(1, 2);
            
            ceres::CostFunction* costFunc = SnavelyReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(costFunc, NULL, c->val, t->points3D[i].val, &focalLength);

            if (!isCameraLocked) {
                //problem.SetParameterBlockConstant(&camPoses6d[0](0));

                isCameraLocked = true;
            }
        }
    }

    ceres::Solver::Options options;
    if (solverType == "DENSE_SCHUR")
        options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
    else
        options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;

    options.minimizer_progress_to_stdout = true;
    options.eta = 1e-2;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 100;

    /*options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.use_inner_iterations = true;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;*/

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    cv::Mat K; camera.K.copyTo(K);

    K.at<double>(0,0) = focalLength;
    K.at<double>(1,1) = focalLength;

    camera.updateCameraParameters(K, camera.distCoeffs);

    /*if (!summary.IsSolutionUsable()) {
        std::cout << "Bundle Adjustment failed." << std::endl;
    } else {
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << camPoses.size() << "\n"
            << " #num_residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << std::endl;
    }*/

    for (auto [c, cEnd, c6, c6End, it] = std::tuple{camPoses.begin(), camPoses.end(), camPoses6d.begin(), camPoses6d.end(), 0}; c != cEnd && c6 != c6End && it < maxIter; ++c, ++c6, ++it) {
        cv::Matx34f& cam = (cv::Matx34f&)*c;
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

    std::cout << "[DONE]\n";
}
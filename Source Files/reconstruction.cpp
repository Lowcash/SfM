#include "reconstruction.h"

Reconstruction::Reconstruction(const std::string triangulateMethod, const std::string baMethod, const float minDistance, const float maxDistance, const float maxProjectionError, const bool useNormalizePts)
    : m_triangulateMethod(triangulateMethod), m_baMethod(baMethod), m_minDistance(minDistance), m_maxDistance(maxDistance), m_maxProjectionError(maxProjectionError), m_useNormalizePts(useNormalizePts) {}

void Reconstruction::pointsToRGBCloud(Camera camera, cv::Mat imgColor, cv::Mat R, cv::Mat t, cv::Mat points3D, cv::Mat inputPts2D, std::vector<cv::Vec3d>& cloud3D, std::vector<cv::Vec3b>& cloudRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask) {
    //  Project 3D points back to image plane for validation
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

        //  set mask to filter bad projected points, points behind camera and points far away from camera
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

void Reconstruction::adjustBundle(Camera& camera, std::vector<cv::Vec3d>& pCloud, std::vector<CloudTrack>& pCloudTracks, std::list<cv::Matx34d>& camPoses) {
    std::cout << "Bundle adjustment...\n" << std::flush;

    if (pCloud.empty()) {
        std::cout << "Empty cloud -> interrupting bundle adjustment!" << "\n";

        return;
    }

    cv::Matx14d intrinsics4d (
        camera.focal.x,
        camera.focal.y,
        camera.pp.x,
        camera.pp.y
    );

    std::vector<cv::Matx16d> extrinsics6d;

    for (auto [c, cEnd, it] = std::tuple{camPoses.cbegin(), camPoses.cend(), 0}; c != cEnd; ++c, ++it) {
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
    for (auto [c, ct, cEnd, ctEnd] = std::tuple{pCloud.begin(), pCloudTracks.begin(), pCloud.end(), pCloudTracks.end()}; c != cEnd && ct != ctEnd; ++c, ++ct) {
        for (size_t idx = 0; idx < ct->numTracks; ++idx) {
            cv::Point2f p2d = ct->projKeys[idx];
            cv::Matx16d* ext = &extrinsics6d[ct->extrinsicsIdxs[idx]];
            
            ceres::CostFunction* costFunc = SnavelyReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(costFunc, NULL, intrinsics4d.val, ext->val, c->val);

            if (!isCameraLocked) {
                problem.SetParameterBlockConstant(ext->val);

                isCameraLocked = true;
            }
        }
    }

    problem.SetParameterBlockConstant(&intrinsics4d(0));

    ceres::Solver::Options options;

    options.linear_solver_type = m_baMethod == "DENSE_SCHUR" ? 
        ceres::LinearSolverType::DENSE_SCHUR : ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;

    options.minimizer_progress_to_stdout = true;
    options.eta = 1e-2;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 100;

    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.use_inner_iterations = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    cv::Mat K; camera.K.copyTo(K);

    K.at<double>(0,0) = intrinsics4d(0);
    K.at<double>(1,1) = intrinsics4d(1);
    K.at<double>(0,2) = intrinsics4d(2);
    K.at<double>(1,2) = intrinsics4d(3);

    camera.updateCameraParameters(K, camera.distCoeffs);

    for (auto [c, cEnd, c6, c6End, it] = std::tuple{camPoses.begin(), camPoses.end(), extrinsics6d.begin(), extrinsics6d.end(), 0}; c != cEnd && c6 != c6End; ++c, ++c6, ++it) {
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

    std::cout << "[DONE]\n";
}
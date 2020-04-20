#include "pch.h"
#include "tracking.h"
#include "feature_processing.h"
#include "visualization.h"
#include "camera.h"

#pragma region CLASSES

class UserInput {
private:
    const float m_maxRange;

    cv::Point2f m_medianMove;

    void computePointsRangeMove(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, std::map<std::pair<float, float>, float>& pointsDist, cv::Point2f& move) {
        for (int i = 0; i < prevPts.size(); ++i) {
            float dist = std::pow(
                std::pow(currPts[i].x - prevPts[i].x, 2) + 
                std::pow(currPts[i].y - prevPts[i].y, 2)
            , 0.5);

            if (dist < m_maxRange) {
                pointsDist[std::pair{currPts[i].x, currPts[i].y}] = dist;
                move += currPts[i] - prevPts[i];
            }
        }

        if (!pointsDist.empty()) {
            move.x /= pointsDist.size();
            move.y /= pointsDist.size();
        }
    }
public:
    std::vector<cv::Vec3d> m_usrPts3D;
    std::vector<cv::Point2f> m_usrPts2D;
    
    UserInput(const float maxRange)
        : m_maxRange(maxRange) {}

    void recoverPoints(cv::Mat R, cv::Mat t, Camera camera, cv::Mat& imOutUsr) {
        if (!m_usrPts3D.empty()) {
            std::vector<cv::Point2f> pts2D; cv::projectPoints(m_usrPts3D, R, t, camera._K, cv::Mat(), pts2D);

            for (const auto p : pts2D) {
                std::cout << "Point projected to: " << p << "\n";

                cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
            }
        }
    }

    void addPoints(const std::vector<cv::Vec3d> pts3D) {
        m_usrPts3D.insert(m_usrPts3D.end(), pts3D.begin(), pts3D.end());
    }

    void addPoints(const std::vector<cv::Point2f> prevPts2D, const std::vector<cv::Point2f> currPts2D) {
        std::map<std::pair<float, float>, float> pointsDist;

        cv::Point2f move; computePointsRangeMove(prevPts2D, currPts2D, pointsDist, move);

        for (auto [it, end, idx] = std::tuple{currPts2D.cbegin(), currPts2D.cend(), 0}; it != end; ++it, ++idx) {
            cv::Point2f p = (cv::Point2f)*it;
            
            //if (pointsDist.find(std::pair{p.x, p.y}) != pointsDist.end())
                m_usrPts2D.push_back(p);
            /*else {
                m_usrPts2D.push_back(prevPts[idx] + m_medianMove);
            }*/
        }
    }

    void updatePoints(const std::vector<cv::Point2f> currPts2D, const cv::Rect boundary, const uint offset) {
        std::map<std::pair<float, float>, float> pointsDist;

        cv::Point2f move; computePointsRangeMove(m_usrPts2D, currPts2D, pointsDist, move);

        for (auto [it, end, idx] = std::tuple{currPts2D.cbegin(), currPts2D.cend(), 0}; it != end; ++it, ++idx) {
            cv::Point2f p = (cv::Point2f)*it;
            
            //if (pointsDist.find(std::pair{p.x, p.y}) != pointsDist.end())
                m_usrPts2D[idx] = currPts2D[idx];
            /*else {
                m_usrPts2D[idx] = m_usrPts2D[idx] + m_medianMove;
            }*/
        }

        for (int i = 0, idxCorrection = 0; i < m_usrPts2D.size(); ++i) {
            auto p = m_usrPts2D[i];

            if (p.x < boundary.x + offset || p.y < boundary.y + offset || p.x > boundary.width - offset || p.y > boundary.height - offset) {
                m_usrPts2D.erase(m_usrPts2D.begin() + (i - idxCorrection));

                idxCorrection++;
            } 
        }
    }
    
    void updateMedianMove(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts) {
        std::map<std::pair<float, float>, float> pointsDist;

        computePointsRangeMove(prevPts, currPts, pointsDist, m_medianMove);

        std::cout << m_medianMove << "\n";
    }

    void recoverPoints(cv::Mat& imOutUsr) {
        for (const auto& p : m_usrPts2D) {
            //std::cout << "Point projected to: " << p << "\n";
                
            cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
        }
    }

    void recoverPoints(cv::Mat& imOutUsr, cv::Mat cameraK, cv::Mat R, cv::Mat t) {
        if (!m_usrPts3D.empty()) {
            cv::Mat recoveredPts;

            cv::projectPoints(m_usrPts3D, R, t, cameraK, cv::Mat(), recoveredPts);

            for (int i = 0; i < recoveredPts.rows; ++i) {
                auto p = recoveredPts.at<cv::Point2d>(i);
                //std::cout << "Point projected to: " << p << "\n";
                    
                cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
            }
        } 
    }
};

struct MouseUsrDataParams {
public:
    const std::string m_inputWinName;

    cv::Mat* m_inputMat;

    std::vector<cv::Point2f> m_clickedPoint;

    MouseUsrDataParams(const std::string inputWinName, cv::Mat* inputMat)
        : m_inputWinName(inputWinName) {
        m_inputMat = inputMat;
    }
};

#pragma endregion CLASSES

#pragma region STRUCTS

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 7 parameters: 3 for rotation, 3 for translation, 1 for
// focal length. The principal point is not modeled (assumed be located at the
// image center, and already subtracted from 'observed'), and focal_x = focal_y.
struct SimpleReprojectionError {
    SimpleReprojectionError(double observed_x, double observed_y) :
            observed_x(observed_x), observed_y(observed_y) {
    }
    template<typename T>
    bool operator()(const T* const camera,
    				const T* const point,
					const T* const focal,
						  T* residuals) const {
        T p[3];
        // Rotate: camera[0,1,2] are the angle-axis rotation.
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate: camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective divide
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        // Compute final projected point position.
        const T predicted_x = *focal * xp;
        const T predicted_y = *focal * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
                new SimpleReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3]; 
        p[1] += camera[4]; 
        p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                        const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                    new SnavelyReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};

#pragma endregion STRUCTS

#pragma region METHODS

static void onUsrWinClick (int event, int x, int y, int flags, void* params) {
    if (event != cv::EVENT_LBUTTONDOWN) { return; }

    MouseUsrDataParams* mouseParams = (MouseUsrDataParams*)params;

    const cv::Point clickedPoint(x, y);

    mouseParams->m_clickedPoint.push_back(clickedPoint);

    std::cout << "Clicked to: " << clickedPoint << "\n";
    
    cv::circle(*mouseParams->m_inputMat, clickedPoint, 3, CV_RGB(200, 0, 0), cv::FILLED, cv::LINE_AA);

    cv::imshow(mouseParams->m_inputWinName, *mouseParams->m_inputMat);
}

void pointsToMat(std::vector<cv::Point2f>& points, cv::Mat& pointsMat) {
    pointsMat = (cv::Mat_<double>(2,1) << 1, 1);;

    for (const auto& p : points) {
        cv::Mat _pointMat = (cv::Mat_<double>(2,1) << p.x, p.y);

        cv::hconcat(pointsMat, _pointMat, pointsMat);
    } 

    pointsMat = pointsMat.colRange(1, pointsMat.cols);
}

void pointsToMat(cv::Mat& points, cv::Mat& pointsMat) {
    pointsMat = (cv::Mat_<double>(2,1) << 1, 1);;

    for (size_t i = 0; i < points.cols; ++i) {
        cv::Point2f _point = points.at<cv::Point2f>(i);

        cv::Mat _pointMat = (cv::Mat_<double>(2,1) << _point.x, _point.y);

        cv::hconcat(pointsMat, _pointMat, pointsMat);
    } 

    pointsMat = pointsMat.colRange(1, pointsMat.cols);
}

bool findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers) {
    if (prevPts.size() <= 5 || currPts.size() <= 5) { return false; }

    cv::Mat E = cv::findEssentialMat(prevPts, currPts, cameraK, recPose.recPoseMethod, recPose.prob, recPose.threshold, recPose.mask);

    if (!(E.cols == 3 && E.rows == 3)) { return false; }

    cv::Mat p0 = cv::Mat( { 3, 1 }, {
		double( prevPts[0].x ), double( prevPts[0].y ), 1.0
		} );
	cv::Mat p1 = cv::Mat( { 3, 1 }, {
		double( currPts[0].x ), double( currPts[0].y ), 1.0
		} );
	const double E_error = cv::norm( p1.t() * cameraK.inv().t() * E * cameraK.inv() * p0, cv::NORM_L2 );

    if (E_error > 1e-03) { return false; }

    numInliers = cv::recoverPose(E, prevPts, currPts, cameraK, recPose.R, recPose.t, recPose.mask);

    return numInliers > minInliers;
}

void pointsToRGBCloud(cv::Mat imgColor, Camera camera, cv::Mat R, cv::Mat t, cv::Mat points3D, cv::Mat inputPts2D, std::vector<cv::Vec3d>& cloud3D, std::vector<cv::Vec3b>& cloudRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask) {
    cv::Mat _pts2D; cv::projectPoints(points3D, R, t, camera._K, cv::Mat(), _pts2D);

    cloud3D.clear();
    cloudRGB.clear();
    
    for(size_t i = 0; i < points3D.rows; ++i) {
        const cv::Vec3d point3D = points3D.at<cv::Vec3d>(i);
        const cv::Vec2d point2D = inputPts2D.at<cv::Vec2d>(i);
        const cv::Vec2d _point2D = _pts2D.at<cv::Vec2d>(i);
        const cv::Vec3b imPoint2D = imgColor.at<cv::Vec3b>(cv::Point(point2D));

        const double err = cv::norm(_point2D - point2D);

        cloud3D.push_back( point3D );
        cloudRGB.push_back( imPoint2D );

        mask.push_back( err <  maxProjErr && point3D[2] > minDist && point3D[2] < maxDist );
    }
}

void adjustBundle(std::vector<TrackView>& tracks, Camera& camera, std::vector<cv::Matx34f>& camPoses, std::string solverType, std::string lossType, double lossFunctionScale, uint maxIter = 999) {
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

    for (auto [t, tEnd, c, cEnd, it] = std::tuple{tracks.begin(), tracks.end(), camPoses6d.begin(), camPoses6d.end(), 0}; t != tEnd && c != cEnd && it < maxIter; ++t, ++c, ++it) {
        for (size_t i = 0; i < t->numTracks; ++i) {
            cv::Point2f p2d = t->points2D[i];
            p2d.x -= camera.K33d(0, 2);
            p2d.y -= camera.K33d(1, 2);

            ceres::CostFunction* costFunc = SimpleReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(costFunc, lossType != "NONE" ? lossFunction : NULL, c->val, t->points3D[i].val, &focalLength);
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
    options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    cv::Mat K; camera._K.copyTo(K);

    K.at<double>(0,0) = focalLength;
    K.at<double>(1,1) = focalLength;

    camera.updateCameraParameters(K, camera.distCoeffs);

    std::cout << "Focal length: " << focalLength << "\n";

    if (!summary.IsSolutionUsable()) {
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
	}
    
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

bool loadImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray, float downSample = 1.0f) {
    cap >> imColor; if (imColor.empty()) { return false; }

    if (downSample != 1.0f)
        cv::resize(imColor, imColor, cv::Size(imColor.cols*downSample, imColor.rows*downSample));

    cv::cvtColor(imColor, imGray, cv::COLOR_BGR2GRAY);

    return true;
}

bool findGoodImagePair(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, Camera camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView, float imDownSampling = 1.0f, bool useCUDA = false) {
    std::cout << "Finding good image pair" << std::flush;

    cv::Mat _imColor, _imGray;

    std::vector<cv::Point2f> _prevCorners, _currCorners;

    int numSkippedFrames = -1, numHomInliers = 0;
    do {
        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        //cv::GaussianBlur(_imGray, _imGray, cv::Size(5, 5), 0);
        //cv::medianBlur(_imGray, _imGray, 7);

        std::cout << "." << std::flush;
        
        if (viewContainer.isEmpty()) {
            viewContainer.addItem(ViewData(_imColor, _imGray));

            featDetector.generateFlowFeatures(_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

            if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        }

        _prevCorners = ofPrevView.corners;
        _currCorners = ofCurrView.corners;

        optFlow.computeFlow(viewContainer.getLastOneItem()->imGray, _imGray, _prevCorners, _currCorners, optFlow.statusMask, true, true);

        numSkippedFrames++;
    } while(!findCameraPose(recPose, _prevCorners, _currCorners, camera._K, recPose.minInliers, numHomInliers));

    viewContainer.addItem(ViewData(_imColor, _imGray));

    ofPrevView.setCorners(_prevCorners);
    ofCurrView.setCorners(_currCorners);

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return true;
}

bool findGoodImagePair(cv::VideoCapture cap, ViewDataContainer& viewContainer, float imDownSampling = 1.0f) {
    cv::Mat _imColor, _imGray;

    if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 

    if (viewContainer.isEmpty()) {
        viewContainer.addItem(ViewData(_imColor, _imGray));

        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; }
    }

    viewContainer.addItem(ViewData(_imColor, _imGray));

    return true;
}

void composeExtrinsicMat(cv::Matx33d R, cv::Matx31d t, cv::Matx34d& pose) {
    pose = cv::Matx34d(
        R(0,0), R(0,1), R(0,2), t(0),
        R(1,0), R(1,1), R(1,2), t(1),
        R(2,0), R(2,1), R(2,2), t(2)
    );
}

void drawTrajectory(cv::Mat& imOutTraj, cv::Matx31d t) {
    const int x = int(t(0) + (imOutTraj.cols / 2));
    const int y = int(t(2) + (imOutTraj.rows / 2));
    cv::circle(imOutTraj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

    cv::rectangle(imOutTraj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
    char text[100]; sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t(0), t(1), t(2));
    cv::putText(imOutTraj, text, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255), 1, 8);
}

void applyBoundaryFilter(std::vector<cv::Point2f>& points, const cv::Rect boundary) {
    for (int i = 0, idxCorrection = 0; i < points.size(); ++i) {
        auto point = points[i];

        if (!boundary.contains(point)) { 
            points.erase(points.begin() + (i - idxCorrection)); 

            idxCorrection++;
        }
    }
}

void triangulateCloud(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const cv::Mat colorImage, std::vector<cv::Vec3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, std::vector<bool>& mask, Camera camera, const cv::Matx34d prevPose, const cv::Matx34d currPose, RecoveryPose& recPose, const std::string method, float minDistance, float maxDistance, float maxProjectionError) {
    cv::Mat _prevPtsN; cv::undistort(prevPts, _prevPtsN, camera.K33d, cv::Mat());
    cv::Mat _currPtsN; cv::undistort(currPts, _currPtsN, camera.K33d, cv::Mat());

    cv::Mat _prevPtsMat; pointsToMat(_prevPtsN, _prevPtsMat);
    cv::Mat _currPtsMat; pointsToMat(_currPtsN, _currPtsMat);

    cv::Mat _homogPts, _pts3D;

    if (method == "DLT") {
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

    pointsToRGBCloud(colorImage, camera, cv::Mat(recPose.R), cv::Mat(recPose.t), _pts3D, _currPtsMat.t(), points3D, pointsRGB, minDistance, maxDistance, maxProjectionError, mask);
}

#pragma endregion METHODS

enum Method { KLT_2D = 0, KLT_3D, KLT_3D_PNP };

int main(int argc, char** argv) {
#pragma region INIT
    std::cout << "Using OpenCV " << cv::getVersionString().c_str() << std::flush;

    cv::CommandLineParser parser(argc, argv,
		"{ help h ?  |             | help }"
        "{ bSource   | .           | source video file [.mp4, .avi ...] }"
		"{ bcalib    | .           | camera intric parameters file path }"
        "{ bDownSamp | 1           | downsampling of input source images }"
        "{ bWinWidth | 1024        | debug windows width }"
        "{ bWinHeight| 768         | debug windows height }"
        "{ bUseMethod| KLT_2D      | method to use KLT_2D/KLT_3D/KLT_3D_PNP }"
        "{ bUseCuda  | false       | is nVidia CUDA used }"

        "{ fDecType  | AKAZE       | used detector type }"
        "{ fMatchType| BRUTEFORCE  | used matcher type }"
        "{ fKnnRatio | 0.75        | knn ration match }"

        "{ ofMinKPts | 500         | optical flow min descriptor to generate new one }"
        "{ ofWinSize | 21          | optical flow window size }"
        "{ ofMaxLevel| 3           | optical flow max pyramid level }"
        "{ ofMaxItCt | 30          | optical flow max iteration count }"
        "{ ofItEps   | 0.1         | optical flow iteration epsilon }"
        "{ ofMaxError| 0           | optical flow max error }"
        "{ ofMaxCorn | 500         | optical flow max generated corners }"
        "{ ofQualLvl | 0.1         | optical flow generated corners quality level }"
        "{ ofMinDist | 25.0        | optical flow generated corners min distance }"

        "{ peMethod  | LMEDS       | pose estimation fundamental matrix computation method [RANSAC/LMEDS] }"
        "{ peProb    | 0.999       | pose estimation confidence/probability }"
        "{ peThresh  | 1.0         | pose estimation threshold }"
        "{ peMinInl  | 50          | pose estimation in number of homography inliers user for reconstruction }"
        "{ peMinMatch| 50          | pose estimation min matches to break }"
       
        "{ pePMetrod | 50          | pose estimation method SOLVEPNP_ITERATIVE/SOLVEPNP_P3P/SOLVEPNP_AP3P }"
        "{ peExGuess | false       | pose estimation use extrinsic guess }"
        "{ peNumIteR | 250         | pose estimation max iteration }"

        "{ baMethod  | DENSE_SCHUR | bundle adjustment used solver type DENSE_SCHUR/SPARSE_NORMAL_CHOLESKY }"
        "{ baNumIter | 50          | bundle adjustment max iteration }"
        "{ baLossFunc| NONE        | bundle adjustment used loss function NONE/HUBER/CAUCHY }"
        "{ baLossSc  | 1.0         | bundle adjustment loss function scaling parameter }"

        "{ tMethod   | ITERATIVE   | triangulation method ITERATIVE/DLT }"
        "{ tMinDist  | 1.0         | triangulation points min distance }"
        "{ tMaxDist  | 100.0       | triangulation points max distance }"
        "{ tMaxPErr  | 100.0       | triangulation points max reprojection error }"
    );

    if (parser.has("help")) {
        parser.printMessage();
        exit(0);
    }

    //--------------------------------- BASE --------------------------------//
    const std::string bSource = parser.get<std::string>("bSource");
    const std::string bcalib = parser.get<std::string>("bcalib");
    const float bDownSamp = parser.get<float>("bDownSamp");
    const int bWinWidth = parser.get<int>("bWinWidth");
    const int bWinHeight = parser.get<int>("bWinHeight");
    const std::string bUseMethod = parser.get<std::string>("bUseMethod");
    const bool bUseCuda = parser.get<bool>("bUseCuda");

    //------------------------------- FEATURES ------------------------------//
    const std::string fDecType = parser.get<std::string>("fDecType");
    const std::string fMatchType = parser.get<std::string>("fMatchType");
    const float fKnnRatio = parser.get<float>("fKnnRatio");

    //----------------------------- OPTICAL FLOW ----------------------------//
    const int ofMinKPts = parser.get<int>("ofMinKPts");
    const int ofWinSize = parser.get<int>("ofWinSize");
    const int ofMaxLevel = parser.get<int>("ofMaxLevel");
    const float ofMaxItCt = parser.get<float>("ofMaxItCt");
    const float ofItEps = parser.get<float>("ofItEps");
    const float ofMaxError = parser.get<float>("ofMaxError");
    const int ofMaxCorn = parser.get<int>("ofMaxCorn");
    const float ofQualLvl = parser.get<float>("ofQualLvl");
    const float ofMinDist = parser.get<float>("ofMinDist");

    //--------------------------- POSE ESTIMATION ---------------------------//
    const std::string peMethod = parser.get<std::string>("peMethod");
    const float peProb = parser.get<float>("peProb");
    const float peThresh = parser.get<float>("peThresh");
    const int peMinInl = parser.get<int>("peMinInl");
    const int peMinMatch = parser.get<int>("peMinMatch");
    
    const std::string pePMetrod = parser.get<std::string>("pePMetrod");
    const bool peExGuess = parser.get<bool>("peExGuess");
    const int peNumIteR = parser.get<int>("peNumIteR");

    //-------------------------- BUNDLE ADJUSTMENT --------------------------//
    const std::string baMethod = parser.get<std::string>("baMethod");
    const int baNumIter = parser.get<int>("baNumIter");
    const std::string baLossFunc = parser.get<std::string>("baLossFunc");
    const double baLossSc = parser.get<double>("baLossSc");

    //---------------------------- TRIANGULATION ----------------------------//
    const std::string tMethod = parser.get<std::string>("tMethod");
    const float tMinDist = parser.get<float>("tMinDist");
    const float tMaxDist = parser.get<float>("tMaxDist");
    const float tMaxPErr = parser.get<float>("tMaxPErr");

    bool useCUDA = false;

#pragma ifdef OPENCV_CORE_CUDA_HPP
    if (bUseCuda) {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            std::cout << " with CUDA support\n";

            cv::cuda::setDevice(0);
            cv::cuda::printShortCudaDeviceInfo(0);

            useCUDA = true;
        }
        else
            std::cout << "\nCannot use nVidia CUDA -> no devices" << "\n"; 
    }
#pragma endif

    const cv::FileStorage fs(bcalib, cv::FileStorage::READ);
    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;

    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;
    Camera camera(cameraK, distCoeffs, bDownSamp);
    
    const std::string ptCloudWinName = "Point cloud";
    const std::string usrInpWinName = "User input/output";
    const std::string recPoseWinName = "Recovery pose";
    const std::string matchesWinName = "Matches";
    const std::string trajWinName = "Trajectory";

    cv::VideoCapture cap; if(!cap.open(bSource)) {
        std::cerr << "Error opening video stream or file!!" << "\n";
        exit(1);
    }
    
    Method usedMethod;
    if (bUseMethod == "KLT_2D") 
        usedMethod = Method::KLT_2D;
    else if (bUseMethod == "KLT_3D")
        usedMethod = Method::KLT_3D;
    else
        usedMethod = Method::KLT_3D_PNP;

    FeatureDetector featDetector(fDecType, useCUDA);
    DescriptorMatcher descMatcher(fMatchType, fKnnRatio, useCUDA);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, ofMaxItCt, ofItEps);
    OptFlow optFlow(flowTermCrit, ofWinSize, ofMaxLevel, ofMaxError, ofMaxCorn, ofQualLvl, ofMinDist, ofMinKPts, useCUDA);

    RecoveryPose recPose(peMethod, peProb, peThresh, peMinInl, pePMetrod, peExGuess, peNumIteR);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;
    cv::Mat imOutTraj = cv::Mat::zeros(cv::Size(bWinWidth, bWinHeight), CV_8UC3);

    cv::startWindowThread();

    cv::namedWindow(usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(trajWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(matchesWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(usrInpWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(recPoseWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(trajWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(matchesWinName, cv::Size(bWinWidth, bWinHeight));

    MouseUsrDataParams mouseUsrDataParams(usrInpWinName, &imOutUsrInp);

    cv::setMouseCallback(usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);

    ViewDataContainer viewContainer(usedMethod == Method::KLT_2D ? 100 : INT32_MAX);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 
    
    Tracking tracking;
    UserInput userInput(ofMaxError);

    VisPCL visPCL(ptCloudWinName + " PCL", cv::Size(bWinWidth, bWinHeight));
    //boost::thread visPCLthread(boost::bind(&VisPCL::visualize, &visPCL));
    //std::thread visPCLThread(&VisPCL::visualize, &visPCL);

    VisVTK visVTK(ptCloudWinName + " VTK", cv::Size(bWinWidth, bWinHeight));
    //std::thread visVTKThread(&VisVTK::visualize, &visVTK);
#pragma endregion INIT 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    for ( ; ; ) {
        if (!viewContainer.isEmpty() && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
            featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
        }

        bool isPtAdded = false;
        if (!mouseUsrDataParams.m_clickedPoint.empty()) {
            ofPrevView.corners.insert(ofPrevView.corners.end(), mouseUsrDataParams.m_clickedPoint.begin(), mouseUsrDataParams.m_clickedPoint.end());

            isPtAdded = true;
        }

        if (usedMethod == Method::KLT_2D) {
            if (!userInput.m_usrPts2D.empty()) {
                ofPrevView.corners.insert(ofPrevView.corners.end(), userInput.m_usrPts2D.begin(), userInput.m_usrPts2D.end());
            }

            if (!findGoodImagePair(cap, viewContainer, bDownSamp)) { break; }
        }
            
        if ((usedMethod == Method::KLT_3D || usedMethod == Method::KLT_3D_PNP) && !findGoodImagePair(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView, bDownSamp, useCUDA)) { break; }

        ofPrevView.setView(viewContainer.getLastButOneItem());
        ofCurrView.setView(viewContainer.getLastOneItem());

        ofPrevView.viewPtr->imColor.copyTo(imOutRecPose);
        ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

        if (usedMethod == Method::KLT_2D && !ofPrevView.corners.empty()) {
            optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

            optFlow.drawOpticalFlow(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);

            if (!userInput.m_usrPts2D.empty()) {
                std::vector<cv::Point2f> _newPts;
                _newPts.insert(_newPts.end(), ofCurrView.corners.end() - userInput.m_usrPts2D.size(), ofCurrView.corners.end());

                for (int i = 0; i < userInput.m_usrPts2D.size(); ++i) { ofCurrView.corners.pop_back(); }

                userInput.updatePoints(_newPts, cv::Rect(cv::Point(), ofCurrView.viewPtr->imColor.size()), 10);
            }

            if (!mouseUsrDataParams.m_clickedPoint.empty() && isPtAdded) {
                std::vector<cv::Point2f> _newPts;
                _newPts.insert(_newPts.end(), ofCurrView.corners.end() - mouseUsrDataParams.m_clickedPoint.size(), ofCurrView.corners.end());

                userInput.addPoints(mouseUsrDataParams.m_clickedPoint, _newPts);

                for (int i = 0; i < mouseUsrDataParams.m_clickedPoint.size(); ++i) { ofCurrView.corners.pop_back(); }

                mouseUsrDataParams.m_clickedPoint.clear();
            }

            userInput.recoverPoints(imOutUsrInp);
        }
        if ((usedMethod == Method::KLT_3D || usedMethod == Method::KLT_3D_PNP)) {
            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);
        }

        cv::imshow(recPoseWinName, imOutRecPose);

        if ((usedMethod == Method::KLT_3D || usedMethod == Method::KLT_3D_PNP)) {
            std::vector<cv::Vec3d> _points3D;
            std::vector<cv::Vec3b> _pointsRGB;
            std::vector<bool> _mask;

            cv::Matx34d _prevPose, _currPose; 

            if (usedMethod == Method::KLT_3D) {
                composeExtrinsicMat(tracking.R, tracking.t, _prevPose);

                tracking.t = tracking.t + (tracking.R * recPose.t);
                tracking.R = tracking.R * recPose.R;

                composeExtrinsicMat(tracking.R, tracking.t, _currPose);
                tracking.addCamPose(_currPose);

                triangulateCloud(ofPrevView.corners, ofCurrView.corners, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, camera, _prevPose, _currPose, recPose, tMethod, tMinDist, tMaxDist, tMaxPErr);

                if (!mouseUsrDataParams.m_clickedPoint.empty() && isPtAdded) {
                    std::vector<cv::Vec3d> _newPts;
                    _newPts.insert(_newPts.end(), _points3D.end() - mouseUsrDataParams.m_clickedPoint.size(), _points3D.end());

                    userInput.addPoints(_newPts);

                    for (int i = 0; i < mouseUsrDataParams.m_clickedPoint.size(); ++i) { ofCurrView.corners.pop_back(); }

                    mouseUsrDataParams.m_clickedPoint.clear();
                    
                    visVTK.addPoints(_newPts);
                    visPCL.addPoints(_newPts);
                }

                userInput.recoverPoints(imOutUsrInp, camera._K, cv::Mat(tracking.R), cv::Mat(tracking.t));

                visVTK.addPointCloud(_points3D, _pointsRGB);

                visPCL.addPointCloud(_points3D, _pointsRGB);
                
            }

            if (usedMethod == Method::KLT_3D_PNP) {
                featPrevView.setView(viewContainer.getLastButOneItem());
                featCurrView.setView(viewContainer.getLastOneItem());

                if (featPrevView.keyPts.empty()) {
                    featDetector.generateFeatures(featPrevView.viewPtr->imGray, featPrevView.keyPts, featPrevView.descriptor);
                }

                featDetector.generateFeatures(featCurrView.viewPtr->imGray, featCurrView.keyPts, featCurrView.descriptor);

                if (featPrevView.keyPts.empty() || featCurrView.keyPts.empty()) { 
                    std::cerr << "None keypoints to match, skip matching/triangulation!\n";

                    continue; 
                }

                std::vector<cv::Point2f> _prevPts, _currPts;
                std::vector<cv::DMatch> _matches;
                std::vector<int> _prevIdx, _currIdx;

                descMatcher.recipAligMatches(featPrevView.keyPts, featCurrView.keyPts, featPrevView.descriptor, featCurrView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

                if (_prevPts.empty() || _currPts.empty()) { 
                    std::cerr << "None points to triangulate, skip triangulation!\n";

                    continue; 
                }

                if(!tracking.findRecoveredCameraPose(descMatcher, peMinMatch, camera, featCurrView, recPose)) {
                    std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";
        
                    std::swap(ofPrevView, ofCurrView);
                    std::swap(featPrevView, featCurrView);

                    continue;
                }
            
                if (tracking.isCamPosesEmpty())
                    composeExtrinsicMat(cv::Matx33d::eye(), cv::Matx31d::eye(), _prevPose);
                else
                    _prevPose = tracking.getLastCamPose();

                composeExtrinsicMat(recPose.R, recPose.t, _currPose);
                tracking.addCamPose(_currPose);

                triangulateCloud(_prevPts, _currPts, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, camera, _prevPose, _currPose, recPose, tMethod, tMinDist, tMaxDist, tMaxPErr);

                tracking.addTrackView(featCurrView.viewPtr, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, _currIdx);

                adjustBundle(tracking.trackViews, camera, *tracking.getCamPoses(), baMethod, baLossFunc, baLossSc, baNumIter);

                visVTK.addPointCloud(tracking.trackViews);

                visPCL.addPointCloud(tracking.trackViews);
            }

            visVTK.updateCameras(*tracking.getCamPoses(), camera._K);
            visVTK.visualize();

            visPCL.updateCameras(*tracking.getCamPoses());
            visPCL.visualize();
        }

        drawTrajectory(imOutTraj, tracking.t);

        cv::imshow(trajWinName, imOutTraj);
        cv::imshow(usrInpWinName, imOutUsrInp);

        std::cout << "Iteration count: " << tracking.trackViews.size() << "\n"; cv::waitKey(29);

        std::swap(ofPrevView, ofCurrView);
        std::swap(featPrevView, featCurrView);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\n----------------------------------------------------------\n\n";
    std::cout << "Total computing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds!\n";

    cv::waitKey(); exit(0);
}
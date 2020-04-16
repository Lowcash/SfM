#include "pch.h"
#include "tracking.h"
#include "feature_processing.h"
#include "visualization.h"
#include "camera.h"

#pragma region CLASSES

class UserInput {
private:
public:
    std::vector<cv::Vec3d> m_usrPts;
    std::vector<cv::Vec3f> m_projUsrPts;
    std::vector<cv::Point> m_pts2D;

    void attach2DPoints(std::vector<cv::Point> userPrevPts, std::vector<cv::Point> userCurrPts, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
        prevPts.insert(prevPts.end(), userPrevPts.begin(), userPrevPts.end());
        currPts.insert(currPts.end(), userCurrPts.begin(), userCurrPts.end());
    }

    void detach2DPoints(std::vector<cv::Point> userPts2D, std::vector<cv::Point2f>& points2D) {
        points2D.erase(points2D.end() - userPts2D.size(), points2D.end());
    }

    void detach3DPoints(std::vector<cv::Point> userPts2D, std::vector<cv::Vec3d>& points3D) {
        points3D.erase(points3D.end() - userPts2D.size(), points3D.end());
    }

    void insert3DPoints(std::vector<cv::Vec3d> points3D) {
        m_usrPts.insert(m_usrPts.end(), points3D.begin(), points3D.end());
        m_projUsrPts.insert(m_projUsrPts.end(), points3D.begin(), points3D.end());
    }

    void recoverPoints(cv::Mat R, cv::Mat t, Camera camera, cv::Mat& imOutUsr) {
        if (!m_usrPts.empty()) {
            std::vector<cv::Point2f> pts2D; cv::projectPoints(m_projUsrPts, R, t, camera._K, cv::Mat(), pts2D);

            for (const auto p : pts2D) {
                std::cout << "Point projected to: " << p << "\n";

                cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
            }
        }
    }

    void recoverPoints(cv::Mat& imOutUsr, std::vector<cv::Point> pts2D) {
        for (const auto p : pts2D) {
            std::cout << "Point projected to: " << p << "\n";

            cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
        }
    }

    void movePoints(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const std::vector<cv::Point> inputPts, std::vector<cv::Point>& outputPts) {
        cv::Point2f move; int ptsSize = MIN(prevPts.size(), currPts.size());

        for (int i = 0; i < ptsSize; ++i) 
            move += prevPts[i] - currPts[i];
        
        move.x /= ptsSize;
        move.y /= ptsSize;

        for (int i = 0; i < inputPts.size(); ++i) {
            outputPts.push_back(
                cv::Point2f(
                    inputPts[i].x + move.x, 
                    inputPts[i].y + move.y
                )
            );
        }
    }

    void movePoints(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const cv::Mat mask, const std::vector<cv::Point> inputPts, std::vector<cv::Point>& outputPts) {
        cv::Point2f move; int ptsSize = MIN(prevPts.size(), currPts.size());

        int usedPts = 0;

        for (int i = 0; i < ptsSize; ++i)  {
            if (mask.at<uchar>(i) == 1) {
                move += prevPts[i] - currPts[i];

                usedPts++;
            }
        }
            
        move.x /= usedPts;
        move.y /= usedPts;

        for (int i = 0; i < inputPts.size(); ++i) {
            outputPts.push_back(
                cv::Point2f(
                    inputPts[i].x - move.x, 
                    inputPts[i].y - move.y
                )
            );
        }
    }
};

struct MouseUsrDataParams {
public:
    const std::string m_inputWinName;

    cv::Mat* m_inputMat;

    std::vector<cv::Point> m_clickedPoint;

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

    //cv::Mat _prevPtsMat; pointsToMat(prevPts, _prevPtsMat);
    //cv::Mat _currPtsMat; pointsToMat(currPts, _currPtsMat);

    //cv::Mat F, E; std::vector<int> inl;
    //double bestError = cv::sfm::fundamentalFromCorrespondences8PointRobust(_prevPtsMat, _currPtsMat, recPose.threshold, F, inl);
    //cv::sfm::essentialFromFundamental(F, cameraK, cameraK, E);
    cv::Mat E = cv::findEssentialMat(prevPts, currPts, cameraK, recPose.recPoseMethod, recPose.prob, recPose.threshold, recPose.mask);

    //std::vector<cv::Mat> R, t;
    //cv::sfm::motionFromEssential(E, R, t);

    if (!(E.cols == 3 && E.rows == 3)) { return false; }

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

    for (auto [c, cEnd, it] = std::tuple{camPoses.crbegin(), camPoses.crend(), 0}; c != cEnd && it < maxIter; ++c, ++it) {
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

    for (auto [t, tEnd, c, cEnd, it] = std::tuple{tracks.rbegin(), tracks.rend(), camPoses6d.begin(), camPoses6d.end(), 0}; t != tEnd && c != cEnd && it < maxIter; ++t, ++c, ++it) {
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
    //options.eta = 1e-2;
    options.num_threads = std::thread::hardware_concurrency();

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    //std::cout << summary.FullReport() << "\n";

    cv::Mat K; camera._K.copyTo(K);

    K.at<double>(0,0) = focalLength;
    K.at<double>(1,1) = focalLength;

    camera.updateCameraParameters(K, camera.distCoeffs);

    /*std::cout << "Focal length: " << focalLength << "\n";

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
	}*/
    
    for (auto [c, cEnd, c6, c6End, it] = std::tuple{camPoses.rbegin(), camPoses.rend(), camPoses6d.begin(), camPoses6d.end(), 0}; c != cEnd && c6 != c6End && it < maxIter; ++c, ++c6, ++it) {
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

bool findGoodImagePair(cv::VideoCapture cap, OptFlow optFlow, FeatureDetector featDetector, RecoveryPose& recPose, Camera camera, ViewDataContainer& viewContainer, FlowView& ofPrevView, FlowView& ofCurrView, uint flowMinFeatures, float imDownSampling = 1.0f, bool isUsingCUDA = false) {
    std::cout << "Finding good image pair" << std::flush;

    cv::Mat _imColor, _imGray;
    cv::cuda::GpuMat _d_imColor, _d_imGray;

    std::vector<cv::Point2f> _prevCorners, _currCorners;

    int numSkippedFrames = -1, numHomInliers = 0;
    do {
        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        //cv::GaussianBlur(_imGray, _imGray, cv::Size(5, 5), 0);
        //cv::medianBlur(_imGray, _imGray, 7);

        std::cout << "." << std::flush;
        
        if (isUsingCUDA) {
            _d_imColor.upload(_imColor);
            _d_imGray.upload(_imGray);
        }

        // if (ofPrevView.corners.size() < flowMinFeatures) {
        //     if (isUsingCUDA) {
        //         featDetector.generateFlowFeatures(_d_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
        //     } else 
        //         featDetector.generateFlowFeatures(_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

        //     if (isUsingCUDA) 
        //         viewContainer.addItem(ViewData(_imColor, _imGray, _d_imColor, _d_imGray));
        //     else 
        //         viewContainer.addItem(ViewData(_imColor, _imGray));

        //     continue; 
        // }

        ofPrevView.setView(viewContainer.getLastOneItem());

        _prevCorners = ofPrevView.corners;
        _currCorners = ofCurrView.corners;

        if (isUsingCUDA) {
            //optFlow.computeFlow(ofPrevView.viewPtr->d_imGray, _d_imGray, _prevCorners, _currCorners);
        } else
            //optFlow.computeFlow(ofPrevView.viewPtr->imGray, _imGray, _prevCorners, _currCorners);

        numSkippedFrames++;
    } while(!findCameraPose(recPose, _prevCorners, _currCorners, camera._K, recPose.minInliers, numHomInliers));

    if (isUsingCUDA) 
        viewContainer.addItem(ViewData(_imColor, _imGray, _d_imColor, _d_imGray));
    else 
        viewContainer.addItem(ViewData(_imColor, _imGray));

    ofCurrView.setView(viewContainer.getLastOneItem());

    ofPrevView.setCorners(_prevCorners);
    ofCurrView.setCorners(_currCorners);

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return true;
}

bool findGoodImagePair(cv::VideoCapture cap, FeatureDetector featDetector, DescriptorMatcher matcher, RecoveryPose& recPose, Camera camera, ViewDataContainer& viewContainer, FeatureView& prevView, FeatureView& currView, std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<int>& prevIdx, std::vector<int>& currIdx, float imDownSampling = 1.0f, bool isUsingCUDA = false) {
    std::vector<ViewData> _views;
    std::vector<cv::Mat> _imGrays;

    std::map<std::pair<int, int>, std::vector<cv::DMatch>> _matchingMatrix;
    std::map<std::pair<int, int>, std::vector<cv::Point2f>> _prevPts;
    std::map<std::pair<int, int>, std::vector<cv::Point2f>> _currPts;
    std::map<std::pair<int, int>, std::vector<int>> _prevIdx;
    std::map<std::pair<int, int>, std::vector<int>> _currIdx;
    std::map<int, std::pair<int, int>> _inliers;
    std::map<std::pair<int, int>, cv::Matx33d> _R;
    std::map<std::pair<int, int>, cv::Matx31d> _t;
    std::map<std::pair<int, int>, cv::Mat> _masks;

    for (int i = 0; i < 6; ++i) {
        cv::Mat _imColor, _imGray;

        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        //cv::GaussianBlur(_imGray, _imGray, cv::Size(5, 5), 0);
        //cv::medianBlur(_imGray, _imGray, 7);

        _views.push_back(ViewData(_imColor, _imGray));
        _imGrays.push_back(_imGray);

        for (int j = i; j < 6; ++j) 
            _matchingMatrix[std::pair{i, j}] = std::vector<cv::DMatch>();
    }

    std::vector<std::vector<cv::KeyPoint>> _vecKeyPts;
    std::vector<cv::Mat> _vecDesc;

    featDetector.detector->detect(_imGrays, _vecKeyPts);
    featDetector.extractor->compute(_imGrays, _vecKeyPts, _vecDesc);

    for (auto& m : _matchingMatrix) {
        matcher.recipAligMatches(_vecKeyPts[m.first.first], _vecKeyPts[m.first.second], _vecDesc[m.first.first], _vecDesc[m.first.second], _prevPts[std::pair{m.first.first, m.first.second}], _currPts[std::pair{m.first.first, m.first.second}], m.second, _prevIdx[std::pair{m.first.first, m.first.second}], _currIdx[std::pair{m.first.first, m.first.second}]);

        int numInliers; findCameraPose(recPose, _prevPts[std::pair{m.first.first, m.first.second}], _currPts[std::pair{m.first.first, m.first.second}], camera._K, recPose.minInliers, numInliers);

        _R[std::pair{m.first.first, m.first.second}] = recPose.R;
        _t[std::pair{m.first.first, m.first.second}] = recPose.t;
        _masks[std::pair{m.first.first, m.first.second}] = recPose.mask;

        _inliers[numInliers] = std::pair{m.first.first, m.first.second};
    }

    std::pair<int, int> _bestInlierPair = _inliers.rbegin()->second;

    viewContainer.addItem(_views[_bestInlierPair.first]);
    viewContainer.addItem(_views[_bestInlierPair.second]);

    recPose.R = _R[_bestInlierPair];
    recPose.t = _t[_bestInlierPair];
    recPose.mask = _masks[_bestInlierPair];

    matches = _matchingMatrix[_bestInlierPair];
    prevPts = _prevPts[_bestInlierPair];
    currPts = _currPts[_bestInlierPair];
    prevIdx = _prevIdx[_bestInlierPair];
    currIdx = _currIdx[_bestInlierPair];

    prevView.setFeatures(_vecKeyPts[_bestInlierPair.first], _vecDesc[_bestInlierPair.first]);
    currView.setFeatures(_vecKeyPts[_bestInlierPair.second], _vecDesc[_bestInlierPair.second]);

    prevView.setView(viewContainer.getLastButOneItem());
    currView.setView(viewContainer.getLastOneItem());

    return true;
}

void composePoseEstimation(cv::Matx33d R, cv::Matx31d t, cv::Matx34d& pose) {
    pose = cv::Matx34d(
        R(0,0), R(0,1), R(0,2), t(0),
        R(1,0), R(1,1), R(1,2), t(1),
        R(2,0), R(2,1), R(2,2), t(2)
    );
}

#pragma endregion METHODS

enum Method { KLT_2D, KLT_3D };

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
        "{ bUseMethod| KLT         | method to use }"
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

    bool isUsingCUDA = false;

    if (bUseCuda) {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            std::cout << " with CUDA support\n";

            cv::cuda::setDevice(0);
            cv::cuda::printShortCudaDeviceInfo(0);

            isUsingCUDA = true;
        }
        else
            std::cout << "\nCannot use nVidia CUDA -> no devices" << "\n"; 
    }
     
    const cv::FileStorage fs(bcalib, cv::FileStorage::READ);
    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;

    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;
    Camera camera(cameraK, distCoeffs, bDownSamp);
    
    const std::string ptCloudWinName = "Point cloud";
    const std::string usrInpWinName = "User input";
    const std::string recPoseWinName = "Recovery pose";
    const std::string matchesWinName = "Matches";

    cv::VideoCapture cap; if(!cap.open(bSource)) {
        std::cerr << "Error opening video stream or file!!" << "\n";
        exit(1);
    }
    
    Method usedMethod;
    if (bUseMethod == "KLT_2D") 
        usedMethod = Method::KLT_2D;

    FeatureDetector featDetector(fDecType, isUsingCUDA);
    DescriptorMatcher descMatcher(fMatchType, fKnnRatio, isUsingCUDA);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, ofMaxItCt, ofItEps);
    OptFlow optFlow(flowTermCrit, ofWinSize, ofMaxLevel, ofMaxError, ofMaxCorn, ofQualLvl, ofMinDist, isUsingCUDA);

    RecoveryPose recPose(peMethod, peProb, peThresh, peMinInl, pePMetrod, peExGuess, peNumIteR);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;
    
    cv::startWindowThread();

    cv::namedWindow(usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(matchesWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(usrInpWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(recPoseWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(matchesWinName, cv::Size(bWinWidth, bWinHeight));

    MouseUsrDataParams mouseUsrDataParams(usrInpWinName, &imOutUsrInp);

    cv::setMouseCallback(usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 

    std::vector<FeatureView> featureViews;
    ViewDataContainer viewContainer;

    std::vector<cv::Matx34f> camPoses;
    std::vector<cv::Point3f> usrPts;

    VisPCL visPCL(ptCloudWinName + " PCL", cv::Size(bWinWidth, bWinHeight));
    //boost::thread visPCLthread(boost::bind(&VisPCL::visualize, &visPCL));
    //std::thread visPCLThread(&VisPCL::visualize, &visPCL);
    //visPCL.visualize();

    VisVTK visVTK(ptCloudWinName + " VTK", cv::Size(bWinWidth, bWinHeight));
    //std::thread visVTKThread(&VisVTK::visualize, &visVTK);

    Tracking tracking;
    UserInput userInput;

#pragma endregion INIT 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cv::Matx33d R = cv::Matx33d::eye();
    cv::Matx31d t = cv::Matx31d::eye();

    for ( ; ; ) {
        std::vector<cv::Point2f> _prevPts, _currPts;
        std::vector<cv::DMatch> _matches;
        std::vector<int> _prevIdx, _currIdx;

        //cv::Matx34d _prevPose; composePoseEstimation(recPose.R, recPose.t, _prevPose);

        /*if (usedMethod == Method::KLT_3D) {
            if (!findGoodImagePair(cap, optFlow, featDetector, recPose, camera, viewContainer, ofPrevView, ofCurrView, ofMinKPts, bDownSamp, isUsingCUDA)) { break; }           

            ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);
            ofCurrView.viewPtr->imColor.copyTo(imOutRecPose);

            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);
        } else {
            if (!findGoodImagePair(cap, featDetector, descMatcher, recPose, camera, viewContainer, featPrevView, featCurrView, _matches, _prevPts, _currPts, _prevIdx, _currIdx, bDownSamp, isUsingCUDA)) { break; }

            featCurrView.viewPtr->imColor.copyTo(imOutUsrInp);
            featCurrView.viewPtr->imColor.copyTo(imOutRecPose);

            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, featPrevView.pts, featCurrView.pts, recPose.mask);
        }*/

        if (usedMethod == Method::KLT_2D) {
            cv::Mat _imColor, _imGray;

            if (!loadImage(cap, _imColor, _imGray, bDownSamp)) { break; } 

            if (viewContainer.isEmpty()) {
                viewContainer.addItem(ViewData(_imColor, _imGray));

                continue;
            }

            ofPrevView.setView(viewContainer.getLastOneItem());

            viewContainer.addItem(ViewData(_imColor, _imGray));

            ofCurrView.setView(viewContainer.getLastOneItem());

            if (!userInput.m_pts2D.empty()) {
                if (ofPrevView.corners.size() < ofMinKPts) {
                    featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
                }

                ofPrevView.corners.insert(ofPrevView.corners.end(), userInput.m_pts2D.begin(), userInput.m_pts2D.end());

                optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, recPose.mask);

                std::vector<cv::Point> _usrPts2D;
                _usrPts2D.insert(_usrPts2D.end(), ofCurrView.corners.end() - userInput.m_pts2D.size(), ofCurrView.corners.end());

                userInput.m_pts2D = _usrPts2D;
            }
        }

        ofPrevView.viewPtr->imColor.copyTo(imOutRecPose);
        ofPrevView.viewPtr->imColor.copyTo(imOutUsrInp);

        if (!userInput.m_pts2D.empty()) {
            userInput.recoverPoints(imOutUsrInp, userInput.m_pts2D);
        }
        
        recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);
        
        cv::imshow(recPoseWinName, imOutRecPose); 
        cv::imshow(usrInpWinName, imOutUsrInp);

        cv::waitKey();

        if (!mouseUsrDataParams.m_clickedPoint.empty()) {
            userInput.m_pts2D.insert(userInput.m_pts2D.end(), mouseUsrDataParams.m_clickedPoint.begin(), mouseUsrDataParams.m_clickedPoint.end());
            mouseUsrDataParams.m_clickedPoint.clear();
        }

       /* t = t + (R * recPose.t);
        R = recPose.R * R;

        cv::Matx34d camPose; composePoseEstimation(R, t, camPose);
        camPoses.push_back(camPose);
        
        visPCL.updateCameras(camPoses);
        visPCL.visualize();

        visVTK.updateCameras(camPoses, camera.K33d);
        visVTK.visualize();

        ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

        std::vector<cv::Point> _movePts;
        userInput.movePoints(ofPrevView.corners, ofCurrView.corners, recPose.mask, mouseUsrDataParams.m_clickedPoint, _movePts);
        userInput.recoverPoints(imOutUsrInp, _movePts);

        cv::imshow(usrInpWinName, imOutUsrInp);

        cv::waitKey();*/
        /*if (bUseOptFl) {
            if (featureViews.empty()) {
                if (isUsingCUDA) 
                    featDetector.generateFeatures(ofPrevView.viewPtr->imGray, ofPrevView.viewPtr->d_imGray, featPrevView.keyPts, featPrevView.descriptor);             
                 else
                    featDetector.generateFeatures(ofPrevView.viewPtr->imGray, featPrevView.keyPts, featPrevView.descriptor);

                featureViews.push_back(featPrevView);
            }

            if (isUsingCUDA) 
                featDetector.generateFeatures(ofCurrView.viewPtr->imGray, ofCurrView.viewPtr->d_imGray, featCurrView.keyPts, featCurrView.descriptor); 
             else
                featDetector.generateFeatures(ofCurrView.viewPtr->imGray, featCurrView.keyPts, featCurrView.descriptor);
                
            featureViews.push_back(featCurrView);

            featPrevView.setView(viewContainer.getLastButOneItem());
            featCurrView.setView(viewContainer.getLastOneItem());
        }

        if (featPrevView.keyPts.empty() || featCurrView.keyPts.empty()) { 
            std::cerr << "None keypoints to match, skip matching/triangulation!\n";

            continue; 
        }

        if (bUseOptFl) {
            descMatcher.recipAligMatches(featPrevView.keyPts, featCurrView.keyPts, featPrevView.descriptor, featCurrView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);
        }

        cv::drawMatches(featPrevView.viewPtr->imColor, featPrevView.keyPts, featCurrView.viewPtr->imColor, featCurrView.keyPts, _matches, imOutMatches);

        cv::imshow(matchesWinName, imOutMatches);

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

        userInput.recoverPoints(cv::Mat(recPose.R), cv::Mat(recPose.t), camera, imOutUsrInp);

        cv::imshow(usrInpWinName, imOutUsrInp);

        std::vector<cv::Point> movedPts;
        userInput.movePoints(_prevPts, _currPts, mouseUsrDataParams.m_clickedPoint, movedPts);

        userInput.attach2DPoints(mouseUsrDataParams.m_clickedPoint, movedPts, _prevPts, _currPts);

        cv::Matx34d _currPose; composePoseEstimation(recPose.R, recPose.t, _currPose);
        camPoses.push_back(_currPose);

        cv::Mat _prevPtsN; cv::undistort(_prevPts, _prevPtsN, camera.K33d, cv::Mat());
        cv::Mat _currPtsN; cv::undistort(_currPts, _currPtsN, camera.K33d, cv::Mat());

        cv::Mat _prevPtsMat; pointsToMat(_prevPts, _prevPtsMat);
        cv::Mat _currPtsMat; pointsToMat(_currPts, _currPtsMat);

        cv::Mat _homogPts, _pts3D;

        if (tMethod == "DLT") {
            std::vector<cv::Mat> _pts, _projMats;
            _pts.push_back(_prevPtsMat);
            _pts.push_back(_currPtsMat);
            _projMats.push_back(cv::Mat(camera.K33d * _prevPose));
            _projMats.push_back(cv::Mat(camera.K33d * _currPose));

            cv::sfm::triangulatePoints(_pts, _projMats, _pts3D); _pts3D = _pts3D.t();
        } else {
            cv::triangulatePoints(camera.K33d * _prevPose, camera.K33d * _currPose, _prevPtsMat, _currPtsMat, _homogPts);
            cv::convertPointsFromHomogeneous(_homogPts.t(), _pts3D);
        }

        std::vector<cv::Vec3d> _points3D;
        std::vector<cv::Vec3b> _pointsRGB;
        std::vector<bool> _mask; pointsToRGBCloud(featCurrView.viewPtr->imColor, camera, cv::Mat(recPose.R), cv::Mat(recPose.t), _pts3D, _currPtsMat.t(), _points3D, _pointsRGB, tMinDist, tMaxDist, tMaxPErr, _mask);

        std::vector<cv::Vec3d> _usrPts;
        _usrPts.insert(_usrPts.end(), _points3D.end() - mouseUsrDataParams.m_clickedPoint.size(), _points3D.end());
        userInput.insert3DPoints(_usrPts);

        userInput.detach2DPoints(mouseUsrDataParams.m_clickedPoint, _currPts);
        userInput.detach3DPoints(mouseUsrDataParams.m_clickedPoint, _points3D);

        visPCL.addPoints(_usrPts);
        //visVTK.addPoints(_usrPts);
        mouseUsrDataParams.m_clickedPoint.clear();

        tracking.addTrackView(featCurrView.viewPtr, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, _currIdx);

        adjustBundle(tracking.trackViews, camera, camPoses, baMethod, baLossFunc, baLossSc, baNumIter);
        
        std::cout << "\n Cam pose: " << _currPose << "\n"; cv::waitKey(1);

        visPCL.addPointCloud(tracking.trackViews);
        visPCL.updateCameras(camPoses);
        visPCL.visualize();
        
        visVTK.addPointCloud(tracking.trackViews);
        //visVTK.updateCameras(camPoses, camera.K33d);
        visVTK.visualize();*/

        std::swap(ofPrevView, ofCurrView);
        std::swap(featPrevView, featCurrView);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\n----------------------------------------------------------\n\n";
    std::cout << "Total computing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds!\n";

    cv::waitKey(); exit(0);
}
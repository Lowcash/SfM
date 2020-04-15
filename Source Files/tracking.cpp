#include "tracking.h"

RecoveryPose::RecoveryPose(std::string recPoseMethod, const double prob, const double threshold, const uint minInliers, std::string poseEstMethod, const bool useExtrinsicGuess, const uint numIter) 
    : prob(prob), threshold(threshold), minInliers(minInliers), useExtrinsicGuess(useExtrinsicGuess), numIter(numIter) {
    R = cv::Matx33d::eye();
    t = cv::Matx31d::eye();

    std::for_each(recPoseMethod.begin(), recPoseMethod.end(), [](char& c){
        c = ::toupper(c);
    });

    if (recPoseMethod == "RANSAC")
        this->recPoseMethod = cv::RANSAC;
    else
        this->recPoseMethod = cv::LMEDS;

    std::for_each(poseEstMethod.begin(), poseEstMethod.end(), [](char& c){
        c = ::toupper(c);
    });

    if (poseEstMethod == "SOLVEPNP_P3P")
        this->poseEstMethod = cv::SolvePnPMethod::SOLVEPNP_P3P;
    else if (poseEstMethod == "SOLVEPNP_AP3P")
        this->poseEstMethod = cv::SolvePnPMethod::SOLVEPNP_AP3P;
    else
        this->poseEstMethod = cv::SolvePnPMethod::SOLVEPNP_ITERATIVE;
}

void RecoveryPose::drawRecoveredPose(cv::Mat inputImg, cv::Mat& outputImg, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, cv::Mat mask) {
    bool isUsingMask = mask.rows == prevPts.size();

    inputImg.copyTo(outputImg);

    for (int i = 0; i < prevPts.size() && i < currPts.size(); ++i) {
        cv::arrowedLine(outputImg, currPts[i], prevPts[i], CV_RGB(0,200,0), 2);

        if (isUsingMask && mask.at<uchar>(i) == 0)
            cv::arrowedLine(outputImg, currPts[i], prevPts[i], CV_RGB(200,0,0), 2);
    }
}

Tracking::Tracking() {
    initKalmanFilter(KF, 18, 6, 0, 0.125);
}

void Tracking::initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt) {
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
                    /* DYNAMIC MODEL */
    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    KF.transitionMatrix.at<double>(3,6) = dt;
    KF.transitionMatrix.at<double>(4,7) = dt;
    KF.transitionMatrix.at<double>(5,8) = dt;
    KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
    // orientation
    KF.transitionMatrix.at<double>(9,12) = dt;
    KF.transitionMatrix.at<double>(10,13) = dt;
    KF.transitionMatrix.at<double>(11,14) = dt;
    KF.transitionMatrix.at<double>(12,15) = dt;
    KF.transitionMatrix.at<double>(13,16) = dt;
    KF.transitionMatrix.at<double>(14,17) = dt;
    KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
        /* MEASUREMENT MODEL */
    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,9) = 1;  // roll
    KF.measurementMatrix.at<double>(4,10) = 1; // pitch
    KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}

void Tracking::updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated ) {
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();

    std::cout << prediction << "\n";

    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_32F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
}

void Tracking::fillMeasurements( cv::Mat &measurements, const cv::Mat translation_measured, const cv::Mat rotation_measured) {
    // Set measurement to predict
    measurements.at<float>(0) = translation_measured.at<double>(0); // x
    std::cout << measurements.at<double>(0) << "\n";
    std::cout << translation_measured.at<float>(0,0) << "\n";
    measurements.at<float>(1) = translation_measured.at<float>(1); // y
    std::cout << measurements.at<float>(1) << "\n";
    std::cout << translation_measured.at<float>(1) << "\n";
    measurements.at<float>(2) = translation_measured.at<float>(2); // z
    std::cout << measurements.at<float>(2) << "\n";
    std::cout << translation_measured.at<float>(2) << "\n";
    measurements.at<float>(3) = rotation_measured.at<float>(0);      // roll
    std::cout << measurements.at<float>(3) << "\n";
    std::cout << measurements.at<float>(0) << "\n";
    measurements.at<float>(4) = rotation_measured.at<float>(1);      // pitch
    std::cout << measurements.at<float>(4) << "\n";
    std::cout << measurements.at<float>(1) << "\n";
    measurements.at<float>(5) = rotation_measured.at<float>(2);      // yaw
    std::cout << measurements.at<float>(5) << "\n";
    std::cout << measurements.at<float>(2) << "\n";
}

void Tracking::addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer) {
    TrackView _trackView;

    for (uint idx = 0; idx < points3D.size(); ++idx) {
        const cv::KeyPoint _keypoint = featureIndexer.empty() ? keyPoints[idx] : keyPoints[featureIndexer[idx]];
        const cv::Mat _descriptor = featureIndexer.empty() ? descriptor.row(idx) : descriptor.row(featureIndexer[idx]);

        if (mask[idx])
            _trackView.addTrack(points2D[idx], points3D[idx], pointsRGB[idx], _keypoint, _descriptor);
    }

    _trackView.setView(view);

    trackViews.push_back(_trackView);
}

 bool Tracking::findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, RecoveryPose& recPose) {
    if (trackViews.empty()) { return true; }
        
    std::cout << "Recovering pose..." << std::flush;
    
    std::vector<cv::Point2f> _posePoints2D;
    std::vector<cv::Point3f> _posePoints3D;
    cv::Mat _R, _t, _inliers;

    for (auto t = trackViews.rbegin(); t != trackViews.rend(); ++t) {
        if (t->points2D.empty() || featView.keyPts.empty()) { continue; }

        std::vector<cv::Point2f> _prevPts, _currPts;
        std::vector<cv::DMatch> _matches;
        std::vector<int> _prevIdx, _currIdx;
        matcher.recipAligMatches(t->keyPoints, featView.keyPts, t->descriptor, featView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

        std::cout << "Recover pose matches: " << _matches.size() << "\n";

        if (_matches.size() < minMatches)
            break;

        cv::Mat _imOutMatch; cv::drawMatches(t->viewPtr->imColor, t->keyPoints, featView.viewPtr->imColor, featView.keyPts, _matches, _imOutMatch);

        for (const auto& m : _matches) {
            cv::Point3f _point3D = (cv::Point3f)t->points3D[m.queryIdx];
            cv::Point2f _point2D = (cv::Point2f)featView.keyPts[m.trainIdx].pt;

            _posePoints2D.push_back(_point2D);
            _posePoints3D.push_back(_point3D);
        }

        cv::imshow("Matches", _imOutMatch); cv::waitKey(1);
    }

    if (_posePoints2D.size() < 4 || _posePoints3D.size() < 4) { return false; }

    if (!cv::solvePnPRansac(_posePoints3D, _posePoints2D, camera._K, cv::Mat(), _R, _t, recPose.useExtrinsicGuess, recPose.numIter, recPose.threshold, recPose.prob, _inliers, recPose.poseEstMethod)) { return false; }
    //if (!cv::solvePnP(_posePoints3D, _posePoints2D, camera._K, cv::Mat(), _R, _t, false, cv::SolvePnPMethod::SOLVEPNP_EPNP)) { return false; }

    std::cout << "Recover pose inliers: " << _inliers.rows << "\n";

    if (_inliers.rows < recPose.minInliers) { return false; }

    //cv::Mat _measurements = (cv::Mat_<double>(3,3) << _t.at<double>(0), _t.at<double>(1), _t.at<double>(2), _R.at<double>(0), _R.at<double>(1), _R.at<double>(2), 0, 0, 0);
    //std::cout << _t << "\n";
    //std::cout << _R << "\n";
    //fillMeasurements(_measurements, _t, _R);

    //std::cout << _measurements << "\n";

    //updateKalmanFilter(KF, _measurements, _t, _R);

    cv::Rodrigues(_R, recPose.R); recPose.t = _t;

    std::cout << "[DONE]\n";

    return true;
 }
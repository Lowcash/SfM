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

void Tracking::addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer) {
    TrackView _trackView;

    for (uint idx = 0; idx < points3D.size(); ++idx) {
        const cv::KeyPoint _keypoint = featureIndexer.empty() ? keyPoints[idx] : keyPoints[featureIndexer[idx]];
        const cv::Mat _descriptor = featureIndexer.empty() ? descriptor.row(idx) : descriptor.row(featureIndexer[idx]);

        if (mask[idx]) {
            m_pCloud.push_back(points3D[idx]);
            m_pClRGB.push_back(pointsRGB[idx]);
            //m_graph.push_back();

            _trackView.addTrack(&m_pCloud.back(), _keypoint, _descriptor);
        }
            
    }

    if (!_trackView.keyPoints2D.empty() && !_trackView.keyPoints3D.empty()) {
        _trackView.setView(view);

        m_trackViews.push_back(_trackView);
    }
}

bool Tracking::findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers) {
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

bool Tracking::findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, RecoveryPose& recPose) {
    if (m_trackViews.empty()) { return true; }
    
    std::cout << "Recovering pose..." << std::flush;
    
    std::vector<cv::Point2f> _posePoints2D;
    std::vector<cv::Vec3d> _posePoints3D;

    cv::Mat _R, _t, _inliers;

    for (auto t = m_trackViews.rbegin(); t != m_trackViews.rend(); ++t) {
        if (t->keyPoints2D.empty() || featView.keyPts.empty()) { continue; }

        std::vector<cv::Point2f> _prevPts, _currPts;
        std::vector<cv::DMatch> _matches;
        std::vector<int> _prevIdx, _currIdx;
        matcher.findRobustMatches(t->keyPoints2D, featView.keyPts, t->descriptor, featView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

        std::cout << "Recover pose matches: " << _matches.size() << "\n";

        if (_matches.size() < minMatches)
            break;

        cv::Mat _imOutMatch; cv::drawMatches(t->viewPtr->imColor, t->keyPoints2D, featView.viewPtr->imColor, featView.keyPts, _matches, _imOutMatch);

        for (const auto& m : _matches) {
            cv::Vec3d _point3D = (cv::Vec3d)*t->keyPoints3D[m.queryIdx];
            cv::Point2f _point2D = (cv::Point2f)featView.keyPts[m.trainIdx].pt;

            _posePoints2D.push_back(_point2D);
            _posePoints3D.push_back(_point3D);
        }

        cv::imshow("Matches", _imOutMatch); cv::waitKey(1);
    }

    if (_posePoints2D.size() < 7 || _posePoints3D.size() < 7) { return false; }

    if (!cv::solvePnPRansac(_posePoints3D, _posePoints2D, camera.K, cv::Mat(), _R, _t, recPose.useExtrinsicGuess, recPose.numIter, recPose.threshold, recPose.prob, _inliers, recPose.poseEstMethod)) { return false; }
    //if (!cv::solvePnP(_posePoints3D, _posePoints2D, camera._K, cv::Mat(), _R, _t, recPose.useExtrinsicGuess, recPose.poseEstMethod)) { return false; }

    std::cout << "Recover pose inliers: " << _inliers.rows << "\n";

    //if (_inliers.rows < recPose.minInliers) { return false; }

    cv::Rodrigues(_R, recPose.R); recPose.t = _t;

    std::cout << "[DONE]\n";

    return true;
 }
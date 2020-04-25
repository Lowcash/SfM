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

        if (mask[idx])
            _trackView.addTrack(points2D[idx], points3D[idx], pointsRGB[idx], _keypoint, _descriptor);
    }

    _trackView.setView(view);

    trackViews.push_back(_trackView);
}

void Tracking::clusterTracks() {
    if (trackViews.size() > 1) {
        auto prevTracks = &(trackViews.rbegin()[1]);
        auto currTracks = &(trackViews.rbegin()[0]);

        std::vector<cv::Point2f> _prevPts, _currPts;
        std::vector<int> _prevIdx, _currIdx;
        std::vector<cv::DMatch> _matches;

        m_matcher.findRobustMatches(prevTracks->keyPoints, currTracks->keyPoints, prevTracks->descriptor, currTracks->descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

        for (const auto& m : _matches) {
            std::cout << prevTracks->points3D[m.queryIdx] << " " << currTracks->points3D[m.trainIdx] << "\n";

            currTracks->points3D[m.trainIdx] = prevTracks->points3D[m.queryIdx];
            
            std::cout << prevTracks->points3D[m.queryIdx] << " " << currTracks->points3D[m.trainIdx] << "\n";
        }
    }
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
        matcher.findRobustMatches(t->keyPoints, featView.keyPts, t->descriptor, featView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

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

    //if (_inliers.rows < recPose.minInliers) { return false; }

    cv::Rodrigues(_R, recPose.R); recPose.t = _t;

    std::cout << "[DONE]\n";

    return true;
 }
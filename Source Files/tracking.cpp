#include "tracking.h"

void Tracking::addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Point3f> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer) {
    TrackView _trackView;
    
    for (uint idx = 0; idx < mask.size(); ++idx) {
        const cv::KeyPoint _keypoint = featureIndexer.empty() ? keyPoints[idx] : keyPoints[featureIndexer[idx]];
        const cv::Mat _descriptor = featureIndexer.empty() ? descriptor.row(idx) : descriptor.row(featureIndexer[idx]);

        if (mask[idx])
            _trackView.addTrack(points2D[idx], points3D[idx], pointsRGB[idx], _keypoint, _descriptor);
    }

    _trackView.setView(view);

    trackViews.push_back(_trackView);
}

 bool Tracking::findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, cv::Matx33d& R, cv::Matx31d& t) {
    if (trackViews.empty()) { return true; }
        
    std::cout << "Recovering pose..." << std::flush;
    
    std::vector<cv::Point2f> _posePoints2D;
    std::vector<cv::Point3f> _posePoints3D;
    cv::Mat _R, _t;

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

    if (!cv::solvePnPRansac(_posePoints3D, _posePoints2D, camera._K, cv::Mat(), _R, _t, true)) { return false; }

    cv::Rodrigues(_R, R); t = _t;

    std::cout << "[DONE]\n";

    return true;
 }
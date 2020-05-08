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
        //  Green arrow -> point used for pose estimation
        cv::arrowedLine(outputImg, currPts[i], prevPts[i], CV_RGB(0,200,0), 2);

        //  Red arrow -> point filtered by SVD cv::recoveryPose
        if (isUsingMask && mask.at<uchar>(i) == 0)
            cv::arrowedLine(outputImg, currPts[i], prevPts[i], CV_RGB(200,0,0), 2);
    }
}

void Tracking::addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, std::map<std::pair<float, float>, size_t>& cloudMap, const std::vector<int>& ptsToKeyIdx) {
    TrackView _trackView;

    size_t newPtsAdded = 0;
    for (uint idx = 0; idx < points3D.size(); ++idx) {
        //  mapping from triangulated point (alligned) to corresponding keypoint/descriptor (not alligned)
        const cv::KeyPoint _keypoint = ptsToKeyIdx.empty() ? keyPoints[idx] : keyPoints[ptsToKeyIdx[idx]];
        const cv::Mat _descriptor = ptsToKeyIdx.empty() ? descriptor.row(idx) : descriptor.row(ptsToKeyIdx[idx]);

        //  Add only good reprojected points, points in front of camera, points not far away from camera
        if (mask[idx]) {
            //  Check if point is new -> add to cloud otherwise add to seen points
            if (cloudMap.find(std::pair{_keypoint.pt.x, _keypoint.pt.y}) == cloudMap.end()) {
                cloudTracks.push_back(CloudTrack(_keypoint.pt, trackViews.size()));

                _trackView.addTrack(_keypoint, _descriptor, cloud3D.size());

                cloud3D.push_back(points3D[idx]);
                cloudRGB.push_back(pointsRGB[idx]);

                newPtsAdded++;
            } else {
                size_t cloudIdx = cloudMap[std::pair{_keypoint.pt.x, _keypoint.pt.y}];

                cloudTracks[cloudIdx].addTrack(_keypoint.pt, trackViews.size());

                _trackView.addTrack(_keypoint, _descriptor, cloudIdx);
            }
        }    
    }

    //  Min points filter
    if (_trackView.keyPoints.size() > 7 && _trackView.cloudIdxs.size() > 7) {
        std::cout << "New points were added to cloud: " << newPtsAdded << "; Total points: " << cloud3D.size() << "\n";

        _trackView.setView(view);

        trackViews.push_back(_trackView);
    } else {
        std::cout << "Points were not added to cloud!!" << "\n";
    }
}

bool Tracking::findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers) {
    if (prevPts.size() <= 5 || currPts.size() <= 5) { return false; }

    cv::Mat E = cv::findEssentialMat(prevPts, currPts, cameraK, recPose.recPoseMethod, recPose.prob, recPose.threshold, recPose.mask);

    // Essential matrix check
    if (!(E.cols == 3 && E.rows == 3)) { return false; }

    cv::Mat p0 = cv::Mat( { 3, 1 }, {
		double( prevPts[0].x ), double( prevPts[0].y ), 1.0
		} );
	cv::Mat p1 = cv::Mat( { 3, 1 }, {
		double( currPts[0].x ), double( currPts[0].y ), 1.0
		} );
	const double E_error = cv::norm( p1.t() * cameraK.inv().t() * E * cameraK.inv() * p0, cv::NORM_L2 );

    if (E_error > 1e-03) { return false; }

    // numInliers used for camera pose pruning
    numInliers = cv::recoverPose(E, prevPts, currPts, cameraK, recPose.R, recPose.t, recPose.mask);

    return numInliers > minInliers;
}

bool Tracking::findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, RecoveryPose& recPose, std::map<std::pair<float, float>, size_t>& cloudMap) {
    if (trackViews.empty()) { return true; }
    
    std::cout << "Recovering pose..." << std::flush;
    
    // 3D - 2D structures for PnP mapping
    std::vector<cv::Point2f> _posePoints2D;
    std::vector<cv::Vec3d> _posePoints3D;

    cv::Mat _R, _t, _inliers;

    for (auto t = trackViews.rbegin(); t != trackViews.rend(); ++t) {
        if (t->keyPoints.empty() || featView.keyPts.empty()) { continue; }

        std::vector<cv::Point2f> _prevPts, _currPts;
        std::vector<cv::DMatch> _matches;
        std::vector<int> _prevIdx, _currIdx;
        //  matches between new view and old trackViews
        matcher.findRobustMatches(t->keyPoints, featView.keyPts, t->descriptor, featView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

        std::cout << "Recover pose matches: " << _matches.size() << "\n";

        cv::Mat _imOutMatch; cv::drawMatches(t->viewPtr->imColor, t->keyPoints, featView.viewPtr->imColor, featView.keyPts, _matches, _imOutMatch);

        for (const auto& m : _matches) {
            //  2D point from new view
            cv::Point2f _point2D = (cv::Point2f)featView.keyPts[m.trainIdx].pt;

            //  prevent duplicities
            if (cloudMap.find(std::pair{_point2D.x, _point2D.y}) == cloudMap.end()) {
                //  3D point from old view
                cv::Vec3d _point3D = (cv::Vec3d)cloud3D[t->cloudIdxs[m.queryIdx]];
            
                _posePoints2D.push_back(_point2D);
                _posePoints3D.push_back(_point3D);
                
                cloudMap[std::pair{_point2D.x, _point2D.y}] = t->cloudIdxs[m.queryIdx];
            }
        }

        cv::imshow("Matches", _imOutMatch); cv::waitKey(1);
    }

    //  Min point filter
    if (_posePoints2D.size() < 7 || _posePoints3D.size() < 7) { return false; }

    //  Use solvePnPRansac instead of solvePnP -> RANSAC is more robustness
    if (!cv::solvePnPRansac(_posePoints3D, _posePoints2D, camera.K, cv::Mat(), _R, _t, recPose.useExtrinsicGuess, recPose.numIter, recPose.threshold, recPose.prob, _inliers, recPose.poseEstMethod)) { return false; }
    //if (!cv::solvePnP(_posePoints3D, _posePoints2D, camera.K, cv::Mat(), _R, _t, recPose.useExtrinsicGuess, recPose.poseEstMethod)) { return false; }

    std::cout << "Recover pose inliers: " << _inliers.rows << "\n";

    //  Min PnP inliers filter
    if (_inliers.rows < recPose.minInliers) { return false; }

    //  Rotation matrix to rotation vector
    cv::Rodrigues(_R, recPose.R); recPose.t = _t;

    std::cout << "[DONE]\n";

    return true;
 }
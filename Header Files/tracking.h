#ifndef TRACKING_H
#define TRACKING_H
#pragma once

#include "pch.h"
#include "common.h"
#include "reconstruction.h"
#include "view.h"
#include "feature_processing.h"
#include "camera.h"

/** 
 * RecoveryPose helper
 * 
 * It holds pose estimation settings and calculation output result
 */
class RecoveryPose {
public:
    int recPoseMethod, poseEstMethod;

    const double prob, threshold;
    const uint minInliers, numIter;
    const bool useExtrinsicGuess;
    
    cv::Matx33d R;
    cv::Matx31d t;
    cv::Mat mask;

    RecoveryPose(std::string method, const double prob, const double threshold, const uint minInliers, std::string poseEstMethod, const bool useExtrinsicGuess, const uint numIter);

    void drawRecoveredPose(cv::Mat inputImg, cv::Mat& outputImg, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, cv::Mat mask = cv::Mat());
};

/** 
 * TrackView helper used for PnP matching
 * 
 * Only good tracks with cloud reference are created
 */
class TrackView : public View {
public:
    // Each point is mapped to world point cloud idx
    std::vector<size_t> cloudIdxs;
    
    // KeyPoints and descriptors for PnP matching
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;

    void addTrack(const cv::KeyPoint keyPoint, cv::Mat descriptor, size_t cloudIdx) {
        this->keyPoints.push_back(keyPoint);
        this->descriptor.push_back(descriptor);

        this->cloudIdxs.push_back(cloudIdx);
    }
};

class Tracking {
public:
    // Good track used for matching
    std::list<TrackView> trackViews;

    // Result camera poses -> updated by bundle adjuster
    std::list<cv::Matx34d> camPoses;

    // CloudTracks same size as cloud3D
    // Cameras and 2D point projections which affect cloud
    std::vector<CloudTrack> cloudTracks;

    PointCloud pointCloud;

    cv::Matx33d R; cv::Matx31d t;

    Tracking()
        : R(cv::Matx33d::eye()), t(cv::Matx31d::eye()) {}

    bool addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, std::map<std::pair<float, float>, size_t>& cloudMap, const std::vector<int>& ptsToKeyIdx = std::vector<int>());

    /** 
     * Find pose between two views
     * It creates essential matrix and return camera pose by SVD
     */
    bool findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers);

    /** 
     * Find pose between trackViews 
     * It uses PnP alghoritm to return camera pose
     */
    bool findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, RecoveryPose& recPose, std::map<std::pair<float, float>, size_t>& cloudMap);

    bool addCamPose(const cv::Matx34d camPose) { 
        camPoses.push_back(camPose);

        decomposeExtrinsicMat(camPose, R, t);
    }
};

#endif //TRACKING_H
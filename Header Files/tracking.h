#ifndef TRACKING_H
#define TRACKING_H
#pragma once

#include "pch.h"
#include "view.h"
#include "feature_processing.h"
#include "camera.h"

/** RecoveryPose helper
 *  It holds pose estimation settings and calculation output result
 * */
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

/** TrackView helper used for PnP matching
 *  Only good tracks with cloud reference are created
 * */
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

/** CloudTrack helper used by bundle adjuster
 *  Track is created for each 3D cloud point
 *  It holds camera indexes and 2D point projections, which affect 3D cloud point
 * */
class CloudTrack {
public:
    // 2D projections
    std::vector<cv::Point2f> projKeys;

    // camera indexes
    std::vector<uint> extrinsicsIdxs;

    size_t numTracks;

    void addTrack(const cv::Point2f projKey, const uint extrinsicsIdx) {
        projKeys.push_back(projKey);
        extrinsicsIdxs.push_back(extrinsicsIdx);

        numTracks++;
    }

    CloudTrack(const cv::Point2f projKey, const uint extrinsicsIdx) {
        numTracks = 0;

        addTrack(projKey, extrinsicsIdx);
    }
};

class Tracking {
public:
    // Good track used for matching
    std::list<TrackView> trackViews;

    // Result camera poses -> updated by bundle adjuster
    std::list<cv::Matx34d> camPoses;

    // Result cloud -> updated by bundle adjuster
    std::vector<cv::Vec3d> cloud3D;

    // Result cloud colors -> not updated
    std::vector<cv::Vec3b> cloudRGB;

    // CloudTracks same size as cloud3D
    // Cameras and 2D point projections which affect cloud
    std::vector<CloudTrack> cloudTracks;

    cv::Matx33d R; cv::Matx31d t;

    Tracking()
        : R(cv::Matx33d::eye()), t(cv::Matx31d::eye()) {}

    void addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, std::map<std::pair<float, float>, size_t>& cloudMap, const std::vector<int>& ptsToKeyIdx = std::vector<int>());

    bool findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers);

    bool findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, RecoveryPose& recPose, std::map<std::pair<float, float>, size_t>& cloudMap);

    void addCamPose(const cv::Matx34d camPose) { 
        camPoses.push_back(camPose);

        R = camPose.get_minor<3, 3>(0, 0);
        t = cv::Matx31d(camPose(0,3), camPose(1,3), camPose(2,3));
    }
};

#endif //TRACKING_H
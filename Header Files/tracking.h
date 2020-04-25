#ifndef TRACKING_H
#define TRACKING_H
#pragma once

#include "pch.h"
#include "view.h"
#include "feature_processing.h"
#include "camera.h"

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

class TrackView : public View {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;

    std::vector<cv::Point2f> points2D;
    std::vector<cv::Vec3d> points3D;
    std::vector<cv::Vec3b> pointsRGB;

    size_t numTracks;

    TrackView() { numTracks = 0; }

    void addTrack(const cv::Point2f point2D, const cv::Vec3d point3D, const cv::Vec3b pointRGB, const cv::KeyPoint keyPoint, cv::Mat descriptor) {
        points2D.push_back(point2D);
        points3D.push_back(point3D);
        pointsRGB.push_back(pointRGB);

        this->keyPoints.push_back(keyPoint);
        this->descriptor.push_back(descriptor);

        numTracks++;
    }
};

class Tracking {
private:
    std::vector<cv::Matx34f> m_camPoses;

    DescriptorMatcher m_matcher;
public:
    std::vector<TrackView> trackViews;

    cv::Matx33d R; cv::Matx31d t;

    Tracking(DescriptorMatcher descriptorMatcher)
        : R(cv::Matx33d::eye()), t(cv::Matx31d::eye()), m_matcher(descriptorMatcher) {}

    void addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer = std::vector<int>());

    void clusterTracks();

    bool findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camParams, FeatureView& featView, RecoveryPose& recPose);

    void addCamPose(const cv::Matx34f camPose) { 
        m_camPoses.push_back(camPose);

        R = camPose.get_minor<3, 3>(0, 0);
        t = cv::Matx31d(camPose(0,3), camPose(1,3), camPose(2,3));
    }

    std::vector<cv::Matx34f>* getCamPoses() { return &m_camPoses; }

    cv::Matx34f getLastCamPose() const { return m_camPoses.back(); }

    bool isCamPosesEmpty() { return m_camPoses.empty(); }
};

#endif //TRACKING_H
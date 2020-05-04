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
    std::vector<cv::Vec3d*> keyPoints3D;
    std::vector<cv::KeyPoint> keyPoints2D;
    cv::Mat descriptor;

    void addTrack(cv::Vec3d* keyPoint3D, const cv::KeyPoint keyPoint2D, cv::Mat descriptor) {
        this->keyPoints3D.push_back(keyPoint3D);
        this->keyPoints2D.push_back(keyPoint2D);
        this->descriptor.push_back(descriptor);
    }
};

class Track {
public:
    std::vector<cv::Point2f*> projKey;
    std::vector<uint*> extrinsicsKey;

    size_t trackSize;
};

class Tracking {
private:
    
public:
    std::list<TrackView> m_trackViews;

    std::list<cv::Matx34d> m_camPoses;

    std::list<cv::Vec3d> m_pCloud;

    std::list<cv::Vec3b> m_pClRGB;

    std::vector<Track> m_graph;

    cv::Matx33d R; cv::Matx31d t;

    Tracking()
        : R(cv::Matx33d::eye()), t(cv::Matx31d::eye()) {}

    void addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer = std::vector<int>());

    bool findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers);

    bool findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camera, FeatureView& featView, RecoveryPose& recPose);

    void addCamPose(const cv::Matx34d camPose) { 
        m_camPoses.push_back(camPose);

        R = camPose.get_minor<3, 3>(0, 0);
        t = cv::Matx31d(camPose(0,3), camPose(1,3), camPose(2,3));
    }

    /*std::list<cv::Matx34f>* getCamPoses() { return &m_camPoses; }

    cv::Matx34f* getLastCamPose() { return &*m_camPoses.rbegin(); }

    bool isCamPosesEmpty() { return m_camPoses.empty(); }

    std::list<TrackView>* getTrackViews() { return &m_trackViews; }

    TrackView* getLastTrackView() { return &*m_trackViews.rbegin(); }*/
};

#endif //TRACKING_H
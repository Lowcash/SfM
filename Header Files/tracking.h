#ifndef TRACKING_H
#define TRACKING_H
#pragma once

#include "pch.h"
#include "view.h"
#include "feature_processing.h"
#include "camera.h"

class TrackView : public View {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;

    std::vector<cv::Point2f> points2D;
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> pointsRGB;

    void addTrack(const cv::Point2f point2D, const cv::Point3d point3D, const cv::Vec3b pointRGB, const cv::KeyPoint keyPoint, cv::Mat descriptor) {
        points2D.push_back(point2D);
        points3D.push_back(point3D);
        pointsRGB.push_back(pointRGB);

        this->keyPoints.push_back(keyPoint);
        this->descriptor.push_back(descriptor);
    }
};

class Tracking {
public:
    std::vector<TrackView> trackViews;

    void addTrackView(const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Point3f> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer = std::vector<int>());

    bool findRecoveredCameraPose(DescriptorMatcher matcher, float knnRatio, Camera camParams, FeatureView& featView, cv::Matx33d& R, cv::Matx31d& t);
};

#endif //TRACKING_H
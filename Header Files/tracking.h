#pragma once

#ifndef _TRACKING_H
#define _TRACKING_H

// Ratio to the second neighbor to consider a good match.
#define RATIO 0.6

#include "pch.h"

class Track {
public:
    cv::Point2f basePosition2d;
    cv::Point2f activePosition2d;

    cv::Point3d basePosition3d;
    cv::Point3d activePosition3d;

    cv::Vec3b color;

    cv::Mat descriptor;

    int missedFrames;

    Track(const cv::Point2f point2D, const cv::Point3d point3D, const cv::Mat descriptor, const cv::Vec3b color) {
        basePosition2d = point2D;
        activePosition2d = point2D;
        basePosition3d = point3D;
        activePosition3d = point3D;

        descriptor.copyTo(this->descriptor);
        //this->descriptor = descriptor;
        this->color = color;

        missedFrames = 0;
    };
};

class Tracking {
private:
    std::vector<int> m_matchIdx;

    int missedFrames, updatedFrames;

    static bool isTrackingStale(const Track& track);
public:
    std::list<Track> m_tracks;
    
    Tracking(const float matchRatioThreshold, const bool isDebugVisualization = false);

    ~Tracking();

    void updateTracks(const std::vector<cv::Point2f> points2D, const std::vector<cv::Point3d> points3D, const cv::Mat descriptor, const std::vector<cv::Vec3b> colors);

    void matchTracks(const cv::Mat descriptor);

    void triangulate_matches(std::vector<cv::DMatch>& matches, const std::vector<cv::Point2f>&points1, const std::vector<cv::Point2f>& points2,
		cv::Matx34f& cam1P, cv::Matx34f& cam2P, std::vector<cv::Point3f>& pnts3D);

    void transformationFromTracks(cv::Matx33f& R, cv::Matx31f& T);
    
    void tracksToPointCloud(std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& colors);

    // void tracksToPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud);

    float getMedianFeatureMovement();
};

#endif /* _TRACKING_H */
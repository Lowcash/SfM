#ifndef IVISUALIZABLE_H
#define IVISUALIZABLE_H
#pragma once

#include "pch.h"
#include "tracking.h"

class IVisualizable {
protected:
    int m_numClouds, m_numCams;
public:
    virtual void addPointCloud(const std::vector<cv::Point3f>& points3D, const std::vector<cv::Vec3b>& pointsRGB) = 0;

    virtual void addPointCloud(const std::vector<TrackView>& trackViews) = 0;

    virtual void addCamera(const cv::Matx34f  camPose) = 0;

    virtual void visualize() = 0;
};

#endif //IVISUALIZABLE_H
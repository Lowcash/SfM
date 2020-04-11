#ifndef VISUALIZATION_H
#define VISUALIZATION_H
#pragma once

#include "pch.h"
#include "tracking.h"
#include "visualizable.h"

class VisPCLUtils {
protected:
    void cvPoseToPCLPose(cv::Matx34d cvPose, pcl::PointXYZ& pclPose) {
        pclPose = pcl::PointXYZ(cvPose(0, 3), cvPose(1, 3), cvPose(2, 3));
    }

    void cvPoseToInversePCLPose(cv::Matx34d cvPose, pcl::PointXYZ& pclPose) {
        pclPose = pcl::PointXYZ(-cvPose(0, 3), -cvPose(1, 3), -cvPose(2, 3));
    }
};

class VisPCL : public VisPCLUtils, public IVisualizable {
private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> getNewViewer (const std::string windowName, const cv::Size windowSize, const cv::viz::Color bColor = cv::viz::Color::black());
public:
    VisPCL(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor = cv::viz::Color::black());

    void addPointCloud(const std::vector<cv::Point3f>& points3D, const std::vector<cv::Vec3b>& pointsRGB);

    void addPointCloud(const std::vector<TrackView>& trackViews);

    void updateCameras(const std::vector<cv::Matx34f> camPoses);

    void addCamera(const cv::Matx34f camPose);

    void visualize();
};

class VisVTK : public IVisualizable {
private:
    cv::viz::Viz3d m_viewer;
public:
    VisVTK(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor = cv::viz::Color::black());

    void addPointCloud(const std::vector<cv::Point3f>& points3D, const std::vector<cv::Vec3b>& pointsRGB);

    void addPointCloud(const std::vector<TrackView>& trackViews);

    void updateCameras(const std::vector<cv::Matx34f> camPoses);

    void addCamera(const cv::Matx34f camPose);

    void visualize();
};

#endif //VISUALIZATION_H
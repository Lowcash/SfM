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
    bool m_isUpdate;

    boost::mutex m_updateModelMutex;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> getNewViewer (const std::string windowName, const cv::Size windowSize, const cv::viz::Color bColor = cv::viz::Color::black());
public:
    VisPCL(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor = cv::viz::Color::black());

    void addPointCloud(const std::vector<TrackView>& trackViews);

    void addPointCloud(const std::vector<cv::Vec3d>& points3D, const std::vector<cv::Vec3b>& pointsRGB);

    void addPoints(const std::vector<cv::Vec3d> points3D);
    
    void updateCameras(const std::vector<cv::Matx34f> camPoses);

    void addCamera(const cv::Matx34f camPose);

    void visualize();
};

class VisVTKUtils {
private:
    void decomposeCvPose(cv::Matx34f cvPose, cv::Matx33f& R, cv::Vec3f& t) {
        R = cvPose.get_minor<3, 3>(0, 0);
        t = cv::Vec3f(cvPose(0, 3), cvPose(1, 3), cvPose(2, 3));
    }

protected:
    void cvPoseToVTKPose(cv::Matx34f cvPose, cv::Affine3d& vtkPose) {
        cv::Matx33f R; cv::Vec3f t; decomposeCvPose(cvPose, R, t);

        vtkPose = cv::Affine3d(R, t);
    }

    void cvPoseToInverseVTKPose(cv::Matx34f cvPose, cv::Affine3d& vtkPose) {
        cv::Matx33f R; cv::Vec3f t; decomposeCvPose(cvPose, R, t);

        vtkPose = cv::Affine3d(-R, -t);
    }
};

class VisVTK : public VisVTKUtils, public IVisualizable {
private:
    cv::viz::Viz3d m_viewer;
public:
    VisVTK(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor = cv::viz::Color::black());

    void addPointCloud(const std::vector<TrackView>& trackViews);

    void addPointCloud(const std::vector<cv::Vec3d>& points3D, const std::vector<cv::Vec3b>& pointsRGB);
    
    void addPoints(const std::vector<cv::Vec3d> points3D);

    void updateCameras(const std::vector<cv::Matx34f> camPoses, const cv::Matx33d K33d);

    void addCamera(const cv::Matx34f camPose);

    void visualize();
};

#endif //VISUALIZATION_H
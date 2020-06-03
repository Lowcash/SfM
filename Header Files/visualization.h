#ifndef VISUALIZATION_H
#define VISUALIZATION_H
#pragma once

#include "pch.h"
#include "common.h"
#include "tracking.h"
#include "visualizable.h"

class VisPCLUtils {
protected:
    /** 
     * OpenCV pose to PCL pose
     */
    void cvPoseToPCLPose(cv::Matx34d cvPose, pcl::PointXYZ& pclPose) {
        pclPose = pcl::PointXYZ(cvPose(0, 3), cvPose(1, 3), cvPose(2, 3));
    }

    /** 
     * OpenCV pose to PCL pose
     */
    void cvPoseToInversePCLPose(cv::Matx34d cvPose, pcl::PointXYZ& pclPose) {
        pclPose = pcl::PointXYZ(-cvPose(0, 3), -cvPose(1, 3), -cvPose(2, 3));
    }
};

class VisPCL : public VisPCLUtils, public IVisualizable {
private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> getNewViewer (const std::string windowName, const cv::Size windowSize, const cv::viz::Color bColor = cv::viz::Color::black());
public:
    VisPCL(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor = cv::viz::Color::black(), const bool isEnable = true);

    ~VisPCL();

    void updatePointCloud(const std::list<cv::Vec3d>& points3D, const std::list<cv::Vec3b>& pointsRGB);

    void addPoints(const std::vector<cv::Vec3d> points3D);
    
    void updateCameras(const std::list<cv::Matx34d> camPoses);

    void visualize(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor);
};

class VisVTKUtils {
protected:
    /** 
     * OpenCV pose to OpenCV VTK pose
     */
    void cvPoseToVTKPose(cv::Matx34d cvPose, cv::Affine3d& vtkPose) {
        cv::Matx33d R; cv::Vec3d t; decomposeExtrinsicMat(cvPose, R, t);

        vtkPose = cv::Affine3d(R, t);
    }
    
    /** 
     * OpenCV pose to OpenCV VTK pose
     */
    void cvPoseToInverseVTKPose(cv::Matx34d cvPose, cv::Affine3d& vtkPose) {
        cv::Matx33d R; cv::Vec3d t; decomposeExtrinsicMat(cvPose, R, t);

        cv::Vec3d _t = cv::Vec3d(-t[0], -t[1], -t[2]);

        vtkPose = cv::Affine3d(-R, _t);
    }
};

class VisVTK : public VisVTKUtils, public IVisualizable {
private:
    cv::viz::Viz3d m_viewer;
public:
    VisVTK(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor = cv::viz::Color::black(), const bool isEnable = true);

    ~VisVTK();

    void updatePointCloud(const std::list<cv::Vec3d>& points3D, const std::list<cv::Vec3b>& pointsRGB);
    
    void addPoints(const std::vector<cv::Vec3d> points3D);

    void updateCameras(const std::list<cv::Matx34d> camPoses, const cv::Matx33d K33d);

    void addCamera(const std::list<cv::Matx34d> camPoses, const cv::Matx33d K33d);

    void visualize(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor);
};

#endif //VISUALIZATION_H
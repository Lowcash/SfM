#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H
#pragma once

#include "pch.h"
#include "camera.h"
#include "tracking.h"

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const extrinsics,
                    const T* const point,
                    const T* const focal,
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T x[3];
        ceres::AngleAxisRotatePoint(extrinsics, point, x);
        x[0] += extrinsics[3];
        x[1] += extrinsics[4];
        x[2] += extrinsics[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xn = x[0] / x[2];
        T yn = x[1] / x[2];

        // Compute final projected point position.
        T predicted_x = *focal * xn;
        T predicted_y = *focal * yn;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                        const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3, 1>(
                    new SnavelyReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};

class Reconstruction {
private:
    void pointsToMat(std::vector<cv::Point2f> points, cv::Mat& pointsMat) {
        pointsMat = (cv::Mat_<double>(2,1) << 1, 1);;

        for (const auto& p : points) {
            cv::Mat _pointMat = (cv::Mat_<double>(2,1) << p.x, p.y);

            cv::hconcat(pointsMat, _pointMat, pointsMat);
        } 

        pointsMat = pointsMat.colRange(1, pointsMat.cols);
    }

    void pointsToMat(cv::Mat points, cv::Mat& pointsMat) {
        pointsMat = (cv::Mat_<double>(2,1) << 1, 1);;

        for (size_t i = 0; i < points.cols; ++i) {
            cv::Point2f _point = points.at<cv::Point2f>(i);

            cv::Mat _pointMat = (cv::Mat_<double>(2,1) << _point.x, _point.y);

            cv::hconcat(pointsMat, _pointMat, pointsMat);
        } 

        pointsMat = pointsMat.colRange(1, pointsMat.cols);
    }

    const std::string m_method;

    const float m_minDistance, m_maxDistance, m_maxProjectionError;

    const bool m_useNormalizePts;

    void pointsToRGBCloud(cv::Mat imgColor, Camera camera, cv::Mat R, cv::Mat t, cv::Mat points3D, cv::Mat inputPts2D, std::vector<cv::Vec3d>& cloud3D, std::vector<cv::Vec3b>& cloudRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask);
public:
    Reconstruction(const std::string method, const float minDistance, const float maxDistance, const float maxProjectionError, const bool useNormalizePts = true);

    void triangulateCloud(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const cv::Mat colorImage, std::vector<cv::Vec3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, std::vector<bool>& mask, Camera camera, const cv::Matx34d prevPose, const cv::Matx34d currPose, RecoveryPose& recPose);

    void adjustBundle(std::vector<TrackView>& tracks, Camera& camera, std::vector<cv::Matx34f>& camPoses, std::string solverType, std::string lossType, double lossFunctionScale, uint maxIter = 999);
};

#endif //RECONSTRUCTION_H
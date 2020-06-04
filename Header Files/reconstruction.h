#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H
#pragma once

#include "pch.h"
#include "camera.h"
#include "common.h"

class CloudTrack {
public:
    // reference to cloud point
    cv::Vec3d* ptrPoint3D;

    // 2D projections in the camera
    std::vector<cv::Point2f> projKeys;

    // extrinsics camera indexes for mapping
    std::vector<uint> extrinsicsIdxs;

    void addTrack(const cv::Point2f projKey, const uint extrinsicsIdx) {
        projKeys.push_back(projKey);
        extrinsicsIdxs.push_back(extrinsicsIdx);
    }

    CloudTrack(cv::Vec3d* ptrPoint3D, const cv::Point2f projKey, const uint extrinsicsIdx) {
        this->ptrPoint3D = ptrPoint3D;

        addTrack(projKey, extrinsicsIdx);
    }
};

class PointCloud {
    const double m_clearRatio;

    size_t m_numCloudPts;
public:
    std::vector<cv::Vec3d*> cloudMapper;

    // Result cloud -> updated by bundle adjuster
    std::list<cv::Vec3d> cloud3D;

    // Result cloud colors -> not updated
    std::list<cv::Vec3b> cloudRGB;

    // Cloud render/computing visibility mask
    std::vector<bool> cloudMask;

    // Registered views and projections for each cloud point
    std::vector<CloudTrack> cloudTracks;
    
    PointCloud(const double clearRatio = .5) 
        : m_clearRatio(clearRatio), m_numCloudPts(0) {}

    void addCloudPoint(const cv::Point2f projPosition2D, const cv::Vec3d cloudPoint3D, const cv::Vec3b cloudPointRGB, const size_t cameraIdx) {
        cloud3D.push_back(cloudPoint3D);
        cloudRGB.push_back(cloudPointRGB);
        cloudMask.push_back(true);

        // create and register cloud view
        cloudTracks.push_back(CloudTrack(&*cloud3D.rbegin(), projPosition2D, cameraIdx));

        // map to cloud to enable random access
        cloudMapper.push_back(&*cloud3D.rbegin());

        m_numCloudPts++;
    }

    void removeCloudPoint(bool& cloudMaskPt) {
        cloudMaskPt = false;

        m_numCloudPts--;
    }

    void registerCloudView(const size_t cloudPointIdx, const cv::Point2f projPosition2D, const size_t cameraIdx) {
        cloudTracks[cloudPointIdx].addTrack(projPosition2D, cameraIdx);
    }

    void clearCloud() {
        size_t clearedTracks = 0;

        for (auto [pMask, pMaskEnd, pTr, pTrEnd] = std::tuple{cloudMask.begin(), cloudMask.end(), cloudTracks.begin(), cloudTracks.end()}; pMask != pMaskEnd && pTr != pTrEnd; ++pMask, ++pTr) {
            if ((bool)*pMask) {
                if(pTr->extrinsicsIdxs.size() < 2) {
                    clearedTracks++;

                    *pMask = false;

                    m_numCloudPts--;
                    //removeCloudPoint((bool)&*pMask);
                }
            }
        }

        std::cout << "Cleared tracks: [" << clearedTracks << "/" << m_numCloudPts << "]\n"; 
    }

    size_t getNumCloudPoints() const { return m_numCloudPts; }
};

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const intrinsics,
                    const T* const extrinsics,
                    const T* const point,
                    T* residuals) const {
        const T& focalX = intrinsics[0];
        const T& focalY = intrinsics[1];
        const T& ppX = intrinsics[2];
        const T& ppY = intrinsics[3];

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
        T predicted_x = (focalX * xn) + ppX;
        T predicted_y = (focalY * yn) + ppY;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                        const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 4, 6, 3>(
                    new SnavelyReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};

class Reconstruction {
private:
    const std::string m_triangulateMethod, m_baMethod;

    const double m_baMaxRMSE;

    const float m_minDistance, m_maxDistance, m_maxProjectionError;

    const bool m_useNormalizePts;

    void pointsToRGBCloud(Camera camera, cv::Mat imgColor, cv::Matx33d R, cv::Matx31d t, cv::Mat points3D, cv::Mat inputPts2D, std::vector<cv::Vec3d>& cloud3D, std::vector<cv::Vec3b>& cloudRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask);
public:
    Reconstruction(const std::string triangulateMethod, const std::string baMethod, const double baMaxRMSE, const float minDistance, const float maxDistance, const float maxProjectionError, const bool useNormalizePts);

    void triangulateCloud(Camera camera, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const cv::Mat colorImage, std::vector<cv::Vec3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, std::vector<bool>& mask, const cv::Matx34d prevPose, const cv::Matx34d currPose, cv::Matx33d& R, cv::Matx31d& t);

    void adjustBundle(Camera& camera, std::list<cv::Matx34d>& camPoses, PointCloud& pointCloud);
};

#endif //RECONSTRUCTION_H
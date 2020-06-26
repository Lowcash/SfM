#ifndef CAMERA_H
#define CAMERA_H
#pragma once

#include "pch.h"
#include "common.h"

/** 
 * Camera helper
 */
class CameraParameters {
public:
    cv::Mat K, distCoeffs;

    cv::Matx33d K33d;

    cv::Point2d pp;

    cv::Point2d focal;

    CameraParameters(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        updateCameraParameters(K, distCoeffs, downSample);
    }

    void updateCameraParameters(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        this->K = K / downSample;
        this->K.at<double>(2,2) = 1.0;

        this->distCoeffs = distCoeffs;

        this->K33d = cv::Matx33d(
            this->K.at<double>(0,0), this->K.at<double>(0,1), this->K.at<double>(0,2),
            this->K.at<double>(1,0), this->K.at<double>(1,1), this->K.at<double>(1,2), 
            this->K.at<double>(2,0), this->K.at<double>(2,1), this->K.at<double>(2,2)
        );

        this->pp = cv::Point2d(this->K.at<double>(0, 2), this->K.at<double>(1, 2));
        this->focal = cv::Point2d(this->K.at<double>(0, 0), this->K.at<double>(1, 1));

        std::cout << "\nCamera intrices: " << this->K << "\n";
    }
};

class CameraData {
public:
    CameraParameters* intrinsics;

    std::list<cv::Matx34d> extrinsics;

    std::vector<uint> extrinsicsCounter;

    cv::Matx33d actualR; cv::Matx31d actualT;

    uint numCameras;

    CameraData(CameraParameters* cameraIntrinsics) 
        : actualR(cv::Matx33d()), actualT(cv::Matx31d()), numCameras(0) {
       intrinsics = cameraIntrinsics;     
    }

    void addCamPose(const cv::Matx34d camPose) { 
        extrinsics.push_back(camPose);
        extrinsicsCounter.push_back(0);

        decomposeExtrinsicMat(camPose, actualR, actualT);

        numCameras++;
    }
};

#endif //CAMERA_H
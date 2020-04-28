#ifndef CAMERA_H
#define CAMERA_H
#pragma once

#include "pch.h"

class Camera {
public:
    cv::Mat K, distCoeffs;

    cv::Matx33d K33d;

    cv::Point2d pp;

    double focalLength;

    Camera(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        updateCameraParameters(K, distCoeffs, downSample);
    }

    void updateCameraParameters(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        this->K = K * downSample;

        this->K.at<double>(2,2) = 1.0;
    
        std::cout << "\nCamera intrices: " << this->K << "\n";

        this->distCoeffs = distCoeffs;

        K33d = cv::Matx33d(
            K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
            K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2), 
            K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2)
        );

        pp = cv::Point2d(K.at<double>(0, 2), K.at<double>(1, 2));
        focalLength = ((K.at<double>(0, 0) + K.at<double>(1, 1)) / 2.0);
    }
};

#endif //CAMERA_H
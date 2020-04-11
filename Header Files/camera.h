#ifndef CAMERA_H
#define CAMERA_H
#pragma once

#include "pch.h"

class Camera {
public:
    cv::Mat _K, distCoeffs;

    cv::Matx33d K33d;

    cv::Point2d pp;

    double focalLength;

    Camera(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        updateCameraParameters(K, distCoeffs, downSample);
    }

    void updateCameraParameters(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        _K = K * downSample;

        _K.at<double>(2,2) = 1.0;
    
        std::cout << "\nCamera intrices: " << _K << "\n";

        this->distCoeffs = distCoeffs;

        K33d = cv::Matx33d(
            _K.at<double>(0,0), _K.at<double>(0,1), _K.at<double>(0,2),
            _K.at<double>(1,0), _K.at<double>(1,1), _K.at<double>(1,2), 
            _K.at<double>(2,0), _K.at<double>(2,1), _K.at<double>(2,2)
        );

        pp = cv::Point2d(_K.at<double>(0, 2), _K.at<double>(1, 2));
        focalLength = ((_K.at<double>(0, 0) + _K.at<double>(1, 1)) / 2.0);
    }
};

#endif //CAMERA_H
#pragma once

#ifndef _CAMERA_H
#define _CAMERA_H

#include "pch.h"

class Camera {
private:
    const cv::Mat m_K;
    const bool m_isProjective;
public:
    Camera(const cv::Mat K, const bool isProjective = true)
        : m_K(K), m_isProjective(isProjective) {}

    static cv::Mat getCameraK(const double angleOfView, cv::Size cameraSize) {
        const cv::Point2f alpha(angleOfView / 2, angleOfView / 2);

        const cv::Point2f c(cameraSize.width / 2, cameraSize.height / 2);
        const cv::Point2f f(
            cameraSize.width / (2 * std::tan(alpha.x)),
            cameraSize.height / (2 * std::tan(alpha.y))
        );

        auto calibrationMat = cv::Matx33f{
            f.x, 0, c.x,
            0, f.y, c.y,
            0, 0, 1
        };

        return cv::Mat (calibrationMat);
    }

    cv::Mat getK() const { return m_K; }

    static float getFocalLength(const double angleOfView, cv::Size cameraSize) { 
        const cv::Point2f alpha(angleOfView / 2, angleOfView / 2);

        return cameraSize.width / (2 * std::tan(alpha.x));
    }

    bool getIsProjective() const { return m_isProjective; }
};

#endif /* _CAMERA_H */
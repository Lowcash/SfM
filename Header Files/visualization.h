#pragma once

#ifndef _VISUALIZATION_H
#define _VISUALIZATION_H

#include "pch.h"

class Visualization {
public:
    struct Settings {
        private:
            const cv::Matx33f m_cameraK;

            const std::string m_windowName;
            const cv::Size m_windowSize;
            const cv::Point m_windowPosition;
            const cv::viz::Color m_backgroundColor, m_cameraColor;
        public:
            Settings(const cv::Matx33f cameraK, const std::string windowName, const cv::Size windowSize, const cv::Point windowPosition = cv::Point(), const cv::viz::Color backgroundColor = cv::viz::Color::black(), const cv::viz::Color cameraColor = cv::viz::Color::yellow())
                : m_cameraK(cameraK), m_windowName(windowName), m_windowSize(windowSize), m_windowPosition(windowPosition), m_backgroundColor(backgroundColor), m_cameraColor(cameraColor) {}

            cv::Matx33f getCameraK() const { return m_cameraK; }
            
            std::string getWindowName() const { return m_windowName; };

            cv::Size getWindowSize() const { return m_windowSize; }

            cv::Point getWindowPosition() const { return m_windowPosition; }

            cv::viz::Color getBackgroundColor() const { return m_backgroundColor; }

            cv::viz::Color getCameraColor() const { return m_cameraColor; }
    };

    struct ColoredPoints {
    public:
        const std::vector<cv::Point3f> m_points;
        const std::vector<cv::Vec3b> m_colors;

        ColoredPoints(const std::vector<cv::Point3f> points, const std::vector<cv::Vec3b> colors)
            : m_points(points), m_colors(colors) {}

        std::vector<cv::Point3f> getPoints() const { return m_points; }

        std::vector<cv::Vec3b> getColors() const { return m_colors; }
    };

private:
    cv::viz::Viz3d m_window;

    std::vector<ColoredPoints> m_pointClouds;

    std::vector<cv::Affine3d> m_camPoses;
public:
    Visualization(Settings visSettings);

    cv::viz::Viz3d getWindow() { return m_window; }

    void updateVieverPose(cv::Affine3d viewerPose);

    void updatePointCloud(ColoredPoints coloredPointCloud, cv::Affine3d cam);

    void updateCameraPose(cv::Affine3d cam);
};

struct VisualizationCore {
    void visualizationCoroutine(cv::viz::Viz3d window) {
        while(true){
            window.spinOnce(1, true);
        }
    }
};

#endif /* _VISUALIZATION_H */
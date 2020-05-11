#ifndef COMMON_H
#define COMMON_H
#pragma once

#include "pch.h"

inline void composeExtrinsicMat(cv::Matx33d R, cv::Matx31d t, cv::Matx34d& pose) {
    pose = cv::Matx34d(
        R(0,0), R(0,1), R(0,2), t(0),
        R(1,0), R(1,1), R(1,2), t(1),
        R(2,0), R(2,1), R(2,2), t(2)
    );
}

inline void composeExtrinsicMat(cv::Matx33d R, cv::Vec3d t, cv::Matx34d& pose) {
    pose = cv::Matx34d(
        R(0,0), R(0,1), R(0,2), t(0),
        R(1,0), R(1,1), R(1,2), t(1),
        R(2,0), R(2,1), R(2,2), t(2)
    );
}

inline void decomposeExtrinsicMat(cv::Matx34d pose, cv::Matx33d& R, cv::Matx31d& t) {
    R = pose.get_minor<3, 3>(0, 0);

    t = cv::Matx31d(
        pose(0,3),
        pose(1,3),
        pose(2,3)
    );
}

inline void decomposeExtrinsicMat(cv::Matx34d pose, cv::Matx33d& R, cv::Vec3d& t) {
    R = pose.get_minor<3, 3>(0, 0);

    t = cv::Vec3d(
        pose(0, 3), 
        pose(1, 3), 
        pose(2, 3)
    );
}

/** 
 *  Convert points to mat
 *  Usually for data preparation for OpenCV SFM structure
 * */
inline void pointsToMat(std::vector<cv::Point2f> points, cv::Mat& pointsMat) {
    pointsMat = (cv::Mat_<double>(2,1) << 1, 1);;

    for (const auto& p : points) {
        cv::Mat _pointMat = (cv::Mat_<double>(2,1) << p.x, p.y);

        cv::hconcat(pointsMat, _pointMat, pointsMat);
    } 

    pointsMat = pointsMat.colRange(1, pointsMat.cols);
}

/** Convert points to mat
 *  Usually for data preparation for OpenCV SFM structure
 * */
inline void pointsToMat(cv::Mat points, cv::Mat& pointsMat) {
    pointsMat = (cv::Mat_<double>(2,1) << 1, 1);;

    for (size_t i = 0; i < points.cols; ++i) {
        cv::Point2f _point = points.at<cv::Point2f>(i);

        cv::Mat _pointMat = (cv::Mat_<double>(2,1) << _point.x, _point.y);

        cv::hconcat(pointsMat, _pointMat, pointsMat);
    } 

    pointsMat = pointsMat.colRange(1, pointsMat.cols);
}

#endif //COMMON_H
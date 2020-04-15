#ifndef TRACKING_H
#define TRACKING_H
#pragma once

#include "pch.h"
#include "view.h"
#include "feature_processing.h"
#include "camera.h"

class RecoveryPose {
public:
    int method;

    const double prob, threshold;
    const uint minInliers;

    cv::Matx33d R;
    cv::Matx31d t;
    cv::Mat mask;

    RecoveryPose(std::string method, const double prob, const double threshold, const uint minInliers);

    void drawRecoveredPose(cv::Mat inputImg, cv::Mat& outputImg, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, cv::Mat mask = cv::Mat());
};

class TrackView : public View {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;

    std::vector<cv::Point2f> points2D;
    std::vector<cv::Vec3d> points3D;
    std::vector<cv::Vec3b> pointsRGB;

    size_t numTracks;

    TrackView() { numTracks = 0; }

    void addTrack(const cv::Point2f point2D, const cv::Vec3d point3D, const cv::Vec3b pointRGB, const cv::KeyPoint keyPoint, cv::Mat descriptor) {
        points2D.push_back(point2D);
        points3D.push_back(point3D);
        pointsRGB.push_back(pointRGB);

        this->keyPoints.push_back(keyPoint);
        this->descriptor.push_back(descriptor);

        numTracks++;
    }
};

class Tracking {
private:
    cv::KalmanFilter KF;

    void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);

    void fillMeasurements( cv::Mat &measurements, const cv::Mat translation_measured, const cv::Mat rotation_measured);

    void updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement, cv::Mat &translation_estimated, cv::Mat &rotation_estimated );

    // Converts a given Rotation Matrix to Euler angles
    cv::Mat rot2euler(const cv::Mat & rotationMatrix) {
        cv::Mat euler(3,1,CV_32F);

        float m00 = rotationMatrix.at<float>(0,0);
        float m02 = rotationMatrix.at<float>(0,2);
        float m10 = rotationMatrix.at<float>(1,0);
        float m11 = rotationMatrix.at<float>(1,1);
        float m12 = rotationMatrix.at<float>(1,2);
        float m20 = rotationMatrix.at<float>(2,0);
        float m22 = rotationMatrix.at<float>(2,2);

        float x, y, z;

        // Assuming the angles are in radians.
        if (m10 > 0.998) { // singularity at north pole
            x = 0;
            y = CV_PI/2;
            z = atan2(m02,m22);
        }
        else if (m10 < -0.998) { // singularity at south pole
            x = 0;
            y = -CV_PI/2;
            z = atan2(m02,m22);
        }
        else
        {
            x = atan2(-m12,m11);
            y = asin(m10);
            z = atan2(-m20,m00);
        }

        euler.at<float>(0) = x;
        euler.at<float>(1) = y;
        euler.at<float>(2) = z;

        return euler;
    }

    // Converts a given Euler angles to Rotation Matrix
    cv::Mat euler2rot(const cv::Mat & euler) {
        cv::Mat rotationMatrix(3,3,CV_32F);

        float x = euler.at<float>(0);
        float y = euler.at<float>(1);
        float z = euler.at<float>(2);

        // Assuming the angles are in radians.
        float ch = cos(z);
        float sh = sin(z);
        float ca = cos(y);
        float sa = sin(y);
        float cb = cos(x);
        float sb = sin(x);

        float m00, m01, m02, m10, m11, m12, m20, m21, m22;

        m00 = ch * ca;
        m01 = sh*sb - ch*sa*cb;
        m02 = ch*sa*sb + sh*cb;
        m10 = sa;
        m11 = ca*cb;
        m12 = -ca*sb;
        m20 = -sh*ca;
        m21 = sh*sa*cb + ch*sb;
        m22 = -sh*sa*sb + ch*cb;

        rotationMatrix.at<float>(0,0) = m00;
        rotationMatrix.at<float>(0,1) = m01;
        rotationMatrix.at<float>(0,2) = m02;
        rotationMatrix.at<float>(1,0) = m10;
        rotationMatrix.at<float>(1,1) = m11;
        rotationMatrix.at<float>(1,2) = m12;
        rotationMatrix.at<float>(2,0) = m20;
        rotationMatrix.at<float>(2,1) = m21;
        rotationMatrix.at<float>(2,2) = m22;

        return rotationMatrix;
    }
public:
    Tracking();

    std::vector<TrackView> trackViews;

    void addTrackView(ViewData* view, const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Vec3d> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer = std::vector<int>());

    bool findRecoveredCameraPose(DescriptorMatcher matcher, int minMatches, Camera camParams, FeatureView& featView, RecoveryPose& recPose);
};

#endif //TRACKING_H
#ifndef FEATURE_PROCESSING_H
#define FEATURE_PROCESSING_H
#pragma once

#include "pch.h"
#include "cuda_usable.h"
#include "view.h"
class FeatureDetector : protected CUDAUsable {
private:
    enum DetectorType { AKAZE = 0, ORB, FAST, STAR, SIFT, SURF, KAZE, BRISK };

    DetectorType m_detectorType;
public:
    cv::Ptr<cv::FeatureDetector> detector, extractor;

    FeatureDetector(std::string method, bool isUsingCUDA = false);

    void generateFeatures(cv::Mat& imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor);

    void generateFeatures(cv::Mat& imGray, cv::cuda::GpuMat& d_imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor);

    void generateFlowFeatures(cv::Mat& imGray, std::vector<cv::Point2f>& corners, int maxCorners, double qualityLevel, double minDistance);
};

class DescriptorMatcher : protected CUDAUsable {
public:
    const float m_ratioThreshold;

    cv::Ptr<cv::DescriptorMatcher> matcher;

    DescriptorMatcher(std::string method, const float ratioThreshold, bool isUsingCUDA = false);

    void ratioMaches(const cv::Mat lDesc, const cv::Mat rDesc, std::vector<cv::DMatch>& matches);

    void findRobustMatches(std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, cv::Mat prevDesc, cv::Mat currDesc, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<cv::DMatch>& matches, std::vector<int>& prevIdx, std::vector<int>& currIdx);
};

class OptFlowAddSettings {
public:
    float maxError, qualLvl, minDist;
    uint maxCorn, minFeatures;

    void setMaxError(float maxError) { this->maxError = maxError; }

    void setMaxCorners(uint maxCorn) { this->maxCorn = maxCorn; }

    void setQualityLvl(float qualLvl) { this->qualLvl = qualLvl; }

    void setMinDistance(float minDist) { this->minDist = minDist; }

    void setMinFeatures(uint minFeatures) { this->minFeatures = minFeatures; }
};

class OptFlow : protected CUDAUsable {
public:
    cv::Ptr<cv::SparsePyrLKOpticalFlow> optFlow;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_optFlow;

    OptFlowAddSettings additionalSettings;

    std::vector<uchar> statusMask;

    OptFlow(cv::TermCriteria termcrit, int winSize, int maxLevel, float maxError, uint maxCorners, float qualityLevel, float minCornersDistance, uint minFeatures, bool isUsingCUDA = false);

    void computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool useImageCorrection = false, bool useErrorCorrection = false);

    void drawOpticalFlow(cv::Mat inputImg, cv::Mat& outputImg, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, std::vector<uchar> statusMask);
};

class FlowView : public View {
public:
    std::vector<cv::Point2f> corners;

    void setCorners(const std::vector<cv::Point2f> corners) { this->corners = corners; }
};

class FeatureView : public View {
public:
    std::vector<cv::KeyPoint> keyPts;
    std::vector<cv::Point2f> pts;

    cv::Mat descriptor;

    void setFeatures(const std::vector<cv::KeyPoint> keyPts, const cv::Mat descriptor) {
        this->keyPts = keyPts;
        this->descriptor = descriptor;

        cv::KeyPoint::convert(this->keyPts, this->pts);
    }
};

#endif //FEATURE_PROCESSING_H
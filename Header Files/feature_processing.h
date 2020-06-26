#ifndef FEATURE_PROCESSING_H
#define FEATURE_PROCESSING_H
#pragma once

#include "pch.h"
#include "view.h"

class FeatureDetector {
private:
    enum DetectorType { AKAZE = 0, ORB, FAST, STAR, SIFT, SURF, KAZE, BRISK };

    DetectorType m_detectorType;
public:
    cv::Ptr<cv::FeatureDetector> detector, extractor;

    FeatureDetector(std::string method);

    /** 
     *  Generate features for triangulation
     * 
     *  It uses AKAZE, FAST, STAR, SIFT, SURF, KAZE, BRISK detector
     */
    void generateFeatures(cv::Mat& imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor);

    /** 
     *  Generate features for flow tracking
     * 
     *  It uses Shi-Tomasi corner detector
     */
    void generateFlowFeatures(cv::Mat& imGray, std::vector<cv::Point2f>& corners, int maxCorners, double qualityLevel, double minDistance);
};

class DescriptorMatcher {
private:
    void drawMatches(const cv::Mat prevFrame, const cv::Mat currFrame, cv::Mat& outFrame, std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, std::vector<cv::DMatch> matches, const std::string matchType);

public:
    const bool m_isVisDebug;

    const float m_ratioThreshold;

    const cv::Size m_visDebugWinSize;

    cv::Ptr<cv::DescriptorMatcher> matcher;

    DescriptorMatcher(std::string method, const float ratioThreshold, const bool isVisDebug = false, const cv::Size visDebugWinSize = cv::Size());

    /** 
     * Knn ratio match by threshold
     */
    void ratioMaches(const cv::Mat lDesc, const cv::Mat rDesc, std::vector<cv::DMatch>& matches);

    /** 
     * Robust matching by knn match, crossmatching, epipolar filter
     */
    void findRobustMatches(std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, cv::Mat prevDesc, cv::Mat currAligPts, std::vector<cv::Point2f>& prevAligPts, std::vector<cv::Point2f>& currPts, std::vector<cv::DMatch>& matches, std::vector<int>& prevPtsToKeyIdx, std::vector<int>& currPtsToKeyIdx, cv::Mat debugPrevFrame, cv::Mat debugCurrFrame, bool useEpipolarFilter);
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

class OptFlow {
private:
    void correctComputedPoints(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts);
public:
    cv::Ptr<cv::SparsePyrLKOpticalFlow> optFlow;

    OptFlowAddSettings additionalSettings;

    std::vector<uchar> statusMask;

    OptFlow(cv::TermCriteria termcrit, int winSize, int maxLevel, float maxError, uint maxCorners, float qualityLevel, float minCornersDistance, uint minFeatures);

    /** 
     *  Compute optical flow between grayscale images
     *  It also filters bad points -> depending on the settings
     * 
     *  @param useBoundaryCorrection filter by image boudary
     *  @param useErrorCorrection filter by optical flow error
     *  @param useOutliersCorrection filter by outliers
     */
    void computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask);

    void filterPoints(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, cv::Rect boundary, bool useBoundaryCorrection, bool useErrorCorrection);

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

struct PointsMove {
    float Q1, Q2, Q3;

    float lowerInFence, upperInFence;
    float lowerOutFence, upperOutFence;

    cv::Point2f medianMove;
};

class ProcesingAdds {
public:
    static void filterPointsByBoundary(std::vector<cv::Point2f>& points, const cv::Rect boundary);

    static void filterPointsByStatusMask(std::vector<cv::Point2f>& points, const std::vector<uchar>& statusMask);

    static void filterPointsByBoundary(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, const cv::Rect boundary);

    static void filterPointsByStatusMask(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, const std::vector<uchar>& statusMask);

    static void analyzePointsMove(std::vector<cv::Point2f>& inPrevPts, std::vector<cv::Point2f>& inCurrPts, PointsMove& outPointsMove);

    static void correctPointsByMoveAnalyze(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, PointsMove& pointsMove);
};

#endif //FEATURE_PROCESSING_H
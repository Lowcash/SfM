#ifndef USR_INP_MANAGER_H
#define USR_INP_MANAGER_H
#pragma once

#include "pch.h"
#include "reconstruction.h"

/** 
 * UserInput to managing selected points
 */
class UserInput {
private:
    const std::string m_winName;

    const float m_maxRange;
    
    const int m_pointSize;
    
    cv::Mat* m_inputImage;

    PointCloud* m_pointCloud;

    bool m_isClickPtsLocked;

    void drawSelectedPoint(const cv::Point point) {
        // draw red point to image
        cv::circle(*m_inputImage, point, m_pointSize, CV_RGB(200, 0, 0), cv::FILLED, cv::LINE_AA);
    }

    void drawRecoveredPoint(const cv::Point point) {
        // draw green point to image
        cv::circle(*m_inputImage, point, m_pointSize, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
    }
public:
    std::vector<size_t> usrCloudPtsIdx;

    std::vector<cv::Point2f> waitClickedPts;
    std::vector<cv::Point2f> doneClickedPts;
    std::vector<cv::Point2f> moveClickedPts;
    std::vector<cv::Point2f> doneUsrPts;
    std::vector<cv::Point2f> moveUsrPts;

    UserInput(const std::string winName, cv::Mat* imageSource, PointCloud* pointCloud, const float maxRange, const int pointSize = 5);

    void addClickedPoint(const cv::Point point, bool forceRedraw = false);

    /** 
     * Add points to 3D
     */
    void addPoints(const std::vector<cv::Point2f> pts2D, const std::vector<cv::Vec3d> pts3D, uint iter);

    void storeClickedPoints();
    
    /** 
     * It returns if there are any clicked user points
     */
    bool anyClickedPoint() const;

    /** 
     * It returns if there are any stored user points
     */
    bool anyUserPoint() const;

    void lockClickedPoints();

    void unlockClickedPoints();

    void updateWaitingPoints();

    /** 
     * It clears clicker points buffer
     * Usually after end of loop cycle iteration
     */
    void clearClickedPoints();

    /** 
     * Filter points by boundary
     */
    void filterPointsByBoundary(const cv::Rect boundary, const uint offset);

    /** 
     * Recover points from 2D
     */
    void recoverPoints(cv::Mat& imOutUsr);

    /** 
     * Recover points from 3D
     */
    void recoverPoints(cv::Mat& imOutUsr, cv::Mat cameraK, cv::Mat R, cv::Mat t);

    void attachPointsToMove(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool clickPts, bool usrPts);

    void detachPointsFromMove(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool clickPts, bool usrPts);
};

#endif //USR_INP_MANAGER_H
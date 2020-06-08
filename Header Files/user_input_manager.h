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
    bool m_isClickPtsLocked;

    const std::string m_winName;

    const float m_maxRange;
    
    const int m_pointSize;

    cv::Mat* m_inputImage;

    std::vector<cv::Point2f> m_tmpClickedPts;

    std::vector<cv::Point2f>* m_usrClickedPts;
    std::vector<cv::Point2f>* m_usr2dPts;

    std::vector<size_t> m_usrCloudPtsIdx;

    PointCloud* m_pointCloud;

    void drawSelectedPoint(const cv::Point point) {
        // draw red point to image
        cv::circle(*m_inputImage, point, m_pointSize, CV_RGB(200, 0, 0), cv::FILLED, cv::LINE_AA);
    }

    void drawRecoveredPoint(const cv::Point point) {
        // draw green point to image
        cv::circle(*m_inputImage, point, m_pointSize, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
    }
public:
    std::vector<std::vector<cv::Point2f>> usr2dPtsToMove;

    UserInput(const std::string winName, cv::Mat* imageSource, PointCloud* pointCloud, const float maxRange, const int pointSize = 5);

    void addClickedPoint(const cv::Point point, bool forceRedraw = false);

    /** 
     * Add points to 3D
     */
    void addPoints(const std::vector<cv::Point2f> pts2D, const std::vector<cv::Vec3d> pts3D, uint iter);

    void storeClickedPoints() const;
    
    /** 
     * It returns if there are any clicked user points
     */
    bool anyClickedPoint() const;

    void lockClickPoints();

    void unlockClickPoints();

    void updateClickedPoints();

    /** 
     * It returns if there are any stored user points
     */
    bool anyUserPoint() const;

    /** 
     * It clears clicker points buffer
     * Usually after end of loop cycle iteration
     */
    void clearClickedPoints() const;

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

    /** 
     * Attach points, usually to optical flow
     */
    void attachPointsToMove(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& move);

    /** 
     * Detach points, usually from optical flow
     */
    void detachPointsFromMove(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& move, uint numPtsToDetach) ;

    void detachPointsFromReconstruction(std::vector<cv::Vec3d>& points, std::vector<cv::Vec3d>& reconstPts, std::vector<cv::Vec3b>& reconstRGB, std::vector<bool>& reconstMask, uint numPtsToDetach);
};

struct UserInputDataParams {
public:
    UserInput* userInput;

    UserInputDataParams(UserInput* userInput) {
        this->userInput = userInput;
    }
};

#endif //USR_INP_MANAGER_H
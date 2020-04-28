#ifndef USR_INP_MANAGER_H
#define USR_INP_MANAGER_H
#pragma once

#include "pch.h"
#include "camera.h"

class UserInput {
private:
    const float m_maxRange;
public:
    std::vector<cv::Point2f> m_usrClickedPts2D;
    std::vector<cv::Point2f> m_usrPts2D;
    std::vector<cv::Vec3d> m_usrPts3D;
    
    UserInput(const float maxRange);

    void recoverPoints(cv::Mat R, cv::Mat t, Camera camera, cv::Mat& imOutUsr);

    void addPoints(const std::vector<cv::Vec3d> pts3D);

    void addPoints(const std::vector<cv::Point2f> prevPts2D, const std::vector<cv::Point2f> currPts2D);

    void updatePoints(const std::vector<cv::Point2f> currPts2D, const cv::Rect boundary, const uint offset);

    void recoverPoints(cv::Mat& imOutUsr);

    void recoverPoints(cv::Mat& imOutUsr, cv::Mat cameraK, cv::Mat R, cv::Mat t);

    void attachPointsToMove(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& move);

    void detachPointsFromMove(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& move, uint numPtsToDetach) ;

    void detachPointsFromReconstruction(std::vector<cv::Vec3d>& points, std::vector<cv::Vec3d>& reconstPts, std::vector<cv::Vec3b>& reconstRGB, std::vector<bool>& reconstMask, uint numPtsToDetach);
};

struct UserInputDataParams {
private:
    const std::string m_inputWinName;
public:
    UserInput* userInput;
    cv::Mat* inputImage;

    UserInputDataParams(const std::string inputWinName, cv::Mat* inputImage, UserInput* userInput)
        : m_inputWinName(inputWinName) {
        this->inputImage = inputImage;
        this->userInput = userInput;
    }

    std::string getWinName() const { return m_inputWinName; }
};

#endif //USR_INP_MANAGER_H
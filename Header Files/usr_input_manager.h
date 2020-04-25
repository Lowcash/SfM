#ifndef USR_INP_MANAGER_H
#define USR_INP_MANAGER_H
#pragma once

#include "pch.h"
#include "camera.h"

class UserInput {
private:
    const float m_maxRange;

    cv::Point2f m_medianMove;

    void computePointsRangeMove(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, std::map<std::pair<float, float>, float>& pointsDist, cv::Point2f& move) {
        for (int i = 0; i < prevPts.size(); ++i) {
            float dist = std::pow(
                std::pow(currPts[i].x - prevPts[i].x, 2) + 
                std::pow(currPts[i].y - prevPts[i].y, 2)
            , 0.5);

            if (dist < m_maxRange) {
                pointsDist[std::pair{currPts[i].x, currPts[i].y}] = dist;
                move += currPts[i] - prevPts[i];
            }
        }

        if (!pointsDist.empty()) {
            move.x /= pointsDist.size();
            move.y /= pointsDist.size();
        }
    }
public:
    std::vector<cv::Vec3d> m_usrPts3D;
    std::vector<cv::Point2f> m_usrPts2D;
    
    UserInput(const float maxRange)
        : m_maxRange(maxRange) {}

    void recoverPoints(cv::Mat R, cv::Mat t, Camera camera, cv::Mat& imOutUsr) {
        if (!m_usrPts3D.empty()) {
            std::vector<cv::Point2f> pts2D; cv::projectPoints(m_usrPts3D, R, t, camera._K, cv::Mat(), pts2D);

            for (const auto p : pts2D) {
                std::cout << "Point projected to: " << p << "\n";

                cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
            }
        }
    }

    void addPoints(const std::vector<cv::Vec3d> pts3D) {
        m_usrPts3D.insert(m_usrPts3D.end(), pts3D.begin(), pts3D.end());
    }

    void addPoints(const std::vector<cv::Point2f> prevPts2D, const std::vector<cv::Point2f> currPts2D) {
        std::map<std::pair<float, float>, float> pointsDist;

        cv::Point2f move; computePointsRangeMove(prevPts2D, currPts2D, pointsDist, move);

        for (auto [it, end, idx] = std::tuple{currPts2D.cbegin(), currPts2D.cend(), 0}; it != end; ++it, ++idx) {
            cv::Point2f p = (cv::Point2f)*it;
            
            //if (pointsDist.find(std::pair{p.x, p.y}) != pointsDist.end())
                m_usrPts2D.push_back(p);
            /*else {
                m_usrPts2D.push_back(prevPts[idx] + m_medianMove);
            }*/
        }
    }

    void updatePoints(const std::vector<cv::Point2f> currPts2D, const cv::Rect boundary, const uint offset) {
        std::map<std::pair<float, float>, float> pointsDist;

        cv::Point2f move; computePointsRangeMove(m_usrPts2D, currPts2D, pointsDist, move);

        for (auto [it, end, idx] = std::tuple{currPts2D.cbegin(), currPts2D.cend(), 0}; it != end; ++it, ++idx) {
            cv::Point2f p = (cv::Point2f)*it;
            
            //if (pointsDist.find(std::pair{p.x, p.y}) != pointsDist.end())
                m_usrPts2D[idx] = currPts2D[idx];
            /*else {
                m_usrPts2D[idx] = m_usrPts2D[idx] + m_medianMove;
            }*/
        }

        for (int i = 0, idxCorrection = 0; i < m_usrPts2D.size(); ++i) {
            auto p = m_usrPts2D[i];

            if (p.x < boundary.x + offset || p.y < boundary.y + offset || p.x > boundary.width - offset || p.y > boundary.height - offset) {
                m_usrPts2D.erase(m_usrPts2D.begin() + (i - idxCorrection));

                idxCorrection++;
            } 
        }
    }
    
    void updateMedianMove(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts) {
        std::map<std::pair<float, float>, float> pointsDist;

        computePointsRangeMove(prevPts, currPts, pointsDist, m_medianMove);

        std::cout << m_medianMove << "\n";
    }

    void recoverPoints(cv::Mat& imOutUsr) {
        for (const auto& p : m_usrPts2D) {
            //std::cout << "Point projected to: " << p << "\n";
                
            cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
        }
    }

    void recoverPoints(cv::Mat& imOutUsr, cv::Mat cameraK, cv::Mat R, cv::Mat t) {
        if (!m_usrPts3D.empty()) {
            cv::Mat recoveredPts;

            cv::projectPoints(m_usrPts3D, R, t, cameraK, cv::Mat(), recoveredPts);

            for (int i = 0; i < recoveredPts.rows; ++i) {
                auto p = recoveredPts.at<cv::Point2d>(i);
                //std::cout << "Point projected to: " << p << "\n";
                    
                cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
            }
        } 
    }
};

#endif //USR_INP_MANAGER_H
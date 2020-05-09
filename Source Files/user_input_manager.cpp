#include "user_input_manager.h"

UserInput::UserInput(const float maxRange)
    : m_maxRange(maxRange) {}

void UserInput::addPoints(const std::vector<cv::Point2f> pts2D) {
    for (auto [it, end, idx] = std::tuple{pts2D.cbegin(), pts2D.cend(), 0}; it != end; ++it, ++idx) {
        cv::Point2f p = (cv::Point2f)*it;
        
        usrPts2D.push_back(p);
    }
}

void UserInput::addPoints(const std::vector<cv::Point2f> pts2D, const std::vector<cv::Vec3d> pts3D, std::vector<cv::Vec3d>& pCloud, std::vector<cv::Vec3b>& pCloudRGB, std::vector<CloudTrack>& cloudTracks, uint iter) {
    for (auto [p2d, p2dEnd, p3d, p3dEnd] = std::tuple{pts2D.cbegin(), pts2D.cend(), pts3D.cbegin(), pts3D.cend()}; p2d != p2dEnd && p3d != p3dEnd; ++p2d, ++p3d) {
        cloudTracks.push_back(CloudTrack((cv::Point2f)*p2d, iter));

        m_usrCloudPts3DIdx.push_back(pCloud.size());

        pCloud.push_back((cv::Vec3d)*p3d);
        pCloudRGB.push_back(cv::Vec3b());
    }
}

void UserInput::filterPoints(const std::vector<cv::Point2f> pts2D, const cv::Rect boundary, const uint offset) {
    std::map<std::pair<float, float>, float> pointsDist;

    for (auto [it, end, idx] = std::tuple{pts2D.cbegin(), pts2D.cend(), 0}; it != end; ++it, ++idx) {
        cv::Point2f p = (cv::Point2f)*it;
        
        usrPts2D[idx] = pts2D[idx];
    }

    for (int i = 0, idxCorrection = 0; i < usrPts2D.size(); ++i) {
        auto p = usrPts2D[i];

        if (p.x < boundary.x + offset || p.y < boundary.y + offset || p.x > boundary.width - offset || p.y > boundary.height - offset) {
            usrPts2D.erase(usrPts2D.begin() + (i - idxCorrection));

            idxCorrection++;
        } 
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr) {
    for (const auto& p : usrPts2D) {
        //std::cout << "Point projected to: " << p << "\n";
            
        cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr, std::vector<cv::Vec3d>& pCloud, cv::Mat cameraK, cv::Mat R, cv::Mat t) {
    if (!pCloud.empty() && !m_usrCloudPts3DIdx.empty()) {
        std::vector<cv::Vec3d> usrPts3D;

        for (const auto& idx : m_usrCloudPts3DIdx)
            usrPts3D.push_back(pCloud[idx]);

        cv::Mat recoveredPts; cv::projectPoints(usrPts3D, R, t, cameraK, cv::Mat(), recoveredPts);

        for (int i = 0; i < recoveredPts.rows; ++i) {
            auto p = recoveredPts.at<cv::Point2d>(i);
            //std::cout << "Point projected to: " << p << "\n";
                
            cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
        }
    } 
}

void UserInput::attachPointsToMove(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& move) {
    if (!points.empty()) {
        move.insert(move.end(), points.begin(), points.end());
    }
}

void UserInput::detachPointsFromMove(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& move, uint numPtsToDetach) {
    points.insert(points.end(), move.end() - numPtsToDetach, move.end());

    for (int i = 0; i < numPtsToDetach; ++i)
        move.pop_back();
}

void UserInput::detachPointsFromReconstruction(std::vector<cv::Vec3d>& points, std::vector<cv::Vec3d>& reconstPts, std::vector<cv::Vec3b>& reconstRGB, std::vector<bool>& reconstMask, uint numPtsToDetach) {
    points.insert(points.end(), reconstPts.end() - numPtsToDetach, reconstPts.end());

    for (int i = 0; i < numPtsToDetach; ++i) {
        reconstPts.pop_back();
        reconstRGB.pop_back();
        reconstMask.pop_back();
    }
}
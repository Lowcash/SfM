#include "user_input_manager.h"

UserInput::UserInput(const std::string winName, cv::Mat* imageSource, const float maxRange, const int pointSize)
    : m_winName(winName), m_inputImage(imageSource), m_maxRange(maxRange), m_pointSize(pointSize) {}

void UserInput::addClickedPoint(const cv::Point point, bool forceRedraw) { 
    usrClickedPts2D.push_back(point);

    if (forceRedraw) {
        drawSelectedPoint(point);

        cv::imshow(m_winName, *m_inputImage);
    }
}

void UserInput::addPoints(const std::vector<cv::Point2f> pts2D) {
    usrPts2D.insert(usrPts2D.end(), pts2D.begin(), pts2D.end());
}

void UserInput::addPoints(const std::vector<cv::Point2f> pts2D, const std::vector<cv::Vec3d> pts3D, PointCloud& pointCloud, uint iter) {
    for (auto [p2d, p2dEnd, p3d, p3dEnd] = std::tuple{pts2D.cbegin(), pts2D.cend(), pts3D.cbegin(), pts3D.cend()}; p2d != p2dEnd && p3d != p3dEnd; ++p2d, ++p3d) {
        m_usrCloudPtsIdx.push_back(pointCloud.cloudTracks.size());

        pointCloud.addCloudPoint(*p2d, *p3d, cv::Vec3b(), iter);
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
        
        drawRecoveredPoint(p);
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr, PointCloud& pointCloud, cv::Mat cameraK, cv::Mat R, cv::Mat t) {
    if (!pointCloud.cloud3D.empty() && !m_usrCloudPtsIdx.empty()) {
        std::vector<cv::Vec3d> usrPts3D;

        for (const auto& idx : m_usrCloudPtsIdx) {
            cv::Vec3d* cloudMapper = pointCloud.cloudMapper[idx];

            usrPts3D.push_back(*cloudMapper);
        }

        cv::Mat recoveredPts; cv::projectPoints(usrPts3D, R, t, cameraK, cv::Mat(), recoveredPts);

        for (int i = 0; i < recoveredPts.rows; ++i) {
            auto p = recoveredPts.at<cv::Point2d>(i);
            //std::cout << "Point projected to: " << p << "\n";
                
            drawRecoveredPoint(p);
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
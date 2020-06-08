#include "user_input_manager.h"

UserInput::UserInput(const std::string winName, cv::Mat* imageSource, PointCloud* pointCloud, const float maxRange, const int pointSize)
    : m_winName(winName), m_inputImage(imageSource), m_maxRange(maxRange), m_pointSize(pointSize) {
    m_isClickPtsLocked = false;
    
    m_pointCloud = pointCloud;

    usr2dPtsToMove.resize(2);

    m_usrClickedPts = &usr2dPtsToMove[0];
    m_usr2dPts = &usr2dPtsToMove[1];
}

void UserInput::addClickedPoint(const cv::Point point, bool forceRedraw) {
    if (m_isClickPtsLocked) 
        m_tmpClickedPts.push_back(point);
    else
        m_usrClickedPts->push_back(point);

    if (forceRedraw) {
        drawSelectedPoint(point);

        cv::imshow(m_winName, *m_inputImage);
    }
}

void UserInput::addPoints(const std::vector<cv::Point2f> pts2D, const std::vector<cv::Vec3d> pts3D, uint iter) {
    for (auto [p2d, p2dEnd, p3d, p3dEnd] = std::tuple{pts2D.cbegin(), pts2D.cend(), pts3D.cbegin(), pts3D.cend()}; p2d != p2dEnd && p3d != p3dEnd; ++p2d, ++p3d) {
        m_usrCloudPtsIdx.push_back(m_pointCloud->getNumCloudPoints());

        m_pointCloud->addCloudPoint(*p2d, *p3d, cv::Vec3b(), iter);
    }
}

void UserInput::lockClickPoints() { m_isClickPtsLocked = true; }

void UserInput::unlockClickPoints() { m_isClickPtsLocked = false; }

void UserInput::updateClickedPoints() { 
    std::swap(m_tmpClickedPts, *m_usrClickedPts);
}

void UserInput::storeClickedPoints() const {
    m_usr2dPts->insert(m_usr2dPts->end(), m_usrClickedPts->begin(), m_usrClickedPts->end());
}

bool UserInput::anyClickedPoint() const {
    return !m_usrClickedPts->empty();
}

bool UserInput::anyUserPoint() const {
    return !m_usr2dPts->empty();
}

void UserInput::clearClickedPoints() const {
    m_usrClickedPts->clear();
}

void UserInput::filterPointsByBoundary(const cv::Rect boundary, const uint offset) {
    for (int i = 0, idxCorrection = 0; i < m_usr2dPts->size(); ++i) {
        auto p = m_usr2dPts->at(i);

        if (p.x < boundary.x + offset || p.y < boundary.y + offset || p.x > boundary.width - offset || p.y > boundary.height - offset) {
            m_usr2dPts->erase(m_usr2dPts->begin() + (i - idxCorrection));

            idxCorrection++;
        } 
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr) {
    for (const auto& p : *m_usr2dPts) {
        //std::cout << "Point projected to: " << p << "\n";
        
        drawRecoveredPoint(p);
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr, cv::Mat cameraK, cv::Mat R, cv::Mat t) {
    if (!m_pointCloud->cloud3D.empty() && !m_usrCloudPtsIdx.empty()) {
        std::vector<cv::Vec3d> usrPts3D;

        for (const auto& idx : m_usrCloudPtsIdx) {
            cv::Vec3d* cloudMapper = m_pointCloud->cloudMapper[idx];

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
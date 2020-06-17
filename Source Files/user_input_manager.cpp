#include "user_input_manager.h"

UserInput::UserInput(const std::string winName, cv::Mat* imageSource, PointCloud* pointCloud, const float maxRange, const int pointSize)
    : m_winName(winName), m_inputImage(imageSource), m_maxRange(maxRange), m_pointSize(pointSize), m_pointCloud(pointCloud), m_isClickPtsLocked(false) {}

void UserInput::addClickedPoint(const cv::Point point, bool forceRedraw) {
    if (m_isClickPtsLocked) 
        waitClickedPts.push_back(point);
    else
        doneClickedPts.push_back(point);

    if (forceRedraw) {
        drawSelectedPoint(point);

        cv::imshow(m_winName, *m_inputImage);

        // render
        cv::waitKey(1);
    }
}

void UserInput::addPoints(const std::vector<cv::Point2f> pts2D, const std::vector<cv::Vec3d> pts3D, uint iter) {
    for (auto [p2d, p2dEnd, p3d, p3dEnd] = std::tuple{pts2D.cbegin(), pts2D.cend(), pts3D.cbegin(), pts3D.cend()}; p2d != p2dEnd && p3d != p3dEnd; ++p2d, ++p3d) {
        usrCloudPtsIdx.push_back(m_pointCloud->getNumCloudPoints());

        m_pointCloud->addCloudPoint(*p2d, *p3d, cv::Vec3b(), iter);
    }
}

void UserInput::lockClickedPoints() { m_isClickPtsLocked = true; }

void UserInput::unlockClickedPoints() { m_isClickPtsLocked = false; }

void UserInput::updateWaitingPoints() {
    doneClickedPts.insert(doneClickedPts.end(), waitClickedPts.begin(), waitClickedPts.end());
    
    waitClickedPts.clear();
}

void UserInput::storeClickedPoints() {
    doneUsrPts.insert(doneUsrPts.end(), doneClickedPts.begin(), doneClickedPts.end());
}

bool UserInput::anyClickedPoint() const {
    return !doneClickedPts.empty();
}

bool UserInput::anyUserPoint() const {
    return !doneUsrPts.empty();
}

void UserInput::clearClickedPoints() {
    doneClickedPts.clear();
    moveClickedPts.clear();
}

void UserInput::filterPointsByBoundary(const cv::Rect boundary, const uint offset) {
    for (int i = 0, idxCorrection = 0; i < doneUsrPts.size(); ++i) {
        auto p = doneUsrPts.at(i);

        if (p.x < boundary.x + offset || p.y < boundary.y + offset || p.x > boundary.width - offset || p.y > boundary.height - offset) {
            doneUsrPts.erase(doneUsrPts.begin() + (i - idxCorrection));

            idxCorrection++;
        } 
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr) {
    for (const auto& p : doneUsrPts) {
        //std::cout << "Point projected to: " << p << "\n";
        
        drawRecoveredPoint(p);
    }
}

void UserInput::recoverPoints(cv::Mat& imOutUsr, cv::Mat cameraK, cv::Mat R, cv::Mat t) {
    if (!m_pointCloud->cloud3D.empty() && !usrCloudPtsIdx.empty()) {
        std::vector<cv::Vec3d> usrPts3D;

        for (const auto& idx : usrCloudPtsIdx) {
            cv::Vec3d* cloudMapper = m_pointCloud->cloudMapper[idx];

            usrPts3D.push_back(*cloudMapper);

            m_pointCloud->cloudMask[idx] = true;
        }

        cv::Mat recoveredPts; cv::projectPoints(usrPts3D, R, t, cameraK, cv::Mat(), recoveredPts);

        for (int i = 0; i < recoveredPts.rows; ++i) {
            auto p = recoveredPts.at<cv::Point2d>(i);
            //std::cout << "Point projected to: " << p << "\n";
                
            drawRecoveredPoint(p);
        }
    } 
}

void UserInput::attachPointsToMove(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool clickPts, bool usrPts) {
    if (!prevPts.empty()) {
        if (clickPts)
            prevPts.insert(prevPts.end(), doneClickedPts.begin(), doneClickedPts.end());
        if (usrPts)
            prevPts.insert(prevPts.end(), doneUsrPts.begin(), doneUsrPts.end());
    }
}

void UserInput::detachPointsFromMove(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool clickPts, bool usrPts) {
    size_t _ptsSize = 0;

    if (!currPts.empty()) {
        if (usrPts) {
            moveUsrPts.insert(moveUsrPts.end(), currPts.end() - doneUsrPts.size(), currPts.end());

            _ptsSize += doneUsrPts.size();
        }
        if (clickPts) {
            moveClickedPts.insert(moveClickedPts.end(), currPts.end() - doneClickedPts.size(), currPts.end());

            _ptsSize += doneClickedPts.size();
        }
    }

    for (int i = 0; i < _ptsSize; ++i) {
        prevPts.pop_back();
        currPts.pop_back();
        statusMask.pop_back();
    }
}

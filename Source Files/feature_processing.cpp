#include "feature_processing.h"

FeatureDetector::FeatureDetector(std::string method) {
    std::for_each(method.begin(), method.end(), [](char& c){
        c = ::toupper(c);
    });

    // std::string to Enum mapping by Mark Ransom
    // https://stackoverflow.com/questions/7163069/c-string-to-enum
    static std::unordered_map<std::string, DetectorType> const table = { 
        {"ORB", DetectorType::ORB}, 
        {"FAST", DetectorType::FAST},  
        {"STAR", DetectorType::STAR},
        {"SIFT", DetectorType::SIFT}, 
        {"SURF", DetectorType::SURF}, 
        {"KAZE", DetectorType::KAZE}, 
        {"BRISK", DetectorType::BRISK}, 
        {"AKAZE", DetectorType::AKAZE}
    };

    if (auto it = table.find(method); it != table.end())
        m_detectorType = it->second;
    else
        m_detectorType = DetectorType::AKAZE;

    switch (m_detectorType) {
        case DetectorType::AKAZE: {
            detector = extractor = cv::AKAZE::create(cv::AKAZE::DescriptorType::DESCRIPTOR_MLDB, 0, 3, 0.001f, 3, 3);

            break;
        }
        case DetectorType::STAR: {
            detector = cv::xfeatures2d::StarDetector::create();
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

            break;
        }
        case DetectorType::SIFT: {
            detector = extractor = cv::xfeatures2d::SIFT::create();

            break;
        }

        case DetectorType::ORB: {
            detector = extractor = cv::ORB::create(2000);

            break;
        }
        
        case DetectorType::FAST: {
            detector = cv::FastFeatureDetector::create();

            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

            break;
        }
        
        case DetectorType::SURF: {
            detector = extractor = cv::xfeatures2d::SURF::create();

            break;
        }
        case DetectorType::KAZE: {
            detector = extractor = cv::KAZE::create();

            break;
        }
        case DetectorType::BRISK: {
            detector = extractor = cv::BRISK::create();
            
            break;
        }
    }
}

void FeatureDetector::generateFeatures(cv::Mat& imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor) {
    //  detectAndCompute is faster than detect/compute
    //  use detectAndCompute if same detector and extractor
    if (detector != extractor){
        detector->detect(imGray, keyPts);
        extractor->compute(imGray, keyPts, descriptor);
    } else
        detector->detectAndCompute(imGray, cv::noArray(), keyPts, descriptor);
}

void FeatureDetector::generateFlowFeatures(cv::Mat& imGray, std::vector<cv::Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
    std::vector<cv::Point2f> _corners;

    std::cout << "Generating flow features..." << std::flush;

    //  Use Shi-Tomasi corner detector
    cv::goodFeaturesToTrack(imGray, _corners, maxCorners, qualityLevel, minDistance);

    // Add new points at the end. Do not remove good points
    corners.insert(corners.end(), _corners.begin(), _corners.end());

    std::cout << "[DONE]";
}

DescriptorMatcher::DescriptorMatcher(std::string method, const float ratioThreshold, const bool isVisDebug, const cv::Size visDebugWinSize)
    : m_ratioThreshold(ratioThreshold), m_isVisDebug(isVisDebug), m_visDebugWinSize(cv::Size(visDebugWinSize.width * 2, visDebugWinSize.height * 3)) {

    std::for_each(method.begin(), method.end(), [](char& c){
        c = ::toupper(c);
    });

    if (method == "BRUTEFORCE_HAMMING")
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);
    else if (method == "BRUTEFORCE_SL2")
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_SL2);
    else if (method == "FLANNBASED")
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::FLANNBASED);
    else 
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE);
}

void DescriptorMatcher::ratioMaches(const cv::Mat lDesc, const cv::Mat rDesc, std::vector<cv::DMatch>& matches) {
    std::vector<std::vector<cv::DMatch>> knnMatches;

    matcher->knnMatch(lDesc, rDesc, knnMatches, 2);

    matches.clear();

    //  Match by knn the neighbor and the second best neighbor ratio
    for (const auto& k : knnMatches) {
        if (k[0].distance < m_ratioThreshold * k[1].distance) 
            matches.push_back(k[0]);
    }
}

void DescriptorMatcher::drawMatches(const cv::Mat prevFrame, const cv::Mat currFrame, cv::Mat& outFrame, std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, std::vector<cv::DMatch> matches, const std::string matchType) {
    cv::drawMatches(prevFrame, prevKeyPts, currFrame, currKeyPts, matches, outFrame, CV_RGB(255,255,0));

    const std::string headedText = matchType;
    const std::string matchesText = "# Matches: " + std::to_string(matches.size());

    cv::rectangle(outFrame, cv::Rect(10, 20, 300, 100), cv::Scalar(75,75,75), cv::FILLED);

    cv::putText(outFrame, headedText, cv::Point(10,50), cv::FONT_HERSHEY_COMPLEX, 1.0, CV_RGB(255, 255, 255), 2);
    cv::putText(outFrame, matchesText, cv::Point(10,100), cv::FONT_HERSHEY_COMPLEX, 1.0, CV_RGB(255, 255, 255));
}

void DescriptorMatcher::findRobustMatches(std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, cv::Mat prevDesc, cv::Mat currDesc, std::vector<cv::Point2f>& prevAligPts, std::vector<cv::Point2f>& currAligPts, std::vector<cv::DMatch>& matches, std::vector<int>& prevPtsToKeyIdx, std::vector<int>& currPtsToKeyIdx, cv::Mat debugPrevFrame, cv::Mat debugCurrFrame) {
    std::vector<cv::DMatch> fMatches, bMatches;

    // knn matches
    ratioMaches(prevDesc, currDesc, fMatches);
    ratioMaches(currDesc, prevDesc, bMatches);

    cv::Mat _imKnnMatch, _imCrossMatching, _imEpipolarFilter, _imOutFilter;     
    if (m_isVisDebug && (!debugPrevFrame.empty() && !debugCurrFrame.empty())) {
        const std::string _matchHeader = "Knn Match";

        drawMatches(debugPrevFrame, debugCurrFrame, _imKnnMatch, prevKeyPts, currKeyPts, fMatches, _matchHeader);
    }

    // crossmatching
    for (const auto& bM : bMatches) {
        bool isFound = false;

        for (const auto& fM : fMatches) {
            if (bM.queryIdx == fM.trainIdx && bM.trainIdx == fM.queryIdx) {
                prevAligPts.push_back(prevKeyPts[fM.queryIdx].pt);
                currAligPts.push_back(currKeyPts[fM.trainIdx].pt);

                matches.push_back(fM);

                isFound = true;

                break;
            }
        }

        if (isFound) { continue; }
    }

    if (m_isVisDebug && (!debugPrevFrame.empty() && !debugCurrFrame.empty())) {
        const std::string _matchHeader = "CrossMatching";

        drawMatches(debugPrevFrame, debugCurrFrame, _imCrossMatching, prevKeyPts, currKeyPts, matches, _matchHeader);
    }

    if (prevAligPts.empty() || currAligPts.empty()) { return; }

    // epipolar filter -> use fundamental mat
    std::vector<uint8_t> inliersMask(matches.size()); 
    cv::findFundamentalMat(prevAligPts, currAligPts, inliersMask);
    
    std::vector<cv::DMatch> _epipolarMatch;
    std::vector<cv::Point2f> _epipolarPrevPts, _epipolarCurrPts;

    for (size_t m = 0; m < matches.size(); ++m) {
        // filter by fundamental mask
        //if (inliersMask[m]) {
            _epipolarPrevPts.push_back(prevKeyPts[matches[m].queryIdx].pt);
            _epipolarCurrPts.push_back(currKeyPts[matches[m].trainIdx].pt);

            prevPtsToKeyIdx.push_back(matches[m].queryIdx);
            currPtsToKeyIdx.push_back(matches[m].trainIdx);

            _epipolarMatch.push_back(matches[m]);
        //}
    }

    if (m_isVisDebug && (!debugPrevFrame.empty() && !debugCurrFrame.empty())) {
        const std::string _matchHeader = "Epipolar filter";

        drawMatches(debugPrevFrame, debugCurrFrame, _imEpipolarFilter, prevKeyPts, currKeyPts, _epipolarMatch, _matchHeader);

        _imKnnMatch.copyTo(_imOutFilter);
        cv::vconcat(_imOutFilter, _imCrossMatching, _imOutFilter);
        cv::vconcat(_imOutFilter, _imEpipolarFilter, _imOutFilter);

        cv::resize(_imOutFilter, _imOutFilter, m_visDebugWinSize);

        cv::imshow("Matches", _imOutFilter);

        cv::waitKey(29);
    }
    
    //  update informations to output structures
    std::swap(matches, _epipolarMatch);
    std::swap(prevAligPts, _epipolarPrevPts);
    std::swap(currAligPts, _epipolarCurrPts);
}

OptFlow::OptFlow(cv::TermCriteria termcrit, int winSize, int maxLevel, float maxError, uint maxCorners, float qualityLevel, float minCornersDistance, uint minFeatures) {
    optFlow = cv::SparsePyrLKOpticalFlow::create(cv::Size(winSize, winSize), maxLevel, termcrit);

    additionalSettings.setMaxError(maxError);
    additionalSettings.setMaxCorners(maxCorners);
    additionalSettings.setQualityLvl(qualityLevel);
    additionalSettings.setMinDistance(minCornersDistance);
    additionalSettings.setMinFeatures(minFeatures);
}

void OptFlow::computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask) {
    std::vector<float> _err;

    optFlow->calc(imPrevGray, imCurrGray, prevPts, currPts, statusMask, _err);

    const size_t numPoints = statusMask.size();

    for (size_t i = 0; i < numPoints; ++i) {
        if (_err[i] > additionalSettings.maxError) 
            statusMask[i] = false;
    }
}

void OptFlow::correctComputedPoints(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
    if (prevPts.size() != currPts.size()) {
        std::cout << "The number of points is not the same!" << "\n";

        return; 
    }

    if (prevPts.size() < 4) {
        std::cout << "Not enough points for correction!" << "\n";
        
        return; 
    }

    const size_t numPoints = prevPts.size();

    std::vector<float> dist(numPoints);
    std::map<float, size_t> distIdxMap;

    for (size_t i = 0; i < numPoints; ++i) {
        dist[i] = cv::norm(prevPts[i] - currPts[i]);

        distIdxMap[dist[i]] = i;
    }

    std::sort(dist.begin(), dist.end(), std::less<float>());

    const uint quarter = (dist.size() / 4);

    const float Q1 = (dist[quarter * 1 - 1] + dist[quarter * 1]) / 2.0f;
    const float Q2 = (dist[quarter * 2 - 1] + dist[quarter * 2]) / 2.0f;
    const float Q3 = (dist[quarter * 3 - 1] + dist[quarter * 3]) / 2.0f;

    const float IQR = (Q3 - Q1) * 3.0f;

    const float dOutFence = Q1 - IQR;
    const float uOutFence = Q3 + IQR;

    const cv::Point2f medianMove = currPts[distIdxMap[dist[quarter * 2 - 1]]] - prevPts[distIdxMap[dist[quarter * 2 - 1]]];

    for (size_t i = 0; i < numPoints; ++i) {
        // point distance
        const float dist = cv::norm(prevPts[i] - currPts[i]);

        // Correct by provided parameters
        bool isOutliOK = numPoints < 4 || (dist >= dOutFence && dist <= uOutFence);

        if (!isOutliOK)
            currPts[i] = prevPts[i] + medianMove;
    }
}

void OptFlow::drawOpticalFlow(cv::Mat inputImg, cv::Mat& outputImg, const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, std::vector<uchar> statusMask) {
    bool isUsingMask = statusMask.size() == prevPts.size();

    inputImg.copyTo(outputImg);


    for (int i = 0; i < prevPts.size() && i < currPts.size(); ++i) {
        cv::Point2f dir = (prevPts[i] - currPts[i]) * 5;
        //  Green arrow -> good optical flow (mask, error)
        cv::arrowedLine(outputImg, currPts[i], currPts[i] + dir, CV_RGB(0,200,0), 1, cv::LineTypes::LINE_AA);

        //  Red arrow -> bad optical flow (mask, error)
        if (isUsingMask && statusMask[i] == 0)
            cv::arrowedLine(outputImg, currPts[i], currPts[i] + dir, CV_RGB(200,0,0),1, cv::LineTypes::LINE_AA);
    }
}

void ProcesingAdds::filterPointsByBoundary(std::vector<cv::Point2f>& points, const cv::Rect boundary) {
    std::vector<cv::Point2f> _emptyPoints;

    filterPointsByBoundary(_emptyPoints, points, boundary);
}

void ProcesingAdds::filterPointsByStatusMask(std::vector<cv::Point2f>& points, const std::vector<uchar>& statusMask) {
    std::vector<cv::Point2f> _emptyPoints;

    filterPointsByStatusMask(_emptyPoints, points, statusMask);
}

void ProcesingAdds::filterPointsByBoundary(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, const cv::Rect boundary) {
    const size_t numPoints = currPts.size();

    for (size_t i = 0, idxCorrection = 0; i < numPoints; ++i) {
        cv::Point2f pt = currPts[i - idxCorrection];

        if (!boundary.contains(pt)) {
            currPts.erase(currPts.begin() + (i - idxCorrection));
            
            if (!prevPts.empty())
                prevPts.erase(prevPts.begin() + (i - idxCorrection));
        }
    }
}

void ProcesingAdds::filterPointsByStatusMask(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, const std::vector<uchar>& statusMask) {
    const size_t numPoints = currPts.size();

    for (size_t i = 0, idxCorrection = 0; i < numPoints; ++i) {
        cv::Point2f pt = currPts[i - idxCorrection];

        if (!statusMask[i]) {
            currPts.erase(currPts.begin() + (i - idxCorrection));
            
            if (!prevPts.empty())
                prevPts.erase(prevPts.begin() + (i - idxCorrection));
        }
    }
}

void ProcesingAdds::analyzePointsMove(std::vector<cv::Point2f>& inPrevPts, std::vector<cv::Point2f>& inCurrPts, PointsMove& outPointsMove) {
    if (inPrevPts.size() != inCurrPts.size()) {
        std::cout << "The number of points is not the same!" << "\n";

        return; 
    }

    if (inPrevPts.size() < 4) {
        std::cout << "Not enough points for correction!" << "\n";
        
        return; 
    }

    const size_t numPoints = inPrevPts.size();

    std::vector<float> dist(numPoints);
    std::map<float, size_t> distIdxMap;

    for (size_t i = 0; i < numPoints; ++i) {
        dist[i] = cv::norm(inPrevPts[i] - inCurrPts[i]);

        distIdxMap[dist[i]] = i;
    }

    std::sort(dist.begin(), dist.end(), std::less<float>());

    const uint quarter = (dist.size() / 4);

    outPointsMove.Q1 = (dist[quarter * 1 - 1] + dist[quarter * 1]) / 2.0f;
    outPointsMove.Q2 = (dist[quarter * 2 - 1] + dist[quarter * 2]) / 2.0f;
    outPointsMove.Q3 = (dist[quarter * 3 - 1] + dist[quarter * 3]) / 2.0f;

    const float IQR = (outPointsMove.Q3 - outPointsMove.Q1);

    outPointsMove.lowerInFence = outPointsMove.Q1 - IQR * 1.5f;
    outPointsMove.upperInFence = outPointsMove.Q3 + IQR * 1.5f;

    outPointsMove.lowerOutFence = outPointsMove.Q1 - IQR * 3.0f;
    outPointsMove.upperOutFence = outPointsMove.Q3 + IQR * 3.0f;

    outPointsMove.medianMove = inCurrPts[distIdxMap[dist[quarter * 2 - 1]]] - inPrevPts[distIdxMap[dist[quarter * 2 - 1]]];
}

void ProcesingAdds::correctPointsByMoveAnalyze(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, PointsMove& pointsMove) {
    if (prevPts.size() != currPts.size()) {
        //std::cout << "The number of points is not the same!" << "\n";

        return; 
    }

    if (prevPts.size() < 4) {
        //std::cout << "Not enough points for correction!" << "\n";
        
        return; 
    }

    const size_t numPoints = prevPts.size();

    for (size_t i = 0; i < numPoints; ++i) {
        // point distance
        const float dist = cv::norm(prevPts[i] - currPts[i]);

        // correct points by median move, if they are not inside fence range
        if (!(numPoints < 4 || (dist >= pointsMove.lowerOutFence && dist <= pointsMove.upperOutFence))) {

            currPts[i] = prevPts[i] + pointsMove.medianMove;
        }
    }
}
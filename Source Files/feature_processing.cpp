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
            detector = extractor = cv::AKAZE::create(cv::AKAZE::DescriptorType::DESCRIPTOR_MLDB, 0, 3, 0.001f, 3, 3, cv::KAZE::DiffusivityType::DIFF_WEICKERT);

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
            detector = extractor = cv::ORB::create();

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
    : m_ratioThreshold(ratioThreshold), m_isVisDebug(isVisDebug), m_visDebugWinSize(cv::Size(visDebugWinSize.width * 2, visDebugWinSize.height)) {

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
        
    if (m_isVisDebug && (!debugPrevFrame.empty() && !debugCurrFrame.empty())) {
        const std::string _matchHeader = "Knn Match";
        cv::Mat _imKnnMatch;

        drawMatches(debugPrevFrame, debugCurrFrame, _imKnnMatch, prevKeyPts, currKeyPts, fMatches, _matchHeader);

        cv::resize(_imKnnMatch, _imKnnMatch, m_visDebugWinSize);

        cv::imshow(_matchHeader, _imKnnMatch);

        cv::waitKey(29);
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
        cv::Mat _imCrossMatching;

        drawMatches(debugPrevFrame, debugCurrFrame, _imCrossMatching, prevKeyPts, currKeyPts, matches, _matchHeader);

        cv::resize(_imCrossMatching, _imCrossMatching, m_visDebugWinSize);

        cv::imshow(_matchHeader, _imCrossMatching);

        cv::waitKey(29);
    }

    if (prevAligPts.empty() || currAligPts.empty()) { return; }

    // epipolar filter -> use fundamental mat
    std::vector<uint8_t> inliersMask(matches.size()); 
    cv::findFundamentalMat(prevAligPts, currAligPts, inliersMask);
    
    std::vector<cv::DMatch> _epipolarMatch;
    std::vector<cv::Point2f> _epipolarPrevPts, _epipolarCurrPts;

    for (size_t m = 0; m < matches.size(); ++m) {
        // filter by fundamental mask
        if (inliersMask[m]) {
            _epipolarPrevPts.push_back(prevKeyPts[matches[m].queryIdx].pt);
            _epipolarCurrPts.push_back(currKeyPts[matches[m].trainIdx].pt);

            prevPtsToKeyIdx.push_back(matches[m].queryIdx);
            currPtsToKeyIdx.push_back(matches[m].trainIdx);

            _epipolarMatch.push_back(matches[m]);
        }
    }

    if (m_isVisDebug && (!debugPrevFrame.empty() && !debugCurrFrame.empty())) {
        const std::string _matchHeader = "Epipolar filter";
        cv::Mat _imEpipolarFilter;

        drawMatches(debugPrevFrame, debugCurrFrame, _imEpipolarFilter, prevKeyPts, currKeyPts, _epipolarMatch, _matchHeader);

        cv::resize(_imEpipolarFilter, _imEpipolarFilter, m_visDebugWinSize);

        cv::imshow(_matchHeader, _imEpipolarFilter);

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

void OptFlow::computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool useBoundaryCorrection, bool useErrorCorrection, bool useDistanceCorrection) {
    std::vector<std::vector<cv::Point2f>> _emptyAddPts;

    computeFlow(imPrevGray, imCurrGray, prevPts, currPts, _emptyAddPts, statusMask, useBoundaryCorrection, useErrorCorrection, useDistanceCorrection);
}

void OptFlow::computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<std::vector<cv::Point2f>>& addPts, std::vector<uchar>& statusMask, bool useBoundaryCorrection, bool useErrorCorrection, bool useDistanceCorrection) {
    std::vector<float> _err;
    
    if (!addPts.empty()) {
        for (auto &p : addPts) {
            if (!p.empty()) {
                // assign additional points for flow computing
                prevPts.insert(prevPts.end(), p.begin(), p.end());
            }
        }

        std::reverse(addPts.begin(), addPts.end());
    }
    
    optFlow->calc(imPrevGray, imCurrGray, prevPts, currPts, statusMask, _err);

    //  Image boundary filter
    cv::Rect boundary(cv::Point(), imCurrGray.size());

    if (!addPts.empty()) {
        for (auto& p : addPts) {
            if (!p.empty()) {
                std::vector<float> _addErr;
                std::vector<uchar> _addStatusMask;
                std::vector<cv::Point2f> _addMovedPts;

                _addErr.insert(_addErr.end(), _err.end() - p.size(), _err.end());
                _addStatusMask.insert(_addStatusMask.end(), statusMask.end() - p.size(), statusMask.end());
                _addMovedPts.insert(_addMovedPts.end(), currPts.end() - p.size(), currPts.end());

                for (int i = 0; i < p.size(); ++i) {
                    _err.pop_back();
                    statusMask.pop_back();
                    currPts.pop_back();
                }

                filterComputedPoints(p, _addMovedPts, _addStatusMask, _addErr, boundary, useBoundaryCorrection, useErrorCorrection, useDistanceCorrection);

                std::swap(p, _addMovedPts);
            }
        }

        std::reverse(addPts.begin(), addPts.end());
    }

    filterComputedPoints(prevPts, currPts, statusMask, _err, boundary, useBoundaryCorrection, useErrorCorrection, useDistanceCorrection);
}

void OptFlow::filterComputedPoints(std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, std::vector<float> err, cv::Rect boundary, bool useBoundaryCorrection, bool useErrorCorrection, bool useDistanceCorrection) {
    for (uint i = 0, idxCorrection = 0; i < statusMask.size() && i < err.size(); ++i) {
        cv::Point2f pt = currPts[i - idxCorrection];

        //  Filter by provided parameters
        bool isErrorOK = statusMask[i] == 1 && err[i] < additionalSettings.maxError;
        bool isBoundOK = boundary.contains(pt);

        if (!isErrorOK || !isBoundOK) {
            if ((useErrorCorrection && !isErrorOK) || (useBoundaryCorrection && !isBoundOK)) {
                //  Throw away bad points
                prevPts.erase(prevPts.begin() + (i - idxCorrection));
                currPts.erase(currPts.begin() + (i - idxCorrection));

                idxCorrection++;
            } else
                statusMask[i] = 0;
        }

        bool isDistaOK = cv::norm(prevPts[i] - currPts[i]) < additionalSettings.maxError;

        if(!isDistaOK) {
            if ((useDistanceCorrection && !isDistaOK)) {
                currPts[i] = prevPts[i];
            }
        }
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
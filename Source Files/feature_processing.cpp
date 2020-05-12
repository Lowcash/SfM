#include "feature_processing.h"

FeatureDetector::FeatureDetector(std::string method, bool isUsingCUDA) {
    m_isUsingCUDA = isUsingCUDA;

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
            m_isUsingCUDA = false;

            detector = extractor = cv::AKAZE::create();

            break;
        }
        case DetectorType::STAR: {
            m_isUsingCUDA = false;

            detector = cv::xfeatures2d::StarDetector::create();
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

            break;
        }
        case DetectorType::SIFT: {
            m_isUsingCUDA = false;

            detector = extractor = cv::xfeatures2d::SIFT::create();

            break;
        }

        case DetectorType::ORB: {
            if (m_isUsingCUDA)
                detector = extractor = cv::cuda::ORB::create();
            else 
                detector = extractor = cv::ORB::create();

            break;
        }
        
        case DetectorType::FAST: {
            if (m_isUsingCUDA)
                detector = cv::cuda::FastFeatureDetector::create();
            else 
                detector = cv::FastFeatureDetector::create();

            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

            break;
        }
        
        case DetectorType::SURF: {
            if (!m_isUsingCUDA)
                detector = extractor = cv::xfeatures2d::SURF::create(400);

            break;
        }
        case DetectorType::KAZE: {
            m_isUsingCUDA = false;

            detector = extractor = cv::KAZE::create();

            break;
        }
        case DetectorType::BRISK: {
            m_isUsingCUDA = false;

            detector = extractor = cv::BRISK::create();
            
            break;
        }
    }
}

void FeatureDetector::generateFeatures(cv::Mat& imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor) {
    //  SURF CUDA not working properly
    //  possible solution -> use CUDA keypoints and descriptors for matching as GpuMat
    if (m_isUsingCUDA && m_detectorType != DetectorType::SURF) {
        //  Upload from host to device
        cv::cuda::GpuMat d_imGray; d_imGray.upload(imGray);

        //std::cout << "Generating CUDA features..." << std::flush;
        /*if(m_detectorType == DetectorType::SURF){
            cv::cuda::SURF_CUDA d_surf;
            cv::cuda::GpuMat d_keyPts, d_descriptor;
            std::vector<float> _descriptor;

            d_surf(d_imGray, cv::cuda::GpuMat(), d_keyPts, d_descriptor);
            d_surf.downloadKeypoints(d_keyPts, keyPts);
            d_surf.downloadDescriptors(d_descriptor, _descriptor);

            descriptor = cv::Mat(_descriptor);
        } */

        //  detectAndCompute is faster than detect/compute
        //  use detectAndCompute if same detector and extractor
        if (detector != extractor){
            detector->detect(d_imGray, keyPts);
            extractor->compute(imGray, keyPts, descriptor);
        } else {
            cv::cuda::GpuMat d_descriptor;
            detector->detectAndCompute(d_imGray, cv::noArray(), keyPts, d_descriptor);
            d_descriptor.download(descriptor);
        }

        //  Download from device to host (clear memory)
        d_imGray.download(imGray);
        // std::cout << "[DONE]";
    } else {
        //  detectAndCompute is faster than detect/compute
        //  use detectAndCompute if same detector and extractor
        if (detector != extractor){
            detector->detect(imGray, keyPts);
            extractor->compute(imGray, keyPts, descriptor);
        } else
            detector->detectAndCompute(imGray, cv::noArray(), keyPts, descriptor);
    }
}

void FeatureDetector::generateFlowFeatures(cv::Mat& imGray, std::vector<cv::Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
    std::vector<cv::Point2f> _corners;

    if (m_isUsingCUDA) {
        std::cout << "Generating CUDA flow features..." << std::flush;

        //  Upload from host to device
        cv::cuda::GpuMat d_imGray, d_corners;
        d_imGray.upload(imGray);

        //  Use Shi-Tomasi corner detector
        cv::Ptr<cv::cuda::CornersDetector> cudaCornersDetector = cv::cuda::createGoodFeaturesToTrackDetector(d_imGray.type(), maxCorners, qualityLevel, minDistance);

        cudaCornersDetector->detect(d_imGray, d_corners);

        //  Download from device to host (clear memory)
        d_corners.download(_corners);
        d_imGray.download(imGray);
    } else {
        std::cout << "Generating flow features..." << std::flush;

        //  Use Shi-Tomasi corner detector
        cv::goodFeaturesToTrack(imGray, _corners, maxCorners, qualityLevel, minDistance);
    }

    // Add new points at the end. Do not remove good points
    corners.insert(corners.end(), _corners.begin(), _corners.end());

    std::cout << "[DONE]";
}

DescriptorMatcher::DescriptorMatcher(std::string method, const float ratioThreshold, bool isUsingCUDA)
    : m_ratioThreshold(ratioThreshold) {
    m_isUsingCUDA = isUsingCUDA;

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

void DescriptorMatcher::findRobustMatches(std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, cv::Mat prevDesc, cv::Mat currDesc, std::vector<cv::Point2f>& prevAligPts, std::vector<cv::Point2f>& currAligPts, std::vector<cv::DMatch>& matches, std::vector<int>& prevPtsToKeyIdx, std::vector<int>& currPtsToKeyIdx, cv::Mat prevFrame, cv::Mat currFrame, bool showMatch) {
    std::vector<cv::DMatch> fMatches, bMatches;

    // knn matches
    ratioMaches(prevDesc, currDesc, fMatches);
    ratioMaches(currDesc, prevDesc, bMatches);

    auto fontFace = cv::FONT_HERSHEY_COMPLEX;
    float fontScale = 1.0;

    cv::Size textSize(15,15);
    cv::Rect boxCoords(10, 20, 300, 100);
        
    if (showMatch) {
        cv::Mat out;
        cv::drawMatches(prevFrame, prevKeyPts, currFrame, currKeyPts, fMatches, out, CV_RGB(255,255,0));

        const std::string headedText = "Knn match";
        const std::string matchesText = "# Matches: " + std::to_string(fMatches.size());

        cv::rectangle(out, boxCoords, cv::Scalar(75,75,75), cv::FILLED);

        cv::putText(out, headedText, cv::Point(10,50), fontFace, fontScale, CV_RGB(255, 255, 255), 2);
        cv::putText(out, matchesText, cv::Point(10,100), fontFace, fontScale, CV_RGB(255, 255, 255));

        cv::imshow("Matches", out);
        //cv::waitKey();
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

    if (showMatch) {
        cv::Mat out;
        cv::drawMatches(prevFrame, prevKeyPts, currFrame, currKeyPts, matches, out, CV_RGB(255,255,0));

        const std::string headedText = "Crossmatching";
        const std::string matchesText = "# Matches: " + std::to_string(matches.size());

       cv::rectangle(out, boxCoords, cv::Scalar(75,75,75), cv::FILLED);

        cv::putText(out, headedText, cv::Point(10,50), fontFace, fontScale, CV_RGB(255, 255, 255), 2);
        cv::putText(out, matchesText, cv::Point(10,100), fontFace, fontScale, CV_RGB(255, 255, 255));

        cv::imshow("Matches", out);
        //cv::waitKey();
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

    if (showMatch) {
        cv::Mat out;
        cv::drawMatches(prevFrame, prevKeyPts, currFrame, currKeyPts, _epipolarMatch, out, CV_RGB(255,255,0));

        const std::string headedText = "Epipolar filter";
        const std::string matchesText = "# Matches: " + std::to_string(_epipolarMatch.size());

        cv::rectangle(out, boxCoords, cv::Scalar(75,75,75), cv::FILLED);

        cv::putText(out, headedText, cv::Point(10,50), fontFace, fontScale, CV_RGB(255, 255, 255), 2);
        cv::putText(out, matchesText, cv::Point(10,100), fontFace, fontScale, CV_RGB(255, 255, 255));

        cv::imshow("Matches", out);
        //cv::waitKey();
    }
    
    //  update informations to output structures
    std::swap(matches, _epipolarMatch);
    std::swap(prevAligPts, _epipolarPrevPts);
    std::swap(currAligPts, _epipolarCurrPts);
}

OptFlow::OptFlow(cv::TermCriteria termcrit, int winSize, int maxLevel, float maxError, uint maxCorners, float qualityLevel, float minCornersDistance, uint minFeatures, bool isUsingCUDA) {
    m_isUsingCUDA = isUsingCUDA;

    if (m_isUsingCUDA)
        d_optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(winSize, winSize), maxLevel, termcrit.MAX_ITER);
    else
        optFlow = cv::SparsePyrLKOpticalFlow::create(cv::Size(winSize, winSize), maxLevel, termcrit);

    additionalSettings.setMaxError(maxError);
    additionalSettings.setMaxCorners(maxCorners);
    additionalSettings.setQualityLvl(qualityLevel);
    additionalSettings.setMinDistance(minCornersDistance);
    additionalSettings.setMinFeatures(minFeatures);
}

void OptFlow::computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<uchar>& statusMask, bool useBoundaryCorrection, bool useErrorCorrection) {
    std::vector<float> err;

    if (m_isUsingCUDA) {
        cv::cuda::GpuMat d_imPrevGray, d_imCurrGray;
        cv::cuda::GpuMat d_prevPts, d_currPts;
        cv::cuda::GpuMat d_statusMask, d_err;

        //  Upload from host to device
        d_imPrevGray.upload(imPrevGray);
        d_imCurrGray.upload(imCurrGray);

        d_prevPts.upload(prevPts);

        // Calculate sparse Lucas-Kanade optical flow
        d_optFlow->calc(d_imPrevGray, d_imCurrGray, d_prevPts, d_currPts, d_statusMask, d_err);

        //  Download from device to host (clear memory)
        d_imPrevGray.download(imPrevGray);
        d_imCurrGray.download(imCurrGray);

        d_prevPts.download(prevPts);
        d_currPts.download(currPts);
        d_statusMask.download(statusMask);
        d_err.download(err);
    } else 
        optFlow->calc(imPrevGray, imCurrGray, prevPts, currPts, statusMask, err);
    
    //  Image boundary filter
    cv::Rect boundary(cv::Point(), imCurrGray.size());

    for (uint i = 0, idxCorrection = 0; i < statusMask.size() && i < err.size(); ++i) {
        cv::Point2f pt = currPts[i - idxCorrection];

        //  Filter by provided parameters
        bool isErrorCorr = statusMask[i] == 1 && err[i] < additionalSettings.maxError;
        bool isBoundCorr = boundary.contains(pt);

        if (!isErrorCorr || !isBoundCorr) {
            if ((useErrorCorrection && !isErrorCorr) || (useBoundaryCorrection && !isBoundCorr)) {
                //  Throw away bad points
                prevPts.erase(prevPts.begin() + (i - idxCorrection));
                currPts.erase(currPts.begin() + (i - idxCorrection));

                idxCorrection++;
            } else
                statusMask[i] = 0;
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
#include "feature_processing.h"

FeatureDetector::FeatureDetector(std::string method, bool isUsingCUDA) {
    m_isUsingCUDA = isUsingCUDA;

    std::for_each(method.begin(), method.end(), [](char& c){
        c = ::toupper(c);
    });

    if (method == "ORB")
        m_detectorType = DetectorType::ORB;
    else if (method == "FAST")
        m_detectorType = DetectorType::FAST;
    else if (method == "STAR")
        m_detectorType = DetectorType::STAR;
    else if (method == "SIFT")
        m_detectorType = DetectorType::SIFT;
    else if (method == "SURF")
        m_detectorType = DetectorType::SURF;
    else if (method == "KAZE")
        m_detectorType = DetectorType::KAZE;
    else if (method == "BRISK")
        m_detectorType = DetectorType::BRISK;
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
                detector = extractor = cv::xfeatures2d::SURF::create();

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
    //std::cout << "Generating features..." << std::flush;

    if (detector != extractor){
        detector->detect(imGray, keyPts);
        extractor->compute(imGray, keyPts, descriptor);
    } else
        detector->detectAndCompute(imGray, cv::noArray(), keyPts, descriptor);

    //std::cout << "[DONE]";
}

void FeatureDetector::generateFeatures(cv::Mat& imGray, cv::cuda::GpuMat& d_imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor) {
    if (m_isUsingCUDA) {
        //std::cout << "Generating CUDA features..." << std::flush;
        if(m_detectorType == DetectorType::SURF){
            cv::cuda::SURF_CUDA d_surf;
            cv::cuda::GpuMat d_keyPts, d_descriptor;
            std::vector<float> _descriptor;

            d_surf(d_imGray, cv::cuda::GpuMat(), d_keyPts, d_descriptor);
            d_surf.downloadKeypoints(d_keyPts, keyPts);
            d_surf.downloadDescriptors(d_descriptor, _descriptor);

            descriptor = cv::Mat(_descriptor);
        } else if (detector != extractor){
            detector->detect(d_imGray, keyPts);
            extractor->compute(imGray, keyPts, descriptor);
        } else {
            cv::cuda::GpuMat d_descriptor;
            detector->detectAndCompute(d_imGray, cv::noArray(), keyPts, d_descriptor);
            d_descriptor.download(descriptor);
        }

        // std::cout << "[DONE]";
    } else
        generateFeatures(imGray, keyPts, descriptor);
}

void FeatureDetector::generateFlowFeatures(cv::Mat& imGray, std::vector<cv::Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
    std::cout << "Generating flow features..." << std::flush;

    cv::goodFeaturesToTrack(imGray, corners, maxCorners, qualityLevel, minDistance);

    std::cout << "[DONE]";
}

void FeatureDetector::generateFlowFeatures(cv::cuda::GpuMat& d_imGray, cv::cuda::GpuMat& corners, int maxCorners, double qualityLevel, double minDistance) {
    std::cout << "Generating CUDA flow features..." << std::flush;

    cv::Ptr<cv::cuda::CornersDetector> cudaCornersDetector = cv::cuda::createGoodFeaturesToTrackDetector(d_imGray.type(), maxCorners, qualityLevel, minDistance);

    cudaCornersDetector->detect(d_imGray, corners);

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
    for (const auto& k : knnMatches) {
        if (k[0].distance < m_ratioThreshold * k[1].distance) 
            matches.push_back(k[0]);
    }
}

void DescriptorMatcher::recipAligMatches(std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, cv::Mat prevDesc, cv::Mat currDesc, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<cv::DMatch>& matches, std::vector<int>& prevIdx, std::vector<int>& currIdx) {
    std::vector<cv::DMatch> fMatches, bMatches;

    ratioMaches(prevDesc, currDesc, fMatches);
    ratioMaches(currDesc, prevDesc, bMatches);

    for (const auto& bM : bMatches) {
        bool isFound = false;

        for (const auto& fM : fMatches) {
            if (bM.queryIdx == fM.trainIdx && bM.trainIdx == fM.queryIdx) {
                prevPts.push_back(prevKeyPts[fM.queryIdx].pt);
                currPts.push_back(currKeyPts[fM.trainIdx].pt);

                prevIdx.push_back(fM.queryIdx);
                currIdx.push_back(fM.trainIdx);

                matches.push_back(fM);

                isFound = true;

                break;
            }
        }

        if (isFound) { continue; }
    }
}

OptFlow::OptFlow(cv::TermCriteria termcrit, int winSize, int maxLevel, float maxError, uint maxCorners, float qualityLevel, float minCornersDistance, bool isUsingCUDA) {
    m_isUsingCUDA = isUsingCUDA;

    if (m_isUsingCUDA)
        d_optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(winSize, winSize), maxLevel, termcrit.MAX_ITER);
    else
        optFlow = cv::SparsePyrLKOpticalFlow::create(cv::Size(winSize, winSize), maxLevel, termcrit);

    additionalSettings.setMaxError(maxError);
    additionalSettings.setMaxCorners(maxCorners);
    additionalSettings.setQualityLvl(qualityLevel);
    additionalSettings.setMinDistance(minCornersDistance);
}

void OptFlow::computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
    std::vector<uchar> status; std::vector<float> err; 
        
    optFlow->calc(imPrevGray, imCurrGray, prevPts, currPts, status, err);

    for (uint i = 0, idxCorrection = 0; i < status.size() && i < err.size(); ++i) {
        cv::Point2f pt = currPts[i - idxCorrection];

        if (status[i] == 0 || err[i] > additionalSettings.maxError ||
            pt.x < 0 || pt.y < 0 || pt.x > imCurrGray.cols || pt.y > imCurrGray.rows) {

            prevPts.erase(prevPts.begin() + (i - idxCorrection));
            currPts.erase(currPts.begin() + (i - idxCorrection));

            idxCorrection++;
        }
    }
}

void OptFlow::computeFlow(cv::cuda::GpuMat& d_imPrevGray, cv::cuda::GpuMat& d_imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
    cv::cuda::GpuMat d_prevPts, d_currPts;
    cv::cuda::GpuMat d_status, d_err;
    
    std::vector<uchar> status; std::vector<float> err; 

    d_prevPts.upload(prevPts);

    d_optFlow->calc(d_imPrevGray, d_imCurrGray, d_prevPts, d_currPts, d_status, d_err);

    d_prevPts.download(prevPts);
    d_currPts.download(currPts);
    d_status.download(status);
    d_err.download(err);

    for (uint i = 0, idxCorrection = 0; i < status.size() && i < err.size(); ++i) {
        cv::Point2f pt = currPts[i - idxCorrection];

        if (status[i] == 0 || err[i] > additionalSettings.maxError ||
            pt.x < 0 || pt.y < 0 || pt.x > d_imCurrGray.cols || pt.y > d_imCurrGray.rows) {

            prevPts.erase(prevPts.begin() + (i - idxCorrection));
            currPts.erase(currPts.begin() + (i - idxCorrection));

            idxCorrection++;
        }
    }
}
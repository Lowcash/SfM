#include "pch.h"

#pragma region CLASSES

class OptFlowAddSettings {
public:
    float maxError, qualLvl, minDist;
    uint maxCorn;

    void setMaxError(float maxError) { this->maxError = maxError; }

    void setMaxCorners(uint maxCorn) { this->maxCorn = maxCorn; }

    void setQualityLvl(float qualLvl) { this->qualLvl = qualLvl; }

    void setMinDistance(float minDist) { this->minDist = minDist; }
};

class UsingCUDA {
protected:
    bool m_isUsingCUDA;

    bool getIsUsingCUDA() const { return m_isUsingCUDA; }
};

class FeatureDetector : protected UsingCUDA {
private:
    enum DetectorType { AKAZE = 0, ORB, FAST, STAR, SIFT, SURF, KAZE, BRISK };

    DetectorType m_detectorType;
public:
    cv::Ptr<cv::FeatureDetector> detector, extractor;

    FeatureDetector(std::string method, bool isUsingCUDA = false) {
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

    void generateFeatures(cv::Mat& imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor) {
        //std::cout << "Generating features..." << std::flush;

        if (detector != extractor){
            detector->detect(imGray, keyPts);
            extractor->compute(imGray, keyPts, descriptor);
        } else
            detector->detectAndCompute(imGray, cv::noArray(), keyPts, descriptor);

        //std::cout << "[DONE]";
    }

    void generateFeatures(cv::Mat& imGray, cv::cuda::GpuMat& d_imGray, std::vector<cv::KeyPoint>& keyPts, cv::Mat& descriptor) {
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

    void generateFlowFeatures(cv::Mat& imGray, std::vector<cv::Point2f>& corners, int maxCorners, double qualityLevel, double minDistance) {
        std::cout << "Generating flow features..." << std::flush;

        cv::goodFeaturesToTrack(imGray, corners, maxCorners, qualityLevel, minDistance);

        std::cout << "[DONE]";
    }

    void generateFlowFeatures(cv::cuda::GpuMat& d_imGray, cv::cuda::GpuMat& corners, int maxCorners, double qualityLevel, double minDistance) {
        std::cout << "Generating CUDA flow features..." << std::flush;

        cv::Ptr<cv::cuda::CornersDetector> cudaCornersDetector = cv::cuda::createGoodFeaturesToTrackDetector(d_imGray.type(), maxCorners, qualityLevel, minDistance);

        cudaCornersDetector->detect(d_imGray, corners);

        std::cout << "[DONE]";
    }
};

class DescriptorMatcher : protected UsingCUDA {
public:
    const float m_ratioThreshold;

    cv::Ptr<cv::DescriptorMatcher> matcher;

    DescriptorMatcher(std::string method, const float ratioThreshold, bool isUsingCUDA = false)
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

    void ratioMaches(const cv::Mat lDesc, const cv::Mat rDesc, std::vector<cv::DMatch>& matches) {
        std::vector<std::vector<cv::DMatch>> knnMatches;

        matcher->knnMatch(lDesc, rDesc, knnMatches, 2);

        matches.clear();
        for (const auto& k : knnMatches) {
            if (k[0].distance < m_ratioThreshold * k[1].distance) 
                matches.push_back(k[0]);
        }
    }

    void recipAligMatches(std::vector<cv::KeyPoint> prevKeyPts, std::vector<cv::KeyPoint> currKeyPts, cv::Mat prevDesc, cv::Mat currDesc, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts, std::vector<int>& prevIdx, std::vector<int>& currIdx, std::vector<cv::DMatch>& matches) {
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

                    isFound = true;

                    matches.push_back(fM);

                    //break;
                }
            }

            if (isFound) { continue; }
        }
    }
};

class OptFlow : protected UsingCUDA {
public:
    cv::Ptr<cv::SparsePyrLKOpticalFlow> optFlow;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_optFlow;

    OptFlowAddSettings additionalSettings;

    OptFlow(cv::TermCriteria termcrit, int winSize, int maxLevel, float maxError, uint maxCorners, float qualityLevel, float minCornersDistance, bool isUsingCUDA = false) {
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

    void computeFlow(cv::Mat imPrevGray, cv::Mat imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
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

    void computeFlow(cv::cuda::GpuMat& d_imPrevGray, cv::cuda::GpuMat& d_imCurrGray, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
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
};

class CameraParameters {
public:
    cv::Mat _K, distCoeffs;

    cv::Matx33d K33d;

    cv::Point2d pp;

    double focalLength;

    CameraParameters(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        updateCameraParameters(K, distCoeffs, downSample);
    }

    void updateCameraParameters(const cv::Mat K, const cv::Mat distCoeffs, const double downSample = 1.0f) {
        _K = K * downSample;

        _K.at<double>(2,2) = 1.0;
    
        std::cout << "\nCamera intrices: " << _K << "\n";

        this->distCoeffs = distCoeffs;

        K33d = cv::Matx33d(
            _K.at<double>(0,0), _K.at<double>(0,1), _K.at<double>(0,2),
            _K.at<double>(1,0), _K.at<double>(1,1), _K.at<double>(1,2), 
            _K.at<double>(2,0), _K.at<double>(2,1), _K.at<double>(2,2)
        );

        pp = cv::Point2d(_K.at<double>(0, 2), _K.at<double>(1, 2));
        focalLength = ((_K.at<double>(0, 0) + _K.at<double>(1, 1)) / 2.0);
    }
};

class RecoveryPose {
public:
    int method;

    const double prob, threshold;
    const uint minInliers;

    cv::Matx33d R;
    cv::Matx31d t;
    cv::Mat mask;

    RecoveryPose(std::string method, const double prob, const double threshold, const uint minInliers) 
        : prob(prob), threshold(threshold), minInliers(minInliers) {
        R = cv::Matx33d::eye();
        t = cv::Matx31d::eye();

        std::for_each(method.begin(), method.end(), [](char& c){
            c = ::toupper(c);
        });

        if (method == "RANSAC")
            this->method = cv::RANSAC;
        if (method == "LMEDS")
            this->method = cv::LMEDS;
    }

    void drawRecoveredPose(cv::Mat inputImg, cv::Mat& outputImg, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat mask = cv::Mat()) {
        bool isUsingMask = mask.rows == prevPts.size();

        inputImg.copyTo(outputImg);

        for (int i = 0; i < prevPts.size() && i < currPts.size(); ++i) {
            cv::arrowedLine(outputImg, currPts[i], prevPts[i], CV_RGB(0,200,0), 2);

            if (isUsingMask && mask.at<uchar>(i) == 0)
                cv::arrowedLine(outputImg, currPts[i], prevPts[i], CV_RGB(200,0,0), 2);
        }
    }
};

class ViewData {
public:
    cv::Mat imColor, imGray;
    cv::cuda::GpuMat d_imColor, d_imGray;
private:
    void setView(const cv::Mat imColor, const cv::Mat imGray) {
        imColor.copyTo(this->imColor);
        imGray.copyTo(this->imGray);
    }

    void setView(const cv::cuda::GpuMat d_imColor, const cv::cuda::GpuMat d_imGray) {
        d_imColor.copyTo(this->d_imColor);
        d_imGray.copyTo(this->d_imGray);
    }
public:
    ViewData() {}

    ViewData(const cv::Mat imColor, const cv::Mat imGray) {
        setView(imColor, imGray);
    }

    ViewData(const cv::cuda::GpuMat d_imColor, const cv::cuda::GpuMat d_imGray) {
        setView(d_imColor, d_imGray);
    }

    ViewData(const cv::Mat imColor, const cv::Mat imGray, const cv::cuda::GpuMat d_imColor, const cv::cuda::GpuMat d_imGray) {
        setView(imColor, imGray);
        setView(d_imColor, d_imGray);
    }
};

class View {
public:
    ViewData* viewPtr;

    void setView(ViewData* view) { this->viewPtr = view; }
};

class FlowView : public View {
public:
    std::vector<cv::Point2f> corners;
    cv::cuda::GpuMat d_corners;

    void setPts(const std::vector<cv::Point2f> corners) { this->corners = corners; }
    void setPts(const cv::cuda::GpuMat corners) { this->d_corners = corners; }
};

class FeatureView : public View {
public:
    std::vector<cv::KeyPoint> keyPts;
    std::vector<cv::Point2f> pts;

    cv::Mat descriptor;

    void setFeatures(const std::vector<cv::KeyPoint> keyPts, const cv::Mat descriptor) {
        this->keyPts = keyPts;
        this->descriptor = descriptor;

        cv::KeyPoint::convert(this->keyPts, this->pts);
    }
};

class TrackView : public View {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptor;

    std::vector<cv::Point2f> points2D;
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> pointsRGB;

    void addTrack(const cv::Point2f point2D, const cv::Point3d point3D, const cv::Vec3b pointRGB, const cv::KeyPoint keyPoint, cv::Mat descriptor) {
        points2D.push_back(point2D);
        points3D.push_back(point3D);
        pointsRGB.push_back(pointRGB);

        this->keyPoints.push_back(keyPoint);
        this->descriptor.push_back(descriptor);
    }
};

class Tracking {
public:
    std::vector<TrackView> trackViews;

    void addTrackView(const std::vector<bool>& mask, const std::vector<cv::Point2f>& points2D, const std::vector<cv::Point3f> points3D, const std::vector<cv::Vec3b>& pointsRGB, const std::vector<cv::KeyPoint>& keyPoints, const cv::Mat& descriptor, const std::vector<int>& featureIndexer = std::vector<int>()) {
        TrackView _trackView;

        for (uint idx = 0; idx < mask.size(); ++idx) {
            const cv::KeyPoint _keypoint = featureIndexer.empty() ? keyPoints[idx] : keyPoints[featureIndexer[idx]];
            const cv::Mat _descriptor = featureIndexer.empty() ? descriptor.row(idx) : descriptor.row(featureIndexer[idx]);

            if (mask[idx])
                _trackView.addTrack(points2D[idx], points3D[idx], pointsRGB[idx], _keypoint, _descriptor);
        }

        trackViews.push_back(_trackView);
    }

    bool findRecoveredCameraPose(DescriptorMatcher matcher, float knnRatio, CameraParameters camParams, FeatureView& featView, cv::Matx33d& R, cv::Matx31d& t) {
        if (trackViews.empty()) { return true; }
        
        std::cout << "Recovering pose..." << std::flush;
        
        std::vector<cv::Point2f> _posePoints2D;
        std::vector<cv::Point3f> _posePoints3D;
        cv::Mat _R, _t;

        int it = 0;

        for (auto t = trackViews.rbegin(); t != trackViews.rend(); ++t) {
            if (t->points2D.empty() || featView.keyPts.empty()) { continue; }

            std::vector<cv::DMatch> _matches;
            std::vector<cv::Point2f> _prevPts, _currPts;
            std::vector<int> _prevIdx, _currIdx;
            matcher.recipAligMatches(t->keyPoints, featView.keyPts, t->descriptor, featView.descriptor, _prevPts, _currPts, _prevIdx, _currIdx, _matches);

            //cv::Mat _inOutMatch; t->viewPtr->imColor.copyTo(_inOutMatch); 
            //cv::hconcat(_inOutMatch, featView.viewPtr->imColor, _inOutMatch);

            for (int i = 0; i < _prevPts.size() && i < _currPts.size(); ++i) {
                cv::Point3f _point3D = t->points3D[_prevIdx[i]];
                cv::Point2f _point2D = _currPts[i];

                //cv::line(_inOutMatch, _prevPts[i], cv::Point2f(_currPts[i].x + featView.viewPtr->imColor.cols, _currPts[i].y) , CV_RGB(0, 0, 0), 2);

                _posePoints2D.push_back(_point2D);
                _posePoints3D.push_back(_point3D);
            }

            if (it > 2) break;

            it++;

            //cv::imshow("Matches", _inOutMatch); cv::waitKey(1);
        }

        if (_posePoints2D.size() < 4 || _posePoints3D.size() < 4) { return false; }

        if (!cv::solvePnPRansac(_posePoints3D, _posePoints2D, camParams._K, cv::Mat(), _R, _t, true)) { return false; }

        cv::Rodrigues(_R, R); t = _t;

        std::cout << "[DONE]\n";

        return true;
    }
};

#pragma endregion CLASSES

#pragma region STRUCTS

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 7 parameters: 3 for rotation, 3 for translation, 1 for
// focal length. The principal point is not modeled (assumed be located at the
// image center, and already subtracted from 'observed'), and focal_x = focal_y.
struct SimpleReprojectionError {
    SimpleReprojectionError(double observed_x, double observed_y) :
            observed_x(observed_x), observed_y(observed_y) {
    }
    template<typename T>
    bool operator()(const T* const camera,
    				const T* const point,
					const T* const focal,
						  T* residuals) const {
        T p[3];
        // Rotate: camera[0,1,2] are the angle-axis rotation.
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate: camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective divide
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        // Compute final projected point position.
        const T predicted_x = *focal * xp;
        const T predicted_y = *focal * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
                new SimpleReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y) {
     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                 new SnavelyReprojectionError(observed_x, observed_y)));
   }

  double observed_x;
  double observed_y;
};

struct VisualizationPCL {
private:
    int m_numClouds, m_numCams, m_numPts;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> getNewViewer (const std::string windowName, const cv::viz::Color bColor) {
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud);
	    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
        viewer->setBackgroundColor(bColor.val[0], bColor.val[1], bColor.val[1]);
        viewer->initCameraParameters();
        viewer->setCameraPosition(0, 0, -1, 0, -1, 0);

        return (viewer);
    }
public:
    VisualizationPCL(const std::string windowName, const cv::viz::Color backgroundColor = cv::viz::Color::black()) {
        m_viewer = getNewViewer(windowName, backgroundColor);
        
        m_numClouds = 0;
        m_numCams = 0;
        m_numPts = 0;
    }

    void addPointCloud(std::vector<cv::Point3f> points3D, std::vector<cv::Vec3b> pointsRGB) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (auto [it, end, i] = std::tuple{points3D.cbegin(), points3D.cend(), 0}; it != end; ++it, ++i) {
            pcl::PointXYZRGB rgbPoint;

            auto p3d = (cv::Point3d)*it;
            auto pClr = pointsRGB[i];

            rgbPoint.x = p3d.x;
            rgbPoint.y = p3d.y;
            rgbPoint.z = p3d.z;

            rgbPoint.r = pClr[2];
            rgbPoint.g = pClr[1];
            rgbPoint.b = pClr[0];

            pointCloud->push_back(rgbPoint);
        }
        
        m_viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud, "cloud_" + std::to_string(m_numClouds));
        
        m_numClouds++;
    }

    void addPointCloud(std::vector<TrackView>& trackViews) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (auto [tIt, tEnd, tIdx] = std::tuple{trackViews.crbegin(), trackViews.crend(), 0}; tIt != tEnd; ++tIt, ++tIdx) {
            auto t = (TrackView)*tIt;

            for (auto [it, end, i] = std::tuple{t.points3D.cbegin(), t.points3D.cend(), 0}; it != end; ++it, ++i) {
                pcl::PointXYZRGB rgbPoint;

                auto p3d = (cv::Point3d)*it;
                auto pClr = t.pointsRGB[i];

                rgbPoint.x = p3d.x;
                rgbPoint.y = p3d.y;
                rgbPoint.z = p3d.z;

                rgbPoint.r = pClr[2];
                rgbPoint.g = pClr[1];
                rgbPoint.b = pClr[0];

                pointCloud->push_back(rgbPoint);
            }

            //break;
        }

        m_viewer->updatePointCloud(pointCloud);
    }

    void addPoints(std::vector<cv::Point3f> points3D) {
        for (const auto& p : points3D) {
            m_viewer->addSphere(p, 1.0, 0, 255, 255, "point_" + std::to_string(m_numPts));

            m_numPts++;
        }
    }

    void addCamera(pcl::PointXYZ camPose) {        
        m_viewer->addSphere(camPose, 0.25, "cam_pose_" + std::to_string(m_numCams));

        m_numCams++;
    }

    void visualize() {
        //while(!m_viewer->wasStopped())
            m_viewer->spinOnce(60);
        //m_viewer->spin();
    }
};

class UserInput {
private:
    std::vector<cv::Point3f> m_usrPts;

public:
    void attach2DPoints(std::vector<cv::Point> userPrevPts, std::vector<cv::Point> userCurrPts, std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& currPts) {
        prevPts.insert(prevPts.end(), userPrevPts.begin(), userPrevPts.end());
        currPts.insert(currPts.end(), userCurrPts.begin(), userCurrPts.end());
    }

    void detach2DPoints(std::vector<cv::Point> userPts2D, std::vector<cv::Point2f>& points2D) {
        points2D.erase(points2D.end() - userPts2D.size(), points2D.end());
    }

    void detach3DPoints(std::vector<cv::Point> userPts2D, std::vector<cv::Point3f>& points3D) {
        points3D.erase(points3D.end() - userPts2D.size(), points3D.end());
    }

    void insert3DPoints(std::vector<cv::Point3f> points3D) {
        m_usrPts.insert(m_usrPts.end(), points3D.begin(), points3D.end());
    }

    void recoverPoints(cv::Mat R, cv::Mat t, CameraParameters camParams, cv::Mat& imOutUsr) {
        if (!m_usrPts.empty()) {
            std::vector<cv::Point2f> pts2D; cv::projectPoints(m_usrPts, R, t, camParams.K33d, cv::Mat(), pts2D);

            for (const auto p : pts2D) {
                std::cout << "Point projected to: " << p << "\n";

                cv::circle(imOutUsr, p, 3, CV_RGB(150, 200, 0), cv::FILLED, cv::LINE_AA);
            }
        }
    }

    void movePoints(const std::vector<cv::Point2f> prevPts, const std::vector<cv::Point2f> currPts, const std::vector<cv::Point> inputPts, std::vector<cv::Point>& outputPts) {
        cv::Point2f move; int ptsSize = MIN(prevPts.size(), currPts.size());

        for (int i = 0; i < ptsSize; ++i) 
            move += currPts[i] - prevPts[i];
        
        move.x /= ptsSize;
        move.y /= ptsSize;

        for (int i = 0; i < inputPts.size(); ++i) {
            outputPts.push_back(
                cv::Point2f(
                    inputPts[i].x + move.x, 
                    inputPts[i].y + move.y
                )
            );
        }
    }
};

struct MouseUsrDataParams {
public:
    const std::string m_inputWinName;

    cv::Mat* m_inputMat;

    std::vector<cv::Point> m_clickedPoint;

    MouseUsrDataParams(const std::string inputWinName, cv::Mat* inputMat)
        : m_inputWinName(inputWinName) {
        m_inputMat = inputMat;
    }
};

#pragma endregion STRUCTS

#pragma region METHODS

static void onUsrWinClick (int event, int x, int y, int flags, void* params) {
    if (event != cv::EVENT_LBUTTONDOWN) { return; }

    MouseUsrDataParams* mouseParams = (MouseUsrDataParams*)params;

    const cv::Point clickedPoint(x, y);

    mouseParams->m_clickedPoint.push_back(clickedPoint);

    std::cout << "Clicked to: " << clickedPoint << "\n";
    
    cv::circle(*mouseParams->m_inputMat, clickedPoint, 3, CV_RGB(200, 0, 0), cv::FILLED, cv::LINE_AA);

    cv::imshow(mouseParams->m_inputWinName, *mouseParams->m_inputMat);
}

bool findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers) {
    if (prevPts.size() <= 5 || currPts.size() <= 5) { return false; }

    cv::Mat E = cv::findEssentialMat(prevPts, currPts, cameraK, recPose.method, recPose.prob, recPose.threshold, recPose.mask);

    if (!(E.cols == 3 && E.rows == 3)) { return false; }

    numInliers = cv::recoverPose(E, prevPts, currPts, cameraK, recPose.R, recPose.t, recPose.mask);

    return numInliers > minInliers;
}

void homogPtsToRGBCloud(cv::Mat imgColor, CameraParameters camParams, cv::Mat R, cv::Mat t, cv::Mat homPoints, std::vector<cv::Point2f>& points2D, std::vector<cv::Point3f>& points3D, std::vector<cv::Vec3b>& pointsRGB, float minDist, float maxDist, float maxProjErr, std::vector<bool>& mask) {
    cv::Mat point3DMat; cv::convertPointsFromHomogeneous(homPoints.t(), point3DMat);

    std::vector<cv::Point2f> _pts2D; cv::projectPoints(point3DMat, R, t, camParams.K33d, cv::Mat(), _pts2D);

    points3D.clear();
    pointsRGB.clear();
    
    for(int i = 0, idxCorrection = 0; i < point3DMat.rows; ++i) {
        const cv::Point3f point3D = point3DMat.at<cv::Point3f>(i, 0);
        const cv::Vec3b imPoint2D = imgColor.at<cv::Vec3b>(points2D[i]);

        const float err = cv::norm(_pts2D[i] - points2D[i]);

        points3D.push_back( point3D );
        pointsRGB.push_back( imPoint2D );

        mask.push_back( point3D.z > minDist && point3D.z < maxDist && err <  maxProjErr);
    }
}

void adjustBundle(std::vector<TrackView>& tracks, CameraParameters& camParams, std::vector<cv::Matx34f>& camPoses) {
    std::cout << "Bundle adjustment...\n" << std::flush;

    ceres::Problem problem;

    std::vector<cv::Matx16d> camPoses6d;
    camPoses6d.reserve(camPoses.size());

    uint ptsCount = 0;

    for (int i = 0; i < camPoses.size(); ++i) {
        cv::Matx34f c = camPoses[i];

        if ((0, 0) == 0 && c(1, 1) == 0 && c(2, 2) == 0) { 
            camPoses6d.push_back(cv::Matx16d());
            continue; 
        }

        cv::Vec3f t(c(0, 3), c(1, 3), c(2, 3));
        cv::Matx33f R = c.get_minor<3, 3>(0, 0);
        float angleAxis[3]; ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);

        camPoses6d.push_back(cv::Matx16d(
            angleAxis[0],
            angleAxis[1],
            angleAxis[2],
            t(0),
            t(1),
            t(2)
        ));

        ptsCount += tracks[i].points3D.size();
    }

    ceres::LossFunction* lossFunction = new ceres::CauchyLoss(0.5);

    double focalLength = camParams.focalLength;

    std::vector<cv::Vec3d> pts3D(ptsCount);

    for (int j = 0, ptsCount = 0; j < tracks.size(); ++j) {
        for (int i = 0; i < tracks[j].points2D.size() && i < tracks[j].points3D.size(); ++i) {
            cv::Point2f p2d = tracks[j].points2D[i];
            pts3D[ptsCount] = cv::Vec3d(tracks[j].points3D[i].x, tracks[j].points3D[i].y, tracks[j].points3D[i].z);

            p2d.x -= camParams.K33d(0, 2);
            p2d.y -= camParams.K33d(1, 2);

            ceres::CostFunction* costFunc = SimpleReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(costFunc, lossFunction, camPoses6d[j].val, pts3D[ptsCount].val, &focalLength);

            ptsCount++;
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.eta = 1e-2;
    options.max_solver_time_in_seconds = 10;
    options.logging_type = ceres::LoggingType::SILENT;
    options.num_threads = 8;

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    cv::Mat K; camParams._K.copyTo(K);

    K.at<double>(0,0) = focalLength;
    K.at<double>(1,1) = focalLength;

    camParams.updateCameraParameters(K, camParams.distCoeffs);

    std::cout << "Focal length: " << focalLength << "\n";

    if (!summary.IsSolutionUsable()) {
		std::cout << "Bundle Adjustment failed." << std::endl;
	} else {
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << camPoses.size() << "\n"
			<< " #num_residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
	}
    
    for (int i = 0; i < camPoses.size(); ++i) {
        cv::Matx34f& c = camPoses[i];
        
        if (c(0, 0) == 0 && c(1, 1) == 0 && c(2, 2) == 0) { continue; }

        double rotationMat[9] = { 0 };
        ceres::AngleAxisToRotationMatrix(camPoses6d[i].val, rotationMat);

        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                c(column, row) = rotationMat[row * 3 + column];
            }
        }

        c(0, 3) = camPoses6d[i](3); 
        c(1, 3) = camPoses6d[i](4); 
        c(2, 3) = camPoses6d[i](5);
    }

    for (int j = 0, ptsCount = 0; j < tracks.size(); ++j) {
        for (int i = 0; i < tracks[j].points2D.size() && i < tracks[j].points3D.size(); ++i) {
            tracks[j].points3D[i].x = pts3D[ptsCount](0);
            tracks[j].points3D[i].y = pts3D[ptsCount](1);
            tracks[j].points3D[i].z = pts3D[ptsCount](2);

            ptsCount++;
        }
    }

    std::cout << "[DONE]\n";
}

bool loadImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray, float downSample = 1.0f) {
    cap >> imColor; if (imColor.empty()) { return false; }

    if (downSample != 1.0f)
        cv::resize(imColor, imColor, cv::Size(imColor.cols*downSample, imColor.rows*downSample));

    cv::cvtColor(imColor, imGray, cv::COLOR_BGR2GRAY);

    return true;
}

bool findGoodImagePair(cv::VideoCapture cap, OptFlow optFlow, FeatureDetector featDetector, RecoveryPose& recPose, CameraParameters camParams, std::vector<ViewData>& views, FlowView& ofPrevView, FlowView& ofCurrView, uint flowMinFeatures, float imDownSampling = 1.0f, bool isUsingCUDA = false) {
    std::cout << "Finding good image pair" << std::flush;

    cv::Mat _imColor, _imGray;
    cv::cuda::GpuMat _d_imColor, _d_imGray;

    int numSkippedFrames = -1, numHomInliers = 0;
    do {
        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        //cv::GaussianBlur(_imGray, _imGray, cv::Size(5, 5), 0);
        cv::medianBlur(_imGray, _imGray, 7);

        std::cout << "." << std::flush;
        
        if (isUsingCUDA) {
            _d_imColor.upload(_imColor);
            _d_imGray.upload(_imGray);
        }

        if (ofPrevView.corners.size() < flowMinFeatures) {
            if (isUsingCUDA) {
                cv::cuda::GpuMat d_corners;

                featDetector.generateFlowFeatures(_d_imGray, d_corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

                d_corners.download(ofPrevView.corners);
            } else 
                featDetector.generateFlowFeatures(_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

            if (isUsingCUDA) 
                views.push_back(ViewData(_imColor, _imGray, _d_imColor, _d_imGray));
            else 
                views.push_back(ViewData(_imColor, _imGray));

            continue; 
        }

        ofPrevView.setView(&(views.rbegin()[0]));

        if (isUsingCUDA) {
            optFlow.computeFlow(ofPrevView.viewPtr->d_imGray, _d_imGray, ofPrevView.corners, ofCurrView.corners);
        } else
            optFlow.computeFlow(ofPrevView.viewPtr->imGray, _imGray, ofPrevView.corners, ofCurrView.corners);

        numSkippedFrames++;
    } while(!findCameraPose(recPose, ofPrevView.corners, ofCurrView.corners, camParams._K, recPose.minInliers, numHomInliers));

    if (isUsingCUDA) 
        views.push_back(ViewData(_imColor, _imGray, _d_imColor, _d_imGray));
    else 
        views.push_back(ViewData(_imColor, _imGray));

    std::swap(ofPrevView, ofCurrView);

    ofPrevView.setView(&(views.rbegin()[1]));
    ofCurrView.setView(&(views.rbegin()[0]));

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return true;
}

bool findGoodImagePair(cv::VideoCapture cap, FeatureDetector featDetector, DescriptorMatcher matcher, RecoveryPose& recPose, CameraParameters camParams, std::vector<ViewData>& views, FeatureView& prevView, FeatureView& currView, float imDownSampling = 1.0f, bool isUsingCUDA = false) {
    std::vector<ViewData> _views;
    std::vector<cv::Mat> _imGrays;

    std::map<std::pair<int, int>, std::vector<cv::DMatch>> _matchingMatrix;
    std::map<std::pair<int, int>, std::vector<cv::Point2f>> _prevPts;
    std::map<std::pair<int, int>, std::vector<cv::Point2f>> _currPts;
    std::map<int, std::pair<int, int>> _inliers;
    std::map<std::pair<int, int>, cv::Matx33d> _R;
    std::map<std::pair<int, int>, cv::Matx31d> _t;
    for (int i = 0; i < 6; ++i) {
        cv::Mat _imColor, _imGray;

        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        cv::GaussianBlur(_imGray, _imGray, cv::Size(5,5), 0);

        _views.push_back(ViewData(_imColor, _imGray));
        _imGrays.push_back(_imGray);

        for (int j = i; j < 6; ++j) {
            _matchingMatrix[std::pair{i, j}] = std::vector<cv::DMatch>();
            _prevPts[std::pair{i, j}] = std::vector<cv::Point2f>();
            _currPts[std::pair{i, j}] = std::vector<cv::Point2f>();
        }
    }

    std::vector<std::vector<cv::KeyPoint>> _keyPts;
    std::vector<cv::Mat> _decriptor;

    featDetector.detector->detect(_imGrays, _keyPts);
    featDetector.extractor->compute(_imGrays, _keyPts, _decriptor);

    for (auto& m : _matchingMatrix) {
        std::vector<int> _prevIdx, _currIdx;

        matcher.recipAligMatches(_keyPts[m.first.first], _keyPts[m.first.second], _decriptor[m.first.first], _decriptor[m.first.second], _prevPts[std::pair{m.first.first, m.first.second}], _currPts[std::pair{m.first.first, m.first.second}], _prevIdx, _currIdx, m.second);
    }

    for (auto& m : _matchingMatrix) {
        int inliers; findCameraPose(recPose, _prevPts[std::pair{m.first.first, m.first.second}], _currPts[std::pair{m.first.first, m.first.second}], camParams._K, recPose.minInliers, inliers);

        _R[std::pair{m.first.first, m.first.second}] = recPose.R;
        _t[std::pair{m.first.first, m.first.second}] = recPose.t;

        _inliers[inliers] = std::pair{m.first.first, m.first.second};
    }

    views.push_back(_views[_inliers.rbegin()->second.first]);
    views.push_back(_views[_inliers.rbegin()->second.second]);

    recPose.R = _R[std::pair{_inliers.rbegin()->second.first, _inliers.rbegin()->second.second}];
    recPose.t = _t[std::pair{_inliers.rbegin()->second.first, _inliers.rbegin()->second.second}];

    auto i = _inliers.rbegin()->second;

    prevView.setFeatures(_keyPts[_inliers.rbegin()->second.first], _decriptor[_inliers.rbegin()->second.first]);
    currView.setFeatures(_keyPts[_inliers.rbegin()->second.second], _decriptor[_inliers.rbegin()->second.second]);

    prevView.setView(&(views.rbegin()[1]));
    currView.setView(&(views.rbegin()[0]));

    return true;
}

void composePoseEstimation(cv::Matx33d R, cv::Matx31d t, cv::Matx34d& pose) {
    pose = cv::Matx34d(
        R(0,0), R(0,1), R(0,2), t(0),
        R(1,0), R(1,1), R(1,2), t(1),
        R(2,0), R(2,1), R(2,2), t(2)
    );
}

void cvPoseToPCLPose(cv::Matx31d cvPose, pcl::PointXYZ& pclPose) {
    pclPose = pcl::PointXYZ(cvPose(0), cvPose(1), cvPose(2));
}

#pragma endregion METHODS

int main(int argc, char** argv) {
#pragma region INIT
    std::cout << "Using OpenCV " << cv::getVersionString().c_str() << std::flush;

    cv::CommandLineParser parser(argc, argv,
		"{ help h ?  |            | help }"
        "{ bSource   | .          | source video file [.mp4, .avi ...] }"
		"{ bcalib    | .          | camera intric parameters file path }"
        "{ bDownSamp | 1          | downsampling of input source images }"
        "{ bWinWidth | 1024       | debug windows width }"
        "{ bWinHeight| 768        | debug windows height }"
        "{ bUseCuda  | false      | is nVidia CUDA used }"
        "{ bUseOptFl | true       | is optical flow matching used }"

        "{ fDecType  | AKAZE      | used detector type }"
        "{ fMatchType| BRUTEFORCE | used matcher type }"
        "{ fKnnRatio | 0.75       | knn ration match }"

        "{ ofMinKPts | 500        | optical flow min descriptor to generate new one }"
        "{ ofWinSize | 21         | optical flow window size }"
        "{ ofMaxLevel| 3          | optical flow max pyramid level }"
        "{ ofMaxItCt | 30         | optical flow max iteration count }"
        "{ ofItEps   | 0.1        | optical flow iteration epsilon }"
        "{ ofMaxError| 0          | optical flow max error }"
        "{ ofMaxCorn | 500        | optical flow max generated corners }"
        "{ ofQualLvl | 0.1        | optical flow generated corners quality level }"
        "{ ofMinDist | 25.0       | optical flow generated corners min distance }"

        "{ peMethod  | LMEDS      | pose estimation fundamental matrix computation method [RANSAC/LMEDS] }"
        "{ peProb    | 0.999      | pose estimation confidence/probability }"
        "{ peThresh  | 1.0        | pose estimation threshold }"
        "{ peMinInl  | 50         | pose estimation in number of homography inliers user for reconstruction }"

        "{ tMinDist  | 1.0        | triangulation points min distance }"
        "{ tMaxDist  | 100.0      | triangulation points max distance }"
        "{ tMaxPErr  | 100.0      | triangulation points max reprojection error }"
    );

    if (parser.has("help")) {
        parser.printMessage();
        exit(0);
    }

    //--------------------------------- BASE --------------------------------//
    const std::string bSource = parser.get<std::string>("bSource");
    const std::string bcalib = parser.get<std::string>("bcalib");
    const float bDownSamp = parser.get<float>("bDownSamp");
    const int bWinWidth = parser.get<int>("bWinWidth");
    const int bWinHeight = parser.get<int>("bWinHeight");
    const bool bUseCuda = parser.get<bool>("bUseCuda");
    const bool bUseOptFl = parser.get<bool>("bUseOptFl");

    //------------------------------- FEATURES ------------------------------//
    const std::string fDecType = parser.get<std::string>("fDecType");
    const std::string fMatchType = parser.get<std::string>("fMatchType");
    const float fKnnRatio = parser.get<float>("fKnnRatio");

    //----------------------------- OPTICAL FLOW ----------------------------//
    const int ofMinKPts = parser.get<int>("ofMinKPts");
    const int ofWinSize = parser.get<int>("ofWinSize");
    const int ofMaxLevel = parser.get<int>("ofMaxLevel");
    const float ofMaxItCt = parser.get<float>("ofMaxItCt");
    const float ofItEps = parser.get<float>("ofItEps");
    const float ofMaxError = parser.get<float>("ofMaxError");
    const int ofMaxCorn = parser.get<int>("ofMaxCorn");
    const float ofQualLvl = parser.get<float>("ofQualLvl");
    const float ofMinDist = parser.get<float>("ofMinDist");

    //--------------------------- POSE ESTIMATION ---------------------------//
    const std::string peMethod = parser.get<std::string>("peMethod");
    const float peProb = parser.get<float>("peProb");
    const float peThresh = parser.get<float>("peThresh");
    const int peMinInl = parser.get<int>("peMinInl");

    //---------------------------- TRIANGULATION ----------------------------//
    const float tMinDist = parser.get<float>("tMinDist");
    const float tMaxDist = parser.get<float>("tMaxDist");
    const float tMaxPErr = parser.get<float>("tMaxPErr");

    bool isUsingCUDA = false;

    if (bUseCuda) {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            std::cout << " with CUDA support\n";

            cv::cuda::setDevice(0);
            cv::cuda::printShortCudaDeviceInfo(0);

            isUsingCUDA = true;
        }
        else
            std::cout << "\nCannot use nVidia CUDA -> no devices" << "\n"; 
    }
     
    const cv::FileStorage fs(bcalib, cv::FileStorage::READ);
    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;

    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;
    CameraParameters camParams(cameraK, distCoeffs, bDownSamp);
    
    const std::string ptCloudWinName = "Point cloud";
    const std::string usrInpWinName = "User input";
    const std::string recPoseWinName = "Recovery pose";
    const std::string matchesWinName = "Matches";

    cv::VideoCapture cap; if(!cap.open(bSource)) {
        std::cerr << "Error opening video stream or file!!" << "\n";
        exit(1);
    }

    FeatureDetector featDetector(fDecType, isUsingCUDA);
    DescriptorMatcher descMatcher(fMatchType, fKnnRatio, isUsingCUDA);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, ofMaxItCt, ofItEps);
    OptFlow optFlow(flowTermCrit, ofWinSize, ofMaxLevel, ofMaxError, ofMaxCorn, ofQualLvl, ofMinDist, isUsingCUDA);

    RecoveryPose recPose(peMethod, peProb, peThresh, peMinInl);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;
    
    cv::startWindowThread();

    cv::namedWindow(usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(matchesWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(usrInpWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(recPoseWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(matchesWinName, cv::Size(bWinWidth, bWinHeight));

    MouseUsrDataParams mouseUsrDataParams(usrInpWinName, &imOutUsrInp);

    cv::setMouseCallback(usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 

    std::vector<FeatureView> featureViews;
    std::vector<ViewData> views;

    std::vector<cv::Matx34f> camPoses;
    std::vector<cv::Point3f> usrPts;

    VisualizationPCL visPcl( ptCloudWinName);
    //std::thread visPclThread(&VisualizationPCL::visualize, &visPcl);

    Tracking tracking;
    UserInput userInput;

#pragma endregion INIT 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for ( ; ; ) {
        cv::Matx34d _prevPose; composePoseEstimation(recPose.R, recPose.t, _prevPose);
        
        if (bUseOptFl) {
            if (!findGoodImagePair(cap, optFlow, featDetector, recPose, camParams, views, ofPrevView, ofCurrView, ofMinKPts, bDownSamp, isUsingCUDA)) { break; }           

            ofPrevView.viewPtr->imColor.copyTo(imOutUsrInp);
            ofPrevView.viewPtr->imColor.copyTo(imOutRecPose);

            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofCurrView.corners, ofPrevView.corners, recPose.mask);
        } else {
            if (!findGoodImagePair(cap, featDetector, descMatcher, recPose, camParams, views, featPrevView, featCurrView, bDownSamp, isUsingCUDA)) { break; }

            featPrevView.viewPtr->imColor.copyTo(imOutUsrInp);
            featCurrView.viewPtr->imColor.copyTo(imOutRecPose);

            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, featPrevView.pts, featCurrView.pts, recPose.mask);
        }

        cv::imshow(usrInpWinName, imOutUsrInp);
        cv::imshow(recPoseWinName, imOutRecPose); cv::waitKey();

        if (bUseOptFl) {
            if (featureViews.empty()) {
                if (isUsingCUDA) 
                    featDetector.generateFeatures(ofPrevView.viewPtr->imGray, ofPrevView.viewPtr->d_imGray, featPrevView.keyPts, featPrevView.descriptor);             
                 else
                    featDetector.generateFeatures(ofPrevView.viewPtr->imGray, featPrevView.keyPts, featPrevView.descriptor);

                featureViews.push_back(featPrevView);
            }

            if (isUsingCUDA) 
                featDetector.generateFeatures(ofCurrView.viewPtr->imGray, ofCurrView.viewPtr->d_imGray, featCurrView.keyPts, featCurrView.descriptor); 
             else
                featDetector.generateFeatures(ofCurrView.viewPtr->imGray, featCurrView.keyPts, featCurrView.descriptor);
                
            featureViews.push_back(featCurrView);

            featPrevView.setView(&(views.rbegin()[1]));
            featCurrView.setView(&(views.rbegin()[0]));
        }

        if (featPrevView.keyPts.empty() || featCurrView.keyPts.empty()) { 
            std::cerr << "None keypoints to match, skip matching/triangulation!\n";

            continue; 
        }

        std::vector<cv::DMatch> _matches;
        std::vector<cv::Point2f> _prevPts, _currPts;
        std::vector<int> _prevIdx, _currIdx;
        
        descMatcher.recipAligMatches(featPrevView.keyPts, featCurrView.keyPts, featPrevView.descriptor, featCurrView.descriptor, _prevPts, _currPts, _prevIdx, _currIdx, _matches);

        cv::drawMatches(featPrevView.viewPtr->imColor, featPrevView.keyPts, featCurrView.viewPtr->imColor, featCurrView.keyPts, _matches, imOutMatches);

        cv::imshow(matchesWinName, imOutMatches);

        if (_prevPts.empty() || _currPts.empty()) { 
            std::cerr << "None points to triangulate, skip triangulation!\n";

            continue; 
        }

        if(!tracking.findRecoveredCameraPose(descMatcher, fKnnRatio, camParams, featCurrView, recPose.R, recPose.t)) {
            std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";

            std::swap(featPrevView, featCurrView);

            continue;
        }

        userInput.recoverPoints(cv::Mat(recPose.R), cv::Mat(recPose.t), camParams, imOutUsrInp);

        std::vector<cv::Point> movedPts;
        userInput.movePoints(_prevPts, _currPts, mouseUsrDataParams.m_clickedPoint, movedPts);

        userInput.attach2DPoints(mouseUsrDataParams.m_clickedPoint, movedPts, _prevPts, _currPts);

        cv::Matx34d _currPose; composePoseEstimation(recPose.R, recPose.t, _currPose);
        camPoses.push_back(_currPose);

        cv::Mat _prevImgN; cv::undistort(_prevPts, _prevImgN, camParams.K33d, cv::Mat());
        cv::Mat _currImgN; cv::undistort(_currPts, _currImgN, camParams.K33d, cv::Mat());

        cv::Mat _homogPts; cv::triangulatePoints(camParams.K33d * _prevPose, camParams.K33d * _currPose, _prevImgN, _currImgN, _homogPts);

        std::vector<cv::Point3f> _points3D;
        std::vector<cv::Vec3b> _pointsRGB;
        std::vector<bool> _mask; homogPtsToRGBCloud(featCurrView.viewPtr->imColor, camParams, cv::Mat(recPose.R), cv::Mat(recPose.t), _homogPts, _currPts, _points3D, _pointsRGB, tMinDist, tMaxDist, tMaxPErr, _mask);

        std::vector<cv::Point3f> _usrPts;
        _usrPts.insert(_usrPts.end(), _points3D.end() - mouseUsrDataParams.m_clickedPoint.size(), _points3D.end());
        userInput.insert3DPoints(_usrPts);

        userInput.detach2DPoints(mouseUsrDataParams.m_clickedPoint, _currPts);
        userInput.detach3DPoints(mouseUsrDataParams.m_clickedPoint, _points3D);
        mouseUsrDataParams.m_clickedPoint.clear();

        tracking.addTrackView(_mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, _currIdx);

        adjustBundle(tracking.trackViews, camParams, camPoses);

        pcl::PointXYZ pclPose; cvPoseToPCLPose(-recPose.t, pclPose);

        std::cout << "\n" << pclPose << "\n"; cv::waitKey(20);

        visPcl.addPointCloud(tracking.trackViews);
        visPcl.addCamera(pclPose);
        visPcl.visualize();

        std::swap(featPrevView, featCurrView);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\n----------------------------------------------------------\n\n";
    std::cout << "Total computing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds!\n";

    cv::waitKey(); exit(0);
}
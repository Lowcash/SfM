#include "pch.h"
#include "camera.h"

struct Visualization {
private:
    cv::viz::Viz3d m_window;

    int numClouds, numCams;
public:
    Visualization(std::string windowName) {
        m_window = cv::viz::Viz3d(windowName);
        m_window.setBackgroundColor(cv::viz::Color::black());
        m_window.setWindowSize(cv::Size(500, 500));
        m_window.setWindowPosition(cv::Point(3840, 0));
        
        numClouds = 0;
        numCams = 0;
    }

    void updatePointCloud(std::vector<cv::Point3d> points3D, std::vector<cv::Vec3b> pointsRGB, cv::Affine3d cloudPose) {
        const cv::viz::WCloud pointCloudWidget(points3D, pointsRGB);

        //m_window.showWidget("point_cloud_" + std::to_string(numClouds), pointCloudWidget, cloudPose);
        //m_window.showWidget("point_cloud_" + std::to_string(numClouds), pointCloudWidget);
        m_window.showWidget("point_cloud", pointCloudWidget);

        numClouds++;
    }

    void updateCameraPose(cv::Affine3d camPose, cv::Matx33f cameraK33) {
        const cv::viz::WCameraPosition camWidget(cameraK33, 5, cv::viz::Color::yellow());

        m_window.showWidget("cam_pose_" + std::to_string(numCams), camWidget, camPose);

        numCams++;
    }

    void updateVieverPose(cv::Affine3d camPose) {
        m_window.setViewerPose(camPose);
    }

    void clear() {
        m_window.removeAllWidgets();
    }

    void visualize() {
        while(!m_window.wasStopped())
            m_window.spinOnce(60, true);
    }
};

cv::Vec3d CalculateMean(const cv::Mat_<cv::Vec3d> &points) {
    cv::Mat_<cv::Vec3d> result;
    cv::reduce(points, result, 0, CV_REDUCE_AVG);
    return result(0, 0);
}

cv::Mat_<double> FindRigidTransform(const cv::Mat_<cv::Vec3d> points1, const cv::Mat_<cv::Vec3d> points2) {
    /* Calculate centroids. */
    cv::Vec3d t1 = -CalculateMean(points1);
    cv::Vec3d t2 = -CalculateMean(points2);

    cv::Mat_<double> T1 = cv::Mat_<double>::eye(4, 4);
    T1(0, 3) = t1[0];
    T1(1, 3) = t1[1];
    T1(2, 3) = t1[2];

    cv::Mat_<double> T2 = cv::Mat_<double>::eye(4, 4);
    T2(0, 3) = -t2[0];
    T2(1, 3) = -t2[1];
    T2(2, 3) = -t2[2];

    /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
     * which is used for scale calculation.
     */
    cv::Mat_<double> C(3, 3, 0.0);
    double p1Rms = 0, p2Rms = 0;
    for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++) {
        cv::Vec3d p1 = points1(ptIdx, 0) + t1;
        cv::Vec3d p2 = points2(ptIdx, 0) + t2;
        p1Rms += p1.dot(p1);
        p2Rms += p2.dot(p2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C(i, j) += p2[i] * p1[j];
            }
        }
    }

    cv::Mat_<double> u, s, vh;
    cv::SVD::compute(C, s, u, vh);

    cv::Mat_<double> R = u * vh;

    if (cv::determinant(R) < 0) {
        R -= u.col(2) * (vh.row(2) * 2.0);
    }

    double scale = sqrt(p2Rms / p1Rms);
    R *= scale;

    cv::Mat_<double> M = cv::Mat_<double>::eye(4, 4);
    R.copyTo(M.colRange(0, 3).rowRange(0, 3));

    cv::Mat_<double> result = T2 * M * T1;
    result /= result(3, 3);

    return result.rowRange(0, 3);
}

void featureTracking(cv::Mat lImg, cv::Mat rImg, std::vector<cv::Point2f>& lPoints, std::vector<cv::Point2f>& rPoints) { 	
    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    cv::Size winSize=cv::Size(21,21);
    
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(lImg, rImg, lPoints, rPoints, status, err, winSize, 3, termcrit, 0, 0.001);

    int indexCorrection = 0;
    for(int i = 0; i < status.size(); i++) {  
        cv::Point2f pt = rPoints.at(i - indexCorrection);

        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if((pt.x < 0) || (pt.y < 0))
                status.at(i) = 0;
            
            lPoints.erase (lPoints.begin() + (i - indexCorrection));
            rPoints.erase (rPoints.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void featureDetection(cv::Mat img, std::vector<cv::Point2f>& points, int numKeyPts) {
    cv::goodFeaturesToTrack(img, points, 2000, 0.01, 25);
}

void setImageTrajectory(cv::Mat& outTrajectory, cv::Mat translation) {
    const int x = int(translation.at<double>(0)) + (outTrajectory.cols / 2);
    const int y = int(translation.at<double>(1)) + (outTrajectory.rows / 3 * 2);
    cv::circle(outTrajectory, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

    cv::rectangle(outTrajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
    char text[100]; sprintf(text, "Coordinates: x = %.3fm y = %.3fm z = %.3fm", translation.at<double>(0)*0.01, -translation.at<double>(1)*0.01, translation.at<double>(2)*0.01);
    cv::putText(outTrajectory, text, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255), 1, 8);
}

void setMaskedDescriptors(cv::Mat& outArrows, cv::Mat& outKeyPts, cv::Mat mask, std::vector<cv::Point2f> lPoints, std::vector<cv::Point2f> rPoints) {
    for (int i = 0; i < mask.rows; ++i) {
        cv::Point2f direction = rPoints[i] - lPoints[i];

        //direction *= 3;

        if(mask.at<unsigned char>(i)) {
            cv::arrowedLine(outArrows, (cv::Point2i)lPoints[i], (cv::Point2i)(lPoints[i] + direction), CV_RGB(0, 222, 0), 1, 8);

            cv::circle(outKeyPts, lPoints[i], 3, CV_RGB(0, 255, 0), cv::FILLED);
        } else {
            cv::arrowedLine(outArrows, (cv::Point2i)lPoints[i], (cv::Point2i)(lPoints[i] + direction), CV_RGB(222, 0, 0), 1, 8);

            cv::circle(outKeyPts, lPoints[i], 3, CV_RGB(255, 0, 0), cv::FILLED);
        }          
    }
}

size_t getNumHomographyInliers(std::vector<cv::Point2f> prevFeatures, std::vector<cv::Point2f> currFeatures, double focal, cv::Point2d pp, cv::Mat& R, cv::Mat& t, cv::Mat& E, cv::Mat& mask) {
    E = cv::findEssentialMat(prevFeatures, currFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);

    return cv::recoverPose(E, prevFeatures, currFeatures, R, t, focal, pp, mask);
}

size_t getNumHomographyInliers(std::vector<cv::Point2f> prevFeatures, std::vector<cv::Point2f> currFeatures, cv::Mat cameraK, cv::Mat distCoeffs, cv::Mat& R, cv::Mat& t, cv::Mat& E, cv::Mat& mask, cv::Mat& points3D) {
    cv::Mat prevNorm; cv::undistort(prevFeatures, prevNorm, cameraK, cv::Mat());
    cv::Mat currNorm; cv::undistort(currFeatures, currNorm, cameraK, cv::Mat());

    E = cv::findEssentialMat(prevNorm, currNorm, cameraK, cv::RANSAC, 0.999, 1.0, mask);

    return cv::recoverPose(E, prevNorm, currNorm, cameraK, R, t, 100.0, mask, points3D);
}

void homogenousPointsToRGBCloud(cv::Mat imgColor, cv::Mat homPoints, std::vector<cv::Point2f> points2D, std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, std::map<std::pair<float, float>, cv::Point3d>& match3dMap) {
    cv::Mat pointMatTmp; cv::convertPointsFromHomogeneous(homPoints.t(), pointMatTmp);

    points3D.clear();
    pointsRGB.clear();

    for(int i = 0; i < pointMatTmp.rows; ++i) {
        cv::Point3d point3D = (cv::Point3d)pointMatTmp.at<cv::Point3f>(i, 0);
        cv::Point2f point2D = (cv::Point2f)points2D[i];
        cv::Point2i imgPoint2D = (cv::Point2i)points2D[i];
        
        if (point3D.z > 0.0001 && (imgPoint2D.x > 0 && imgPoint2D.y > 0 && imgPoint2D.x < imgColor.cols && imgPoint2D.y < imgColor.rows)) {
            points3D.emplace_back(point3D);
            pointsRGB.emplace_back(imgColor.at<cv::Vec3b>(imgPoint2D));

            match3dMap[std::make_pair(point2D.x, point2D.y)] = point3D;
        }
    }
}

bool findGoodImagePair(cv::VideoCapture cap, uint numOfUsedOffsetFrames, cv::Mat& prevImgColor, cv::Mat& currImgColor, cv::Mat& prevImgGray, cv::Mat& currImgGray, std::vector<cv::Point2f>& prevFeatures, std::vector<cv::Point2f>& currFeatures, uint minNumHomographyInliers, cv::Mat cameraK, cv::Mat distCoeffs, cv::Mat& R, cv::Mat& t, cv::Mat& E, cv::Mat& mask, cv::Mat& homogPoints3D) {
    std::vector<cv::Point2f> prevFeaturesTmp;

    int numSkippedFrames = -1, numHomographyInliers = 0;
    do {
        for (int i = 0; i < numOfUsedOffsetFrames; ++i)
            cap >> currImgColor;
        
        if (currImgColor.empty()) { return false; }

        prevFeaturesTmp.clear(); prevFeaturesTmp.insert(prevFeaturesTmp.end(), prevFeatures.begin(), prevFeatures.end());

        cv::cvtColor(currImgColor, currImgGray, cv::COLOR_BGR2GRAY);

        featureTracking(prevImgGray, currImgGray, prevFeaturesTmp, currFeatures);

        numHomographyInliers = getNumHomographyInliers(prevFeaturesTmp, currFeatures, cameraK, distCoeffs, R, t, E, mask, homogPoints3D);

        numSkippedFrames++;

        std::cout << "Inliers count: " << numHomographyInliers << "\n";
    } while(numHomographyInliers < minNumHomographyInliers);

    if (numSkippedFrames > 0)
        std::cout << "Skipped frames: " << numSkippedFrames << "\n";

    std::swap(prevFeaturesTmp, prevFeatures);

    return true;
}

class Track {
public:
    cv::Point2f basePosition2d;
    cv::Point2f activePosition2d;

    cv::Point3d basePosition3d;
    cv::Point3d activePosition3d;

    cv::Vec3b color;
    
    Track(const cv::Point2f point2D, const cv::Point3d point3D, const cv::Vec3b color) {
        basePosition2d = point2D;
        activePosition2d = point2D;
        basePosition3d = point3D;
        activePosition3d = point3D;

        this->color = color;
    };
};

class Tracking {
public:
    std::vector<Track> tracks;

    std::vector<cv::Point3d> points3D;
    std::vector<cv::Vec3b> pointsRGB;

    cv::Mat descriptor;

    void addTrack(const cv::Point2f point2D, const cv::Point3d point3D, const cv::Vec3b color, const cv::Mat descriptor) {
        tracks.emplace_back(Track(point2D, point3D, color));

        points3D.emplace_back(point3D);
        pointsRGB.emplace_back(color);

        this->descriptor.push_back(descriptor);
    }

    void clear() {
        tracks.clear();
        descriptor.release();
    }
};

void trackDrawMatches(cv::Mat prevImg, std::vector<Track> tracks, cv::Mat currImg, std::vector<cv::KeyPoint> currKeyPts, std::vector<cv::DMatch> matches, cv::Mat& out) {
    out = prevImg.clone();
    cv::hconcat(out, currImg, out);

    for(const auto& m : matches) {
        auto startPoint = tracks[m.queryIdx].basePosition2d;
        auto endPoint = cv::Point2f(currKeyPts[m.trainIdx].pt.x + prevImg.cols, currKeyPts[m.trainIdx].pt.y);

        cv::circle(out, startPoint, 5, CV_RGB(0,0,255));
        cv::circle(out, endPoint, 5, CV_RGB(255,255,0));
    }

    for(const auto& m : matches) {
        auto startPoint = tracks[m.queryIdx].basePosition2d;
        auto endPoint = cv::Point2f(currKeyPts[m.trainIdx].pt.x + prevImg.cols, currKeyPts[m.trainIdx].pt.y);

        cv::line(out, startPoint, endPoint, CV_RGB(200, 0, 0));
    }
}

int main(int argc, char** argv) {
#pragma region INIT
	cv::CommandLineParser parser(argc, argv,
		"{ help h ? |       | help }"
        "{ calibCam | .     | camera intric parameters file path }"
		"{ source   | .     | source video file [.mp4, .avi ...] }"
        "{ numKeyPts| 0     | number of detector used key points/descriptors }"
        "{ minKeyPts| 0     | min number of detector used key points/descriptors }"
        "{ offFrames| 0     | number of offset frames used for reconstruction }"
        "{ absScale | 1     | absolute scale }"
        "{ maxSkipFr| 1     | max skipped frames to use new }"
        "{ minInlier| 50    | minimal number of homography inliers user for reconstruction }"
        "{ maxMaDiff| 0.5   | maximal magnitude diff }"
        "{ visDebug | false | enable debug visualization }"
    );

	if (parser.has("help")) {
        parser.printMessage();
        exit(0);
    }

    const std::string cameraParamsPath = parser.get<std::string>("calibCam");
    const std::string videoPath = parser.get<std::string>("source");
    const int numOfUsedDescriptors = parser.get<int>("numKeyPts");
    const int minNumOfUsedDescriptors = parser.get<int>("minKeyPts");
    const int numOfUsedOffsetFrames = parser.get<int>("offFrames");
    const float absoluteScale = parser.get<float>("absScale");
    const float maxSkipFrames = parser.get<float>("maxSkipFr");
    const int minNumHomographyInliers = parser.get<int>("minInlier");
    const float maxMagnitudeDiff = parser.get<float>("maxMaDiff");
    const bool isDebugVisualization = parser.get<bool>("visDebug");

    const std::string debugWinName = "Debug vis out";
    const std::string ptCloudWinName = "Point cloud";
    const std::string trajWinName = "Trajectory";

    std::cout << "Camera intrices read...";

    const cv::FileStorage fs(cameraParamsPath, cv::FileStorage::READ);

    const cv::FileNode fnTime = fs.root();
    const std::string time = fnTime["calibration_time"];

    const cv::Size camSize((int)fs["image_width"], (int)fs["image_height"]);

    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;

    const cv::Matx33d cameraK33d(
        cameraK.at<double>(0,0), cameraK.at<double>(0,1), cameraK.at<double>(0,2),
        cameraK.at<double>(1,0), cameraK.at<double>(1,1), cameraK.at<double>(1,2), 
        cameraK.at<double>(2,0), cameraK.at<double>(2,1), cameraK.at<double>(2,2)
    );

    const cv::Point2d pp(cameraK.at<double>(0, 2), cameraK.at<double>(1, 2));
    const double focal((cameraK.at<double>(0, 0) + cameraK.at<double>(1, 1)) / 2.0);

    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;

    std::cout << "[DONE]" << "\n";
    std::cout << "Creation time: " << time << "\n";
    std::cout << "-----------------------------------------" << "\n";
    std::cout << "Camera intrices: \n" << cameraK << "\n";
    std::cout << "Distortion coefficients: \n" << distCoeffs << "\n";
    std::cout << "Principal point: " << pp << "\n";
    std::cout << "Focal length: " << focal << "\n";
    std::cout << "-----------------------------------------" << "\n";
    
    cv::Mat prevImgColor, currImgColor, prevImgGray, currImgGray, flowImgPose, flowImgKeyPt, flowImgConc, prevCurrImgMatch, trackImgMatches, debugImgTraj, debugImgOut;
    cv::Mat R_f, t_f, R, t, E, mask, homogPoints3D;
    cv::Mat prevDesc, currDesc;
    std::vector<cv::Point2f> prevFlowPt, currFlowPt, prevTrackPt, currTrackPt;
    std::vector<cv::KeyPoint> prevKeyPts, currKeyPts, trackKeyPts;
    std::vector<cv::Affine3d> camPoses; cv::Affine3d camPose;
    std::vector<std::vector<cv::Point3d>> pointCloud; std::vector<cv::Point3d> points3D;
    std::vector<std::vector<cv::Vec3b>> pointCloudRGB; std::vector<cv::Vec3b> pointsRGB;
    std::vector< std::vector<cv::DMatch>> knnMatches; std::vector<cv::DMatch> matches, trackMatches;
    std::vector<cv::Point2f> prevPtsALigned, currPtsAligned;

    cv::VideoCapture cap;

    if(!cap.open(videoPath)) {
        printf("Error opening video stream or file!!\n");
        exit(1);
    }

    Visualization vis(ptCloudWinName);

    std::thread visThread(&Visualization::visualize, &vis);

#pragma endregion INIT
    cap >> prevImgColor;
    
    cv::cvtColor(prevImgColor, prevImgGray, cv::COLOR_BGR2GRAY);

    featureDetection(prevImgGray, prevFlowPt, numOfUsedDescriptors); 

    if(!findGoodImagePair(cap, numOfUsedOffsetFrames, prevImgColor, currImgColor, prevImgGray, currImgGray, prevFlowPt, currFlowPt, minNumHomographyInliers, cameraK, distCoeffs, R, t, E, mask, homogPoints3D)) { exit(1); }

    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

    detector->detectAndCompute(prevImgGray, cv::noArray(), prevKeyPts, prevDesc);
    detector->detectAndCompute(currImgGray, cv::noArray(), currKeyPts, currDesc);

    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(12, 20, 2));
    cv::BFMatcher matcher(cv::BFMatcher(cv::NORM_HAMMING));

    matcher.knnMatch( prevDesc, currDesc, knnMatches, 2);
    const float ratioThresh = 0.5f;
    
    for (size_t i = 0; i < knnMatches.size(); ++i) {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
            matches.emplace_back(knnMatches[i][0]);
    }

    for (const auto& m : matches) {
        prevPtsALigned.emplace_back(prevKeyPts[m.queryIdx].pt);
        currPtsAligned.emplace_back(currKeyPts[m.trainIdx].pt);
    }

    R_f = R.clone();
    t_f = t.clone();

    camPose = cv::Affine3d(R_f, t_f); camPoses.emplace_back(camPose);

    cv::Matx34d prevProjCam = cameraK33d * cv::Matx34d::eye();

    cv::Matx34d currProjCam = cameraK33d * cv::Matx34d(
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
    );

    cv::Mat prevNorm; cv::undistort(prevPtsALigned, prevNorm, cameraK33d, cv::Mat());
    cv::Mat currNorm; cv::undistort(currPtsAligned, currNorm, cameraK33d, cv::Mat());

    cv::triangulatePoints(prevProjCam, currProjCam, prevNorm, currNorm, homogPoints3D);

    std::map<std::pair<float, float>, cv::Point3d> match3dMap;
    homogenousPointsToRGBCloud(currImgColor, homogPoints3D, currPtsAligned, points3D, pointsRGB, match3dMap);

    Tracking tracking;
    for (int i = 0; i < matches.size(); ++i) {
        if(match3dMap[std::make_pair(currPtsAligned[i].x, currPtsAligned[i].y)] != cv::Point3d()) {
            tracking.addTrack(currPtsAligned[i], points3D[i], pointsRGB[i], currDesc.row(matches[i].trainIdx));
        }
    }
    
    std::cout << "Added " << tracking.tracks.size() << " points to cloud!" << "\n";

    prevImgColor.copyTo(flowImgPose);
    prevImgColor.copyTo(flowImgKeyPt);

    setMaskedDescriptors(flowImgPose, flowImgKeyPt, mask, currFlowPt, prevFlowPt);

    pointCloud.emplace_back(tracking.points3D);
    pointCloudRGB.emplace_back(tracking.pointsRGB);

    vis.updatePointCloud(tracking.points3D, tracking.pointsRGB, camPose);
    vis.updateCameraPose(camPose, cameraK33d);

    vis.updateVieverPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0, 0, -100)));

    cv::drawMatches( prevImgColor, prevKeyPts, currImgColor, currKeyPts, matches, prevCurrImgMatch, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    trackDrawMatches(prevImgColor, tracking.tracks, currImgColor, currKeyPts, trackMatches, trackImgMatches);
    
    debugImgTraj = cv::Mat::zeros(600, 600, CV_8UC3);

    prevCurrImgMatch.copyTo(debugImgOut);
    flowImgKeyPt.copyTo(flowImgConc);
    cv::hconcat(flowImgConc, flowImgPose, flowImgConc);
    cv::vconcat(flowImgConc, debugImgOut, debugImgOut);
    cv::vconcat(debugImgOut, trackImgMatches, debugImgOut);

    cv::resize(debugImgOut, debugImgOut, cv::Size(debugImgOut.cols/4, debugImgOut.rows/4));

    cv::namedWindow(debugWinName, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(trajWinName, cv::WINDOW_AUTOSIZE);

    cv::imshow(debugWinName, debugImgOut);
    cv::imshow(trajWinName, debugImgTraj);

    cv::waitKey();

	for ( ; ; ) {
        cv::swap(prevImgColor, currImgColor);
        cv::swap(prevImgGray, currImgGray);
        cv::swap(prevDesc, currDesc);
        std::swap(prevKeyPts, currKeyPts);
        
        std::cout << "Num features: " << prevFlowPt.size() << "\n";
        if (prevFlowPt.size() < minNumOfUsedDescriptors)	{
            featureDetection(prevImgGray, prevFlowPt, numOfUsedDescriptors);
 	    }

        if(!findGoodImagePair(cap, numOfUsedOffsetFrames, prevImgColor, currImgColor, prevImgGray, currImgGray, prevFlowPt, currFlowPt, minNumHomographyInliers, cameraK, distCoeffs, R, t, E, mask, homogPoints3D)) { exit(1); }

        detector->detectAndCompute(currImgGray, cv::noArray(), currKeyPts, currDesc);
        matches.clear(); knnMatches.clear();

        matcher.knnMatch( prevDesc, currDesc, knnMatches, 2);
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
                matches.emplace_back(knnMatches[i][0]);
        }

        prevPtsALigned.clear(); currPtsAligned.clear(); 
        for (const auto& m : matches) {
            prevPtsALigned.emplace_back(prevKeyPts[m.queryIdx].pt);
            currPtsAligned.emplace_back(currKeyPts[m.trainIdx].pt);
        }

        t_f = t_f - absoluteScale * (R_f * t);
        R_f = R_f * R;

        prevImgColor.copyTo(flowImgPose);
        prevImgColor.copyTo(flowImgKeyPt);

        setMaskedDescriptors(flowImgPose, flowImgKeyPt, mask, currFlowPt, prevFlowPt);

        camPose = cv::Affine3d(R_f, t_f); camPoses.emplace_back(camPose);

        setImageTrajectory(debugImgTraj, t_f);
        
        currProjCam = cameraK33d * cv::Matx34d(
            R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
        );

        cv::undistort(prevPtsALigned, prevNorm, cameraK33d, cv::Mat());
        cv::undistort(currPtsAligned, currNorm, cameraK33d, cv::Mat());

        cv::triangulatePoints(prevProjCam, currProjCam, prevNorm, currNorm, homogPoints3D);

        // homogenousPointsToRGBCloud(prevImgColor, prevFlowPt, homogPoints3D, points3D, pointsRGB);

        cv::Mat_<cv::Vec3d> p1, p2;

        for(const auto p : tracking.tracks)
            p1.push_back(cv::Vec3d(p.basePosition3d.x, p.basePosition3d.y, p.basePosition3d.z));

        for(const auto p : points3D)
            p2.push_back(cv::Vec3d(p.x, p.y, p.z));

        auto m = FindRigidTransform(p1, p2);

        //std::cout << m << "\n";

        std::map<std::pair<float, float>, cv::Point3d> point3DMap;
        for(int i = 0; i < points3D.size(); ++i)
            point3DMap[std::make_pair(currPtsAligned[i].x, currPtsAligned[i].y)] = points3D[i];
        

        trackMatches.clear(); knnMatches.clear();
        matcher.knnMatch( tracking.descriptor, currDesc, knnMatches, 2);
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
                trackMatches.emplace_back(knnMatches[i][0]);
        }
        
        for (const auto& m : trackMatches) {
            std::pair<float, float> mapKey = std::make_pair(currKeyPts[m.trainIdx].pt.x, currKeyPts[m.trainIdx].pt.y);

            tracking.tracks[m.queryIdx].activePosition2d = currKeyPts[m.trainIdx].pt;
            tracking.tracks[m.queryIdx].activePosition3d = point3DMap[mapKey];

            // std::cout << tracking.tracks[m.queryIdx].basePosition2d << "," << tracking.tracks[m.queryIdx].activePosition2d << "\n";
            // std::cout << tracking.tracks[m.queryIdx].basePosition3d << "," << tracking.tracks[m.queryIdx].activePosition3d << "\n";
        }

        cv::drawMatches( prevImgColor, prevKeyPts, currImgColor, currKeyPts, matches, prevCurrImgMatch, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        trackDrawMatches(prevImgColor, tracking.tracks, currImgColor, currKeyPts, trackMatches, trackImgMatches);

        tracking.clear();
        for (int i=0;i<matches.size();++i)
            tracking.addTrack(currPtsAligned[i], points3D[i], pointsRGB[i], currDesc.row(matches[i].trainIdx));
        
        //vis.updateVieverPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0, 0, -200)));
        //vis.clear();
        vis.updatePointCloud(points3D, pointsRGB, camPose);
        vis.updateCameraPose(camPose, cameraK33d);

        prevCurrImgMatch.copyTo(debugImgOut);
        flowImgKeyPt.copyTo(flowImgConc);
        cv::hconcat(flowImgConc, flowImgPose, flowImgConc);
        cv::vconcat(flowImgConc, debugImgOut, debugImgOut);
        cv::vconcat(debugImgOut, trackImgMatches, debugImgOut);

        cv::resize(debugImgOut, debugImgOut, cv::Size(debugImgOut.cols/4, debugImgOut.rows/4));

        cv::imshow(debugWinName, debugImgOut);
        cv::imshow(trajWinName, debugImgTraj);

        cv::waitKey(1);
	}

    return 0;
}
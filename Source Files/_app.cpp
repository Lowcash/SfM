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
        m_window.showWidget("point_cloud_" + std::to_string(numClouds), pointCloudWidget);
        
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

void homogenousPointsToRGBCloud(cv::Mat imgColor, cv::Mat homPoints, std::vector<cv::Point2f> features, std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& pointsRGB) {
    cv::Mat pointMatTmp; cv::convertPointsFromHomogeneous(homPoints.t(), pointMatTmp);

    points3D.clear();
    pointsRGB.clear();

    for(int i = 0; i < pointMatTmp.rows; ++i) {
        cv::Point3d point3D = (cv::Point3d)pointMatTmp.at<cv::Point3f>(i, 0);
        cv::Point2i point2D = (cv::Point2i)features[i];
        
        if (point3D.z > 0.0001 && (point2D.x > 0 && point2D.y > 0 && point2D.x < imgColor.cols && point2D.y < imgColor.rows)) {
            
            points3D.emplace_back(point3D);
            pointsRGB.emplace_back(imgColor.at<cv::Vec3b>(point2D));
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
    cv::Mat descriptor;

    void addTrack(const cv::Point2f point2D, const cv::Point3d point3D, const cv::Vec3b color, const cv::Mat descriptor) {
        tracks.emplace_back(Track(point2D, point3D, color));
        this->descriptor.push_back(descriptor);
    }

    void clear() {
        tracks.clear();
        descriptor.release();
    }
};

void tDrawMatches(cv::Mat prevImg, std::vector<Track> tracks, cv::Mat currImg, std::vector<cv::KeyPoint> currKeyPts, std::vector<cv::DMatch> matches, cv::Mat& out) {
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

    std::cout << "Camera intrices read...";

    cv::FileStorage fs(cameraParamsPath, cv::FileStorage::READ);

    cv::FileNode fnTime = fs.root();
    std::string time = fnTime["calibration_time"];

    cv::Size camSize = cv::Size((int)fs["image_width"], (int)fs["image_height"]);

    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;
    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;

    cv::Point2d pp(cameraK.at<double>(0, 2), cameraK.at<double>(1, 2));
    double focal = (cameraK.at<double>(0, 0) + cameraK.at<double>(1, 1)) / 2.0;

    cv::Matx33d cameraK33d(
        cameraK.at<double>(0,0), cameraK.at<double>(0,1), cameraK.at<double>(0,2),
        cameraK.at<double>(1,0), cameraK.at<double>(1,1), cameraK.at<double>(1,2), 
        cameraK.at<double>(2,0), cameraK.at<double>(2,1), cameraK.at<double>(2,2)
    );

    std::cout << "[DONE]" << "\n";
    std::cout << "Creation time: " << time << "\n";
    std::cout << "-----------------------------------------" << "\n";
    std::cout << "Camera intrices: \n" << cameraK << "\n";
    std::cout << "Distortion coefficients: \n" << distCoeffs << "\n";
    std::cout << "Principal point: " << pp << "\n";
    std::cout << "Focal length: " << focal << "\n";
    std::cout << "-----------------------------------------" << "\n";
    
    cv::Mat prevImgColor, currImgColor, prevImgGray, currImgGray, prevDesc, currDesc;
    cv::Mat R_f, t_f, R, t, E, mask, homogPoints3D;
    std::vector<cv::Point2f> prevFeatures, currFeatures;
    std::vector<cv::KeyPoint> prevKeyPts, currKeyPts, trackKeyPts;
    std::vector<cv::Affine3d> camPoses; cv::Affine3d camPose;
    std::vector<std::vector<cv::Point3d>> pointCloud; std::vector<cv::Point3d> points3D;
    std::vector<std::vector<cv::Vec3b>> pointCloudRGB; std::vector<cv::Vec3b> pointsRGB;

    cv::VideoCapture cap;

    if(!cap.open(videoPath)) {
        printf("Error opening video stream or file!!\n");
        exit(1);
    }

    Visualization vis("Point cloud");

    std::thread visThread(&Visualization::visualize, &vis);

#pragma endregion INIT
    cap >> prevImgColor;
    
    cv::cvtColor(prevImgColor, prevImgGray, cv::COLOR_BGR2GRAY);

    featureDetection(prevImgGray, prevFeatures, numOfUsedDescriptors); 

    if(!findGoodImagePair(cap, numOfUsedOffsetFrames, prevImgColor, currImgColor, prevImgGray, currImgGray, prevFeatures, currFeatures, minNumHomographyInliers, cameraK, distCoeffs, R, t, E, mask, homogPoints3D)) { exit(1); }

    auto detector = cv::AKAZE::create();

    detector->detectAndCompute(prevImgGray, cv::noArray(), prevKeyPts, prevDesc);
    detector->detectAndCompute(currImgGray, cv::noArray(), currKeyPts, currDesc);

    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(12, 20, 2));
    cv::BFMatcher matcher(cv::BFMatcher(cv::NORM_HAMMING));

    std::vector<cv::DMatch> matches;
    std::vector< std::vector<cv::DMatch>> knnMatches;

    matcher.knnMatch( prevDesc, currDesc, knnMatches, 2);
    const float ratioThresh = 0.5f;
    
    for (size_t i = 0; i < knnMatches.size(); ++i) {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
            matches.emplace_back(knnMatches[i][0]);
    }

    std::vector<cv::Point2f> lPointsAligned, rPointsAligned;
    for (const auto& m : matches) {
        lPointsAligned.emplace_back(prevKeyPts[m.queryIdx].pt);
        rPointsAligned.emplace_back(currKeyPts[m.trainIdx].pt);
    }

    R_f = R.clone();
    t_f = t.clone();

    camPose = cv::Affine3d(R_f, t_f); camPoses.emplace_back(camPose);

    cv::Matx34d left = cameraK33d * cv::Matx34d::eye();

    cv::Matx34d right = cameraK33d * cv::Matx34d(
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
    );

    cv::Mat prevNorm; cv::undistort(lPointsAligned, prevNorm, cameraK33d, cv::Mat());
    cv::Mat currNorm; cv::undistort(rPointsAligned, currNorm, cameraK33d, cv::Mat());

    cv::triangulatePoints(left, right, prevNorm, currNorm, homogPoints3D);

    homogenousPointsToRGBCloud(prevImgColor, homogPoints3D, prevFeatures, points3D, pointsRGB);

    Tracking tracking;
    for (int i=0;i<matches.size();++i) {
        tracking.addTrack(rPointsAligned[i], points3D[i], pointsRGB[i], currDesc.row(matches[i].trainIdx));
    }
        
    cv::Mat pose = prevImgColor.clone();
    cv::Mat keyPt = prevImgColor.clone();

    setMaskedDescriptors(pose, keyPt, mask, currFeatures, prevFeatures);

    pointCloud.emplace_back(points3D);
    pointCloudRGB.emplace_back(pointsRGB);

    vis.updatePointCloud(points3D, pointsRGB, camPose);
    vis.updateCameraPose(camPose, cameraK33d);

    vis.updateVieverPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0, 0, -100)));

    cv::Mat prevCurrMatches; cv::drawMatches( prevImgColor, prevKeyPts, currImgColor, currKeyPts, matches, prevCurrMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::resize(prevCurrMatches, prevCurrMatches, cv::Size(prevCurrMatches.cols/4, prevCurrMatches.rows/4));
    cv::resize(pose, pose, cv::Size(pose.cols / 2, pose.rows / 2));
    cv::resize(keyPt, keyPt, cv::Size(keyPt.cols / 2, keyPt.rows / 2));

    cv::imshow("Previous/Current frame matches", prevCurrMatches);
    //cv::imshow("Pose", pose);

    cv::waitKey();

    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
    std::cout << "Tracks: " << tracking.tracks.size() << "\n";

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
	for ( ; ; ) {
        cv::swap(prevImgColor, currImgColor);
        cv::swap(prevImgGray, currImgGray);
        cv::swap(prevDesc, currDesc);
        std::swap(prevKeyPts, currKeyPts);
        
        std::cout << "Num features: " << prevFeatures.size() << "\n";
        if (prevFeatures.size() < minNumOfUsedDescriptors)	{
            featureDetection(prevImgGray, prevFeatures, numOfUsedDescriptors);
 	    }

        if(!findGoodImagePair(cap, numOfUsedOffsetFrames, prevImgColor, currImgColor, prevImgGray, currImgGray, prevFeatures, currFeatures, minNumHomographyInliers, cameraK, distCoeffs, R, t, E, mask, homogPoints3D)) { exit(1); }

        detector->detectAndCompute(currImgGray, cv::noArray(), currKeyPts, currDesc);
        matches.clear(); knnMatches.clear();

        matcher.knnMatch( prevDesc, currDesc, knnMatches, 2);
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
                matches.emplace_back(knnMatches[i][0]);
        }

        lPointsAligned.clear(); rPointsAligned.clear(); 
        for (const auto& m : matches) {
            lPointsAligned.emplace_back(prevKeyPts[m.queryIdx].pt);
            rPointsAligned.emplace_back(currKeyPts[m.trainIdx].pt);
        }

        t_f = t_f - absoluteScale * (R_f * t);
        R_f = R_f * R;

        pose = prevImgColor.clone();
        keyPt = prevImgColor.clone();

        setMaskedDescriptors(pose, keyPt, mask, currFeatures, prevFeatures);

        camPose = cv::Affine3d(R_f, t_f); camPoses.emplace_back(camPose);

        setImageTrajectory(traj, t_f);

        right = cameraK33d * cv::Matx34d(
            R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
        );

        cv::undistort(lPointsAligned, prevNorm, cameraK33d, cv::Mat());
        cv::undistort(rPointsAligned, currNorm, cameraK33d, cv::Mat());

        cv::triangulatePoints(left, right, prevNorm, currNorm, homogPoints3D);

        homogenousPointsToRGBCloud(prevImgColor, homogPoints3D, prevFeatures, points3D, pointsRGB);

        std::map<std::pair<float, float>, cv::Point3d> point3DMap;
        for(int i = 0; i < points3D.size(); ++i)
            point3DMap[std::make_pair(rPointsAligned[i].x, rPointsAligned[i].y)] = points3D[i];
        

        std::vector<cv::DMatch> tMatches; knnMatches.clear();
        matcher.knnMatch( tracking.descriptor, currDesc, knnMatches, 2);
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
                tMatches.emplace_back(knnMatches[i][0]);
        }
        
        for (const auto& m : tMatches) {
            std::pair<float, float> mapKey = std::make_pair(currKeyPts[m.trainIdx].pt.x, currKeyPts[m.trainIdx].pt.y);

            tracking.tracks[m.queryIdx].activePosition2d = currKeyPts[m.trainIdx].pt;
            tracking.tracks[m.queryIdx].activePosition3d = point3DMap[mapKey];

            std::cout << tracking.tracks[m.queryIdx].basePosition2d << "," << tracking.tracks[m.queryIdx].activePosition2d << "\n";
            std::cout << tracking.tracks[m.queryIdx].basePosition3d << "," << tracking.tracks[m.queryIdx].activePosition3d << "\n";
        }

        cv::drawMatches( prevImgColor, prevKeyPts, currImgColor, currKeyPts, matches, prevCurrMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        cv::Mat trackMatches; tDrawMatches(prevImgColor, tracking.tracks, currImgColor, currKeyPts, tMatches, trackMatches);

        tracking.clear();
        for (int i=0;i<matches.size();++i)
            tracking.addTrack(rPointsAligned[i], points3D[i], pointsRGB[i], currDesc.row(matches[i].trainIdx));
        
        //vis.updateVieverPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0, 0, -200)));
        //vis.clear();
        //vis.updatePointCloud(points3D, pointsRGB, camPose);
        vis.updateCameraPose(camPose, cameraK33d);

        cv::resize(prevCurrMatches, prevCurrMatches, cv::Size(prevCurrMatches.cols/4, prevCurrMatches.rows/4));
        cv::resize(trackMatches, trackMatches, cv::Size(trackMatches.cols/4, trackMatches.rows/4));
        cv::resize(pose, pose, cv::Size(pose.cols / 2, pose.rows / 2));
        cv::resize(keyPt, keyPt, cv::Size(keyPt.cols / 2, keyPt.rows / 2));

        cv::imshow("Previous/Current frame matches", prevCurrMatches);
        cv::imshow("Track matches", trackMatches);
        cv::imshow("Pose", pose);
        cv::imshow("KeyPt", keyPt);
        cv::imshow("Trajectory", traj);

        cv::waitKey(1);
	}

    return 0;
}
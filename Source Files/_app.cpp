#include "pch.h"
#include "camera.h"
#include "tracking.h"

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

        m_window.showWidget("point_cloud_" + std::to_string(numClouds), pointCloudWidget, cloudPose);
        //m_window.showWidget("point_cloud_" + std::to_string(numClouds), pointCloudWidget);

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

void filter_stereo_features(const std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& goodMatches, double maxYDistance) {
	if (matches.size() == 0) return;

	goodMatches.clear();

	for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		// Get the position of left keypoints
		float xl = keypoints1[it->queryIdx].pt.x;
		float yl = keypoints1[it->queryIdx].pt.y;

		// Get the position of right keypoints
		float xr = keypoints2[it->trainIdx].pt.x;
		float yr = keypoints2[it->trainIdx].pt.y;

		if (abs(yl - yr) <= maxYDistance) {
			goodMatches.push_back(*it);
		}
	}
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

void featureDetection(cv::Mat img, std::vector<cv::Point2f>& points, int numKeyPts, cv::Mat& descriptor) {
    cv::goodFeaturesToTrack(img, points, 2000, 0.01, 25);
}

void setImageTrajectory(cv::Mat& outTrajectory, cv::Mat translation) {
    const int x = int(-translation.at<double>(0)) + (outTrajectory.cols / 2);
    const int y = int(-translation.at<double>(1)) + (outTrajectory.rows / 3 * 2);
    cv::circle(outTrajectory, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

    cv::rectangle(outTrajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
    char text[100]; sprintf(text, "Coordinates: x = %.3fm y = %.3fm z = %.3fm", -translation.at<double>(0)*0.01, translation.at<double>(1)*0.01, -translation.at<double>(2)*0.01);
    cv::putText(outTrajectory, text, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255), 1, 8);
}

void setMaskedDescriptors(cv::Mat& outArrows, cv::Mat& outKeyPts, cv::Mat mask, std::vector<cv::Point2f> lPoints, std::vector<cv::Point2f> rPoints) {
    for (int i = 0; i < mask.rows; ++i) {
        cv::Point2f direction = rPoints[i] - lPoints[i];

        direction *= 3;

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

size_t getNumHomographyInliers(std::vector<cv::Point2f> prevFeatures, std::vector<cv::Point2f> currFeatures, cv::Mat cameraK, cv::Mat& R, cv::Mat& t, cv::Mat& E, cv::Mat& mask, cv::Mat& points3D) {
    E = cv::findEssentialMat(prevFeatures, currFeatures, cameraK, cv::RANSAC, 0.999, 1.0, mask);

    return cv::recoverPose(E, prevFeatures, currFeatures, cameraK, R, t, 100.0, mask, points3D);
}

void homogenousPointsToRGBCloud(cv::Mat imgColor, cv::Mat homogenPoints, std::vector<cv::Point2f> features, std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& pointsRGB) {
    std::vector<cv::Point3d> points3DTmp;
    cv::convertPointsFromHomogeneous(homogenPoints.t(), points3DTmp);

    points3D.clear();
    pointsRGB.clear();

    for(int i = 0; i < points3DTmp.size(); ++i) {
        cv::Point3d point3D = (cv::Point3d)points3DTmp[i];
        cv::Point2i point2D = (cv::Point2i)features[i];
        
        if (point2D.x > 0 && point2D.y > 0 && point2D.x < imgColor.cols && point2D.y < imgColor.rows) {
            
            points3D.emplace_back(point3D);
            pointsRGB.emplace_back(imgColor.at<cv::Vec3b>(point2D));
        }
    }
}

bool findGoodImagePair(cv::VideoCapture cap, uint numOfUsedOffsetFrames, cv::Mat& prevImgColor, cv::Mat& currImgColor, cv::Mat& prevImgGray, cv::Mat& currImgGray, std::vector<cv::Point2f>& prevFeatures, std::vector<cv::Point2f>& currFeatures, uint minNumHomographyInliers, cv::Mat cameraK, cv::Mat& R, cv::Mat& t, cv::Mat& E, cv::Mat& mask, std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& pointsRGB) {
    std::vector<cv::Point2f> prevFeaturesTmp;

    cv::Mat homogPoints3D;

    int numSkippedFrames = -1, numHomographyInliers = 0;
    do {
        for (int i = 0; i < numOfUsedOffsetFrames; ++i)
            cap >> currImgColor;
        
        if (currImgColor.empty()) { return false; }

        prevFeaturesTmp.clear(); prevFeaturesTmp.insert(prevFeaturesTmp.end(), prevFeatures.begin(), prevFeatures.end());

        cv::cvtColor(currImgColor, currImgGray, cv::COLOR_BGR2GRAY);

        featureTracking(prevImgGray, currImgGray, prevFeaturesTmp, currFeatures);

        numHomographyInliers = getNumHomographyInliers(prevFeaturesTmp, currFeatures, cameraK, R, t, E, mask, homogPoints3D);

        numSkippedFrames++;

        std::cout << "Inliers count: " << numHomographyInliers << "\n";
    } while(numHomographyInliers < minNumHomographyInliers);

    if (numSkippedFrames > 0)
        std::cout << "Skipped frames: " << numSkippedFrames << "\n";

    std::swap(prevFeaturesTmp, prevFeatures);

    homogenousPointsToRGBCloud(prevImgColor, homogPoints3D, prevFeatures, points3D, pointsRGB);

    return true;
}

cv::Mat iterativeLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1) {
    cv::Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
    );

    cv::Matx41d B(-(u.x*P(2,3) -P(0,3)),
                -(u.y*P(2,3)-P(1,3)),
                -(u1.x*P1(2,3)-P1(0,3)),
                -(u1.y*P1(2,3)-P1(1,3)));
    
    cv::Mat X;cv::solve(A, B, X, cv::DECOMP_SVD);

    return X;
}

void triangulate(const std::vector<cv::Point2f>& lPoints, const std::vector<cv::Point2f>& rPoints, const cv::Mat& Kinv, const cv::Matx34d& P, const cv::Matx34d& P1, std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& pointsRGB, cv::Mat img) {
    uint pts_size = lPoints.size();
#pragma omp parallel for
    for (uint i=0; i < pts_size; ++i) {
        cv::Point2f kp = lPoints[i];
        cv::Point3d u(kp.x,kp.y,1.0);
        cv::Mat um = Kinv * cv::Mat(u);
        u = um.at<cv::Point3d>(0);
        cv::Point2f kp1 = rPoints[i];
        cv::Point3d u1(kp1.x,kp1.y,1.0);
        cv::Mat um1 = Kinv * cv::Mat(u1);
        u1 = um1.at<cv::Point3d>(0);

        cv::Mat X = iterativeLinearLSTriangulation(u,P,u1,P1);

#pragma omp critical
        {
            points3D.emplace_back(cv::Point3d(X.at<double>(0), X.at<double>(1), X.at<double>(2)));
            pointsRGB.emplace_back(img.at<cv::Vec3b>(lPoints[i]));
        }
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
    cv::Mat R_f, t_f, R, t, E, mask;
    std::vector<cv::Point2f> prevFeatures, currFeatures;
    std::vector<cv::KeyPoint> prevKeyPts, currKeyPts;
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

    Tracking tracking(0, false);

#pragma endregion INIT
    cap >> prevImgColor;

    cv::cvtColor(prevImgColor, prevImgGray, cv::COLOR_BGR2GRAY);

    featureDetection(prevImgGray, prevFeatures, numOfUsedDescriptors, prevDesc); 

    if(!findGoodImagePair(cap, numOfUsedOffsetFrames, prevImgColor, currImgColor, prevImgGray, currImgGray, prevFeatures, currFeatures, minNumHomographyInliers, cameraK, R, t, E, mask, points3D, pointsRGB)) { exit(1); }

    R_f = R.clone();
    t_f = t.clone();

    camPose = cv::Affine3d(R_f, t_f); camPoses.emplace_back(camPose);

    cv::Matx34d left =  cv::Matx34d::eye();

    cv::Matx34d right =  cv::Matx34d(
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
    );

    cv::Mat pose = prevImgColor.clone();
    cv::Mat keyPt = prevImgColor.clone();

    setMaskedDescriptors(pose, keyPt, mask, prevFeatures, currFeatures);
    
    std::vector<cv::DMatch> matches, goodMatches;
    //cv::Ptr<cv::ORB> detector = cv::ORB::create(numOfUsedDescriptors);
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(numOfUsedDescriptors);
    
    detector->detectAndCompute( prevImgGray, cv::noArray(), prevKeyPts, prevDesc );
    detector->detectAndCompute( currImgGray, cv::noArray(), currKeyPts, currDesc );

    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    // matcher->match(prevDesc, currDesc, matches, cv::noArray());

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knnMatches;

    matcher->knnMatch( prevDesc, currDesc, knnMatches, 2 );
    const float ratioThresh = 0.33f;
    
    for (size_t i = 0; i < knnMatches.size(); ++i) {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) 
            goodMatches.emplace_back(knnMatches[i][0]);
    }

    std::vector<cv::Point2f> lAlignedPoints, rAlignedPoints;

    int i = 0; cv::Mat newDesc;
    for (const auto& m : goodMatches) {
        lAlignedPoints.emplace_back(prevKeyPts[m.queryIdx].pt);
        rAlignedPoints.emplace_back(currKeyPts[m.trainIdx].pt);

        newDesc.push_back(currDesc.row(i));
        i++;
    }
    
    points3D.clear();
    pointsRGB.clear();

    triangulate(lAlignedPoints, rAlignedPoints, cameraK.inv(), left, right, points3D, pointsRGB, prevImgColor);

    pointCloud.emplace_back(points3D);
    pointCloudRGB.emplace_back(pointsRGB);

    vis.updatePointCloud(points3D, pointsRGB, camPose);
    //vis.updateCameraPose(camPose, cameraK33d);

    vis.updateVieverPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0, 0, -100)));

    cv:: Mat imgMatches; cv::drawMatches( prevImgColor, prevKeyPts, currImgColor, currKeyPts, goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::resize(imgMatches, imgMatches, cv::Size(imgMatches.cols/4, imgMatches.rows/4));
    cv::resize(pose, pose, cv::Size(pose.cols / 2, pose.rows / 2));
    cv::resize(keyPt, keyPt, cv::Size(keyPt.cols / 2, keyPt.rows / 2));

    cv::imshow("Good Matches", imgMatches);
    cv::imshow("KeyPt", keyPt);

    cv::waitKey();

    cv::swap(prevImgColor, currImgColor);
    cv::swap(prevImgGray, currImgGray);
    cv::swap(prevDesc, currDesc);
    std::swap(prevKeyPts, currKeyPts);
    std::swap(prevFeatures, currFeatures);

    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);
	for ( ; ; ) {
        if (prevFeatures.size() < minNumOfUsedDescriptors)	{
            featureDetection(prevImgGray, prevFeatures, numOfUsedDescriptors, prevDesc);
            featureTracking(prevImgGray, currImgGray, prevFeatures, currFeatures);
 	    }

        if(!findGoodImagePair(cap, numOfUsedOffsetFrames, prevImgColor, currImgColor, prevImgGray, currImgGray, prevFeatures, currFeatures, minNumHomographyInliers, cameraK, R, t, E, mask, points3D, pointsRGB)) { exit(1); }

        t_f = t_f - absoluteScale * (R_f * t);
        R_f = R_f * R;

        pose = prevImgColor.clone();
        keyPt = prevImgColor.clone();

        setMaskedDescriptors(pose, keyPt, mask, currFeatures, prevFeatures);

        camPose = cv::Affine3d(R_f, t_f); camPoses.emplace_back(camPose);

        setImageTrajectory(traj, t_f);

        detector->detectAndCompute( currImgGray, cv::noArray(), currKeyPts, currDesc );

        knnMatches.clear(); 
        matcher->knnMatch( prevDesc, currDesc, knnMatches, 2 );
        
        goodMatches.clear();
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) 
                goodMatches.emplace_back(knnMatches[i][0]);
        }

        //newDesc.release();
        lAlignedPoints.clear();
        rAlignedPoints.clear();
        for (const auto& m : goodMatches) {
            lAlignedPoints.emplace_back(prevKeyPts[m.queryIdx].pt);
            rAlignedPoints.emplace_back(currKeyPts[m.trainIdx].pt);
        }

        right = cv::Matx34d(
            R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2)
        );

        points3D.clear();
        pointsRGB.clear();

        triangulate(lAlignedPoints, rAlignedPoints, cameraK.inv(), left, right, points3D, pointsRGB, prevImgColor);

        pointCloud.emplace_back(points3D);
        pointCloudRGB.emplace_back(pointsRGB);

        cv::drawMatches( prevImgColor, prevKeyPts, currImgColor, currKeyPts, goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //vis.updateVieverPose(cv::Affine3d(cv::Vec3d(), cv::Vec3d(0, 0, -200)));
        vis.clear();
        vis.updatePointCloud(points3D, pointsRGB, camPose);
        //vis.updateCameraPose(camPose, cameraK33d);

        cv::resize(imgMatches, imgMatches, cv::Size(imgMatches.cols/4, imgMatches.rows/4));
        cv::resize(pose, pose, cv::Size(pose.cols / 2, pose.rows / 2));
        cv::resize(keyPt, keyPt, cv::Size(keyPt.cols / 2, keyPt.rows / 2));

        cv::imshow("Good Matches", imgMatches);
        cv::imshow("Pose", pose);
        cv::imshow("KeyPt", keyPt);
        cv::imshow("Trajectory", traj);

        cv::waitKey(1);

        cv::swap(prevImgColor, currImgColor);
        cv::swap(prevImgGray, currImgGray);
        cv::swap(prevDesc, currDesc);
        std::swap(prevKeyPts, currKeyPts);
        std::swap(prevFeatures, currFeatures);
	}

    /*cv::viz::Viz3d window("Camera trajectory");
    window.setBackgroundColor(cv::viz::Color::black());
    window.setWindowSize(cv::Size(500, 500));

    window.setWindowPosition(cv::Point(3840, 0));

    for (int i = 0; i < pointCloud.size() && i < pointCloudRGB.size(); ++i) {
        if(pointCloud[i].size() == pointCloudRGB[i].size()) {
            const cv::viz::WCloud cloudWidget(pointCloud[i], pointCloudRGB[i]);

            window.showWidget("point_cloud_" + std::to_string(i), cloudWidget, camPoses[i]);
        }
    } 

    const cv::viz::WCameraPosition camWidget(cameraK33, 15, cv::viz::Color::yellow());

    // cv::Mat rotVec = cv::Mat::zeros(1,3,CV_32F);

	// rotVec.at<float>(0,2) += CV_PI * 1.0f;

	// cv::Mat rotMat; cv::Rodrigues(rotVec, rotMat);

    // cv::Affine3f rotation(rotMat, cv::Vec3d());

    // window.setViewerPose(camPoses[0].rotate(rotation.rotation()).translate(cv::Vec3d(0, 0, -100)));
    
    int idx = 0, n = static_cast<int>(camPoses.size());
    while(!window.wasStopped()) {
        camPose = camPoses[idx];

        window.showWidget("camera_trajectory", cv::viz::WTrajectory(camPoses, cv::viz::WTrajectory::PATH, 1.0, cv::viz::Color::green()));
        window.showWidget("cam_widget", camWidget, camPose);

        idx = idx == n ? 0 : idx + 1;

        window.spinOnce(30, true);
    }*/

    return 0;
}
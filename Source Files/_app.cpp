#include "pch.h"
#include "camera.h"

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

void featureDetection(cv::Mat img, std::vector<cv::Point2f>& points, int numKeyPts)	{
    // std::vector<cv::KeyPoint> keyPts;

    // cv::FAST(img, keyPts, 20, true);
    // cv::KeyPoint::convert(keyPts, points, std::vector<int>());

    cv::goodFeaturesToTrack(img, points, numKeyPts, 0.01, 25);
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
    cv::convertPointsFromHomogeneous(homogenPoints.t(), points3D);

    pointsRGB.clear();

    int indexCorrection = 0;
    for(int i = 0; i < points3D.size(); ++i) {
        const cv::Point3i point3D = (cv::Point3i)points3D[i];
        const cv::Point2i point2D = (cv::Point2i)features[i];

        if (point3D.z < 0) {
            points3D.erase(points3D.begin() + (i - indexCorrection));
            indexCorrection++;
        } else
            pointsRGB.emplace_back(imgColor.at<cv::Vec3b>(point2D));
    }
}

int main(int argc, char** argv) {
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

    std::cout << "[DONE]" << "\n";
    std::cout << "Creation time: " << time << "\n";
    std::cout << "-----------------------------------------" << "\n";
    std::cout << "Camera intrices: \n" << cameraK << "\n";
    std::cout << "Distortion coefficients: \n" << distCoeffs << "\n";
    std::cout << "Principal point: " << pp << "\n";
    std::cout << "Focal length: " << focal << "\n";
    std::cout << "-----------------------------------------" << "\n";
    
    cv::Mat prevImgColor, currImgColor, prevImgGray, currImgGray;
    cv::Mat R_f, t_f, R, t, E, mask, homogPoints3D;
    std::vector<cv::Point2f> prevFeatures, currFeatures;
    std::vector<cv::Affine3d> camPoses; cv::Affine3d camPose;
    std::vector<std::vector<cv::Point3d>> pointCloud; std::vector<cv::Point3d> points3D;
    std::vector<std::vector<cv::Vec3b>> pointCloudRGB; std::vector<cv::Vec3b> pointsRGB;

    size_t numHomographyInliers;
    int numSkippedFrames;

    cv::VideoCapture cap;

    if(!cap.open(videoPath)) {
        printf("Error opening video stream or file!!\n");
        exit(1);
    }

    cap >> prevImgColor;

    // cv::pyrDown(prevImgColor, prevImgColor, cv::Size(prevImgColor.cols/2, prevImgColor.rows/2));
    // cv::pyrDown(prevImgColor, prevImgColor, cv::Size(prevImgColor.cols/2, prevImgColor.rows/2));

    cv::cvtColor(prevImgColor, prevImgGray, cv::COLOR_BGR2GRAY);
    
    featureDetection(prevImgGray, prevFeatures, numOfUsedDescriptors); 

    numSkippedFrames = -1;
    do {
        if (numSkippedFrames > maxSkipFrames) {
            cv::swap(prevImgColor, currImgColor);
            cv::swap(prevImgGray, currImgGray);
            std::swap(prevFeatures, currFeatures);

            numSkippedFrames = -1;
        }

        for (int i = 0; i < numOfUsedOffsetFrames; ++i)
            cap >> currImgColor;
        
        if (currImgColor.empty()) { exit(1); }

        // cv::pyrDown(currImgColor, currImgColor, cv::Size(currImgColor.cols/2, currImgColor.rows/2));
        // cv::pyrDown(currImgColor, currImgColor, cv::Size(currImgColor.cols/2, currImgColor.rows/2));
    
        cv::cvtColor(currImgColor, currImgGray, cv::COLOR_BGR2GRAY);

        featureTracking(prevImgGray, currImgGray, prevFeatures, currFeatures);

        // numHomographyInliers = getNumHomographyInliers(prevFeatures, currFeatures, focal, pp, R, t, E, mask);

        numHomographyInliers = getNumHomographyInliers(prevFeatures, currFeatures, cameraK, R, t, E, mask, homogPoints3D);

        numSkippedFrames++;

        std::cout << "Inliers count: " << numHomographyInliers << "\n";
    } while(numHomographyInliers < minNumHomographyInliers);

    if (numSkippedFrames > 0)
        std::cout << "Skipped frames: " << numSkippedFrames << "\n";

    homogenousPointsToRGBCloud(prevImgColor, homogPoints3D, prevFeatures, points3D, pointsRGB);

    pointCloud.emplace_back(points3D);
    pointCloudRGB.emplace_back(pointsRGB);

    cv::Mat pose = prevImgColor.clone();
    cv::Mat keyPt = prevImgColor.clone();

    setMaskedDescriptors(pose, keyPt, mask, prevFeatures, currFeatures);
    
    cv::resize(pose, pose, cv::Size(pose.cols / 2, pose.rows / 2));
    cv::resize(keyPt, keyPt, cv::Size(keyPt.cols / 2, keyPt.rows / 2));

    cv::swap(prevImgColor, currImgColor);
    cv::swap(prevImgGray, currImgGray);
    std::swap(prevFeatures, currFeatures);

    R_f = R.clone();
    t_f = t.clone();

    camPose.matrix = cv::Matx44d(
        R_f.at<double>(0,0), R_f.at<double>(0,1), R_f.at<double>(0,2), t_f.at<double>(0),
        R_f.at<double>(1,0), R_f.at<double>(1,1), R_f.at<double>(1,2), t_f.at<double>(1),
        R_f.at<double>(2,0), R_f.at<double>(2,1), R_f.at<double>(2,2), -t_f.at<double>(2),
        0, 0, 0, 1
    );

    camPoses.emplace_back(camPose);

    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

	for ( ; ; ) {

        numSkippedFrames = -1;
        do {
            if (numSkippedFrames > maxSkipFrames) {
                cv::swap(prevImgColor, currImgColor);
                cv::swap(prevImgGray, currImgGray);
                std::swap(prevFeatures, currFeatures);

                numSkippedFrames = -1;
            }

            for (int i = 0; i < numOfUsedOffsetFrames; ++i)
                cap >> currImgColor;

            if (currImgColor.empty()) { break; }

            // cv::pyrDown(currImgColor, currImgColor, cv::Size(currImgColor.cols/2, currImgColor.rows/2));
            // cv::pyrDown(currImgColor, currImgColor, cv::Size(currImgColor.cols/2, currImgColor.rows/2));

            cv::cvtColor(currImgColor, currImgGray, cv::COLOR_BGR2GRAY);

            featureTracking(prevImgGray, currImgGray, prevFeatures, currFeatures);

            //numHomographyInliers = getNumHomographyInliers(prevFeatures, currFeatures, focal, pp, R, t, E, mask);

            numHomographyInliers = getNumHomographyInliers(prevFeatures, currFeatures, cameraK, R, t, E, mask, homogPoints3D);

            numSkippedFrames++;

            //std::cout << "Inliers count: " << numHomographyInliers << "\n";
        } while(numHomographyInliers < minNumHomographyInliers);

        if (currImgColor.empty()) { break; }

        //if (numSkippedFrames > 0)
        //    std::cout << "Skipped frames: " << numSkippedFrames << "\n";

        homogenousPointsToRGBCloud(prevImgColor, homogPoints3D, prevFeatures, points3D, pointsRGB);

        pointCloud.emplace_back(points3D);
        pointCloudRGB.emplace_back(pointsRGB);

        pose = prevImgColor.clone();
        keyPt = prevImgColor.clone();

        setMaskedDescriptors(pose, keyPt, mask, currFeatures, prevFeatures);

        t_f = t_f + absoluteScale * (R_f * t);
        R_f = R * R_f;
        
        camPose.matrix = cv::Matx44d(
            R_f.at<double>(0,0), R_f.at<double>(0,1), R_f.at<double>(0,2), t_f.at<double>(0),
            R_f.at<double>(1,0), R_f.at<double>(1,1), R_f.at<double>(1,2), t_f.at<double>(1),
            R_f.at<double>(2,0), R_f.at<double>(2,1), R_f.at<double>(2,2), -t_f.at<double>(2),
            0, 0, 0, 1
        );

        camPoses.emplace_back(camPose);

        if (prevFeatures.size() < minNumOfUsedDescriptors)	{
            featureDetection(prevImgGray, prevFeatures, numOfUsedDescriptors);
            featureTracking(prevImgGray, currImgGray, prevFeatures, currFeatures);
 	    }

        cv::swap(prevImgColor, currImgColor);
        cv::swap(prevImgGray, currImgGray);
        std::swap(prevFeatures, currFeatures);

        setImageTrajectory(traj, t_f);

        cv::resize(pose, pose, cv::Size(pose.cols / 2, pose.rows / 2));
        cv::resize(keyPt, keyPt, cv::Size(keyPt.cols / 2, keyPt.rows / 2));

        cv::imshow("Pose", pose);
        //cv::imshow("KeyPt", keyPt);
        cv::imshow("Trajectory", traj);

        cv::waitKey(1);
	}

    cv::viz::Viz3d window("Camera trajectory");
    window.setBackgroundColor(cv::viz::Color::black());
    window.setWindowSize(cv::Size(500, 500));

    window.setWindowPosition(cv::Point(3840, 0));

    cv::Matx33d cameraK33(
        cameraK.at<double>(0, 0), cameraK.at<double>(0, 1), cameraK.at<double>(0, 2),
        cameraK.at<double>(1, 0), cameraK.at<double>(1, 1), cameraK.at<double>(1, 2),
        cameraK.at<double>(2, 0), cameraK.at<double>(2, 1), cameraK.at<double>(2, 2)
    );

    int idx = 0, forw = -1, n = static_cast<int>(camPoses.size());
    
    // for (int i = 0; i < camPoses.size(); ++i) {
    //     const cv::viz::WSphere camPosWidget(cv::Point3d(), 1, 10, cv::viz::Color::orange());

    //     window.showWidget("cam_pose_" + std::to_string(i), camPosWidget, camPoses[i]);
    // }  

    for (int i = 0; i < pointCloud.size() && i < pointCloudRGB.size(); ++i) {
        if(pointCloud[i].size() == pointCloudRGB[i].size()) {
            const cv::viz::WCloud cloudWidget(pointCloud[i], pointCloudRGB[i]);

            window.showWidget("point_cloud_" + std::to_string(i), cloudWidget, camPoses[i]);
        }
    } 

    const cv::viz::WCameraPosition camWidget(cameraK33, 15, cv::viz::Color::yellow());

    cv::Mat rotVec = cv::Mat::zeros(1,3,CV_32F);

	rotVec.at<float>(0,2) += CV_PI * 1.0f;

	cv::Mat rotMat; cv::Rodrigues(rotVec, rotMat);

    cv::Affine3f rotation(rotMat, cv::Vec3d());

    window.setViewerPose(camPoses[0].rotate(rotation.rotation()).translate(cv::Vec3d(0, 0, -100)));
    
    while(!window.wasStopped()) {
        camPose = camPoses[idx];

        window.showWidget("camera_trajectory", cv::viz::WTrajectory(camPoses, cv::viz::WTrajectory::PATH, 1.0, cv::viz::Color::green()));
        window.showWidget("cam_widget", camWidget, camPose);

        idx = idx == n ? 0 : idx + 1;

        window.spinOnce(30, true);
    }

    return 0;
}
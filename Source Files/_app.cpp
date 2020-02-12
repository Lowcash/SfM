#include "pch.h"
#include "camera.h"

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  std::vector<float> err;					
  cv::Size winSize=cv::Size(21,21);																								
  cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}

void featureDetection(cv::Mat img_1, std::vector<cv::Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  std::vector<cv::KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());
}

// float getAbsoluteScale(std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts) {
//     float sumScale = 0;

//     for (int i = 0; i < prevPts.size() && i < currPts.size(); ++i) {
//         sumScale += sqrt(pow((currPts[i].x - prevPts[i].x),2) + pow((currPts[i].y - prevPts[i].y),2));
//     }

//     return sumScale / prevPts.size();
// }

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {
    std::string line;
    int i = 0;
    std::ifstream myfile ("/home/lowcash/Documents/SfM_Flow/Resource Files/00/times.txt");
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()) {

        while (( getline (myfile,line) ) && (i<=frame_id)) {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++) {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }
        
            i++;
        }

        myfile.close();
    }

    else {
        std::cout << "Unable to open file";
        return 0;
    }

    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{help h ? |       | help }"
        "{calibCam | .     | camera intric parameters file path }"
		"{source   | .     | source video file [.mp4, .avi ...] }"
        "{numKeyPts| 0     | number of detector used key points/descriptors }"
        "{offFrames| 0     | number of offset frames used for reconstruction }"
        "{minAbsSc | 10     | minimal absolute scale }"
        "{visDebug | false | enable debug visualization }"
    );

	if (parser.has("help")) {
        parser.printMessage();
        exit(0);
    }

    const std::string cameraParamsPath = parser.get<std::string>("calibCam");
    const std::string videoPath = parser.get<std::string>("source");
    const int numOfUsedDescriptors = parser.get<int>("numKeyPts");
    const int numOfUsedOffsetFrames = parser.get<int>("offFrames");
    const int minAbsoluteScale = parser.get<int>("minAbsSc");
    const bool isDebugVisualization = parser.get<bool>("visDebug");

    std::cout << "Camera intrices read...";

    cv::FileStorage fs(cameraParamsPath, cv::FileStorage::READ);

    cv::FileNode fnTime = fs.root();
    std::string time = fnTime["calibration_time"];

    cv::Size imageWidth = cv::Size((int)fs["image_width"], (int)fs["image_height"]);

    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;
    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;

    std::cout << "[DONE]" << "\n";
    std::cout << "Creation time: " << time << "\n";
    std::cout << "-----------------------------------------" << "\n";
    std::cout << "Camera intrices: \n" << cameraK << "\n";
    std::cout << "Distortion coefficients: \n" << distCoeffs << "\n";
    std::cout << "-----------------------------------------" << "\n";
    
    cv::Mat img_1, img_2, img_1_c, img_2_c;
    cv::Mat R_f, t_f; //the final rotation and tranlation vectors containing the 
    double scale = 1.00;

    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;  
    cv::Point textOrg(10, 50);

    // img_1_c = cv::imread(videoPath + "0000.png");
    // img_2_c = cv::imread(videoPath + "0001.png");

    // if ( !img_1_c.data || !img_2_c.data ) { 
    //     std::cout<< " --(!) Error reading images " << std::endl; return -1;
    // }
    
    cv::VideoCapture cap;
    cv::Vec3d prevVelocity, currVelocity;

    if(!cap.open(videoPath)) {
        printf("Error opening video stream or file!!\n");
        exit(1);
    }

    cap >> img_1_c;
    cap >> img_2_c;

    cv::cvtColor(img_1_c, img_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_2_c, img_2, cv::COLOR_BGR2GRAY);

    // feature detection, tracking
    std::vector<cv::Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureDetection(img_1, points1);        //detect features in img_1
    std::vector<uchar> status;
    featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2

    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    //double focal = 718.8560;
    //cv::Point2d pp(607.1928, 185.2157);
    double focal = 567.7815;
    cv::Point2d pp(1614.4588, 1002.8441);
    //recovering the pose and the essential matrix
    cv::Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);

    prevVelocity = cv::Vec3d(t.at<double>(0), t.at<double>(1), t.at<double>(2));

    cv::Mat prevImage = img_2;
    cv::Mat currImage;
    std::vector<cv::Point2f> prevFeatures = points2;
    std::vector<cv::Point2f> currFeatures;

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    cv::namedWindow( "Road facing camera", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::namedWindow( "Trajectory", cv::WINDOW_AUTOSIZE );// Create a window for display.

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

	for (int numFrame = 2; ; numFrame++) {
        // std::string offset;
        // if (numFrame < 1000) offset += "0";
        // if (numFrame < 100) offset += "0";
        // if (numFrame < 10) offset += "0";

        // cv::Mat currImage_c = cv::imread(videoPath + offset + std::to_string(numFrame) + ".png");
        cv::Mat currImage_c;
        cap >> currImage_c;

        if(currImage_c.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);

            break;
        }

        if (currImage_c.empty()) { break; }

        cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY);
        std::vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        E = findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

        currVelocity = cv::Vec3d(t.at<double>(0), t.at<double>(1), t.at<double>(2));

        //scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
        scale = minAbsoluteScale;

        //if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
        if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
            t_f = t_f + scale*(R_f*t);
            R_f = R*R_f;

            std::cout << "Scale is " << scale << "\n";
        }
        else {
            std::cout << "Incorrect scale is " << scale << "\n";
        }
        
        // lines for printing results
        // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

        // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT)	{
            //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
            //cout << "trigerring redection" << endl;
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);
 	    }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        int x = int(t_f.at<double>(0)) + (traj.cols / 2);
        int y = int(-t_f.at<double>(2)) + (traj.rows / 3 * 2);
        cv::circle(traj, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

        cv::rectangle( traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

        cv::imshow( "Road facing camera", currImage_c );
        cv::imshow( "Trajectory", traj );

        cv::waitKey(1);
	}

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Total time taken: " << elapsed_secs << "s" << std::endl;

    //cout << R_f << endl;
    //cout << t_f << endl;

    return 0;
}
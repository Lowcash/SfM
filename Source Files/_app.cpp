#include "pch.h"
#include "app_solver.h"

int main(int argc, char** argv) {
#pragma region INIT
    std::cout << "Using OpenCV " << cv::getVersionString().c_str() << std::flush;

    //  Prepare input parameters
    cv::CommandLineParser parser(argc, argv,
		"{ help h ?  |             | help }"
        "{ bSource   | .           | source video file [.mp4, .avi ...] }"
		"{ bcalib    | .           | camera intrics parameters file path }"
        "{ bDownSamp | 0.5         | downsampling of input source images }"
        "{ bWinWidth | 960         | debug windows width }"
        "{ bWinHeight| 540         | debug windows height }"
        "{ bUseMethod| PNP         | method to use KLT/VO/PNP }"
        "{ bMaxSkFram| 10          | max number of skipped frames to swap }"
        "{ bDebugVisE| true        | enable debug point cloud visualization by VTK, PCL }"
        "{ bDebugMatE| false       | enable debug matching visualization by GTK/... }"

        "{ fDecType  | AKAZE       | used detector type }"
        "{ fMatchType| BRUTEFORCE_HAMMING  | used matcher type }"
        "{ fKnnRatio | 0.5         | knn ration match }"

        "{ ofMinKPts | 333         | optical flow min descriptor to generate new one }"
        "{ ofWinSize | 30          | optical flow window size }"
        "{ ofMaxLevel| 5           | optical flow max pyramid level }"
        "{ ofMaxItCt | 250         | optical flow max iteration count }"
        "{ ofItEps   | 0.0         | optical flow iteration epsilon }"
        "{ ofMaxError| 10.0        | optical flow max error }"
        "{ ofMaxCorn | 2000        | optical flow max generated corners }"
        "{ ofQualLvl | 0.1         | optical flow generated corners quality level }"
        "{ ofMinDist | 5           | optical flow generated corners min distance }"

        "{ peMethod  | RANSAC      | pose estimation fundamental matrix computation method [RANSAC/LMEDS] }"
        "{ peProb    | 0.99        | pose estimation confidence/probability }"
        "{ peThresh  | 0.5         | pose estimation threshold }"
        "{ peMinInl  | 10          | pose estimation in number of homography inliers user for reconstruction }"
        "{ peMinMatch| 50          | pose estimation min matches to break }"
       
        "{ pePMetrod | SOLVEPNP_P3P| pose estimation method ITERATIVE/SOLVEPNP_P3P/SOLVEPNP_AP3P/SOLVEPNP_EPNP }"
        "{ peExGuess | false       | pose estimation use extrinsic guess }"
        "{ peNumIteR | 500         | pose estimation max iteration }"

        "{ baMethod  | SPARSE_NORMAL_CHOLESKY | bundle adjustment solver type DENSE_SCHUR/SPARSE_NORMAL_CHOLESKY }"
        "{ baMaxRMSE | 10.0        | bundle adjustment max RMSE error to recover from back up }"
        "{ baUpdLck  | 5           | bundle adjustment lock block after %d updates }"
        "{ baProcIt  | 5           | bundle adjustment process each %d iteration }"

        "{ tMethod   | ITERATIVE   | triangulation method ITERATIVE/DLT }"
        "{ tMinDist  | 0.0001      | triangulation points min distance }"
        "{ tMaxDist  | 250         | triangulation points max distance }"
        "{ tMaxPErr  | 3.0         | triangulation points max reprojection error }"

        "{ cSRemThr  | 1.00        | statistical outlier removal stddev multiply threshold }"
        "{ cLSize    | 0.25        | cloud leaf filter size }"
        "{ cSRange   | 1.00        | cloud radius search radius distance }"
        "{ cFProcIt  | 5           | cloud filter process each %d iteration }"
    );

    //  Show help info
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
    const std::string bUseMethod = parser.get<std::string>("bUseMethod");
    const int bMaxSkFram = parser.get<int>("bMaxSkFram");
    const bool bDebugVisE = parser.get<bool>("bDebugVisE");
    const bool bDebugMatE = parser.get<bool>("bDebugMatE");

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
    const int peMinMatch = parser.get<int>("peMinMatch");
    
    const std::string pePMetrod = parser.get<std::string>("pePMetrod");
    const bool peExGuess = parser.get<bool>("peExGuess");
    const int peNumIteR = parser.get<int>("peNumIteR");

    //-------------------------- BUNDLE ADJUSTMENT --------------------------//
    const std::string baMethod = parser.get<std::string>("baMethod");
    const double baMaxRMSE = parser.get<double>("baMaxRMSE");
    const int baUpdLck = parser.get<int>("baUpdLck");
    const int baProcIt = parser.get<int>("baProcIt");

    //---------------------------- TRIANGULATION ----------------------------//
    const std::string tMethod = parser.get<std::string>("tMethod");
    const float tMinDist = parser.get<float>("tMinDist");
    const float tMaxDist = parser.get<float>("tMaxDist");
    const float tMaxPErr = parser.get<float>("tMaxPErr");

    //----------------------------- CLOUD FILTER ----------------------------//
    const float cSRemThr = parser.get<float>("cSRemThr");
    const float cLSize = parser.get<float>("cLSize");
    const double cSRange = parser.get<double>("cSRange");
    const int cFProcIt = parser.get<int>("cFProcIt");

    //  Read camera calibration script
    const cv::FileStorage fs(bcalib, cv::FileStorage::READ);

    int cameraWidth; fs["image_width"] >> cameraWidth;
    int cameraHeight; fs["image_height"] >> cameraHeight;

    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;

    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;
    
    const std::string ptCloudWinName = "Point cloud";
    const std::string usrInpWinName = "User input/output";
    const std::string recPoseWinName = "Recovery pose";
    const std::string matchesWinName = "Matches";

    AppSolver solver(AppSolverDataParams(bUseMethod, ptCloudWinName, usrInpWinName, recPoseWinName, matchesWinName, bSource, bDownSamp, bMaxSkFram, cv::Size(bWinWidth, bWinHeight), cv::Size(cameraWidth, cameraHeight), bDebugVisE, bDebugMatE, fDecType, fMatchType, fKnnRatio, ofMinKPts, ofWinSize, ofMaxLevel, ofMaxItCt, ofItEps, ofMaxError, ofMaxCorn, ofQualLvl, ofMinDist, peMethod, peProb, peThresh, peMinInl, peMinMatch, pePMetrod, peExGuess, peNumIteR, baMethod, baMaxRMSE, baUpdLck, baProcIt, tMethod, tMinDist, tMaxDist, tMaxPErr, cameraK, distCoeffs, cSRemThr, cLSize, cSRange, cFProcIt));

#pragma endregion INIT 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    solver.run();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\n----------------------------------------------------------\n\n";
    std::cout << "Total computing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "milliseconds!\n";

    exit(0);
}
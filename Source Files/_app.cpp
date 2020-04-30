#include "pch.h"
#include "app_solver.h"

int main(int argc, char** argv) {
#pragma region INIT
    std::cout << "Using OpenCV " << cv::getVersionString().c_str() << std::flush;

    cv::CommandLineParser parser(argc, argv,
		"{ help h ?  |             | help }"
        "{ bSource   | .           | source video file [.mp4, .avi ...] }"
		"{ bcalib    | .           | camera intric parameters file path }"
        "{ bDownSamp | 1           | downsampling of input source images }"
        "{ bWinWidth | 1024        | debug windows width }"
        "{ bWinHeight| 768         | debug windows height }"
        "{ bUseMethod| KLT         | method to use KLT/VO/PNP }"
        "{ bUseCuda  | false       | is nVidia CUDA used }"

        "{ fDecType  | AKAZE       | used detector type }"
        "{ fMatchType| BRUTEFORCE  | used matcher type }"
        "{ fKnnRatio | 0.75        | knn ration match }"

        "{ ofMinKPts | 500         | optical flow min descriptor to generate new one }"
        "{ ofWinSize | 21          | optical flow window size }"
        "{ ofMaxLevel| 3           | optical flow max pyramid level }"
        "{ ofMaxItCt | 30          | optical flow max iteration count }"
        "{ ofItEps   | 0.1         | optical flow iteration epsilon }"
        "{ ofMaxError| 0           | optical flow max error }"
        "{ ofMaxCorn | 500         | optical flow max generated corners }"
        "{ ofQualLvl | 0.1         | optical flow generated corners quality level }"
        "{ ofMinDist | 25.0        | optical flow generated corners min distance }"

        "{ peMethod  | RANSAC      | pose estimation fundamental matrix computation method [RANSAC/LMEDS] }"
        "{ peProb    | 0.999       | pose estimation confidence/probability }"
        "{ peThresh  | 1.0         | pose estimation threshold }"
        "{ peMinInl  | 50          | pose estimation in number of homography inliers user for reconstruction }"
        "{ peMinMatch| 50          | pose estimation min matches to break }"
       
        "{ pePMetrod | SOLVEPNP_P3P| pose estimation method SOLVEPNP_ITERATIVE/SOLVEPNP_P3P/SOLVEPNP_AP3P }"
        "{ peExGuess | false       | pose estimation use extrinsic guess }"
        "{ peNumIteR | 250         | pose estimation max iteration }"

        "{ baLibrary | CERES       | bundle adjustment minizer CERES/GTSAM }"
        "{ baCMethod | DENSE_SCHUR | bundle adjustment CERES solver type DENSE_SCHUR/SPARSE_NORMAL_CHOLESKY }"
        
        "{ baNumIter | 50          | bundle adjustment max iteration }"

        "{ tMethod   | ITERATIVE   | triangulation method ITERATIVE/DLT }"
        "{ tMinDist  | 1.0         | triangulation points min distance }"
        "{ tMaxDist  | 100.0       | triangulation points max distance }"
        "{ tMaxPErr  | 100.0       | triangulation points max reprojection error }"
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
    const std::string bUseMethod = parser.get<std::string>("bUseMethod");
    const bool bUseCuda = parser.get<bool>("bUseCuda");

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
    const std::string baLibrary = parser.get<std::string>("baLibrary");
    const std::string baCMethod = parser.get<std::string>("baCMethod");
    const int baNumIter = parser.get<int>("baNumIter");

    //---------------------------- TRIANGULATION ----------------------------//
    const std::string tMethod = parser.get<std::string>("tMethod");
    const float tMinDist = parser.get<float>("tMinDist");
    const float tMaxDist = parser.get<float>("tMaxDist");
    const float tMaxPErr = parser.get<float>("tMaxPErr");

    bool useCUDA = false;

#pragma ifdef OPENCV_CORE_CUDA_HPP
    if (bUseCuda) {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            std::cout << " with CUDA support\n";

            cv::cuda::setDevice(0);
            cv::cuda::printShortCudaDeviceInfo(0);

            useCUDA = true;
        }
        else
            std::cout << "\nCannot use nVidia CUDA -> no devices" << "\n"; 
    }
#pragma endif
    const cv::FileStorage fs(bcalib, cv::FileStorage::READ);
    cv::Mat cameraK; fs["camera_matrix"] >> cameraK;

    cv::Mat distCoeffs; fs["distortion_coefficients"] >> distCoeffs;
    
    const std::string ptCloudWinName = "Point cloud";
    const std::string usrInpWinName = "User input/output";
    const std::string recPoseWinName = "Recovery pose";
    const std::string matchesWinName = "Matches";

    AppSolver solver(AppSolverDataParams(bUseMethod, ptCloudWinName, usrInpWinName, recPoseWinName, matchesWinName, bSource, bDownSamp, cv::Size(bWinWidth, bWinHeight), useCUDA, fDecType, fMatchType, fKnnRatio, ofMinKPts, ofWinSize, ofMaxLevel, ofMaxItCt, ofItEps, ofMaxError, ofMaxCorn, ofQualLvl, ofMinDist, peMethod, peProb, peThresh, peMinInl, peMinMatch, pePMetrod, peExGuess, peNumIteR, baLibrary, baCMethod, baNumIter, tMethod, tMinDist, tMaxDist, tMaxPErr, cameraK, distCoeffs));

#pragma endregion INIT 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    solver.run();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\n----------------------------------------------------------\n\n";
    std::cout << "Total computing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds!\n";

    cv::waitKey(); exit(0);
}
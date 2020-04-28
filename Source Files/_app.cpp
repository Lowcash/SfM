#include "pch.h"
#include "common.h"
#include "tracking.h"
#include "feature_processing.h"
#include "visualization.h"
#include "camera.h"
#include "user_input_manager.h"
#include "reconstruction.h"

#pragma region METHODS

static void onUsrWinClick (int event, int x, int y, int flags, void* params) {
    if (event != cv::EVENT_LBUTTONDOWN) { return; }

    UserInputDataParams* inputDataParams = (UserInputDataParams*)params;

    const cv::Point clickedPoint(x, y);

    inputDataParams->userInput->m_usrClickedPts2D.push_back(clickedPoint);

    std::cout << "Clicked to: " << clickedPoint << "\n";
    
    cv::circle(*inputDataParams->inputImage, clickedPoint, 3, CV_RGB(200, 0, 0), cv::FILLED, cv::LINE_AA);

    cv::imshow(inputDataParams->getWinName(), *inputDataParams->inputImage);
}

bool findCameraPose(RecoveryPose& recPose, std::vector<cv::Point2f> prevPts, std::vector<cv::Point2f> currPts, cv::Mat cameraK, int minInliers, int& numInliers) {
    if (prevPts.size() <= 5 || currPts.size() <= 5) { return false; }

    cv::Mat E = cv::findEssentialMat(prevPts, currPts, cameraK, recPose.recPoseMethod, recPose.prob, recPose.threshold, recPose.mask);

    if (!(E.cols == 3 && E.rows == 3)) { return false; }

    cv::Mat p0 = cv::Mat( { 3, 1 }, {
		double( prevPts[0].x ), double( prevPts[0].y ), 1.0
		} );
	cv::Mat p1 = cv::Mat( { 3, 1 }, {
		double( currPts[0].x ), double( currPts[0].y ), 1.0
		} );
	const double E_error = cv::norm( p1.t() * cameraK.inv().t() * E * cameraK.inv() * p0, cv::NORM_L2 );

    if (E_error > 1e-03) { return false; }

    numInliers = cv::recoverPose(E, prevPts, currPts, cameraK, recPose.R, recPose.t, recPose.mask);

    return numInliers > minInliers;
}

bool loadImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray, float downSample = 1.0f) {
    cap >> imColor; if (imColor.empty()) { return false; }

    if (downSample != 1.0f)
        cv::resize(imColor, imColor, cv::Size(imColor.cols*downSample, imColor.rows*downSample));

    cv::cvtColor(imColor, imGray, cv::COLOR_BGR2GRAY);

    return true;
}

bool findGoodImagePair(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, Camera camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView, float imDownSampling = 1.0f, bool useCUDA = false) {
    std::cout << "Finding good image pair" << std::flush;

    cv::Mat _imColor, _imGray;

    std::vector<cv::Point2f> _prevCorners, _currCorners;

    int numSkippedFrames = -1, numHomInliers = 0;
    do {
        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        std::cout << "." << std::flush;
        
        if (viewContainer.isEmpty()) {
            viewContainer.addItem(ViewData(_imColor, _imGray));

            featDetector.generateFlowFeatures(_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

            if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 
        }

        _prevCorners = ofPrevView.corners;
        _currCorners = ofCurrView.corners;

        optFlow.computeFlow(viewContainer.getLastOneItem()->imGray, _imGray, _prevCorners, _currCorners, optFlow.statusMask, true, true);

        numSkippedFrames++;
    } while(!findCameraPose(recPose, _prevCorners, _currCorners, camera.K, recPose.minInliers, numHomInliers));

    viewContainer.addItem(ViewData(_imColor, _imGray));

    ofPrevView.setCorners(_prevCorners);
    ofCurrView.setCorners(_currCorners);

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return true;
}

bool findGoodImagePair(cv::VideoCapture cap, ViewDataContainer& viewContainer, float imDownSampling = 1.0f) {
    cv::Mat _imColor, _imGray;

    if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 

    if (viewContainer.isEmpty()) {
        viewContainer.addItem(ViewData(_imColor, _imGray));

        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; }
    }

    viewContainer.addItem(ViewData(_imColor, _imGray));

    return true;
}

#pragma endregion METHODS

enum Method { KLT_2D = 0, KLT_3D, KLT_3D_PNP };

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
        "{ bUseMethod| KLT_2D      | method to use KLT_2D/KLT_3D/KLT_3D_PNP }"
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

        "{ peMethod  | LMEDS       | pose estimation fundamental matrix computation method [RANSAC/LMEDS] }"
        "{ peProb    | 0.999       | pose estimation confidence/probability }"
        "{ peThresh  | 1.0         | pose estimation threshold }"
        "{ peMinInl  | 50          | pose estimation in number of homography inliers user for reconstruction }"
        "{ peMinMatch| 50          | pose estimation min matches to break }"
       
        "{ pePMetrod | 50          | pose estimation method SOLVEPNP_ITERATIVE/SOLVEPNP_P3P/SOLVEPNP_AP3P }"
        "{ peExGuess | false       | pose estimation use extrinsic guess }"
        "{ peNumIteR | 250         | pose estimation max iteration }"

        "{ baMethod  | DENSE_SCHUR | bundle adjustment used solver type DENSE_SCHUR/SPARSE_NORMAL_CHOLESKY }"
        "{ baNumIter | 50          | bundle adjustment max iteration }"
        "{ baLossFunc| NONE        | bundle adjustment used loss function NONE/HUBER/CAUCHY }"
        "{ baLossSc  | 1.0         | bundle adjustment loss function scaling parameter }"

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
    const std::string baMethod = parser.get<std::string>("baMethod");
    const int baNumIter = parser.get<int>("baNumIter");
    const std::string baLossFunc = parser.get<std::string>("baLossFunc");
    const double baLossSc = parser.get<double>("baLossSc");

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
    Camera camera(cameraK, distCoeffs, bDownSamp);
    
    const std::string ptCloudWinName = "Point cloud";
    const std::string usrInpWinName = "User input/output";
    const std::string recPoseWinName = "Recovery pose";
    const std::string matchesWinName = "Matches";
    const std::string trajWinName = "Trajectory";

    cv::VideoCapture cap; if(!cap.open(bSource)) {
        std::cerr << "Error opening video stream or file!!" << "\n";
        exit(1);
    }
    
    Method usedMethod;
    if (bUseMethod == "KLT_2D") 
        usedMethod = Method::KLT_2D;
    else if (bUseMethod == "KLT_3D")
        usedMethod = Method::KLT_3D;
    else
        usedMethod = Method::KLT_3D_PNP;

    FeatureDetector featDetector(fDecType, useCUDA);
    DescriptorMatcher descMatcher(fMatchType, fKnnRatio, useCUDA);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, ofMaxItCt, ofItEps);
    OptFlow optFlow(flowTermCrit, ofWinSize, ofMaxLevel, ofMaxError, ofMaxCorn, ofQualLvl, ofMinDist, ofMinKPts, useCUDA);

    RecoveryPose recPose(peMethod, peProb, peThresh, peMinInl, pePMetrod, peExGuess, peNumIteR);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;

    cv::startWindowThread();

    cv::namedWindow(usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(matchesWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(usrInpWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(recPoseWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(trajWinName, cv::Size(bWinWidth, bWinHeight));
    cv::resizeWindow(matchesWinName, cv::Size(bWinWidth, bWinHeight));

    UserInput userInput(ofMaxError);
    UserInputDataParams mouseUsrDataParams(usrInpWinName, &imOutUsrInp, &userInput);

    cv::setMouseCallback(usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);

    ViewDataContainer viewContainer(usedMethod == Method::KLT_2D ? 100 : INT32_MAX);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 
    
    Tracking tracking;
    Reconstruction reconstruction(tMethod, tMinDist, tMaxDist, tMaxPErr, true);

    VisPCL visPCL(ptCloudWinName + " PCL", cv::Size(bWinWidth, bWinHeight));
    //boost::thread visPCLthread(boost::bind(&VisPCL::visualize, &visPCL));
    //std::thread visPCLThread(&VisPCL::visualize, &visPCL);

    VisVTK visVTK(ptCloudWinName + " VTK", cv::Size(bWinWidth, bWinHeight));
    //std::thread visVTKThread(&VisVTK::visualize, &visVTK);
#pragma endregion INIT 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (uint iteration = 1; ; ++iteration) {
        if (!viewContainer.isEmpty() && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
            featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
        }

        bool isPtAdded = false;
        if (!userInput.m_usrClickedPts2D.empty()) {
            userInput.attachPointsToMove(userInput.m_usrClickedPts2D, ofPrevView.corners);

            isPtAdded = true;
        }

        if (usedMethod == Method::KLT_2D) {
            if (!userInput.m_usrPts2D.empty()) {
                userInput.attachPointsToMove(userInput.m_usrPts2D, ofPrevView.corners);
            }

            if (!findGoodImagePair(cap, viewContainer, bDownSamp)) { break; }
        }
            
        if ((usedMethod == Method::KLT_3D || usedMethod == Method::KLT_3D_PNP) && !findGoodImagePair(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView, bDownSamp, useCUDA)) { break; }

        ofPrevView.setView(viewContainer.getLastButOneItem());
        ofCurrView.setView(viewContainer.getLastOneItem());

        ofPrevView.viewPtr->imColor.copyTo(imOutRecPose);
        ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

        if (usedMethod == Method::KLT_2D && !ofPrevView.corners.empty()) {
            optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

            optFlow.drawOpticalFlow(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);

            if (!userInput.m_usrPts2D.empty()) {
                std::vector<cv::Point2f> _newPts2D;
                userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.m_usrPts2D.size());

                userInput.updatePoints(_newPts2D, cv::Rect(cv::Point(), ofCurrView.viewPtr->imColor.size()), 10);
            }

            if (!userInput.m_usrClickedPts2D.empty() && isPtAdded) {
                std::vector<cv::Point2f> _newPts2D;
                userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.m_usrClickedPts2D.size());

                userInput.addPoints(userInput.m_usrClickedPts2D, _newPts2D);

                userInput.m_usrClickedPts2D.clear();
            }

            userInput.recoverPoints(imOutUsrInp);
        }
        if ((usedMethod == Method::KLT_3D || usedMethod == Method::KLT_3D_PNP)) {
            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);
        }

        cv::imshow(recPoseWinName, imOutRecPose);

        if ((usedMethod == Method::KLT_3D || usedMethod == Method::KLT_3D_PNP)) {
            std::vector<cv::Vec3d> _points3D;
            std::vector<cv::Vec3b> _pointsRGB;
            std::vector<bool> _mask;

            cv::Matx34d _prevPose, _currPose; 

            if (usedMethod == Method::KLT_3D) {
                composeExtrinsicMat(tracking.R, tracking.t, _prevPose);

                tracking.t = tracking.t + (tracking.R * recPose.t);
                tracking.R = tracking.R * recPose.R;

                composeExtrinsicMat(tracking.R, tracking.t, _currPose);
                tracking.addCamPose(_currPose);

                reconstruction.triangulateCloud(ofPrevView.corners, ofCurrView.corners, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, camera, _prevPose, _currPose, recPose);

                if (!userInput.m_usrClickedPts2D.empty() && isPtAdded) {
                    std::vector<cv::Point2f> _newPts2D;
                    std::vector<cv::Vec3d> _newPts3D;
                    
                    userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.m_usrClickedPts2D.size());
                    userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.m_usrClickedPts2D.size());

                    userInput.addPoints(_newPts3D);
                    
                    userInput.m_usrClickedPts2D.clear();
                    
                    visVTK.addPoints(_newPts3D);
                    visPCL.addPoints(_newPts3D);
                }

                userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(tracking.R), cv::Mat(tracking.t));
            }

            if (usedMethod == Method::KLT_3D_PNP) {
                featPrevView.setView(viewContainer.getLastButOneItem());
                featCurrView.setView(viewContainer.getLastOneItem());

                if (featPrevView.keyPts.empty()) {
                    featDetector.generateFeatures(featPrevView.viewPtr->imGray, featPrevView.keyPts, featPrevView.descriptor);
                }

                featDetector.generateFeatures(featCurrView.viewPtr->imGray, featCurrView.keyPts, featCurrView.descriptor);

                if (featPrevView.keyPts.empty() || featCurrView.keyPts.empty()) { 
                    std::cerr << "None keypoints to match, skip matching/triangulation!\n";

                    continue; 
                }

                std::vector<cv::Point2f> _prevPts, _currPts;
                std::vector<cv::DMatch> _matches;
                std::vector<int> _prevIdx, _currIdx;

                descMatcher.findRobustMatches(featPrevView.keyPts, featCurrView.keyPts, featPrevView.descriptor, featCurrView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx);

                std::cout << "Matches count: " << _matches.size() << "\n";

                if (_prevPts.empty() || _currPts.empty()) { 
                    std::cerr << "None points to triangulate, skip triangulation!\n";

                    continue; 
                }

                if(!tracking.findRecoveredCameraPose(descMatcher, peMinMatch, camera, featCurrView, recPose)) {
                    std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";
        
                    std::swap(ofPrevView, ofCurrView);
                    std::swap(featPrevView, featCurrView);

                    continue;
                }
            
                if (tracking.isCamPosesEmpty())
                    composeExtrinsicMat(cv::Matx33d::eye(), cv::Matx31d::eye(), _prevPose);
                else
                    _prevPose = tracking.getLastCamPose();

                composeExtrinsicMat(recPose.R, recPose.t, _currPose);
                tracking.addCamPose(_currPose);

                if (!userInput.m_usrClickedPts2D.empty() && isPtAdded) {
                    userInput.detachPointsFromMove(_prevPts, ofPrevView.corners, userInput.m_usrClickedPts2D.size());
                    userInput.detachPointsFromMove(_currPts, ofCurrView.corners, userInput.m_usrClickedPts2D.size());
                }

                reconstruction.triangulateCloud(_prevPts, _currPts, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, camera, _prevPose, _currPose, recPose);

                if (!userInput.m_usrClickedPts2D.empty() && isPtAdded) {
                    std::vector<cv::Vec3d> _newPts3D;
                    userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.m_usrClickedPts2D.size());

                    userInput.addPoints(_newPts3D);
                    
                    userInput.m_usrClickedPts2D.clear();
                    
                    visVTK.addPoints(_newPts3D);
                    visPCL.addPoints(_newPts3D);
                }

                userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(tracking.R), cv::Mat(tracking.t));

                tracking.addTrackView(featCurrView.viewPtr, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, _currIdx);

                reconstruction.adjustBundle(tracking.trackViews, camera, *tracking.getCamPoses(), baMethod, baLossFunc, baLossSc, baNumIter);

                visVTK.addPointCloud(tracking.trackViews);

                visPCL.addPointCloud(tracking.trackViews);
            }

            visVTK.updateCameras(*tracking.getCamPoses(), camera.K);
            visVTK.visualize();

            visPCL.updateCameras(*tracking.getCamPoses());
            visPCL.visualize();
        }

        cv::imshow(usrInpWinName, imOutUsrInp);

        std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

        std::swap(ofPrevView, ofCurrView);
        std::swap(featPrevView, featCurrView);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "\n----------------------------------------------------------\n\n";
    std::cout << "Total computing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds!\n";

    cv::waitKey(); exit(0);
}
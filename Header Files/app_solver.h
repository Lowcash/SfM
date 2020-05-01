#ifndef APP_SOLVER_H
#define APP_SOLVER_H
#pragma once

#include "pch.h"
#include "tracking.h"
#include "feature_processing.h"
#include "visualization.h"
#include "camera.h"
#include "user_input_manager.h"
#include "reconstruction.h"

struct AppSolverDataParams {
    const std::string bUseMethod, ptCloudWinName, usrInpWinName, recPoseWinName, matchesWinName, bSource, fDecType, fMatchType, peMethod, pePMetrod, baLibrary, baCMethod, tMethod;
    const float bDownSamp, fKnnRatio, ofMaxItCt, ofItEps, ofMaxError, ofQualLvl, ofMinDist, peProb, peThresh, tMinDist, tMaxDist, tMaxPErr;
    const cv::Size winSize;
    const bool useCUDA, peExGuess;
    const int ofMinKPts, ofWinSize, ofMaxLevel, ofMaxCorn, peMinInl, peMinMatch, peNumIteR, baNumIter; 
    const cv::Mat cameraK, distCoeffs;

    /** AppSolverDataParams constructor

    Transfer params to solver
    Usually from cv::CommandLineParser

    @param bUseMethod method to use KLT/VO/PNP
    @param ptCloudWinName point cloud window name
    @param usrInpWinName user input window name
    @param recPoseWinName recovery pose window name
    @param matchesWinName matches window name
    @param bSource source video file [.mp4, .avi ...] 
    @param bDownSamp downsampling of input source images 
    @param winSize debug windows size
    @param useCUDA is nVidia CUDA used 
    @param fDecType used detector type
    @param fMatchType used matcher type
    @param fKnnRatio knn ration match
    @param ofMinKPts optical flow min descriptor to generate new one
    @param ofWinSize optical flow window size
    @param ofMaxLevel optical flow max pyramid level
    @param ofMaxItCt optical flow max iteration count
    @param ofItEps optical flow iteration epsilon
    @param ofMaxError optical flow max error
    @param ofMaxCorn optical flow max generated corners
    @param ofQualLvl optical flow generated corners quality level
    @param ofMinDist optical flow generated corners min distance
    @param peMethod pose estimation fundamental matrix computation method [RANSAC/LMEDS]
    @param peProb pose estimation confidence/probability
    @param peThresh pose estimation threshold
    @param peMinInl pose estimation in number of homography inliers user for reconstruction
    @param peMinMatch pose estimation min matches to break
    @param pePMetrod pose estimation method SOLVEPNP_ITERATIVE/SOLVEPNP_P3P/SOLVEPNP_AP3P
    @param peExGuess pose estimation use extrinsic guess
    @param peNumIteR pose estimation max iteration
    @param baLibrary bundle adjustment used library CERES/GTSAM
    @param baCMethod bundle adjustment CERES solver type DENSE_SCHUR/SPARSE_NORMAL_CHOLESKY
    @param baNumIter bundle adjustment max iteration
    @param tMethod triangulation method ITERATIVE/DLT
    @param tMinDist triangulation points min distance
    @param tMaxDist triangulation points max distance
    @param tMaxPErr triangulation points max reprojection error
    @param cameraK camera intrics parameters
    @param distCoeffs camera distortion parameters
    */
    AppSolverDataParams(const std::string bUseMethod, const std::string ptCloudWinName, const std::string usrInpWinName, std::string recPoseWinName, const std::string matchesWinName, const std::string bSource, const float bDownSamp, const cv::Size winSize, const bool useCUDA, const std::string fDecType, const std::string fMatchType, const float fKnnRatio, const int ofMinKPts, const int ofWinSize, const int ofMaxLevel, const float ofMaxItCt, const float ofItEps, const float ofMaxError, const int ofMaxCorn, const float ofQualLvl, const float ofMinDist, const std::string peMethod, const float peProb, const float peThresh, const int peMinInl, const int peMinMatch, const std::string pePMetrod, const bool peExGuess, const int peNumIteR, const std::string baLibrary, const std::string baCMethod, const int baNumIter, const std::string tMethod, const float tMinDist, const float tMaxDist, const float tMaxPErr, const cv::Mat cameraK, const cv::Mat distCoeffs) 
        : bUseMethod(bUseMethod), ptCloudWinName(ptCloudWinName), usrInpWinName(usrInpWinName), recPoseWinName(recPoseWinName), matchesWinName(matchesWinName), bSource(bSource), bDownSamp(bDownSamp), winSize(winSize), useCUDA(useCUDA), fDecType(fDecType), fMatchType(fMatchType), fKnnRatio(fKnnRatio), ofMinKPts(ofMinKPts), ofWinSize(ofWinSize), ofMaxLevel(ofMaxLevel), ofMaxItCt(ofMaxItCt), ofItEps(ofItEps), ofMaxError(ofMaxError), ofMaxCorn(ofMaxCorn), ofQualLvl(ofQualLvl), ofMinDist(ofMinDist), peMethod(peMethod), peProb(peProb), peThresh(peThresh), peMinInl(peMinInl), peMinMatch(peMinMatch), pePMetrod(pePMetrod), peExGuess(peExGuess), peNumIteR(peNumIteR), baLibrary(baLibrary), baCMethod(baCMethod), baNumIter(baNumIter), tMethod(tMethod), tMinDist(tMinDist), tMaxDist(tMaxDist), tMaxPErr(tMaxPErr), cameraK(cameraK), distCoeffs(distCoeffs) {}
};

class AppSolver {
private:
    enum Method { KLT = 0, VO, PNP };

    Method m_usedMethod;

    const AppSolverDataParams params;

    Tracking m_tracking;

    bool findGoodImagePair(cv::VideoCapture cap, ViewDataContainer& viewContainer, float imDownSampling = 1.0f);

    bool findGoodImagePair(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, Camera camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView, float imDownSampling = 1.0f, bool useCUDA = false);

    void composeExtrinsicMat(cv::Matx33d R, cv::Matx31d t, cv::Matx34d& pose) {
        pose = cv::Matx34d(
            R(0,0), R(0,1), R(0,2), t(0),
            R(1,0), R(1,1), R(1,2), t(1),
            R(2,0), R(2,1), R(2,2), t(2)
        );
    }

    void decomposeExtrinsicMat(cv::Matx34d pose, cv::Matx33d& R, cv::Matx31d& t) {
        R = cv::Matx33d(
            pose(0,0), pose(0,1), pose(0,2),
            pose(1,0), pose(1,1), pose(1,2),
            pose(2,0), pose(2,1), pose(2,2)
        );

        t = cv::Matx31d(
            pose(0,3),
            pose(1,3),
            pose(2,3)
        );
    }

    bool loadImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray, float downSample = 1.0f);

    static void onUsrWinClick (int event, int x, int y, int flags, void* params) {
        if (event != cv::EVENT_LBUTTONDOWN) { return; }

        UserInputDataParams* inputDataParams = (UserInputDataParams*)params;

        const cv::Point clickedPoint(x, y);

        inputDataParams->userInput->m_usrClickedPts2D.push_back(clickedPoint);

        std::cout << "Clicked to: " << clickedPoint << "\n";
        
        cv::circle(*inputDataParams->inputImage, clickedPoint, 3, CV_RGB(200, 0, 0), cv::FILLED, cv::LINE_AA);

        cv::imshow(inputDataParams->getWinName(), *inputDataParams->inputImage);
    }
public:
    AppSolver (const AppSolverDataParams params)
        : params(params) {       
        if (params.bUseMethod == "KLT") 
            m_usedMethod = Method::KLT;
        else if (params.bUseMethod == "VO")
            m_usedMethod = Method::VO;
        else
            m_usedMethod = Method::PNP;
    }

    void run();
};

#endif //APP_SOLVER_H
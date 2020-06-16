#ifndef APP_SOLVER_H
#define APP_SOLVER_H
#pragma once

#include "pch.h"
#include "common.h"
#include "tracking.h"
#include "feature_processing.h"
#include "visualization.h"
#include "camera.h"
#include "user_input_manager.h"
#include "reconstruction.h"

struct AppSolverDataParams {
    const std::string bUseMethod, ptCloudWinName, usrInpWinName, recPoseWinName, matchesWinName, bSource, fDecType, fMatchType, peMethod, pePMetrod, baMethod, tMethod;
    const float bDownSamp, fKnnRatio, ofMaxItCt, ofItEps, ofMaxError, ofQualLvl, ofMinDist, peProb, peThresh, tMinDist, tMaxDist, tMaxPErr, cSRemThr, cLSize;
    const double baMaxRMSE, cSRange;
    const cv::Size winSize, camSize;
    const bool peExGuess, bDebugVisE, bDebugMatE;
    const int ofMinKPts, ofWinSize, ofMaxLevel, ofMaxCorn, peMinInl, peMinMatch, peNumIteR, bMaxSkFram, baProcIt, cFProcIt;
    const cv::Mat cameraK, distCoeffs;

    /** 
     * AppSolverDataParams constructor
     * 
     * Transfer params to solver
     * Typically from cv::CommandLineParser
     * 
     * @param bUseMethod method to use KLT/VO/PNP
     * @param ptCloudWinName point cloud window name
     * @param usrInpWinName user input window name
     * @param recPoseWinName recovery pose window name
     * @param matchesWinName matches window name
     * @param bSource source video file [.mp4, .avi ...]
     * @param bDownSamp downsampling of input source images
     * @param bMaxSkFram max number of skipped frames to swap
     * @param bDebugVisE ienable debug point cloud visualization by VTK, PCL
     * @param bDebugMatE enable debug matching visualization by GTK/...
     * @param winSize debug windows size
     * @param camSize camera/image size
     * @param fDecType used detector type
     * @param fMatchType used matcher type
     * @param fKnnRatio knn ration match
     * @param ofMinKPts optical flow min descriptor to generate new one
     * @param ofWinSize optical flow window size
     * @param ofMaxLevel optical flow max pyramid level
     * @param ofMaxItCt optical flow max iteration count
     * @param ofItEps optical flow iteration epsilon
     * @param ofMaxError optical flow max error
     * @param ofMaxCorn optical flow max generated corners
     * @param ofQualLvl optical flow generated corners quality level
     * @param ofMinDist optical flow generated corners min distance
     * @param peMethod pose estimation fundamental matrix computation method [RANSAC/LMEDS]
     * @param peProb pose estimation confidence/probability
     * @param peThresh pose estimation threshold
     * @param peMinInl pose estimation in number of homography inliers user for reconstruction
     * @param peMinMatch pose estimation min matches to break
     * @param pePMetrod pose estimation method SOLVEPNP_ITERATIVE/SOLVEPNP_P3P/SOLVEPNP_AP3P
     * @param peExGuess pose estimation use extrinsic guess
     * @param peNumIteR pose estimation max iteration
     * @param baMethod bundle adjustment solver type DENSE_SCHUR/SPARSE_NORMAL_CHOLESKY
     * @param baMaxRMSE bundle adjustment max RMSE error to recover from back up
     * @param baProcIt bundle adjustment process each %d iteration
     * @param tMethod triangulation method ITERATIVE/DLT
     * @param tMinDist triangulation points min distance
     * @param tMaxDist triangulation points max distance
     * @param tMaxPErr triangulation points max reprojection error
     * @param cameraK camera intrics parameters
     * @param distCoeffs camera distortion parameters
     * @param cSRemThr statistical outlier removal stddev multiply threshold
     * @param cLSize cloud leaf filter size
     * @param cSRange cloud radius search radius distance
     * @param cFProcIt cloud filter process each %d iteration
     */
    AppSolverDataParams(const std::string bUseMethod, const std::string ptCloudWinName, const std::string usrInpWinName, std::string recPoseWinName, const std::string matchesWinName, const std::string bSource, const float bDownSamp, const int bMaxSkFram, const cv::Size winSize, const cv::Size camSize, const bool bDebugVisE, const bool bDebugMatE, const std::string fDecType, const std::string fMatchType, const float fKnnRatio, const int ofMinKPts, const int ofWinSize, const int ofMaxLevel, const float ofMaxItCt, const float ofItEps, const float ofMaxError, const int ofMaxCorn, const float ofQualLvl, const float ofMinDist, const std::string peMethod, const float peProb, const float peThresh, const int peMinInl, const int peMinMatch, const std::string pePMetrod, const bool peExGuess, const int peNumIteR, const std::string baMethod, const double baMaxRMSE, const int baProcIt, const std::string tMethod, const float tMinDist, const float tMaxDist, const float tMaxPErr, const cv::Mat cameraK, const cv::Mat distCoeffs, const float cSRemThr, const float cLSize, const double cSRange, const int cFProcIt) 
        : bUseMethod(bUseMethod), ptCloudWinName(ptCloudWinName), usrInpWinName(usrInpWinName), recPoseWinName(recPoseWinName), matchesWinName(matchesWinName), bSource(bSource), bDownSamp(bDownSamp), bMaxSkFram(bMaxSkFram), winSize(winSize), camSize(camSize), bDebugVisE(bDebugVisE), bDebugMatE(bDebugMatE), fDecType(fDecType), fMatchType(fMatchType), fKnnRatio(fKnnRatio), ofMinKPts(ofMinKPts), ofWinSize(ofWinSize), ofMaxLevel(ofMaxLevel), ofMaxItCt(ofMaxItCt), ofItEps(ofItEps), ofMaxError(ofMaxError), ofMaxCorn(ofMaxCorn), ofQualLvl(ofQualLvl), ofMinDist(ofMinDist), peMethod(peMethod), peProb(peProb), peThresh(peThresh), peMinInl(peMinInl), peMinMatch(peMinMatch), pePMetrod(pePMetrod), peExGuess(peExGuess), peNumIteR(peNumIteR), baMethod(baMethod), baMaxRMSE(baMaxRMSE), baProcIt(baProcIt), tMethod(tMethod), tMinDist(tMinDist), tMaxDist(tMaxDist), tMaxPErr(tMaxPErr), cameraK(cameraK), distCoeffs(distCoeffs), cSRemThr(cSRemThr), cLSize(cLSize), cSRange(cSRange), cFProcIt(cFProcIt) {}
};

class AppSolver {
private:
    enum ImageFindState { SOURCE_LOST = -1, NOT_FOUND = 0, FOUND = 1};

    enum Method { KLT = 0, VO, PNP };

    const AppSolverDataParams params;

    const cv::Rect m_boundary;

    Method m_usedMethod;

    /** 
     * Find good images
     * 
     * The result will be added to ViewDataContainer
     */
    int findGoodImages(cv::VideoCapture cap, ViewDataContainer& viewContainer);

    /** 
     * Find good images by optical flow
     * 
     * The result will be added to ViewDataContainer
     */
    int findGoodImages(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, Camera camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView);

    /** 
     * Load and convert image to grayscale
     */
    int prepareImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray);

    /**
     * Handle user input
     */
    static void onUsrWinClick (int event, int x, int y, int flags, void* params) {
        if (event != cv::EVENT_LBUTTONDOWN) { return; }

        UserInputDataParams* inputDataParams = (UserInputDataParams*)params;

        const cv::Point clickedPoint(x, y);
        
        std::cout << "Clicked to: " << clickedPoint << "\n";

        // register point and force redraw
        inputDataParams->userInput->addClickedPoint(clickedPoint, true);
    }
public:
    AppSolver (const AppSolverDataParams params)
        : params(params), m_boundary(cv::Rect(cv::Point(), cv::Size(params.camSize.width / params.bDownSamp, params.camSize.height / params.bDownSamp))) {
        // std::string to Enum mapping by Mark Ransom
        // https://stackoverflow.com/questions/7163069/c-string-to-enum
        static std::unordered_map<std::string, Method> const table = { 
            {"KLT", Method::KLT}, {"VO", Method::VO},  {"PNP", Method::PNP}
        };
    
        if (auto it = table.find(params.bUseMethod); it != table.end())
            m_usedMethod = it->second;
        else
            m_usedMethod = Method::PNP;
    }

    /**
     *  Run SfM app
     */
    void run();
};

#endif //APP_SOLVER_H
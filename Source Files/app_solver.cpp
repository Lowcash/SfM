#include "app_solver.h"

bool AppSolver::loadImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray, float downSample) {
    cap >> imColor; if (imColor.empty()) { return false; }

    if (downSample != 1.0f)
        cv::resize(imColor, imColor, cv::Size(imColor.cols*downSample, imColor.rows*downSample));

    cv::cvtColor(imColor, imGray, cv::COLOR_BGR2GRAY);

    return true;
}

bool AppSolver::findGoodImagePair(cv::VideoCapture cap, ViewDataContainer& viewContainer, float imDownSampling) {
    cv::Mat _imColor, _imGray;

    if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; } 

    if (viewContainer.isEmpty()) {
        viewContainer.addItem(ViewData(_imColor, _imGray));

        if (!loadImage(cap, _imColor, _imGray, imDownSampling)) { return false; }
    }

    viewContainer.addItem(ViewData(_imColor, _imGray));

    return true;
}

bool AppSolver::findGoodImagePair(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, Camera camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView, float imDownSampling, bool useCUDA) {
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
    } while(!m_tracking.findCameraPose(recPose, _prevCorners, _currCorners, camera.K, recPose.minInliers, numHomInliers));

    viewContainer.addItem(ViewData(_imColor, _imGray));

    ofPrevView.setCorners(_prevCorners);
    ofCurrView.setCorners(_currCorners);

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return true;
}

void AppSolver::run() {
    cv::VideoCapture cap; if(!cap.open(params.bSource)) {
        std::cerr << "Error opening video stream or file!!" << "\n";
        exit(1);
    }

    Camera camera(params.cameraK, params.distCoeffs, params.bDownSamp);

    FeatureDetector featDetector(params.fDecType, params.useCUDA);
    DescriptorMatcher descMatcher(params.fMatchType, params.fKnnRatio, params.useCUDA);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, params.ofMaxItCt, params.ofItEps);
    OptFlow optFlow(flowTermCrit, params.ofWinSize, params.ofMaxLevel, params.ofMaxError, params.ofMaxCorn, params.ofQualLvl, params.ofMinDist, params.ofMinKPts, params.useCUDA);

    RecoveryPose recPose(params.peMethod, params.peProb, params.peThresh, params.peMinInl, params.pePMetrod, params.peExGuess, params.peNumIteR);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;

    cv::startWindowThread();

    cv::namedWindow(params.usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(params.recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(params.matchesWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(params.usrInpWinName, params.winSize);
    cv::resizeWindow(params.recPoseWinName, params.winSize);
    cv::resizeWindow(params.matchesWinName, params.winSize);
    
    UserInput userInput(params.ofMaxError);
    UserInputDataParams mouseUsrDataParams(params.usrInpWinName, &imOutUsrInp, &userInput);

    cv::setMouseCallback(params.usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);

    ViewDataContainer viewContainer(m_usedMethod == Method::KLT || m_usedMethod == Method::VO ? 100 : INT32_MAX);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 

    Reconstruction reconstruction(params.tMethod, params.tMinDist, params.tMaxDist, params.tMaxPErr, true);

    VisPCL visPCL(params.ptCloudWinName + " PCL", params.winSize);
    //boost::thread visPCLthread(boost::bind(&VisPCL::visualize, &visPCL));
    //std::thread visPCLThread(&VisPCL::visualize, &visPCL);

    VisVTK visVTK(params.ptCloudWinName + " VTK", params.winSize);
    //std::thread visVTKThread(&VisVTK::visualize, &visVTK);

    for (uint iteration = 1; ; ++iteration) {
        if (!viewContainer.isEmpty() && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
            featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
        }

        bool isPtAdded = false;
        if (!userInput.m_usrClickedPts2D.empty()) {
            userInput.attachPointsToMove(userInput.m_usrClickedPts2D, ofPrevView.corners);

            isPtAdded = true;
        }

        if (m_usedMethod == Method::KLT) {
            if (!userInput.m_usrPts2D.empty()) {
                userInput.attachPointsToMove(userInput.m_usrPts2D, ofPrevView.corners);
            }

            if (!findGoodImagePair(cap, viewContainer, params.bDownSamp)) { break; }
        }
            
        if ((m_usedMethod == Method::VO || m_usedMethod == Method::PNP) && !findGoodImagePair(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView, params.bDownSamp, params.useCUDA)) { break; }

        ofPrevView.setView(viewContainer.getLastButOneItem());
        ofCurrView.setView(viewContainer.getLastOneItem());

        ofPrevView.viewPtr->imColor.copyTo(imOutRecPose);
        ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

        if (m_usedMethod == Method::KLT && !ofPrevView.corners.empty()) {
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
        if ((m_usedMethod == Method::VO || m_usedMethod == Method::PNP)) {
            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);
        }

        cv::imshow(params.recPoseWinName, imOutRecPose);

        if ((m_usedMethod == Method::VO || m_usedMethod == Method::PNP)) {
            std::vector<cv::Vec3d> _points3D;
            std::vector<cv::Vec3b> _pointsRGB;
            std::vector<bool> _mask;

            cv::Matx34d _prevPose, _currPose; 

            if (m_usedMethod == Method::VO) {
                composeExtrinsicMat(m_tracking.R, m_tracking.t, _prevPose);

                m_tracking.t = m_tracking.t + (m_tracking.R * recPose.t);
                m_tracking.R = m_tracking.R * recPose.R;

                composeExtrinsicMat(m_tracking.R, m_tracking.t, _currPose);
                m_tracking.addCamPose(_currPose);

                reconstruction.triangulateCloud(camera, ofPrevView.corners, ofCurrView.corners, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose);

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

                userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(m_tracking.R), cv::Mat(m_tracking.t));
            }

            if (m_usedMethod == Method::PNP) {
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

                if(!m_tracking.findRecoveredCameraPose(descMatcher, params.peMinMatch, camera, featCurrView, recPose)) {
                    std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";
        
                    std::swap(ofPrevView, ofCurrView);
                    std::swap(featPrevView, featCurrView);

                    continue;
                }
            
                if (m_tracking.m_camPoses.empty())
                    composeExtrinsicMat(cv::Matx33d::eye(), cv::Matx31d::eye(), _prevPose);
                else
                    _prevPose = m_tracking.m_camPoses.back();

                composeExtrinsicMat(recPose.R, recPose.t, _currPose);
                m_tracking.addCamPose(_currPose);

                if (!userInput.m_usrClickedPts2D.empty() && isPtAdded) {
                    userInput.detachPointsFromMove(_prevPts, ofPrevView.corners, userInput.m_usrClickedPts2D.size());
                    userInput.detachPointsFromMove(_currPts, ofCurrView.corners, userInput.m_usrClickedPts2D.size());
                }

                reconstruction.triangulateCloud(camera, _prevPts, _currPts, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose);

                if (!userInput.m_usrClickedPts2D.empty() && isPtAdded) {
                    std::vector<cv::Vec3d> _newPts3D;
                    userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.m_usrClickedPts2D.size());

                    userInput.addPoints(_newPts3D);
                    
                    userInput.m_usrClickedPts2D.clear();
                    
                    visVTK.addPoints(_newPts3D);
                    visPCL.addPoints(_newPts3D);
                }

                userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(m_tracking.R), cv::Mat(m_tracking.t));

                m_tracking.addTrackView(featCurrView.viewPtr, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, _currIdx);

                reconstruction.adjustBundle(camera, m_tracking.m_trackViews, m_tracking.m_camPoses);

                visVTK.addPointCloud(m_tracking.m_trackViews);

                visPCL.addPointCloud(m_tracking.m_trackViews);
            }

            visVTK.updateCameras(m_tracking.m_camPoses, camera.K);
            visVTK.visualize();

            visPCL.updateCameras(m_tracking.m_camPoses);
            visPCL.visualize();
        }

        cv::imshow(params.usrInpWinName, imOutUsrInp);

        std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

        std::swap(ofPrevView, ofCurrView);
        std::swap(featPrevView, featCurrView);
    }
}
#include "app_solver.h"

int AppSolver::prepareImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray) {
    cap >> imColor; 
    
    if (imColor.empty()) 
        return ImageFindState::SOURCE_LOST;

    if (params.bDownSamp != 1.0f)
        cv::resize(imColor, imColor, cv::Size(imColor.cols*params.bDownSamp, imColor.rows*params.bDownSamp));

    cv::cvtColor(imColor, imGray, cv::COLOR_BGR2GRAY);

    return ImageFindState::FOUND;
}

int AppSolver::findGoodImages(cv::VideoCapture cap, ViewDataContainer& viewContainer) {
    cv::Mat _imColor, _imGray;

    ImageFindState state;

    if (viewContainer.isEmpty()) {
        if ((state = (ImageFindState)prepareImage(cap, _imColor, _imGray)) 
        != ImageFindState::FOUND) 
            return state;

        viewContainer.addItem(ViewData(_imColor, _imGray));
    }

    if ((state = (ImageFindState)prepareImage(cap, _imColor, _imGray)) 
        != ImageFindState::FOUND) 
        return state; 

    viewContainer.addItem(ViewData(_imColor, _imGray));

    return state;
}

int AppSolver::findGoodImages(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, Camera camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView) {
    std::cout << "Finding good images" << std::flush;

    cv::Mat _imColor, _imGray;

    std::vector<cv::Point2f> _prevCorners, _currCorners;

    ImageFindState state;

    int numHomInliers = 0, numSkippedFrames = -1;
    do {
        if ((state = (ImageFindState)prepareImage(cap, _imColor, _imGray)) != ImageFindState::FOUND)
            return state;

        std::cout << "." << std::flush;
        
        if (viewContainer.isEmpty()) {
            viewContainer.addItem(ViewData(_imColor, _imGray));

            featDetector.generateFlowFeatures(_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

            if ((state = (ImageFindState)prepareImage(cap, _imColor, _imGray)) != ImageFindState::FOUND)
                return state;
        }

        _prevCorners = ofPrevView.corners;
        _currCorners = ofCurrView.corners;

        optFlow.computeFlow(viewContainer.getLastOneItem()->imGray, _imGray, _prevCorners, _currCorners, optFlow.statusMask, true, true);

        numSkippedFrames++;

        if (numSkippedFrames > params.bMaxSkFram) {
            viewContainer.addItem(ViewData(_imColor, _imGray));

            return ImageFindState::NOT_FOUND;
        }
    } while(!m_tracking.findCameraPose(recPose, _prevCorners, _currCorners, camera.K, recPose.minInliers, numHomInliers));

    viewContainer.addItem(ViewData(_imColor, _imGray));

    ofPrevView.setCorners(_prevCorners);
    ofCurrView.setCorners(_currCorners);

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return state;
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

    Reconstruction reconstruction(params.tMethod, params.baMethod, params.baMaxRMSE, params.tMinDist, params.tMaxDist, params.tMaxPErr, true);

    VisPCL visPCL(params.ptCloudWinName + " PCL", params.winSize);
    //boost::thread visPCLthread(boost::bind(&VisPCL::visualize, &visPCL));
    //std::thread visPCLThread(&VisPCL::visualize, &visPCL);

    VisVTK visVTK(params.ptCloudWinName + " VTK", params.winSize);
    //std::thread visVTKThread(&VisVTK::visualize, &visVTK);

    for (uint iteration = 1; ; ++iteration) {
        if (m_usedMethod == Method::KLT) {
            bool isPtAdded = false;

            if (iteration != 1 && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                ofPrevView.setView(viewContainer.getLastOneItem());

                featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
            }
            
            if (!userInput.usrClickedPts2D.empty()) {
                userInput.attachPointsToMove(userInput.usrClickedPts2D, ofPrevView.corners);

                isPtAdded = true;
            }

            if (!userInput.usrPts2D.empty()) {
                userInput.attachPointsToMove(userInput.usrPts2D, ofPrevView.corners);
            }

            if (findGoodImages(cap, viewContainer) == ImageFindState::SOURCE_LOST) 
                break;

            ofPrevView.setView(viewContainer.getLastButOneItem());
            ofCurrView.setView(viewContainer.getLastOneItem());

            ofCurrView.viewPtr->imColor.copyTo(imOutRecPose);
            ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

            if (!ofPrevView.corners.empty()) {
                optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

                optFlow.drawOpticalFlow(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);

                if (!userInput.usrPts2D.empty()) {
                    std::vector<cv::Point2f> _newPts2D;
                    userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.usrPts2D.size());

                    userInput.filterPoints(_newPts2D, cv::Rect(cv::Point(), ofCurrView.viewPtr->imColor.size()), 10);
                }

                if (!userInput.usrClickedPts2D.empty() && isPtAdded) {
                    std::vector<cv::Point2f> _newPts2D;
                    userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.usrClickedPts2D.size());

                    userInput.addPoints(_newPts2D);

                    userInput.usrClickedPts2D.clear();
                }

                userInput.recoverPoints(imOutUsrInp);
            }

            cv::imshow(params.recPoseWinName, imOutRecPose);
            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }
        if (m_usedMethod == Method::VO) {
            bool isPtAdded = false;

            if (iteration != 1 && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                ofPrevView.setView(viewContainer.getLastOneItem());

                featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
            }
            
            if (!userInput.usrClickedPts2D.empty()) {
                userInput.attachPointsToMove(userInput.usrClickedPts2D, ofPrevView.corners);

                isPtAdded = true;
            }

            ImageFindState state = (ImageFindState)findGoodImages(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView);
            
            if (state == ImageFindState::SOURCE_LOST) { break; }
            if (state == ImageFindState::NOT_FOUND) {
                ofPrevView.corners.clear();
                ofCurrView.corners.clear();

                std::swap(featPrevView, featCurrView);

                std::cout << "Good images pair not found -> skipping current iteration!" << "\n";

                continue;
            }

            ofPrevView.setView(viewContainer.getLastButOneItem());
            ofCurrView.setView(viewContainer.getLastOneItem());

            ofCurrView.viewPtr->imColor.copyTo(imOutRecPose);
            ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);

            cv::imshow(params.recPoseWinName, imOutRecPose);

            std::vector<cv::Vec3d> _points3D;
            std::vector<cv::Vec3b> _pointsRGB;
            std::vector<bool> _mask;

            cv::Matx34d _prevPose, _currPose;

            composeExtrinsicMat(m_tracking.R, m_tracking.t, _prevPose);

            m_tracking.t = m_tracking.t + (m_tracking.R * recPose.t);
            m_tracking.R = m_tracking.R * recPose.R;

            composeExtrinsicMat(m_tracking.R, m_tracking.t, _currPose);
            m_tracking.addCamPose(_currPose);

            reconstruction.triangulateCloud(camera, ofPrevView.corners, ofCurrView.corners, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose);

            if (!userInput.usrClickedPts2D.empty() && isPtAdded) {
                std::vector<cv::Point2f> _newPts2D;
                std::vector<cv::Vec3d> _newPts3D;
                
                userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.usrClickedPts2D.size());
                userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.usrClickedPts2D.size());

                userInput.addPoints(_newPts2D, _newPts3D, m_tracking.cloud3D, m_tracking.cloudRGB, m_tracking.cloudTracks, m_tracking.trackViews.size());
                
                userInput.usrClickedPts2D.clear();
                
                visVTK.addPoints(_newPts3D);
            }

            userInput.recoverPoints(imOutUsrInp, m_tracking.cloud3D, camera.K, cv::Mat(m_tracking.R), cv::Mat(m_tracking.t));

            visVTK.addCamera(m_tracking.camPoses, camera.K);
            visVTK.visualize(params.bVisEnable);

            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }
        if (m_usedMethod == Method::PNP) {
            bool isPtAdded = false;

            if (iteration != 1) {
                if (iteration % params.baProcIt == 1)
                    reconstruction.adjustBundle(camera, m_tracking.cloud3D, m_tracking.cloudTracks, m_tracking.camPoses);

                if (ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                    ofPrevView.setView(viewContainer.getLastOneItem());

                    featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
                }
            }
            
            if (!userInput.usrClickedPts2D.empty()) {
                userInput.attachPointsToMove(userInput.usrClickedPts2D, ofPrevView.corners);

                isPtAdded = true;
            }

            ImageFindState state = (ImageFindState)findGoodImages(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView);
            
            if (state == ImageFindState::SOURCE_LOST) { break; }
            if (state == ImageFindState::NOT_FOUND) {
                ofPrevView.corners.clear();
                ofCurrView.corners.clear();

                std::swap(featPrevView, featCurrView);

                std::cout << "Good images pair not found -> skipping current iteration!" << "\n";

                continue;
            }

            ofPrevView.setView(viewContainer.getLastButOneItem());
            ofCurrView.setView(viewContainer.getLastOneItem());

            ofCurrView.viewPtr->imColor.copyTo(imOutRecPose);
            ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

            recPose.drawRecoveredPose(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, recPose.mask);

            cv::imshow(params.recPoseWinName, imOutRecPose);

            std::vector<cv::Vec3d> _points3D;
            std::vector<cv::Vec3b> _pointsRGB;
            std::vector<bool> _mask;

            cv::Matx34d _prevPose, _currPose; 

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

            std::map<std::pair<float, float>, size_t> cloudMap;

            if(!m_tracking.findRecoveredCameraPose(descMatcher, params.peMinMatch, camera, featCurrView, recPose, cloudMap)) {
                std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";
    
                std::swap(ofPrevView, ofCurrView);
                std::swap(featPrevView, featCurrView);

                continue;
            }

            if (m_tracking.camPoses.empty())
                composeExtrinsicMat(cv::Matx33d::eye(), cv::Matx31d::eye(), _prevPose);
            else
                _prevPose = m_tracking.camPoses.back();

            composeExtrinsicMat(recPose.R, recPose.t, _currPose);
            m_tracking.addCamPose(_currPose);

            std::vector<cv::Point2f> _newPts2D;
            std::vector<cv::Vec3d> _newPts3D;

            if (!userInput.usrClickedPts2D.empty() && isPtAdded) {
                userInput.detachPointsFromMove(_prevPts, ofPrevView.corners, userInput.usrClickedPts2D.size());

                
                userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.usrClickedPts2D.size());

                _currPts.insert(_currPts.end(), _newPts2D.begin(), _newPts2D.end());
            }

            reconstruction.triangulateCloud(camera, _prevPts, _currPts, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose);

            if (!userInput.usrClickedPts2D.empty() && isPtAdded) {       
                userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.usrClickedPts2D.size());

                userInput.addPoints(_newPts2D, _newPts3D, m_tracking.cloud3D, m_tracking.cloudRGB, m_tracking.cloudTracks, m_tracking.trackViews.size());
                
                userInput.usrClickedPts2D.clear();
                
                visVTK.addPoints(_newPts3D);
                visPCL.addPoints(_newPts3D);
            }

            userInput.recoverPoints(imOutUsrInp, m_tracking.cloud3D, camera.K, cv::Mat(m_tracking.R), cv::Mat(m_tracking.t));

            if (m_tracking.addTrackView(featCurrView.viewPtr, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, cloudMap, _currIdx)) {
                visVTK.updatePointCloud(m_tracking.cloud3D, m_tracking.cloudRGB);
                visPCL.updatePointCloud(m_tracking.cloud3D, m_tracking.cloudRGB);

                visVTK.updateCameras(m_tracking.camPoses, camera.K);
                visVTK.visualize(params.bVisEnable);
                
                visPCL.updateCameras(m_tracking.camPoses);
                visPCL.visualize(params.bVisEnable);
            }

            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }
    }

    if (m_usedMethod == Method::PNP) {
        visVTK.visualize(params.bVisEnable, true);
        visPCL.visualize(params.bVisEnable, true);
    }
}
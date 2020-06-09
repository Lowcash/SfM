#include "app_solver.h"

int AppSolver::prepareImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray) {
    cap >> imColor; 
    
    if (imColor.empty()) 
        return ImageFindState::SOURCE_LOST;

    if (params.bDownSamp != 1.0f)
        cv::resize(imColor, imColor, cv::Size(imColor.cols/params.bDownSamp, imColor.rows/params.bDownSamp));

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

    ViewData _viewData;

    std::vector<cv::Point2f> _prevCorners, _currCorners;

    // search for a good pair of images by min homography inliers -> use optical flow
    int numHomInliers = 0, numSkippedFrames = -1;
    do {
        ImageFindState state;

        if ((state = (ImageFindState)prepareImage(cap, _viewData.imColor, _viewData.imGray)) != ImageFindState::FOUND)
            return state;

        std::cout << "." << std::flush;
        
        // if there is nothing to compare, prepare the first image
        if (viewContainer.isEmpty()) {
            viewContainer.addItem(_viewData);

            featDetector.generateFlowFeatures(_viewData.imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

            if ((state = (ImageFindState)prepareImage(cap, _viewData.imColor, _viewData.imGray)) != ImageFindState::FOUND)
                return state;
        }

        // set corners from back up -> computeFlow func removes corners
        _prevCorners = ofPrevView.corners;
        _currCorners = ofCurrView.corners;

        // flow computing with boundary and error filtering
        optFlow.computeFlow(viewContainer.getLastOneItem()->imGray, _viewData.imGray, _prevCorners, _currCorners, optFlow.statusMask);
        ProcesingAdds::filterPointsByStatusMask(_prevCorners, _currCorners, optFlow.statusMask);
        ProcesingAdds::filterPointsByBoundary(_prevCorners, _currCorners, m_boundary);

        numSkippedFrames++;

        if (numSkippedFrames > params.bMaxSkFram) {
            viewContainer.addItem(_viewData);

            return ImageFindState::NOT_FOUND;
        }
    } while(!Tracking::findCameraPose(recPose, _prevCorners, _currCorners, camera.K, recPose.minInliers, numHomInliers));

    // complete the search for image pairs -> set flow views
    viewContainer.addItem(_viewData);

    ofPrevView.setCorners(_prevCorners);
    ofCurrView.setCorners(_currCorners);

    std::cout << "[DONE]" << " - Inliers count: " << numHomInliers << "; Skipped frames: " << numSkippedFrames << "\t" << std::flush;

    return ImageFindState::FOUND;
}

void AppSolver::run() {
#pragma region INIT
    cv::VideoCapture cap; if(!cap.open(params.bSource)) {
        std::cerr << "Error opening video stream or file!!" << "\n";
        exit(1);
    }

    // initialize structures
    Camera camera(params.cameraK, params.distCoeffs, params.bDownSamp);

    FeatureDetector featDetector(params.fDecType);
    DescriptorMatcher descMatcher(params.fMatchType, params.fKnnRatio, params.bDebugMatE, params.winSize);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, params.ofMaxItCt, params.ofItEps);
    OptFlow optFlow(flowTermCrit, params.ofWinSize, params.ofMaxLevel, params.ofMaxError, params.ofMaxCorn, params.ofQualLvl, params.ofMinDist, params.ofMinKPts);

    RecoveryPose recPose(params.peMethod, params.peProb, params.peThresh, params.peMinInl, params.pePMetrod, params.peExGuess, params.peNumIteR);

    ViewDataContainer viewContainer(params.bDebugMatE ? INT32_MAX : 100);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 

    Tracking tracking;

    Reconstruction reconstruction(params.tMethod, params.baMethod, params.baMaxRMSE, params.tMinDist, params.tMaxDist, params.tMaxPErr, true);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;
    
    UserInput userInput(params.usrInpWinName, &imOutUsrInp, &tracking.pointCloud, params.ofMaxError);

    // run windows in new thread -> avoid rendering white screen
    cv::startWindowThread();

    cv::namedWindow(params.usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(params.recPoseWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(params.matchesWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(params.usrInpWinName, params.winSize);
    cv::resizeWindow(params.recPoseWinName, params.winSize);
    cv::resizeWindow(params.matchesWinName, params.winSize);
    
    UserInputDataParams mouseUsrDataParams(&userInput);

    cv::setMouseCallback(params.usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);
    
    // initialize visualization windows VTK, PCL
    VisPCL visPCL(params.ptCloudWinName + " PCL", params.winSize);

    //VisVTK visVTK(params.ptCloudWinName + " VTK", params.winSize);
#pragma endregion INIT

    for (uint iteration = 1; ; ++iteration) {
        // use if statements instead of switch due to loop breaks
        if (m_usedMethod == Method::KLT) {
            // in the first iteration, the image is not ready yet -> cannot generate features
            // generate features first, to avoid loss of user point in corners stack
            if (iteration != 1 && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                ofPrevView.setView(viewContainer.getLastOneItem());

                featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
            }

            if (findGoodImages(cap, viewContainer) == ImageFindState::SOURCE_LOST) 
                break;
            
            // prepare flow view images for flow computing and debug draw
            ofPrevView.setView(viewContainer.getLastButOneItem());
            ofCurrView.setView(viewContainer.getLastOneItem());

            ofCurrView.viewPtr->imColor.copyTo(imOutRecPose);
            ofCurrView.viewPtr->imColor.copyTo(imOutUsrInp);

            userInput.lockClickedPoints();

            if (!ofPrevView.corners.empty()) {
                userInput.attachPointsToMove(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, true);

                // move user points and corners
                optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);

                userInput.detachPointsFromMove(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, true);

                ProcesingAdds::filterPointsByStatusMask(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);
                ProcesingAdds::filterPointsByBoundary(ofPrevView.corners, ofCurrView.corners, m_boundary);

                PointsMove pointsMove; ProcesingAdds::analyzePointsMove(ofPrevView.corners, ofCurrView.corners, pointsMove);
                ProcesingAdds::correctPointsByMoveAnalyze(userInput.doneClickedPts, userInput.moveClickedPts, pointsMove);
                ProcesingAdds::correctPointsByMoveAnalyze(userInput.doneUsrPts, userInput.moveUsrPts, pointsMove);

                optFlow.drawOpticalFlow(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);
            }

            std::swap(userInput.moveUsrPts, userInput.doneUsrPts);
            userInput.moveUsrPts.clear();

            ProcesingAdds::filterPointsByBoundary(userInput.doneUsrPts, m_boundary);
            
            userInput.storeClickedPoints();
            userInput.clearClickedPoints();

            userInput.updateWaitingPoints();
            userInput.unlockClickedPoints();

            // draw moved points
            userInput.recoverPoints(imOutUsrInp);

            cv::imshow(params.recPoseWinName, imOutRecPose);
            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

            // prepare views to load new frame
            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }
        /*if (m_usedMethod == Method::VO) {
            // in the first iteration, the image is not ready yet -> cannot generate features
            // generate features first, to avoid loss of user point in corners stack
            if (iteration != 1 && ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                ofPrevView.setView(viewContainer.getLastOneItem());

                featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
            }

            // find good image pair by optical flow and essential matrix
            ImageFindState state = (ImageFindState)findGoodImages(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView);
            
            if (state == ImageFindState::SOURCE_LOST) { break; }
            if (state == ImageFindState::NOT_FOUND) {
                ofPrevView.corners.clear();
                ofCurrView.corners.clear();

                std::swap(featPrevView, featCurrView);

                std::cout << "Good images pair not found -> skipping current iteration!" << "\n";

                continue;
            }

            // prepare flow view images for debug draw
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

            composeExtrinsicMat(m_tracking.actualR, m_tracking.actualT, _prevPose);

            // move camera position by computed R a t from optical flow and essential matrix
            m_tracking.actualT = m_tracking.actualT + (m_tracking.actualR * recPose.t);
            m_tracking.actualR = m_tracking.actualR * recPose.R;

            composeExtrinsicMat(m_tracking.actualR, m_tracking.actualT, _currPose);
            m_tracking.addCamPose(_currPose);

            // triangulate corners and user clicked points
            reconstruction.triangulateCloud(camera, ofPrevView.corners, ofCurrView.corners, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose.R, recPose.t);

            userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.usrClickedPts2D.size());

            userInput.addPoints(_newPts2D, _newPts3D, m_tracking.getTrackViews().size());
             
            userInput.storeClickedPoints();
            userInput.clearClickedPoints();

            // draw moved points
            userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(m_tracking.actualR), cv::Mat(m_tracking.actualT));
        
            visPCL.addCamera(m_tracking.getLastCam() , camera.K);
            //visPCL.addPoints();

            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

            // prepare views to load new frame
            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }
        /*if (m_usedMethod == Method::PNP) {
            bool isPtAdded = false;

            if (iteration != 1) {
                // do bundle adjust after loop iteration to avoid "continue" statement
                if (iteration % params.baProcIt == 1 || params.baProcIt == 1) {
                    //m_tracking.pointCloud.clearCloud();

                    reconstruction.adjustBundle(camera, m_tracking.getCamPoses(), m_tracking.pointCloud);
                }

                if (ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                    ofPrevView.setView(viewContainer.getLastOneItem());

                    featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
                }
            }
            
            // attach user clicked points at the end of prev flow corners stack
            // prepare points to move
            if (!userInput.usrClickedPts2D.empty()) {
                userInput.attachPointsToMove(userInput.usrClickedPts2D, ofPrevView.corners);

                isPtAdded = true;
            }

            // find good image pair by optical flow and essential matrix
            ImageFindState state = (ImageFindState)findGoodImages(cap, viewContainer, featDetector, optFlow, camera,recPose, ofPrevView, ofCurrView);
            
            if (state == ImageFindState::SOURCE_LOST) { break; }
            if (state == ImageFindState::NOT_FOUND) {
                ofPrevView.corners.clear();
                ofCurrView.corners.clear();

                std::swap(featPrevView, featCurrView);

                std::cout << "Good images pair not found -> skipping current iteration!" << "\n";

                continue;
            }

            // prepare flow view images for debug draw
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

            // prepare feature view images for generate features
            featPrevView.setView(viewContainer.getLastButOneItem());
            featCurrView.setView(viewContainer.getLastOneItem());

            if (featPrevView.keyPts.empty()) {
                featDetector.generateFeatures(featPrevView.viewPtr->imGray, featPrevView.keyPts, featPrevView.descriptor);
            }

            // prepare features
            featDetector.generateFeatures(featCurrView.viewPtr->imGray, featCurrView.keyPts, featCurrView.descriptor);

            if (featPrevView.keyPts.empty() || featCurrView.keyPts.empty()) { 
                std::cerr << "None keypoints to match, skip matching/triangulation!\n";

                continue; 
            }

            std::vector<cv::Point2f> _prevPts, _currPts;
            std::vector<cv::DMatch> _matches;
            std::vector<int> _prevIdx, _currIdx;

            // match features
            descMatcher.findRobustMatches(featPrevView.keyPts, featCurrView.keyPts, featPrevView.descriptor, featCurrView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx, featPrevView.viewPtr->imColor, featCurrView.viewPtr->imColor);

            std::cout << "Matches count: " << _matches.size() << "\n";

            if (_prevPts.empty() || _currPts.empty()) { 
                std::cerr << "None points to triangulate, skip triangulation!\n";

                continue; 
            }

            // recover camera pose and get a mapping to the cloud points that were used as 3D points to recover pose
            std::map<std::pair<float, float>, size_t> cloudMapping;

            if(!m_tracking.findRecoveredCameraPose(descMatcher, params.peMinMatch, camera, featCurrView, recPose, cloudMapping)) {
                std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";
    
                std::swap(ofPrevView, ofCurrView);
                std::swap(featPrevView, featCurrView);

                continue;
            }

            // prepare previous and current camera poses for triangulation
            // previous camera pose is last camera pose in scene, current is from camera estimation
            if (m_tracking.getCamPoses().empty())
                composeExtrinsicMat(cv::Matx33d::eye(), cv::Matx31d::eye(), _prevPose);
            else
                _prevPose = m_tracking.getLastCam();
    
            composeExtrinsicMat(recPose.R, recPose.t, _currPose);

            std::vector<cv::Point2f> _newPts2D;
            std::vector<cv::Vec3d> _newPts3D;

            // detach user clicked points from optical flow and attach them to feature points to triangulate
            if (!userInput.usrClickedPts2D.empty() && isPtAdded) {
                userInput.detachPointsFromMove(_prevPts, ofPrevView.corners, userInput.usrClickedPts2D.size());
                
                userInput.detachPointsFromMove(_newPts2D, ofCurrView.corners, userInput.usrClickedPts2D.size());

                _currPts.insert(_currPts.end(), _newPts2D.begin(), _newPts2D.end());
            }

            // triangulate feature points and user clicked points
            reconstruction.triangulateCloud(camera, _prevPts, _currPts, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose.R, recPose.t);

            // get triangulated user clicked points and visualize in point cloud
            if (!userInput.usrClickedPts2D.empty() && isPtAdded) {       
                userInput.detachPointsFromReconstruction(_newPts3D, _points3D, _pointsRGB, _mask, userInput.usrClickedPts2D.size());

                userInput.addPoints(_newPts2D, _newPts3D, m_tracking.pointCloud, m_tracking.getTrackViews().size());
                
                userInput.usrClickedPts2D.clear();
                
                //visVTK.addPoints(_newPts3D);
                //visPCL.addPoints(_newPts3D);
            }

            // register tracks for PnP 2D-3D matching and point cloud
            m_tracking.addTrackView(featCurrView.viewPtr, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, cloudMapping, _currIdx);

            // confirm camera pose and add it to stack
            m_tracking.addCamPose(_currPose);

            // draw moved points
            userInput.recoverPoints(imOutUsrInp, m_tracking.pointCloud, camera.K, cv::Mat(m_tracking.actualR), cv::Mat(m_tracking.actualT));

            //visVTK.updatePointCloud(m_tracking.pointCloud.cloud3D, m_tracking.pointCloud.cloudRGB);
            visPCL.updatePointCloud(m_tracking.pointCloud.cloud3D, m_tracking.pointCloud.cloudRGB, m_tracking.pointCloud.cloudMask);

            //visVTK.updateCameras(m_tracking.getCamPoses(), camera.K);
            visPCL.updateCameras(m_tracking.getCamPoses());
            
            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);

            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }*/
    }
}
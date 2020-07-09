#include "app_solver.h"

int AppSolver::prepareImage(cv::VideoCapture& cap, cv::Mat& imColor, cv::Mat& imGray) {
    if (!cap.read(imColor)) 
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

int AppSolver::findGoodImages(cv::VideoCapture& cap, ViewDataContainer& viewContainer, FeatureDetector featDetector, OptFlow optFlow, CameraParameters camera, RecoveryPose& recPose, FlowView& ofPrevView, FlowView& ofCurrView) {
    std::cout << "Finding good images" << std::flush;

    std::vector<cv::Point2f> _prevCorners, _currCorners;
    cv::Mat _imColor, _imGray;

    // search for a good pair of images by min homography inliers -> use optical flow
    int numHomInliers = 0, numSkippedFrames = -1;
    do {
        ImageFindState state;

        if ((state = (ImageFindState)prepareImage(cap, _imColor, _imGray)) != ImageFindState::FOUND)
            return state;

        std::cout << "." << std::flush;
        
        // if there is nothing to compare, prepare the first image
        if (viewContainer.isEmpty()) {
            viewContainer.addItem(ViewData(_imColor, _imGray));

            featDetector.generateFlowFeatures(_imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);

            if ((state = (ImageFindState)prepareImage(cap, _imColor, _imGray)) != ImageFindState::FOUND)
                return state;
        }

        // set corners from back up -> computeFlow func removes corners
        _prevCorners = ofPrevView.corners;
        _currCorners = ofCurrView.corners;
 
        // flow computing with boundary and error filtering
        optFlow.computeFlow(viewContainer.getLastOneItem()->imGray, _imGray, _prevCorners, _currCorners, optFlow.statusMask); 
        ProcesingAdds::filterPointsByStatusMask(_prevCorners, _currCorners, optFlow.statusMask);
        ProcesingAdds::filterPointsByBoundary(_prevCorners, _currCorners, m_boundary);

        numSkippedFrames++;

        if (numSkippedFrames > params.bMaxSkFram) {
            viewContainer.addItem(ViewData(_imColor, _imGray));

            return ImageFindState::NOT_FOUND;
        }
    } while(!Tracking::findCameraPose(recPose, _prevCorners, _currCorners, camera.K, recPose.minInliers, numHomInliers));

    // complete the search for image pairs -> set flow views
    viewContainer.addItem(ViewData(_imColor, _imGray));

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
    
    MJPEGWriter wri(7777); wri.start();
    
    // initialize structures
    CameraParameters camera(params.cameraK, params.distCoeffs, params.bDownSamp);
    CameraData camData(&camera);

    FeatureDetector featDetector(params.fDecType);
    DescriptorMatcher descMatcher(params.fMatchType, params.fKnnRatio, params.bDebugMatE, params.winSize);
    
    cv::TermCriteria flowTermCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, params.ofMaxItCt, params.ofItEps);
    OptFlow optFlow(flowTermCrit, params.ofWinSize, params.ofMaxLevel, params.ofMaxError, params.ofMaxCorn, params.ofQualLvl, params.ofMinDist, params.ofMinKPts);

    RecoveryPose recPose(params.peMethod, params.peProb, params.peThresh, params.peMinInl, params.pePMetrod, params.peExGuess, params.peNumIteR);

    ViewDataContainer viewContainer(params.bDebugMatE ? INT32_MAX : 100);

    FeatureView featPrevView, featCurrView;
    FlowView ofPrevView, ofCurrView; 

    Reconstruction reconstruction(params.tMethod, params.baMethod, params.baMaxRMSE, params.tMinDist, params.tMaxDist, params.tMaxPErr, true);

    PointCloud pointCloud(params.cSRemThr); 
    Tracking tracking(&pointCloud);

    cv::Mat imOutUsrInp, imOutRecPose, imOutMatches;
    
    UserInput userInput(params.usrInpWinName, &imOutUsrInp, &pointCloud, params.ofMaxError);

    WindowInputDataParams mouseUsrDataParams(&m_isUpdating, &userInput);
    
    // run windows in new thread -> avoid rendering white screen
    cv::startWindowThread();

    cv::namedWindow(params.usrInpWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(params.recPoseWinName, cv::WINDOW_NORMAL);
    
    cv::resizeWindow(params.usrInpWinName, params.winSize);
    cv::resizeWindow(params.recPoseWinName, params.winSize);

    if (params.bDebugMatE) {
        cv::namedWindow(params.matchesWinName, cv::WINDOW_NORMAL);
        cv::resizeWindow(params.matchesWinName, params.winSize);
    }

    cv::setMouseCallback(params.usrInpWinName, onUsrWinClick, (void*)&mouseUsrDataParams);

    // initialize visualization windows VTK, PCL
    VisPCL visPCL(params.ptCloudWinName + " PCL", params.winSize);

    //VisVTK visVTK(params.ptCloudWinName + " VTK", params.winSize);
#pragma endregion INIT

    for (uint iteration = 1; ; ++iteration) {
        //while (!m_isUpdating) {}

        // use if statements instead of switch due to loop breaks
#pragma region KLT Tracker

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

            // prepare views to load new frame
            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }
        
#pragma endregion KLT Tracker
#pragma region Visual Odometry

        if (m_usedMethod == Method::VO) {
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

            userInput.lockClickedPoints();

            if (!ofPrevView.corners.empty()) {
                userInput.attachPointsToMove(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

                // move user points and corners
                optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);

                userInput.detachPointsFromMove(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

                PointsMove pointsMove; ProcesingAdds::analyzePointsMove(ofPrevView.corners, ofCurrView.corners, pointsMove);
                ProcesingAdds::correctPointsByMoveAnalyze(userInput.doneClickedPts, userInput.moveClickedPts, pointsMove);

                optFlow.drawOpticalFlow(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);
            }

            std::vector<cv::Vec3d> _points3D, _usrPoints3D;
            std::vector<cv::Vec3b> _pointsRGB, _usrPointsRGB;
            std::vector<bool> _mask, _usrMask;

            cv::Matx34d _prevPose, _currPose;

            composeExtrinsicMat(camData.actualR, camData.actualT, _prevPose);

            // move camera position by computed R a t from optical flow and essential matrix
            camData.actualT = camData.actualT + (camData.actualR * recPose.t);
            camData.actualR = camData.actualR * recPose.R;

            composeExtrinsicMat(camData.actualR, camData.actualT, _currPose);
            camData.addCamPose(_currPose);

            // triangulate corners
            reconstruction.triangulateCloud(camera, ofPrevView.corners, ofCurrView.corners, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose.R, recPose.t);

            // triangulate user clicked points
            reconstruction.triangulateCloud(camera, userInput.doneClickedPts, userInput.moveClickedPts, ofCurrView.viewPtr->imColor, _usrPoints3D, _usrPointsRGB, _usrMask, _prevPose, _currPose, recPose.R, recPose.t);

            userInput.addPoints(userInput.moveClickedPts, _usrPoints3D, tracking.getTrackViews().size());
             
            userInput.clearClickedPoints();
            userInput.updateWaitingPoints();
            userInput.unlockClickedPoints();

            // draw moved points
            userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(camData.actualR), cv::Mat(camData.actualT));
        
            //visPCL.addCamera(camData.extrinsics.back() , camera.K);
            //visPCL.addPoints(_usrPoints3D);

            cv::imshow(params.usrInpWinName, imOutUsrInp);

            // prepare views to load new frame
            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }

#pragma endregion Visual Odometry
#pragma region Perspective-n-Point

        if (m_usedMethod == Method::PNP) {
            if (iteration != 1) {
                // do bundle adjust after loop iteration to avoid "continue" statement
                if (params.baProcIt != 0 && (iteration % params.baProcIt == 1 || params.baProcIt == 1)) {
                    reconstruction.adjustBundle(camData, pointCloud);
                }

                // do filteration after loop iteration to avoid "continue" statement
                if (params.cFProcIt != 0 && (iteration % params.cFProcIt == 1 || params.cFProcIt == 1)) {
                    pointCloud.filterCloud();
                }

                if (ofPrevView.corners.size() < optFlow.additionalSettings.minFeatures) {
                    ofPrevView.setView(viewContainer.getLastOneItem());

                    featDetector.generateFlowFeatures(ofPrevView.viewPtr->imGray, ofPrevView.corners, optFlow.additionalSettings.maxCorn, optFlow.additionalSettings.qualLvl, optFlow.additionalSettings.minDist);
                }
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

            userInput.lockClickedPoints();

            if (!ofPrevView.corners.empty()) {
                userInput.attachPointsToMove(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

                // move user points and corners
                optFlow.computeFlow(ofPrevView.viewPtr->imGray, ofCurrView.viewPtr->imGray, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);

                userInput.detachPointsFromMove(ofPrevView.corners, ofCurrView.corners, optFlow.statusMask, true, false);

                PointsMove pointsMove; ProcesingAdds::analyzePointsMove(ofPrevView.corners, ofCurrView.corners, pointsMove);
                ProcesingAdds::correctPointsByMoveAnalyze(userInput.doneClickedPts, userInput.moveClickedPts, pointsMove);

                optFlow.drawOpticalFlow(imOutRecPose, imOutRecPose, ofPrevView.corners, ofCurrView.corners, optFlow.statusMask);
            }

            std::vector<cv::Vec3d> _points3D, _usrPoints3D;
            std::vector<cv::Vec3b> _pointsRGB, _usrPointsRGB;
            std::vector<bool> _mask, _usrMask;

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
            descMatcher.findRobustMatches(featPrevView.keyPts, featCurrView.keyPts, featPrevView.descriptor, featCurrView.descriptor, _prevPts, _currPts, _matches, _prevIdx, _currIdx, featPrevView.viewPtr->imColor, featCurrView.viewPtr->imColor, true);

            std::cout << "Matches count: " << _matches.size() << "\n";

            if (_prevPts.empty() || _currPts.empty()) { 
                std::cerr << "None points to triangulate, skip triangulation!\n";

                continue; 
            }

            TrackView _trackView;

            if(!tracking.trackViews.empty() && !Tracking::findRecoveredCameraPose(descMatcher, params.peMinMatch, params.peTMaxIter, camera, featCurrView, recPose, tracking.trackViews, _trackView, pointCloud)) {
                std::cout << "Recovering camera fail, skip current reconstruction iteration!\n";
    
                std::swap(ofPrevView, ofCurrView);
                std::swap(featPrevView, featCurrView);

                continue;
            }

            // prepare previous and current camera poses for triangulation
            // previous camera pose is last camera pose in scene, current is from camera estimation
            if (camData.extrinsics.empty())
                composeExtrinsicMat(cv::Matx33d::eye(), cv::Matx31d::eye(), _prevPose);
            else
                _prevPose = camData.extrinsics.back();
    
            composeExtrinsicMat(recPose.R, recPose.t, _currPose);

            // triangulate feature points and user clicked points
            reconstruction.triangulateCloud(camera, _prevPts, _currPts, ofCurrView.viewPtr->imColor, _points3D, _pointsRGB, _mask, _prevPose, _currPose, recPose.R, recPose.t);

            // triangulate user clicked points
            reconstruction.triangulateCloud(camera, userInput.doneClickedPts, userInput.moveClickedPts, ofCurrView.viewPtr->imColor, _usrPoints3D, _usrPointsRGB, _usrMask, _prevPose, _currPose, recPose.R, recPose.t);

            userInput.addPoints(userInput.moveClickedPts, _usrPoints3D, tracking.getTrackViews().size());

            // register tracks for PnP 2D-3D matching and point cloud
            if (tracking.addTrackView(featCurrView.viewPtr, _trackView, _mask, _currPts, _points3D, _pointsRGB, featCurrView.keyPts, featCurrView.descriptor, _currIdx)) {
                camData.addCamPose(_currPose);

                //visVTK.addPoints(_usrPoints3D);
                visPCL.addPoints(_usrPoints3D);
            }

            userInput.clearClickedPoints();
            userInput.updateWaitingPoints();
            userInput.unlockClickedPoints();

            // draw moved points
            userInput.recoverPoints(imOutUsrInp, camera.K, cv::Mat(camData.actualR), cv::Mat(camData.actualT));

            //visVTK.updatePointCloud(pointCloud.cloud3D, pointCloud.cloudRGB, pointCloud.cloudMask);
            visPCL.updatePointCloud(pointCloud.cloud3D, pointCloud.cloudRGB, pointCloud.cloudMask);

            //visVTK.updateCameras(camData.extrinsics, camera.K);
            visPCL.updateCameras(camData.extrinsics);
            //visVTK.visualize(params.ptCloudWinName + " VTK", params.winSize, cv::viz::Color::black());

            cv::imshow(params.usrInpWinName, imOutUsrInp);

            std::swap(ofPrevView, ofCurrView);
            std::swap(featPrevView, featCurrView);
        }

#pragma endregion Perspective-n-Point

        wri.write(imOutUsrInp);

        std::cout << "Iteration: " << iteration << "\n"; cv::waitKey(29);
    }

    cap.release();
    wri.stop();
}
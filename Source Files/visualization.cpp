#include "visualization.h"

VisPCL::VisPCL(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor, const bool isEnable) {
    if (isEnable) {
        m_numClouds = 0;
        m_numCams = 0;
        m_numPoints = 0;

        m_visThread = std::thread(&VisPCL::visualize, this, windowName, windowSize, backgroundColor);
    }
}

VisPCL::~VisPCL() {
    if (m_visThread.joinable()) { m_visThread.join(); }
}

bool VisPCL::isViewerInitialized() const {
    return m_viewer != NULL;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> VisPCL::getNewViewer(const std::string windowName, const cv::Size windowSize, const cv::viz::Color bColor) {
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud);
    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
    viewer->setBackgroundColor(bColor.val[0], bColor.val[1], bColor.val[2]);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, -50, 0, -1, 0);
    viewer->setSize(windowSize.width, windowSize.height);
    
    return (viewer);
}

void VisPCL::updatePointCloud(const std::list<cv::Vec3d>& points3D, const std::list<cv::Vec3b>& pointsRGB, const std::vector<bool>& pointsMask) {
    if (isViewerInitialized()) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (auto [p3d, p3dEnd, pClr, pClrEnd, pIdx] = std::tuple{points3D.cbegin(), points3D.cend(), pointsRGB.cbegin(), pointsRGB.cend(), 0}; p3d != p3dEnd && pClr != pClrEnd; ++p3d, ++pClr, ++pIdx) {
            if (pointsMask[pIdx]) {
                pcl::PointXYZRGB rgbPoint;

                rgbPoint.x = p3d->val[0];
                rgbPoint.y = p3d->val[1];
                rgbPoint.z = p3d->val[2];

                rgbPoint.r = pClr->val[2];
                rgbPoint.g = pClr->val[1];
                rgbPoint.b = pClr->val[0];

                pointCloud->push_back(rgbPoint);
            }
        }

        m_visMutex.lock();

        m_viewer->updatePointCloud(pointCloud);

        m_visMutex.unlock();
    }
}

void VisPCL::addPoints(const std::vector<cv::Vec3d> points3D) {
    m_visMutex.lock();

    for (const auto& p : points3D) {
        pcl::PointXYZ pclPose(p.val[0], p.val[1], p.val[2]);

        if (isViewerInitialized())
            m_viewer->addSphere(pclPose, 1.0, 128, 0, 128, "point_" + std::to_string(m_numPoints));

        m_numPoints++;
    }

    m_visMutex.unlock();
}

void VisPCL::addCamera(const cv::Matx34d camPose, const cv::Matx33d K33d) {
    pcl::PointXYZ pclPose; cvPoseToInversePCLPose(camPose, pclPose);

    if (isViewerInitialized())
        m_viewer->addSphere(pclPose, 1.0, 255, 255, 0, "cam_pose_" + std::to_string(m_numCams));

    m_numCams++;
}

void VisPCL::updateCameras(const std::list<cv::Matx34d> camPoses) {
    if (m_viewer != NULL) {
        m_visMutex.lock();

    for (auto [it, end, idx] = std::tuple{camPoses.cbegin(), camPoses.cend(), 0}; it != end; ++it, ++idx) {
        auto c = (cv::Matx34d)*it;

        pcl::PointXYZ pclPose; cvPoseToInversePCLPose(c, pclPose);

        if (idx == m_numCams) {
            if (isViewerInitialized())
                m_viewer->addSphere(pclPose, 1.0, 255, 0, 0, "cam_pose_" + std::to_string(idx));
            m_numCams++;
        }
        else {
            if (isViewerInitialized())
                m_viewer->updateSphere(pclPose, 0.5, 255, 165, 0, "cam_pose_" + std::to_string(idx));
        }
    }

    m_visMutex.unlock();
    }

    m_visMutex.unlock();
}

void VisPCL::visualize(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor) {
    m_viewer = getNewViewer(windowName, windowSize, backgroundColor);
    
    while (!m_viewer->wasStopped()) {
        m_visMutex.lock();

        if (isViewerInitialized())
            m_viewer->spinOnce();

        m_visMutex.unlock();
    }
}

VisVTK::VisVTK(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor, const bool isEnable) {
    if (isEnable) {
        m_numClouds = 0;
        m_numCams = 0;
        m_numPoints = 0;

        m_visThread = std::thread(&VisVTK::visualize, this, windowName, windowSize, backgroundColor);
    }
}

VisVTK::~VisVTK() {
    if (m_visThread.joinable()) { m_visThread.join(); }
}

bool VisVTK::isViewerInitialized() const {
    return true;
}

void VisVTK::updatePointCloud(const std::list<cv::Vec3d>& points3D, const std::list<cv::Vec3b>& pointsRGB, const std::vector<bool>& pointsMask) {
    const std::vector<cv::Vec3d> _points3D(points3D.begin(), points3D.end());
    const std::vector<cv::Vec3b> _pointsRGB(pointsRGB.begin(), pointsRGB.end());

    const cv::viz::WCloud _pCloud(_points3D, _pointsRGB);
    
    m_visMutex.lock();

    // Update point cloud
    if (m_numClouds > 0)
        m_viewer.removeWidget("cloud");
    m_viewer.showWidget("cloud", _pCloud);

    m_visMutex.unlock();
    

    m_numClouds++;
}

void VisVTK::addPoints(const std::vector<cv::Vec3d> points3D) {
    for (const auto& p : points3D) {
        const cv::viz::WSphere _point(cv::Point3d(), 1.0, 20, cv::viz::Color::purple());

        m_viewer.showWidget("point_" + std::to_string(m_numPoints), _point, cv::Affine3d(cv::Vec3d(), p));

        m_numPoints++;
    }
}

void VisVTK::updateCameras(const std::list<cv::Matx34d> camPoses, const cv::Matx33d K33d) {
    //  update all old cameras and create actual camera as red
    for (auto [it, end, idx] = std::tuple{camPoses.cbegin(), camPoses.cend(), 0}; it != end; ++it, ++idx) {
        auto c = (cv::Matx34d)*it;

        cv::Affine3d vtkPose; cvPoseToInverseVTKPose(c, vtkPose);

        if (idx != m_numCams) {
            const cv::viz::WCameraPosition _cam(K33d, -1, cv::viz::Color::orange());

            //m_viewer.removeWidget("cam_" + std::to_string(idx));

            m_viewer.showWidget("cam_" + std::to_string(idx), _cam, vtkPose);
        } else {
            const cv::viz::WCameraPosition _cam(K33d, -5, cv::viz::Color::red());

            m_viewer.showWidget("cam_" + std::to_string(idx), _cam, vtkPose);
        }
    }

    m_numCams++;
}

void VisVTK::addCamera(const cv::Matx34d camPose, const cv::Matx33d K33d) {
    const cv::viz::WCameraPosition _cam(K33d, -1, cv::viz::Color::orange());

    cv::Affine3d vtkPose; cvPoseToInverseVTKPose(camPose, vtkPose);

    m_viewer.showWidget("cam_" + std::to_string(m_numCams), _cam, vtkPose);

    m_numCams++;
}

void VisVTK::visualize(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor) {
    m_viewer = cv::viz::Viz3d(windowName);
    m_viewer.setBackgroundColor(cv::viz::Color::black());
    m_viewer.setWindowSize(windowSize);

    cv::Mat rotVec = cv::Mat::zeros(1,3,CV_32F);

    //rotVec.at<float>(0,0) += CV_PI * 1.0f;
	//rotVec.at<float>(0,2) += CV_PI * 1.0f;

	cv::Mat rotMat; cv::Rodrigues(rotVec, rotMat);

    cv::Affine3f rotation(rotMat, cv::Vec3d());
    
    //  Move camera a little bit back away from the cloud
    m_viewer.setViewerPose(rotation.translate(cv::Vec3d(0, 0, -150)));

    while (!m_viewer.wasStopped()) {
        m_visMutex.lock();

        m_viewer.spinOnce();

        m_visMutex.unlock();
    }
}
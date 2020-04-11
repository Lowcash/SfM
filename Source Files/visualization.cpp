#include "visualization.h"

VisPCL::VisPCL(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor) {
    m_viewer = getNewViewer(windowName, windowSize, backgroundColor);
        
    m_numClouds = 0;
    m_numCams = 0;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> VisPCL::getNewViewer(const std::string windowName, const cv::Size windowSize, const cv::viz::Color bColor) {
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
    viewer->setBackgroundColor(bColor.val[0], bColor.val[1], bColor.val[1]);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, -1, 0, -1, 0);
    viewer->setSize(windowSize.width, windowSize.height);
    
    return (viewer);
}

void VisPCL::addPointCloud(const std::vector<cv::Point3f>& points3D, const std::vector<cv::Vec3b>& pointsRGB) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (auto [it, end, i] = std::tuple{points3D.cbegin(), points3D.cend(), 0}; it != end; ++it, ++i) {
        pcl::PointXYZRGB rgbPoint;

        auto p3d = (cv::Point3d)*it;
        auto pClr = pointsRGB[i];

        rgbPoint.x = p3d.x;
        rgbPoint.y = p3d.y;
        rgbPoint.z = p3d.z;

        rgbPoint.r = pClr[2];
        rgbPoint.g = pClr[1];
        rgbPoint.b = pClr[0];

        pointCloud->push_back(rgbPoint);
    }
    
    m_viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud, "cloud_" + std::to_string(m_numClouds));
    
    m_numClouds++;
}

void VisPCL::addPointCloud(const std::vector<TrackView>& trackViews) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (auto [tIt, tEnd, tIdx] = std::tuple{trackViews.crbegin(), trackViews.crend(), 0}; tIt != tEnd; ++tIt, ++tIdx) {
        auto t = (TrackView)*tIt;

        for (auto [it, end, i] = std::tuple{t.points3D.cbegin(), t.points3D.cend(), 0}; it != end; ++it, ++i) {
            pcl::PointXYZRGB rgbPoint;

            auto p3d = (cv::Point3d)*it;
            auto pClr = t.pointsRGB[i];

            rgbPoint.x = p3d.x;
            rgbPoint.y = p3d.y;
            rgbPoint.z = p3d.z;

            rgbPoint.r = pClr[2];
            rgbPoint.g = pClr[1];
            rgbPoint.b = pClr[0];

            pointCloud->push_back(rgbPoint);
        }

        break;
    }

    m_viewer->updatePointCloud(pointCloud);
}

void VisPCL::addCamera(const cv::Matx34f camPose) {       
    pcl::PointXYZ pclPose; cvPoseToPCLPose(camPose, pclPose);

    m_viewer->addSphere(pclPose, 0.25, "cam_pose_" + std::to_string(m_numCams));

    m_numCams++;
}

void VisPCL::updateCameras(const std::vector<cv::Matx34f> camPoses) { 
    for (auto [it, end, idx] = std::tuple{camPoses.cbegin(), camPoses.cend(), 0}; it != end; ++it, ++idx)  {
        auto c = (cv::Matx34d)*it;

        pcl::PointXYZ pclPose; cvPoseToInversePCLPose(c, pclPose);
        
        if (idx == m_numCams) {
            m_viewer->addSphere(pclPose, 0.25, "cam_pose_" + std::to_string(idx));
            m_numCams++;
        }
        else
            m_viewer->updateSphere(pclPose, 0.25, 200, 200, 0, "cam_pose_" + std::to_string(idx));
    }
}

void VisPCL::visualize() {
    //while(!m_viewer->wasStopped())
        //m_viewer->spinOnce(60);
    m_viewer->spin();
}

VisVTK::VisVTK(const std::string windowName, const cv::Size windowSize, const cv::viz::Color backgroundColor) {
    m_viewer = cv::viz::Viz3d(windowName);
    m_viewer.setBackgroundColor(cv::viz::Color::black());
    m_viewer.setWindowSize(windowSize);

    m_numClouds = 0;
    m_numCams = 0;
}

void VisVTK::addPointCloud(const std::vector<cv::Point3f>& points3D, const std::vector<cv::Vec3b>& pointsRGB) {}

void VisVTK::addPointCloud(const std::vector<TrackView>& trackViews) {}

void VisVTK::updateCameras(const std::vector<cv::Matx34f> camPoses) {}

void VisVTK::addCamera(const cv::Matx34f camPose) {}

void VisVTK::visualize() {
    //while(!m_viewer->wasStopped())
        //m_viewer->spinOnce(60);
    m_viewer.spin();
}
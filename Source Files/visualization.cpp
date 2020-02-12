#include "visualization.h"

Visualization::Visualization(Settings visSettings) {
	m_window = cv::viz::Viz3d(visSettings.getWindowName());
	m_window.setWindowSize(visSettings.getWindowSize());
	m_window.setWindowPosition(visSettings.getWindowPosition());
	m_window.setBackgroundColor(visSettings.getBackgroundColor());
}

void Visualization::updateVieverPose(cv::Affine3d viewerPose) {
	m_window.setViewerPose(viewerPose);
}

void Visualization::updateCameraPose(cv::Affine3d cam) {
	cv::viz::WSphere sphere(cv::Point3f(), 1);

	m_camPoses.emplace_back(cam);

	m_window.showWidget("cam_pose_" + std::to_string(m_camPoses.size()), sphere, cam);

	m_window.spinOnce(1, true);
}

void Visualization::updatePointCloud(ColoredPoints coloredPointCloud, cv::Affine3d cam) {
	m_pointClouds.emplace_back(coloredPointCloud);

	std::cout << "\n";
	std::cout << "-------------------------------------------\n";
	std::cout << "Vizualized " << coloredPointCloud.m_points.size() << " points" << "\n";
	std::cout << "-------------------------------------------\n";
	std::cout << "\n";

	cv::Mat rotVec = cv::Mat::zeros(1,3,CV_32F);

	rotVec.at<float>(0,1) -= CV_PI * 0.5f;

	cv::Mat rotMat; cv::Rodrigues(rotVec, rotMat);

	cv::Affine3f cloudPose(rotMat, cv::Vec3d());

	//m_window.removeAllWidgets();
	const cv::viz::WCloud pointCloud(coloredPointCloud.getPoints(), coloredPointCloud.getColors());

	//m_window.showWidget("point_cloud", pointCloud, cloudPose);
	//m_window.showWidget("point_cloud_" + std::to_string(m_pointClouds.size()), pointCloud, cam);
	m_window.showWidget("point_cloud_" + std::to_string(m_pointClouds.size()), pointCloud);

	//m_window.spin();
	//m_window.spinOnce(1, true);
}

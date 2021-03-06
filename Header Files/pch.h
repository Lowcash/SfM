#ifndef PCH_H
#define PCH_H
#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define _USE_OPENCV true

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/highgui/highgui_c.h>
#include <opencv4/opencv2/core.hpp>
// #include <opencv4/opencv2/sfm.hpp>
// #include <opencv4/opencv2/sfm/robust.hpp>
// #include <opencv4/opencv2/sfm/reconstruct.hpp>
#include <opencv4/opencv2/viz.hpp>
#include <opencv4/opencv2/core/utils/logger.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>
#include <opencv4/opencv2/tracking.hpp>

#define CERES_FOUND true

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#endif //PCH_H

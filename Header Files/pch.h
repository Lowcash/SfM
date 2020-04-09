#ifndef PCH_H
#define PCH_H

#include <iostream>
#include <thread>
#include <chrono>

#define _USE_OPENCV true

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/highgui/highgui_c.h>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/sfm.hpp>
#include <opencv4/opencv2/sfm/reconstruct.hpp>
#include <opencv4/opencv2/viz.hpp>
#include <opencv4/opencv2/core/utils/logger.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaoptflow.hpp>

#define CERES_FOUND true

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/connected_components.hpp>

// #include <pcl/common/common_headers.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/registration/transformation_estimation_svd.h>
// #include <pcl/common/transforms.h>

#endif //PCH_H

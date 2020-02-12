#ifndef PCH_H
#define PCH_H

#define CERES_FOUND true
#define _USE_OPENCV true

#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/sfm/reconstruct.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/connected_components.hpp>

#endif //PCH_H

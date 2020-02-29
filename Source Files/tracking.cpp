#include "tracking.h"

Tracking::Tracking(const float matchRatioThreshold, const bool isDebugVisualization) {}

Tracking::~Tracking() {}

void Tracking::updateTracks(const std::vector<cv::Point2f> points2D, const std::vector<cv::Point3d> points3D, const cv::Mat descriptor, const std::vector<cv::Vec3b> colors) {
    matchTracks(points2D, points3D, descriptor);

    int i, updated, missed;

    i = updated = missed = 0;

    for (auto track = m_tracks.begin(); track != m_tracks.end(); ++track, ++i) {
        int j = m_matchIdx[i];
        if (j >= 0) {
            cv::Point3f point = points3D[j];

            if (point.x != 0 && point.y != 0 && point.z != 0){
                track->missedFrames = 0;
                track->activePosition2d = points2D[j];
                track->activePosition3d = points3D[j];
                descriptor.row(j).copyTo(track->descriptor);

                updated++;
                continue;
            }
        }
        
        track->missedFrames++;
        missed++;
    }

    std::cout << "Updated Tracks:" << updated << " Missed:" << missed << "\n";

    m_tracks.remove_if(isTrackingStale);
}

void Tracking::matchTracks(const std::vector<cv::Point2f> points2D, const std::vector<cv::Point3d> points3D, const cv::Mat descriptor) {
    // Create a FlannMatcher based on values provided in docs
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(12, 20, 2));

    std::vector<cv::Mat> trainVector;
    trainVector.emplace_back(descriptor);

    matcher.add(trainVector);

    m_matchIdx.resize(m_tracks.size());

    int i, matchCount;

    i = matchCount = 0;

    for (auto track = m_tracks.begin(); track != m_tracks.end(); ++track, ++i) {
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(track->descriptor, matches, 2);

        if (matches[0].size() >= 2) {
            float best_dist = matches[0][0].distance;
            float next_dist = matches[0][1].distance;

            if (best_dist < next_dist * RATIO) {
                m_matchIdx[i] = matches[0][0].trainIdx;
                matchCount++;
            }
            else 
                m_matchIdx[i] = -1;
        }
    }

    std::cout << "Matched features:" << matchCount << "\n";
}

bool Tracking::isTrackingStale(const Track& track) {
    return track.missedFrames > 10;
}

void Tracking::tracksToPointCloud(std::vector<cv::Point3d>& points3D, std::vector<cv::Vec3b>& colors) {
    points3D.clear();
    colors.clear();

    const double maxZ = 1.0e4;

    int i = 0;
	for (auto track = m_tracks.begin(); track != m_tracks.end(); ++track, ++i) {

		if (track->missedFrames == 0) {
			cv::Point3f point = track->activePosition3d;
            
			if (fabs(point.z - maxZ) < FLT_EPSILON || fabs(point.z) > maxZ) continue;

			points3D.emplace_back(point);
            colors.emplace_back(track->color);
		}
	}
}

void Tracking::tracksToPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud) {
    const double maxZ = 1.0e4;

    int i = 0;
	for (auto track = m_tracks.begin(); track != m_tracks.end(); ++track, ++i) {

		if (track->missedFrames == 0) {
			cv::Point3f point = track->activePosition3d;
            cv::Vec3b color = track->color;

			if (fabs(point.z - maxZ) < FLT_EPSILON || fabs(point.z) > maxZ) continue;

			pcl::PointXYZRGB rgbPoint;
			rgbPoint.x = point.x;
			rgbPoint.y = point.y;
			rgbPoint.z = point.z;
            rgbPoint.r = color[2];
            rgbPoint.g = color[1];
            rgbPoint.b = color[0];

            pointCloud->points.push_back(rgbPoint);
		}
	}
}
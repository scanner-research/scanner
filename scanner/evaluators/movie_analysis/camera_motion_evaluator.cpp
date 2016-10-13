#include "camera_motion_evaluator.h"

namespace scanner {

CameraMotionEvaluator::CameraMotionEvaluator() {
  detector = cv::xfeatures2d::SURF::create();
  detector->setHessianThreshold(400);
}

void CameraMotionEvaluator::evaluate(
  std::vector<cv::Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes) {

  std::vector<cv::Mat> imgs_gray;
  for (auto& input : inputs) {
    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    imgs_gray.emplace_back(gray);
  }

  std::vector<std::vector<cv::KeyPoint>> keypoints;
  std::vector<cv::Mat> descriptors;

  double start;

  start = CycleTimer::currentSeconds();
  for (auto& img : imgs_gray) {
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    detector->detectAndCompute(img, cv::Mat(), kps, desc);
    keypoints.emplace_back(std::move(kps));
    descriptors.emplace_back(std::move(desc));
  }
  LOG(INFO) << "Features: " << CycleTimer::currentSeconds() - start;

  std::vector<std::vector<std::pair<cv::Vec2f, cv::Vec2f>>> pairs;

  start = CycleTimer::currentSeconds();
  for (i32 i = 0; i < inputs.size() - 1; ++i) {
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors[i], descriptors[i+1], matches);

    std::vector<std::pair<cv::Vec2f, cv::Vec2f>> plist;

    for (auto& match : matches) {
      if (match.distance > 200) { continue; }
      cv::Point2f p1 = keypoints[i][match.queryIdx].pt;
      cv::Point2f p2 = keypoints[i+1][match.trainIdx].pt;
      plist.emplace_back(std::make_pair(cv::Vec2f(p1.x, p1.y), cv::Vec2f(p2.x, p2.y)));
    }

    pairs.emplace_back(std::move(plist));
  }
  LOG(INFO) << "Matches: " << CycleTimer::currentSeconds() - start;

  std::vector<std::vector<cv::Vec2f>> tracks;

  start = CycleTimer::currentSeconds();
  for (i32 i = 0; i < pairs.size(); ++i) {
    for (auto& pts : pairs[i]) {
      bool found = false;
      for (auto& track : tracks) {
        cv::Vec2f& p1 = track[track.size()-1];
        cv::Vec2f& p2 = pts.first;
        if (p1 == p2) {
          track.emplace_back(pts.second);
          found = true;
          break;
        }
      }
      if (!found) {
        std::vector<cv::Vec2f> track;
        for (i32 j = 0; j < i; ++j) {
          track.emplace_back(cv::Vec2f(-1));
        }
        track.emplace_back(pts.first);
        track.emplace_back(pts.second);
        tracks.emplace_back(track);
      }
    }

    for (auto& track : tracks) {
      if (track.size() != i+1) {
        track.emplace_back(cv::Vec2f(-1));
      }
    }
  }

  tracks.resize(15);

  LOG(INFO) << "Tracks: " << CycleTimer::currentSeconds() - start;
  LOG(INFO) << tracks.size();

  std::vector<cv::Mat> points2d;
  for (i32 i = 0; i < inputs.size()-1; ++i) {
    cv::Mat_<float> frame(2, tracks.size(), CV_32F);
    for (i32 j = 0; j < tracks.size(); ++j) {
      frame(0,j) = tracks[j][i][0];
      frame(1,j) = tracks[j][i][1];
    }
    points2d.emplace_back(cv::Mat(frame));
  }

  start = CycleTimer::currentSeconds();
  double f = 0.05, cx = 0.5, cy = 0.5;
  cv::Matx33d K = cv::Matx33d(f, 0, cx,
                              0, f, cy,
                              0, 0, 1);
  std::vector<cv::Mat> Rs_est, ts_est, points3d_estimated;
  cv::sfm::reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, true);
  LOG(INFO) << "Reconstruct: " << CycleTimer::currentSeconds() - start;

  std::vector<cv::Affine3d> path_est;
  for (i32 i = 0; i < Rs_est.size(); ++i) {
    path_est.emplace_back(cv::Affine3d(Rs_est[i], ts_est[i]));
  }

  for (i32 i = 0; i < path_est.size(); ++i) {
    LOG(INFO) << path_est[i].translation();
  }
}

}

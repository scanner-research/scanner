#include "camera_motion_evaluator.h"

#include <cstring>

namespace scanner {

CameraMotionEvaluator::CameraMotionEvaluator() {
#if CV_MAJOR_VERSION >= 3
  detector = cv::xfeatures2d::SURF::create();
  detector->setHessianThreshold(400);
#endif
}

void CameraMotionEvaluator::from_homography(
  std::vector<cv::Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes) {

#if CV_MAJOR_VERSION >= 3
  std::vector<cv::Mat> imgs_gray;
  cv::Size im_size = inputs[0].size();
  for (auto& input : inputs) {
    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    imgs_gray.emplace_back(gray);
  }

  if (!initial_frame.empty()) {
    cv::Mat gray;
    cv::cvtColor(initial_frame, gray, CV_BGR2GRAY);
    imgs_gray.emplace(imgs_gray.begin(), std::move(gray));
  } else {
    i32 out_size = sizeof(double);
    u8* out_buf = new u8[out_size];
    *((double*)out_buf) = 0;
    output_buffers.push_back(out_buf);
    output_sizes.push_back(out_size);
  }

  assert(imgs_gray.size() > 1);

  std::vector<std::vector<cv::KeyPoint>> keypoints;
  std::vector<cv::Mat> descriptors;

  for (auto& img : imgs_gray) {
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    detector->detectAndCompute(img, cv::Mat(), kps, desc);
    keypoints.emplace_back(std::move(kps));
    descriptors.emplace_back(std::move(desc));
  }

  double focal_length = 28, sensor_width = 4.6; // mm
  double fx = im_size.width * focal_length / sensor_width;
  double fy = im_size.height * focal_length / sensor_width;
  double mx = im_size.width/2.0;
  double my = im_size.height/2.0;
  cv::Matx33d K(fx, 0,  mx,
                0,  fy, my,
                0,  0,  1);

  for (i32 i = 0; i < imgs_gray.size() - 1; ++i) {
    double norm = -1;
    if (!descriptors[i].empty() && !descriptors[i+1].empty()) {
      std::vector<cv::DMatch> matches;
      matcher.match(descriptors[i], descriptors[i+1], matches);

      std::vector<cv::Point2f> a_pts, b_pts;
      for (auto& match : matches) {
        if (match.distance > 600) { continue; }
        a_pts.emplace_back(keypoints[i][match.queryIdx].pt);
        b_pts.emplace_back(keypoints[i+1][match.trainIdx].pt);
      }

      cv::Mat H = cv::findHomography(a_pts, b_pts, CV_RANSAC);
      if (!H.empty()) {
        std::vector<cv::Mat> rotations, translations, normals;
        cv::decomposeHomographyMat(H, K, rotations, translations, normals);
        LOG(INFO) << cv::norm(translations[0]);
        norm = 100.0 * cv::norm(translations[0]);
      }
   }

    double* out_buf = new double;
    *out_buf = norm;
    output_buffers.push_back((u8*)out_buf);
    output_sizes.push_back(sizeof(double));
  }
#else
  LOG(FATAL) << "Need OpenCV 3.x for homographies";
#endif
}

void CameraMotionEvaluator::from_background_subtraction(
  std::vector<cv::Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes)
{
  for (i32 i = 0; i < inputs.size(); ++i) {
    cv::Mat a, b;
    if (i == 0) {
      if (initial_frame.empty()) {
        double* out_buf = new double;
        *out_buf = 0;
        output_buffers.push_back((u8*)out_buf);
        output_sizes.push_back(sizeof(double));
        continue;
      } else {
        a = initial_frame;
        b = inputs[0];
      }
    } else {
      a = inputs[i-1];
      b = inputs[i];
    }
    cv::Mat diff(inputs[0].size(), CV_8UC1);
    cv::absdiff(a, b, diff);
    cv::threshold(diff, diff, 10, 255, CV_THRESH_BINARY);
    double color = 0;
    std::vector<cv::Mat> bgr_planes;
    cv::split(diff, bgr_planes);
    for (auto& plane : bgr_planes) {
      color += ((double) cv::countNonZero(plane)) / (plane.rows * plane.cols);
    }
    color /= 3.0;

    double* out_buf = new double;
    *out_buf = color;
    output_buffers.push_back((u8*)out_buf);
    output_sizes.push_back(sizeof(double));
  }
}

void CameraMotionEvaluator::evaluate(
  std::vector<cv::Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes) {
  from_homography(inputs, output_buffers, output_sizes);
}

}

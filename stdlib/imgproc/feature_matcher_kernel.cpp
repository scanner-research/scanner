#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scanner/util/serialize.h"
#include "scanner/util/cycle_timer.h"
#include "stdlib/stdlib.pb.h"

#include <opencv2/xfeatures2d.hpp>

namespace scanner {

class Constants {
 public:
  int w = 32;
  int g = 4;
  int iw, ih;
  int T;
  int d;
  float tau_c, gamma;

  Constants(int iw, int ih, int T) {
    this->iw = iw;
    this->ih = ih;
    this->T = T;
    d = (int)sqrt((float)(ih * ih + iw * iw));
    tau_c = 0.1 * (float)d;
    gamma = 0.5 * (float)d;
  }
};

class FeatureMatcherKernel : public StenciledKernel, public VideoKernel {
 public:
  FeatureMatcherKernel(const KernelConfig& config)
    : StenciledKernel(config), device_(config.devices[0]), C_(0, 0, 0) {
    set_device();

    matcher_ = cvc::DescriptorMatcher::createBFMatcher();
    features_suffix_.resize(C_.w);
    kps_suffix_.resize(C_.w);
  }

  void new_frame_info() override {
    set_device();

    C_ = Constants(frame_info_.width(), frame_info_.height(), 0);
    features_suffix_.clear();
    kps_suffix_.clear();
    features_suffix_.resize(C_.w);
    kps_suffix_.resize(C_.w);
  }

  void set_device() {
    CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    cvc::setDevice(device_.id);
  }

protected:
  void execute(const StenciledElements& input_columns,
               Elements& output_columns) override {
    set_device();

    auto& features_col = input_columns[0];
    auto& keypoints_col = input_columns[1];
    auto& frame_info_col = input_columns[2];
    check_frame_info(device_, frame_info_col[0]);

    i32 window_size = features_col.size();

    std::vector<cvc::GpuMat> features;
    std::vector<std::vector<proto::Keypoint>> kps;

    for (i32 i = 0; i < window_size; ++i) {
      size_t size = keypoints_col[i].size;
      u8* buf = new_buffer(CPU_DEVICE, size);
      memcpy_buffer(buf, CPU_DEVICE, keypoints_col[i].buffer, device_,
                    size);
      std::vector<proto::Keypoint> kp =
        deserialize_proto_vector<proto::Keypoint>(buf, size);
      kps.push_back(kp);

      size = features_col[i].size;
      if (kp.size() == 0) {
        features.push_back(cvc::GpuMat());
      } else {
        i32 step = size / kp.size();
        i32 cols;
        if (kp.size() == 1) {
          cols = step / sizeof(f32);
        } else {
          cols = step / (sizeof(f32) * 2);
        }
        LOG_IF(FATAL, cols != 64) << "Not 64 cols: " << cols;
        features.push_back(cvc::GpuMat(kp.size(), cols, CV_32F,
                                       features_col[i].buffer, step));
      }
    }

    size_t size = window_size * sizeof(f32);
    f32* cost_buf = (f32*)new_buffer(CPU_DEVICE, size);

    std::vector<std::vector<cv::DMatch>> matches;
    matches.resize(window_size);
    for (i32 j = 1; j < window_size; j++) {
      if (kps[0].size() == 0 || kps[j].size() == 0) {
        continue;
      }
      matcher_->match(features[0], features[j], matches[j]);
    }

#pragma omp parallel for
    for (i32 j = 1; j < window_size; j++) {
      f32 cost = match_cost(kps[0], kps[j], matches[j]);
      cost_buf[j] = cost;
    }

    if (device_.type == DeviceType::GPU) {
      u8* gpu_buf = new_buffer(device_, size);
      memcpy_buffer(gpu_buf, device_, (u8*) cost_buf, CPU_DEVICE, size);
      delete_buffer(CPU_DEVICE, (u8*) cost_buf);
      cost_buf = (f32*) gpu_buf;
    }

    insert_element(output_columns[0], (u8*) cost_buf, size);
  }

 private:
  float reprojection_error(std::vector<cv::Point2f>& src,
                           std::vector<cv::Point2f>& dst, cv::Mat& H) {
    std::vector<cv::Point2f> dst_proj;
    perspectiveTransform(src, dst_proj, H);
    int N = src.size();
    cv::Mat dst_proj_m = cv::Mat::zeros(N, 2, CV_32F),
            dst_m = cv::Mat::zeros(N, 2, CV_32F);
    for (int i = 0; i < N; i++) {
      dst_proj_m.at<float>(i, 0) = dst_proj[i].x;
      dst_proj_m.at<float>(i, 1) = dst_proj[i].y;
      dst_m.at<float>(i, 0) = dst[i].x;
      dst_m.at<float>(i, 1) = dst[i].y;
    }
    cv::Mat diff = dst_m - dst_proj_m;
    cv::Mat summed, sq;
    reduce(diff.mul(diff), summed, 1, CV_REDUCE_SUM);
    sqrt(summed, sq);
    return mean(sq)[0];
  }

  float match_cost(std::vector<proto::Keypoint>& kp1,
                   std::vector<proto::Keypoint>& kp2,
                   std::vector<cv::DMatch>& matches) {
    if (matches.size() == 0) {
      return C_.gamma;
    }

    double min_dist = std::numeric_limits<double>::max();
    for (auto& match : matches) {
      double dist = match.distance;
      if (dist < min_dist) {
        min_dist = dist;
      }
    }

    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> fr1, fr2;
    for (auto& match : matches) {
      if (match.distance <= std::max(1.5 * min_dist, 0.01)) {
        good_matches.push_back(match);
        fr1.push_back(
            cv::Point2f(kp1[match.queryIdx].x(), kp1[match.queryIdx].y()));
        fr2.push_back(
            cv::Point2f(kp2[match.trainIdx].x(), kp2[match.trainIdx].y()));
      }
    }

    // Need at least 4 points to find a homography
    if (fr1.size() < 4) {
      return C_.gamma;
    }

    cv::Mat H = cv::findHomography(fr1, fr2, CV_RANSAC);
    // If H is empty, then homography could not be found
    if (H.rows == 0) {
      return C_.gamma;
    }

    float cr = reprojection_error(fr1, fr2, H);

    cv::Point2f x(C_.ih / 2), y(C_.iw / 2);
    std::vector<cv::Point2f> center = {x, y};
    float co = reprojection_error(center, center, H);

    // LOG(INFO) << "cr: " << cr << ", co: " << co << ", C_.tau_c: " <<
    // C_.tau_c;
    if (cr < C_.tau_c) {
      return co;
    } else {
      return C_.gamma;
    }
  }

  DeviceHandle device_;
  Constants C_;
  cv::Ptr<cvc::DescriptorMatcher> matcher_;
  std::vector<cvc::GpuMat> features_suffix_;
  std::vector<std::vector<proto::Keypoint>> kps_suffix_;
};

REGISTER_OP(FeatureMatcher)
  .input("features")
  .input("keypoints")
  .input("frame_info")
  .stencil()
  .output("cost_matrix");

REGISTER_KERNEL(FeatureMatcher, FeatureMatcherKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}

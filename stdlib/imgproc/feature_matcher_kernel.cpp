#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scanner/util/serialize.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class Constants {
public:
  int w = 24;
  int g = 4;
  int iw, ih;
  int T;
  int d;
  float tau_c, gamma;

  //int lam_s = 200;
  //int lam_a = 80;
  int lam_s = 100;
  int lam_a = 40;

  int tau_s = 200;
  int tau_a = 200;
  int v = 6;

  Constants(int iw, int ih, int T) {
    this->iw = iw;
    this->ih = ih;
    this->T = T;
    d = (int) sqrt((float)(ih * ih + iw * iw));
    tau_c = 0.1 * (float) d;
    gamma = 0.5 * (float) d;
  }
};


class FeatureMatcherKernel : public VideoKernel {
public:
  FeatureMatcherKernel(const Kernel::Config& config)
    : VideoKernel(config),
      device_(config.devices[0]), C_(0,0,0) {
    set_device();
    matcher_ = cvc::DescriptorMatcher::createBFMatcher();
    features_suffix_.resize(C_.w);
    kps_suffix_.resize(C_.w);
  }

  void set_frame_info() {
    C_ = Constants(frame_info_.width(), frame_info_.height(), 0);
  }

  void reset() {
    features_suffix_.clear();
    kps_suffix_.clear();
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    set_device();

    auto& features_col = input_columns[0];
    auto& keypoints_col = input_columns[1];
    auto& frame_info_col = input_columns[2];
    check_frame_info(device_, frame_info_col);

    i32 input_count = input_columns[0].rows.size();
    C_.T = input_count;

    LOG(INFO) << "Count: " << C_.T;

    cv::Mat Cm = cv::Mat::zeros(C_.T+1, C_.T+1, CV_32F);

    std::vector<cvc::GpuMat> features;
    std::vector<std::vector<proto::Keypoint>> kps;

    for (i32 i = 0; i < C_.T; ++i) {
      size_t size = keypoints_col.rows[i].size;
      u8* buf = new_buffer(CPU_DEVICE, size);
      memcpy_buffer(buf, CPU_DEVICE,
                    keypoints_col.rows[i].buffer, device_,
                    size);
      std::vector<proto::Keypoint> kp =
        deserialize_proto_vector<proto::Keypoint>(buf, size);
      kps.push_back(kp);

      size = features_col.rows[i].size;
      if (kp.size() == 0) {
        features.push_back(cvc::GpuMat());
      } else {
        features.push_back(
          cvc::GpuMat(kp.size(), size / (kp.size() * sizeof(float)), CV_32F,
                      features_col.rows[i].buffer));
      }
    }

    size_t size = C_.w * sizeof(f32);
    for (i32 i = 0; i < C_.T; ++i) {
      if (i % 5 == 0) {
        LOG(INFO) << i;
      }
      f32* cost_buf = (f32*) new_buffer(CPU_DEVICE, size);
      for (i32 j = i+1; j <= i + C_.w; j++) {
        f32 cost;
        if (j >= C_.T) {
          i32 offset = j - C_.T;
          cost = match_cost(kps[i], features[i],
                            kps_suffix_[offset], features_suffix_[offset]);
        } else {
          cost = match_cost(kps[i], features[i], kps[j], features[j]);
        }
        cost_buf[j-i-1] = cost;
      }
      u8* output_buf = new_buffer(device_, size);
      memcpy_buffer(output_buf, device_, (u8*) cost_buf, CPU_DEVICE, size);
      INSERT_ROW(output_columns[0], output_buf, size);
    }

    features_suffix_.clear();
    kps_suffix_.clear();
    for (i32 i = 0; i < C_.w; ++i) {
      i32 offset = C_.T - C_.w + i;
      cvc::GpuMat feat;
      features[offset].copyTo(feat);
      features_suffix_.push_back(feat);
      kps_suffix_.push_back(kps[offset]);
    }
  }

  void set_device() {
    CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    cvc::setDevice(device_.id);
  }

private:

  float reprojection_error(std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, cv::Mat& H) {
    std::vector<cv::Point2f> dst_proj;
    perspectiveTransform(src, dst_proj, H);
    int N = src.size();
    cv::Mat dst_proj_m = cv::Mat::zeros(N, 2, CV_32F), dst_m = cv::Mat::zeros(N, 2, CV_32F);
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

  float match_cost(std::vector<proto::Keypoint>& kp1, cvc::GpuMat& desc1_gpu,
                   std::vector<proto::Keypoint>& kp2, cvc::GpuMat& desc2_gpu) {
    if (kp1.size() == 0 || kp2.size() == 0) {
      return C_.gamma;
    }

    std::vector<cv::DMatch> matches;
    matcher_->match(desc1_gpu, desc2_gpu, matches);

    double min_dist = std::numeric_limits<double>::max();
    for (auto& match : matches) {
      double dist = match.distance;
      if (dist < min_dist) { min_dist = dist; }
    }

    std::vector<cv::Point2f> fr1, fr2;
    for (auto& match : matches) {
      if (match.distance <= std::max(2 * min_dist, 0.02)) {
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

    cv::Mat mask;
    cv::Mat H = cv::findHomography(fr1, fr2, CV_RANSAC, 3, mask);

    // If H is empty, then homography could not be found
    if (H.rows == 0) {
      return C_.gamma;
    }

    float cr = reprojection_error(fr1, fr2, H);

    cv::Point2f x(C_.ih/2), y(C_.iw/2);
    std::vector<cv::Point2f> center = {x, y};
    float co = reprojection_error(center, center, H);

    // LOG(INFO) << "cr: " << cr << ", co: " << co << ", C_.tau_c: " << C_.tau_c;
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

REGISTER_OP(FeatureMatcher).inputs({"features", "keypoints", "frame_info"}).outputs({"cost_matrix"});

REGISTER_KERNEL(FeatureMatcher, FeatureMatcherKernel)
.device(DeviceType::GPU)
.num_devices(1);
}

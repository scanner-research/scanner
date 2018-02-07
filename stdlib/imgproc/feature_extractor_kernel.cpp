#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scanner/util/serialize.h"
#include "stdlib/stdlib.pb.h"

#include <opencv2/xfeatures2d.hpp>

namespace scanner {

class FeatureExtractorKernel : public Kernel, public VideoKernel {
 public:
  FeatureExtractorKernel(const KernelConfig& config)
    : Kernel(config), device_(config.devices[0]) {
    set_device();

    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      LOG(FATAL) << "Failed to parse args";
    }

    if (args_.feature_type() == proto::ExtractorType::SIFT) {
      if (device_.type == DeviceType::GPU) {
        LOG(FATAL) << "GPU SIFT not supported yet";
      } else {
        cpu_extractor_ = cv::xfeatures2d::SIFT::create();
      }
    } else if (args_.feature_type() == proto::ExtractorType::SURF) {
      if (device_.type == DeviceType::GPU) {
        gpu_extractor_ = new cvc::SURF_CUDA(100);
      } else {
        cpu_extractor_ = cv::xfeatures2d::SURF::create();
      }
    } else {
      LOG(FATAL) << "Invalid feature type";
    }
  }

  void execute(const Elements& input_columns, Elements& output_columns) override {
    set_device();

    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col);

    std::vector<cv::KeyPoint> keypoints;
    std::tuple<u8*, size_t> features;

    cvc::GpuMat feat_gpus;
    cv::Mat cv_features;
    cvc::GpuMat kp_gpu;

    if (device_.type == DeviceType::GPU) {
      if (args_.feature_type() == proto::ExtractorType::SURF) {
        cvc::SURF_CUDA* surf = (cvc::SURF_CUDA*)gpu_extractor_;
        cvc::GpuMat img = frame_to_gpu_mat(frame_col.as_const_frame());
        cvc::cvtColor(img, img, CV_RGB2GRAY);
        (*surf)(img, cvc::GpuMat(), kp_gpu, feat_gpus);
        surf->downloadKeypoints(kp_gpu, keypoints);

        LOG_IF(FATAL, !feat_gpus.empty() && feat_gpus.cols != 64)
          << "Not 64 SURF columns?";

        features = std::make_tuple(
          feat_gpus.data, feat_gpus.step * feat_gpus.rows);
      } else {
        LOG(FATAL) << "SIFT GPU not supported";
      }
    } else {
      cv::Mat img = frame_to_mat(frame_col.as_const_frame());
      cv::cvtColor(img, img, CV_RGB2GRAY);
      cpu_extractor_->detectAndCompute(img, cv::Mat(), keypoints,
                                       cv_features);

      features = std::make_tuple(
        cv_features.data,
        cv_features.total() * cv_features.elemSize());
    }

#define OR_4(N) std::max((N), (size_t)4)

    u8* cv_buf = std::get<0>(features);
    size_t size = std::get<1>(features);
    u8* output_buf = new_buffer(device_, OR_4(size));
    memcpy_buffer(output_buf, device_, cv_buf, CPU_DEVICE, size);
    insert_element(output_columns[0], output_buf, OR_4(size));

    std::vector<proto::Keypoint> kps_proto;
    for (auto& kp : keypoints) {
      proto::Keypoint kp_proto;
      kp_proto.set_x(kp.pt.x);
      kp_proto.set_y(kp.pt.y);
      kps_proto.push_back(kp_proto);
    }


    output_buf = new_buffer(CPU_DEVICE, OR_4(size));
    serialize_proto_vector(kps_proto, output_buf, size);
    if (device_.type == DeviceType::GPU) {
      u8* gpu_buf = new_buffer(device_, OR_4(size));
      memcpy_buffer(gpu_buf, device_, output_buf, CPU_DEVICE, size);
      delete_buffer(CPU_DEVICE, output_buf);
      output_buf = gpu_buf;
    }
    insert_element(output_columns[1], output_buf, OR_4(size));
  }

  void set_device() {
    CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    cvc::setDevice(device_.id);
  }

 private:
  DeviceHandle device_;
  proto::FeatureExtractorArgs args_;
  void* gpu_extractor_;
  cv::Ptr<cv::Feature2D> cpu_extractor_;
};

REGISTER_OP(FeatureExtractor)
    .frame_input("frame")
    .output("features")
    .output("keypoints");

REGISTER_KERNEL(FeatureExtractor, FeatureExtractorKernel)
    .device(DeviceType::GPU)
    .num_devices(1);

REGISTER_KERNEL(FeatureExtractor, FeatureExtractorKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

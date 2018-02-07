#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class FacenetInputKernelCPU : public BatchedKernel, public VideoKernel {
 public:
  FacenetInputKernelCPU(const KernelConfig& config)
    : BatchedKernel(config),
      device_(config.devices[0])
  {
    proto::FacenetArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    args_.CopyFrom(args.caffe_args());
    scale_ = args.scale();
  }

  void new_frame_info() override {
    net_input_width_ = std::floor(frame_info_.width() * scale_);
    net_input_height_ = std::floor(frame_info_.height() * scale_);
    if (net_input_width_ % 8 != 0) {
      net_input_width_ += 8 - (net_input_width_ % 8);
    };
    if (net_input_height_ % 8 != 0) {
      net_input_height_ += 8 - (net_input_height_ % 8);
    }

    mean_mat_ =
        cv::Mat(net_input_height_, net_input_width_, CV_32FC3,
                cv::Scalar(args_.net_descriptor().mean_colors(0),
                           args_.net_descriptor().mean_colors(1),
                           args_.net_descriptor().mean_colors(2)));

    frame_input_.clear();
    resized_input_.clear();
    float_input_.clear();
    flipped_planes_.clear();
    normalized_input_.clear();
    input_planes_.clear();
    planar_input_.clear();
    flipped_planes_.clear();
    for (size_t i = 0; i < 1; ++i) {
      frame_input_.push_back(
          cv::Mat(frame_info_.height(), frame_info_.width(), CV_8UC3));
      resized_input_.push_back(
        cv::Mat(net_input_height_, net_input_width_, CV_8UC3));
      float_input_.push_back(
          cv::Mat(net_input_height_, net_input_width_, CV_32FC3));
      normalized_input_.push_back(
          cv::Mat(net_input_height_, net_input_width_, CV_32FC3));
      std::vector<cv::Mat> planes;
      std::vector<cv::Mat> flipped_planes;
      for (i32 i = 0; i < 3; ++i) {
        planes.push_back(
            cv::Mat(net_input_height_, net_input_width_, CV_32FC1));
        flipped_planes.push_back(
            cv::Mat(net_input_width_, net_input_height_, CV_32FC1));
      }
      input_planes_.push_back(planes);
      flipped_planes_.push_back(flipped_planes);
      planar_input_.push_back(
          cv::Mat(net_input_width_ * 3, net_input_height_, CV_32FC1));
    }
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    i32 input_count = (i32)frame_col.size();
    FrameInfo net_input_info(3, net_input_width_, net_input_height_,
                             FrameType::F32);
    i32 net_input_size = net_input_info.size();
    std::vector<Frame*> output_frames =
        new_frames(device_, net_input_info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      Frame* output_frame = output_frames[i];
      f32* net_input = (f32*)output_frame->data;

      i32 sid = 0;

      // Convert input frame to gpu mat
      frame_input_[sid] = frame_to_mat(frame_col[i].as_const_frame());

      cv::resize(frame_input_[sid], resized_input_[sid],
                 cv::Size(net_input_width_, net_input_height_), 0, 0,
                 cv::INTER_LINEAR);
      resized_input_[sid].convertTo(float_input_[sid], CV_32FC3);
      cv::subtract(float_input_[sid], mean_mat_, normalized_input_[sid],
                   cv::noArray(), -1);
      // Changed from interleaved RGB to planar RGB
      cv::split(normalized_input_[sid], input_planes_[sid]);
      cv::transpose(input_planes_[sid][0], flipped_planes_[sid][0]);
      cv::transpose(input_planes_[sid][1], flipped_planes_[sid][1]);
      cv::transpose(input_planes_[sid][2], flipped_planes_[sid][2]);
      auto& plane1 = flipped_planes_[sid][0];
      auto& plane2 = flipped_planes_[sid][1];
      auto& plane3 = flipped_planes_[sid][2];
      auto& planar_input = planar_input_[sid];
      plane1.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 0, net_input_height_, net_input_width_)));
      plane2.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 1, net_input_height_, net_input_width_)));
      plane3.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 2, net_input_height_, net_input_width_)));
      assert(planar_input.cols == net_input_height_);
      for (int j = 0; j < net_input_width_ * 3; ++j) {
        memcpy(net_input + j * net_input_height_,
               planar_input.data + j * planar_input.step,
               net_input_height_ * sizeof(float));
      }
      insert_frame(output_columns[0], output_frame);
    }
  }

 private:
  DeviceHandle device_;
  proto::CaffeArgs args_;
  f32 scale_;
  i32 net_input_width_;
  i32 net_input_height_;

  cv::Mat mean_mat_;
  std::vector<cv::Mat> frame_input_;
  std::vector<cv::Mat> resized_input_;
  std::vector<cv::Mat> float_input_;
  std::vector<cv::Mat> normalized_input_;
  std::vector<std::vector<cv::Mat>> input_planes_;
  std::vector<std::vector<cv::Mat>> flipped_planes_;
  std::vector<cv::Mat> planar_input_;
};

REGISTER_OP(FacenetInput).frame_input("frame").frame_output("facenet_input");

REGISTER_KERNEL(FacenetInput, FacenetInputKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);
}

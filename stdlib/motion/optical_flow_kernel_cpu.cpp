#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class OpticalFlowKernelCPU : public StenciledKernel, public VideoKernel {
 public:
  OpticalFlowKernelCPU(const KernelConfig& config)
    : StenciledKernel(config),
      device_(config.devices[0]) {
    flow_finder_ =
        cv::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0);
  }

  void new_frame_info() override {
    grayscale_.resize(0);
    for (i32 i = 0; i < 2; ++i) {
      grayscale_.emplace_back(frame_info_.height(), frame_info_.width(),
                              CV_8UC1);
    }
  }

  void execute(const StenciledElements& input_columns,
               Elements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 2,
                             FrameType::F32);
    Frame* output_frame = new_frame(device_, out_frame_info);

    cv::Mat input0 = frame_to_mat(frame_col[0].as_const_frame());
    cv::Mat input1 = frame_to_mat(frame_col[1].as_const_frame());
    cv::cvtColor(input0, grayscale_[0], CV_BGR2GRAY);
    cv::cvtColor(input1, grayscale_[1], CV_BGR2GRAY);
    cv::Mat flow = frame_to_mat(output_frame);
    flow_finder_->calc(grayscale_[0], grayscale_[1], flow);
    insert_frame(output_columns[0], output_frame);
  }

 private:
  DeviceHandle device_;
  cv::Ptr<cv::DenseOpticalFlow> flow_finder_;
  std::vector<cv::Mat> grayscale_;
};

REGISTER_OP(OpticalFlow)
    .frame_input("frame")
    .frame_output("flow")
    .stencil({0, 1});

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);
}

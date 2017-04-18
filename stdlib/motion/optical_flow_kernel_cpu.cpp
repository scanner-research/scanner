#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cycle_timer.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class OpticalFlowKernelCPU : public VideoKernel {
 public:
  OpticalFlowKernelCPU(const Kernel::Config& config)
      : VideoKernel(config),
        device_(config.devices[0]),
        work_item_size_(config.work_item_size) {
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

  void reset() override { initial_frame_ = cv::Mat(); }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    i32 input_count = (i32)num_rows(frame_col);

    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 2,
                             FrameType::F32);
    std::vector<Frame*> frames =
      new_frames(device_, out_frame_info, input_count);

    double start = CycleTimer::currentSeconds();

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat input = frame_to_mat(input_columns[0][i].as_const_frame());
      cv::cvtColor(input, grayscale_[i % 2], CV_BGR2GRAY);
      cv::Mat flow = frame_to_mat(frames[i]);
      if (i == 0) {
        if (initial_frame_.empty()) {
          insert_frame(output_columns[0], frames[i]);
          continue;
        } else {
          flow_finder_->calc(initial_frame_, grayscale_[0], flow);
        }
      } else {
        flow_finder_->calc(grayscale_[(i - 1) % 2], grayscale_[i % 2], flow);
      }
      insert_frame(output_columns[0], frames[i]);
    }

    grayscale_[(input_count - 1) % 2].copyTo(initial_frame_);
  }

 private:
  DeviceHandle device_;
  cv::Ptr<cv::DenseOpticalFlow> flow_finder_;
  cv::Mat initial_frame_;
  std::vector<cv::Mat> grayscale_;
  i32 work_item_size_;
};

REGISTER_OP(OpticalFlow).frame_input("frame").frame_output("flow");

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelCPU)
  .device(DeviceType::CPU)
  .num_devices(1);
}

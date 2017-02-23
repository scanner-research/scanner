#include "scanner/api/op.h"
#include "scanner/api/kernel.h"
#include "scanner/util/cycle_timer.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class OpticalFlowKernelCPU : public VideoKernel {
public:
  OpticalFlowKernelCPU(const Kernel::Config &config)
      : VideoKernel(config), device_(config.devices[0]),
        work_item_size_(config.work_item_size) {
    flow_finder_ = cv::FarnebackOpticalFlow::create();
  }

  void new_frame_info() override {
    grayscale_.resize(0);
    for (i32 i = 0; i < work_item_size_; ++i) {
      grayscale_.emplace_back(frame_info_.height(), frame_info_.width(),
                              CV_8UC1);
    }
  }

  void reset() override { initial_frame_ = cv::Mat(); }

  void execute(const BatchedColumns &input_columns,
               BatchedColumns &output_columns) override {
    check_frame_info(device_, input_columns[1]);

    i32 input_count = (i32)input_columns[0].rows.size();
    size_t out_buf_size =
        frame_info_.width() * frame_info_.height() * 2 * sizeof(float);

    u8 *output_block =
        new_block_buffer(device_, out_buf_size * input_count, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat input(frame_info_.height(), frame_info_.width(), CV_8UC3,
                    input_columns[0].rows[i].buffer);
      cv::cvtColor(input, grayscale_[i], CV_BGR2GRAY);
    }

    double start = CycleTimer::currentSeconds();

    cv::Ptr<cv::DenseOpticalFlow> flow_finder =
        cv::FarnebackOpticalFlow::create();

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat flow(frame_info_.height(), frame_info_.width(), CV_32FC2,
                   output_block + i * out_buf_size);

      if (i == 0) {
        if (initial_frame_.empty()) {
          output_columns[0].rows.push_back(Row{flow.data, out_buf_size});
          continue;
        } else {
          flow_finder_->calc(initial_frame_, grayscale_[0], flow);
        }
      } else {
        flow_finder_->calc(grayscale_[i - 1], grayscale_[i], flow);
      }

      output_columns[0].rows.push_back(Row{flow.data, out_buf_size});
    }
  }

private:
  DeviceHandle device_;
  cv::Ptr<cv::DenseOpticalFlow> flow_finder_;
  cv::Mat initial_frame_;
  std::vector<cv::Mat> grayscale_;
  i32 work_item_size_;
};

REGISTER_OP(OpticalFlow).inputs({"frame", "frame_info"}).outputs({"flow"});

REGISTER_KERNEL(OpticalFlow, OpticalFlowKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);
}

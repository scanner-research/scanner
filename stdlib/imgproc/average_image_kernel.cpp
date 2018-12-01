#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

#include <opencv2/video.hpp>

namespace scanner {

class AverageImageKernel : public StenciledKernel, VideoKernel {
 public:
  AverageImageKernel(const KernelConfig& config)
    : StenciledKernel(config),
      device_(config.devices[0]) {}

  void new_frame_info() override {
    avg_.resize(0);
    for (i32 i = 0; i < 5; ++i) {
      avg_.emplace_back(frame_info_.height(), frame_info_.width(),
                              CV_8UC3);
    }
  }

  void execute(const StenciledElements& input_columns,
               Elements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    FrameInfo out_frame_info(frame_info_.height(), frame_info_.width(), 3,
                             FrameType::U8);

    cv::Mat input0 = frame_to_mat(frame_col[0].as_const_frame());
    cv::Mat input1 = frame_to_mat(frame_col[1].as_const_frame());
    cv::Mat input2 = frame_to_mat(frame_col[2].as_const_frame());
    cv::Mat input3 = frame_to_mat(frame_col[3].as_const_frame());
    cv::Mat input4 = frame_to_mat(frame_col[4].as_const_frame());
    cv::Mat avg = (input0 + input1 + input2 + input3 + input4) / 5;

    Frame* output_frame = mat_to_frame(avg);
    insert_frame(output_columns[0], output_frame);
  }

 private:
  DeviceHandle device_;
  std::vector<cv::Mat> avg_;
};

REGISTER_OP(AverageImage)
    .frame_input("frame")
    .frame_output("avg")
    .stencil({-2, -1, 0, 1, 2});

REGISTER_KERNEL(AverageImage, AverageImageKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

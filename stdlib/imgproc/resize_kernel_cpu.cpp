#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class ResizeKernelCPU : public VideoKernel {
 public:
  ResizeKernelCPU(const Kernel::Config& config) : VideoKernel(config) {
    proto::ResizeArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    width_ = args.width();
    height_ = args.height();
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(CPU_DEVICE, frame_col[0]);

    i32 input_count = num_rows(frame_col);
    FrameInfo info(height_, width_, 3, FrameType::U8);
    std::vector<Frame*> output_frames =
        new_frames(CPU_DEVICE, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
      cv::Mat out_mat = frame_to_mat(output_frames[i]);
      cv::resize(img, out_mat, cv::Size(width_, height_));
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

 private:
  int width_;
  int height_;
};

REGISTER_OP(Resize).frame_input("frame").frame_output("frame");

REGISTER_KERNEL(Resize, ResizeKernelCPU).device(DeviceType::CPU).num_devices(1);
}

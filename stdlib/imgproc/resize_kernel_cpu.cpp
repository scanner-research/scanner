#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class ResizeKernelCPU : public VideoKernel {
 public:
  ResizeKernelCPU(const Kernel::Config& config) : VideoKernel(config) {
    args_.ParseFromArray(config.args.data(), config.args.size());
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(CPU_DEVICE, frame_col[0]);

    i32 target_width = args_.width();
    i32 target_height = args_.height();
    if (args_.preserve_aspect()) {
      if (target_width == 0) {
        target_width =
            frame_info_.width() * target_height / frame_info_.height();
      } else {
        target_height =
            frame_info_.height() * target_width / frame_info_.width();
      }
    }
    if (args_.min()) {
      if (frame_info_.width() <= target_width &&
          frame_info_.height() <= target_height) {
        target_width = frame_info_.width();
        target_height = frame_info_.height();
      }
    }

    i32 input_count = num_rows(frame_col);
    FrameInfo info(target_height, target_width, 3, FrameType::U8);
    std::vector<Frame*> output_frames =
        new_frames(CPU_DEVICE, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
      cv::Mat out_mat = frame_to_mat(output_frames[i]);

      cv::resize(img, out_mat, cv::Size(target_width, target_height));
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

 private:
  proto::ResizeArgs args_;
};

REGISTER_OP(Resize).frame_input("frame").frame_output("frame");

REGISTER_KERNEL(Resize, ResizeKernelCPU).device(DeviceType::CPU).num_devices(1);
}

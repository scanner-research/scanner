#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class ResizeKernel : public BatchedKernel {
 public:
  ResizeKernel(const KernelConfig& config)
    : BatchedKernel(config), device_(config.devices[0]) {
    args_.ParseFromArray(config.args.data(), config.args.size());
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];
    set_device();

    const Frame* frame = frame_col[0].as_const_frame();

    i32 target_width = args_.width();
    i32 target_height = args_.height();
    if (args_.preserve_aspect()) {
      if (target_width == 0) {
        target_width =
            frame->width() * target_height / frame->height();
      } else {
        target_height =
            frame->height() * target_width / frame->width();
      }
    }
    if (args_.min()) {
      if (frame->width() <= target_width &&
          frame->height() <= target_height) {
        target_width = frame->width();
        target_height = frame->height();
      }
    }

    i32 input_count = num_rows(frame_col);
    FrameInfo info(target_height, target_width, 3, FrameType::U8);
    std::vector<Frame*> output_frames = new_frames(device_, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      if (device_.type == DeviceType::CPU) {
        cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
        cv::Mat out_mat = frame_to_mat(output_frames[i]);
        cv::resize(img, out_mat, cv::Size(target_width, target_height));
      } else {
        CUDA_PROTECT({
          cvc::GpuMat img = frame_to_gpu_mat(frame_col[i].as_const_frame());
          cvc::GpuMat out_mat = frame_to_gpu_mat(output_frames[i]);
          cvc::resize(img, out_mat, cv::Size(target_width, target_height));
        });
      }
      insert_frame(output_columns[0], output_frames[i]);
    }
  }

  void set_device() {
    if (device_.type == DeviceType::GPU) {
      CUDA_PROTECT({
        CU_CHECK(cudaSetDevice(device_.id));
        cvc::setDevice(device_.id);
      });
    }
  }

 private:
  DeviceHandle device_;
  proto::ResizeArgs args_;
};

REGISTER_OP(Resize).frame_input("frame").frame_output("frame");

REGISTER_KERNEL(Resize, ResizeKernel).device(DeviceType::CPU).num_devices(1);

#ifdef HAVE_CUDA
REGISTER_KERNEL(Resize, ResizeKernel).device(DeviceType::GPU).num_devices(1);
#endif
}

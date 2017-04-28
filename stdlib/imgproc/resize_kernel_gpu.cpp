#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class ResizeKernelGPU : public VideoKernel {
 public:
  ResizeKernelGPU(const Kernel::Config& config)
    : VideoKernel(config), device_(config.devices[0]) {
    proto::ResizeArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    width_ = args.width();
    height_ = args.height();
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];

    set_device();
    check_frame(device_, frame_col[0]);

    i32 input_count = num_rows(frame_col);
    FrameInfo info(height_, width_, 3, FrameType::U8);
    std::vector<Frame*> output_frames = new_frames(device_, info, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      insert_frame(output_columns[0], output_frames[i]);
    }

    LOG(FATAL) << "Not yet implemented";

    // cvc::GpuMat img = frame_to_gpu_mat(frame_col[i].as_const_frame());
    // cvc::GpuMat out_mat = frame_to_gpu_mat(output_frames[i]);
    // cvc::resize(img, out_mat, cv::Size(width_, height_));
  }

  void set_device() {
    CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    cvc::setDevice(device_.id);
  }

 private:
  DeviceHandle device_;
  int width_;
  int height_;
};

REGISTER_KERNEL(Resize, ResizeKernelGPU).device(DeviceType::GPU).num_devices(1);
}

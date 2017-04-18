#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class MontageKernelGPU : public VideoKernel {
 public:
  MontageKernelGPU(const Kernel::Config& config)
      : VideoKernel(config),
        device_(config.devices[0]),
        frames_seen_(0),
        montage_width_(0),
        montage_buffer_(nullptr) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      RESULT_ERROR(&valid_, "MontageKernel could not parse protobuf args");
      return;
    }

    num_frames_ = args_.num_frames();
    target_width_ = args_.target_width();
    frames_per_row_ = args_.frames_per_row();
  }

  ~MontageKernelGPU() {
    if (montage_buffer_ != nullptr) {
      delete_buffer(device_, montage_buffer_);
    }
  }

  void reset() {
    if (montage_width_ != 0) {
      montage_buffer_ =
          new_buffer(device_, montage_width_ * montage_height_ * 3);
      montage_image_ = cvc::GpuMat(montage_height_, montage_width_, CV_8UC3,
                                   montage_buffer_);
      montage_image_.setTo(0);
      frames_seen_ = 0;
    }
  }

  void new_frame_info() override {
    set_device();
    frame_width_ = frame_info_.width();
    frame_height_ = frame_info_.height();

    target_height_ = (target_width_ / (1.0 * frame_width_) * frame_height_);

    montage_width_ = frames_per_row_ * target_width_;
    montage_height_ =
        std::ceil(num_frames_ / (1.0 * frames_per_row_)) * target_height_;
    reset();
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(device_, frame_col[0]);

    set_device();

    assert(montage_buffer_ != nullptr);
    i32 input_count = num_rows(frame_col);
    for (i32 i = 0; i < input_count; ++i) {
      cvc::GpuMat img = frame_to_gpu_mat(frame_col[i].as_const_frame());
      i64 x = frames_seen_ % frames_per_row_;
      i64 y = frames_seen_ / frames_per_row_;
      cvc::GpuMat montage_subimg =
          montage_image_(cv::Rect(target_width_ * x, target_height_ * y,
                                  target_width_, target_height_));
      cvc::resize(img, montage_subimg, cv::Size(target_width_, target_height_));

      frames_seen_++;
      if (frames_seen_ == num_frames_) {
        assert(montage_buffer_ != nullptr);
        FrameInfo info(montage_height_, montage_width_, 3, FrameType::U8);
        insert_frame(output_columns[0], new Frame(info, montage_buffer_));
        montage_image_ = cvc::GpuMat();
        montage_buffer_ = nullptr;
      } else {
        insert_element(output_columns[0], new_buffer(device_, 1), 1);
      }
    }
  }

  void set_device() {
    CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    cvc::setDevice(device_.id);
  }

 private:
  proto::Result valid_;
  DeviceHandle device_;
  proto::MontageArgs args_;
  i64 num_frames_;
  i32 frame_width_;
  i32 frame_height_;
  i32 target_width_;
  i32 target_height_;
  i32 frames_per_row_;

  i64 montage_width_;
  i64 montage_height_;

  u8* montage_buffer_;
  cvc::GpuMat montage_image_;
  i64 frames_seen_;
};

REGISTER_OP(Montage).frame_input("frame").frame_output("montage");

REGISTER_KERNEL(Montage, MontageKernelGPU)
  .device(DeviceType::GPU)
  .num_devices(1);
}

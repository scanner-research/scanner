#include "halide_resize/halide_resize.h"
#include "resize.pb.h"
#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/halide.h"
#include "scanner/util/memory.h"

class ResizeKernel : public scanner::VideoKernel {
 public:
  ResizeKernel(const scanner::Kernel::Config& config)
      : scanner::VideoKernel(config), device_(config.devices[0]) {
    ResizeArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    width_ = args.width();
    height_ = args.height();
  }

  void execute(const scanner::BatchedElements& input_columns,
               scanner::BatchedElements& output_columns) override {
    int input_count = input_columns[0].rows.size();

    // This must be called at the top of the execute method in any VideoKernel.
    // See the VideoKernel for the implementation check_frame_info.
    check_frame_info(device_, input_columns[1]);

    size_t output_size = width_ * height_ * 3;
    unsigned char* output_block = scanner::new_block_buffer(
        device_, output_size * input_count, input_count);

    for (int i = 0; i < input_count; ++i) {
      buffer_t input_halide_buf = {0};
      scanner::setup_halide_frame_buf(input_halide_buf, frame_info_);
      scanner::set_halide_buf_ptr(device_, input_halide_buf,
                                  input_columns[0].rows[i].buffer,
                                  input_columns[0].rows[i].size);

      buffer_t output_halide_buf = {0};
      scanner::setup_halide_frame_buf(output_halide_buf, frame_info_);
      scanner::set_halide_buf_ptr(device_, output_halide_buf,
                                  output_block + i * output_size, output_size);

      int error = halide_resize(&input_halide_buf, frame_info_.width(),
                                frame_info_.height(), width_, height_,
                                &output_halide_buf);
      LOG_IF(FATAL, error != 0) << "Halide error " << error;

      scanner::unset_halide_buf_ptr(device_, input_halide_buf);
      scanner::unset_halide_buf_ptr(device_, output_halide_buf);
    }
  }

 private:
  scanner::DeviceHandle device_;
  int width_;
  int height_;
};

REGISTER_OP(Resize).outputs({"frame"});

REGISTER_KERNEL(Resize, ResizeKernel)
    .device(scanner::DeviceType::GPU)
    .num_devices(1);

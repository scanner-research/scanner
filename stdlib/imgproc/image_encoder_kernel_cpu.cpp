#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

namespace scanner {

class ImageEncoderKernel : public VideoKernel {
 public:
  ImageEncoderKernel(const Kernel::Config& config) : VideoKernel(config) {}

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(CPU_DEVICE, frame_col[0]);

    i32 input_count = num_rows(frame_col);
    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img = frame_to_mat(frame_col[i].as_const_frame());
      std::vector<u8> buf;
      bool success = cv::imencode(".png", img, buf);
      LOG_IF(FATAL, !success) << "Failed to encode image";
      u8* output_buf = new_buffer(CPU_DEVICE, buf.size());
      std::memcpy(output_buf, buf.data(), buf.size());
      insert_element(output_columns[0], output_buf, buf.size());
    }
  }
};

REGISTER_OP(ImageEncoder).frame_input("frame").output("png");

REGISTER_KERNEL(ImageEncoder, ImageEncoderKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

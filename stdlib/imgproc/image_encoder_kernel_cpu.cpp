#include "scanner/api/op.h"
#include "scanner/api/kernel.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

namespace scanner {

class ImageEncoderKernel : public VideoKernel {
public:
  ImageEncoderKernel(const Kernel::Config &config) : VideoKernel(config) {}

  void execute(const BatchedColumns &input_columns,
               BatchedColumns &output_columns) override {
    auto& frame_col = input_columns[0];
    auto& frame_info_col = input_columns[1];
    check_frame_info(CPU_DEVICE, frame_info_col);

    i32 input_count = frame_col.rows.size();
    for (i32 i = 0; i < input_count; ++i) {
      cv::Mat img(frame_info_.height(), frame_info_.width(), CV_8UC3,
                  (u8 *)frame_col.rows[i].buffer);
      std::vector<u8> buf;
      bool success = cv::imencode(".png", img, buf);
      LOG_IF(FATAL, !success) << "Failed to encode image";
      u8 *output_buf = new_buffer(CPU_DEVICE, buf.size());
      std::memcpy(output_buf, buf.data(), buf.size());
      output_columns[0].rows.push_back(Row{output_buf, buf.size()});
    }
  }
};

REGISTER_OP(ImageEncoder).inputs({"frame", "frame_info"}).outputs({"png"});

REGISTER_KERNEL(ImageEncoder, ImageEncoderKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

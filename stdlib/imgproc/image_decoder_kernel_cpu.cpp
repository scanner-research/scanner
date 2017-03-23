#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"

namespace scanner {

class ImageDecoderKernel : public Kernel {
 public:
  ImageDecoderKernel(const Kernel::Config& config) : Kernel(config) {}

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = input_columns[0].rows.size();
    for (i32 i = 0; i < input_count; ++i) {
      std::vector<u8> input_buf(
          input_columns[0].rows[i].buffer,
          input_columns[0].rows[i].buffer + input_columns[0].rows[i].size);
      cv::Mat img = cv::imdecode(input_buf, CV_LOAD_IMAGE_COLOR);
      LOG_IF(FATAL, img.empty() || !img.data) << "Failed to decode image";
      size_t size = img.total() * img.elemSize();
      u8* output_buf = new_buffer(CPU_DEVICE, size);
      std::memcpy(output_buf, img.data, size);
      INSERT_ROW(output_columns[0], output_buf, size);

      FrameInfo frame_info;
      frame_info.set_width(img.cols);
      frame_info.set_height(img.rows);
      size = frame_info.ByteSize();
      output_buf = new_buffer(CPU_DEVICE, size);
      frame_info.SerializeToArray(output_buf, size);
      INSERT_ROW(output_columns[1], output_buf, size);
    }
  }
};

REGISTER_OP(ImageDecoder).inputs({"img"}).outputs({"frame", "frame_info"});

REGISTER_KERNEL(ImageDecoder, ImageDecoderKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

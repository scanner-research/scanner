#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

class ImageDecoderKernelCPU : public BatchedKernel {
 public:
  ImageDecoderKernelCPU(const KernelConfig& config) : BatchedKernel(config) {}

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    i32 input_count = num_rows(input_columns[0]);

    for (i32 i = 0; i < input_count; ++i) {
      std::vector<u8> input_buf(
        input_columns[0][i].buffer,
        input_columns[0][i].buffer + input_columns[0][i].size);
      cv::Mat img = cv::imdecode(input_buf, CV_LOAD_IMAGE_COLOR);
      LOG_IF(FATAL, img.empty() || !img.data) << "Failed to decode image";
      size_t size = img.total() * img.elemSize();
      Frame* frame = new_frame(CPU_DEVICE, mat_to_frame_info(img));
      std::memcpy(frame->data, img.data, size);
      insert_frame(output_columns[0], frame);
    }
  }
};

REGISTER_OP(ImageDecoder).input("img").frame_output("frame");

REGISTER_KERNEL(ImageDecoder, ImageDecoderKernelCPU)
    .device(DeviceType::CPU)
    .num_devices(1);
}

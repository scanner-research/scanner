#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

namespace codec = cv::cudacodec;

class ImageSource : public codec::RawVideoSource {
public:
  ImageSource(const BatchedColumns& input_columns) : input_columns(input_columns) {}

  bool getNextPacket(unsigned char** data, int* size, bool* endOfFile) override {
    const Element& element = input_columns[0][i];
    *data = element.buffer;
    *size = element.size;
    *endOfFile = false;
    return true;
  }

  codec::FormatInfo format() const override {
    codec::FormatInfo format_info;
    format_info.codec = codec::Codec::JPEG;
    format_info.chromaFormat = codec::ChromaFormat::YUV420;
    format_info.width = 640; // TODO
    format_info.height = 480;
    return format_info;
  }

private:
  int i = 0;
  const BatchedColumns& input_columns;
};

class ImageDecoderKernel : public Kernel {
 public:
  ImageDecoderKernel(const Kernel::Config& config) : Kernel(config), device_(config.devices[0]) {
    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      LOG(FATAL) << "Failed to parse args";
    }
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = num_rows(input_columns[0]);

    cv::Ptr<codec::RawVideoSource> src =
      cv::Ptr<codec::RawVideoSource>(new ImageSource(input_columns));
    cv::Ptr<codec::VideoReader> d_reader = codec::createVideoReader(src);
    cv::cuda::GpuMat gpu_frame;

    for (i32 i = 0; i < input_count; ++i) {
      proto::ImageDecoderArgs_ImageType image_type = args_.image_type();
      if (image_type == proto::ImageDecoderArgs_ImageType_JPEG) {
        if (!d_reader->nextFrame(gpu_frame)) {
          LOG(FATAL) << "Failed to decode image";
        }
        Frame* frame = new_frame(CPU_DEVICE, gpu_mat_to_frame_info(gpu_frame));
        cv::Mat mat = frame_to_mat(frame);
        gpu_frame.download(mat);
        insert_frame(output_columns[0], frame);
      } else if (image_type == proto::ImageDecoderArgs_ImageType_ANY) {
        std::vector<u8> input_buf(
          input_columns[0][i].buffer,
          input_columns[0][i].buffer + input_columns[0][i].size);
        cv::Mat img = cv::imdecode(input_buf, CV_LOAD_IMAGE_COLOR);
        LOG_IF(FATAL, img.empty() || !img.data) << "Failed to decode image";
        size_t size = img.total() * img.elemSize();
        Frame* frame = new_frame(CPU_DEVICE, mat_to_frame_info(img));
        std::memcpy(frame->data, img.data, size);
        insert_frame(output_columns[0], frame);
      } else {
        LOG(FATAL) << "Invalid image type";
      }
    }
  }

private:
  proto::ImageDecoderArgs args_;
  DeviceHandle device_;
};

REGISTER_OP(ImageDecoder).input("img").frame_output("frame");

REGISTER_KERNEL(ImageDecoder, ImageDecoderKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "scanner/util/cuda.h"
#include "stdlib/stdlib.pb.h"

namespace scanner {

namespace codec = cv::cudacodec;

class ImageSource : public codec::RawVideoSource {
public:
  ImageSource(const BatchedElements& input_columns, const cv::Mat& img)
    : input_columns_(input_columns), img_(img) {}

  bool getNextPacket(unsigned char** data, int* size, bool* endOfFile) override {
    const Element& element = input_columns_[0][i_];
    *data = element.buffer;
    *size = element.size;
    *endOfFile = false;
    // Theoretically we should be able to set endOfFile to true at the last
    // frame, but the OpenCV VideoReader appears to return false on a valid
    // nextFrame request if I do this, so instead I just keep feeding packets
    // until the loader thread dies.
    i_ = (i_ + 1) % input_columns_[0].size();
    return true;
  }

  codec::FormatInfo format() const override {
    codec::FormatInfo format_info;
    format_info.codec = codec::Codec::JPEG;
    format_info.chromaFormat = codec::ChromaFormat::YUV420;
    format_info.width = img_.cols;
    format_info.height = img_.rows;
    return format_info;
  }

private:
  int i_ = 0;
  const cv::Mat& img_;
  const BatchedElements& input_columns_;
};

class ImageDecoderKernelGPU : public Kernel {
 public:
  ImageDecoderKernelGPU(const Kernel::Config& config)
    : Kernel(config), device_(config.devices[0]) {
    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      LOG(FATAL) << "Failed to parse args";
    }
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    i32 input_count = num_rows(input_columns[0]);

    set_device();

    // Assumes all images are the same size
    size_t sz = input_columns[0][0].size;
    u8* cpu_buf = new_buffer(CPU_DEVICE, sz);
    memcpy_buffer(cpu_buf, CPU_DEVICE, input_columns[0][0].buffer,
                  device_, sz);
    std::vector<u8> input_buf(cpu_buf, cpu_buf + sz);
    cv::Mat img = cv::imdecode(input_buf, CV_LOAD_IMAGE_COLOR);
    FrameInfo frame_info(img.rows, img.cols, 3, FrameType::U8);
    delete_buffer(CPU_DEVICE, cpu_buf);

    proto::ImageDecoderArgs_ImageType image_type = args_.image_type();
    std::vector<Frame*> frames = new_frames(device_, frame_info, input_count);

    cv::Ptr<codec::RawVideoSource> src =
      cv::Ptr<codec::RawVideoSource>(new ImageSource(input_columns, img));
    cv::Ptr<codec::VideoReader> d_reader = codec::createVideoReader(src);

    for (i32 i = 0; i < input_count; ++i) {
      if (image_type == proto::ImageDecoderArgs_ImageType_JPEG) {
        cvc::GpuMat gpu_mat = frame_to_gpu_mat(frames[i]);
        if (!d_reader->nextFrame(gpu_mat)) {
          LOG(FATAL) << "Failed to decode image";
        }
        insert_frame(output_columns[0], frames[i]);
      } else if (image_type == proto::ImageDecoderArgs_ImageType_ANY) {
        LOG(FATAL) << "Not yet supported";
      } else {
        LOG(FATAL) << "Invalid image type";
      }
    }
  }

  void set_device() {
    CU_CHECK(cudaSetDevice(device_.id));
    cvc::setDevice(device_.id);
  }

private:
  proto::ImageDecoderArgs args_;
  DeviceHandle device_;
};

REGISTER_KERNEL(ImageDecoder, ImageDecoderKernelGPU)
    .device(DeviceType::GPU)
    .num_devices(1);
}

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
  ImageSource(const BatchedColumns& input_columns, const cv::Mat& img)
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
  const BatchedColumns& input_columns_;
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

    set_device();

    // Assumes all images are the same size
    std::vector<u8> input_buf(
      input_columns[0][0].buffer,
      input_columns[0][0].buffer + input_columns[0][0].size);
    cv::Mat img = cv::imdecode(input_buf, CV_LOAD_IMAGE_COLOR);

    proto::ImageDecoderArgs_ImageType image_type = args_.image_type();

    // TODO(wcrichto): GPU code shouldn't ideally be in CPU kernel
    cv::Ptr<codec::RawVideoSource> src =
      cv::Ptr<codec::RawVideoSource>(new ImageSource(input_columns, img));
    cv::Ptr<codec::VideoReader> d_reader = codec::createVideoReader(src);
    cv::cuda::GpuMat gpu_frame;

    for (i32 i = 0; i < input_count; ++i) {
      if (image_type == proto::ImageDecoderArgs_ImageType_JPEG) {
        if (!d_reader->nextFrame(gpu_frame)) {
          LOG(FATAL) << "Failed to decode image";
        }
        CUDA_PROTECT({
          Frame* frame = new_frame(CPU_DEVICE, gpu_mat_to_frame_info(gpu_frame));
          cv::Mat mat = frame_to_mat(frame);
          gpu_frame.download(mat);
          insert_frame(output_columns[0], frame);
        });
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

  void set_device() {
    // HACK(wcrichto): using CPU id as GPU id...
    proto::ImageDecoderArgs_ImageType image_type = args_.image_type();
    if (image_type == proto::ImageDecoderArgs_ImageType_JPEG) {
      CUDA_PROTECT({
        cvc::setDevice(device_.id);
        CU_CHECK(cudaSetDevice(device_.id % 4));
      });
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

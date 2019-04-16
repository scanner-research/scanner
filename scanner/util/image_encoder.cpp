#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/types.pb.h"
#include "scanner/util/lodepng.h"

namespace scanner {
namespace {
int frame_info_to_lct(const FrameInfo& info, LodePNGColorType& type) {
  if (info.channels() == 1) {
    type = LCT_GREY;
  } else if (info.channels() == 2) {
    type = LCT_GREY_ALPHA;
  } else if (info.channels() == 3) {
    type = LCT_RGB;
  } else if (info.channels() == 4) {
    type = LCT_RGBA;
  } else {
    return 1;
  }
  return 0;
}
int frame_info_to_bitdepth(const FrameInfo& info, unsigned& bitdepth) {
  switch (info.type) {
    case FrameType::U8: {
      bitdepth = 8;
      break;
    }
    case FrameType::U16: {
      bitdepth = 16;
      break;
    }
    default: { return 1; }
  }
  return 0;
}

int encode_png(const Frame* frame, std::vector<unsigned char>& encoded_image) {
  unsigned char* png;
  LodePNGColorType type;
  FrameInfo info = frame->as_frame_info();
  if (frame_info_to_lct(info, type) != 0) {
    return 1;
  }
  unsigned bitdepth;
  if (frame_info_to_bitdepth(info, bitdepth) != 0) {
    return 2;
  }
  {
    u32 error = lodepng::encode(encoded_image, frame->data, frame->width(),
                                frame->height(), type, bitdepth);
    if (error != 0) {
      return 3;
    }
  }
  return 0;
}
}

class ImageEncoderKernel : public BatchedKernel, public VideoKernel {
 public:
  ImageEncoderKernel(const KernelConfig& config) : BatchedKernel(config) {
    proto::ImageEncoderArgs args;
    bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse ImageEncoderArgs");
      return;
    }
    image_type_ = args.format() == "" ? "png" : args.format();
    if (!(image_type_ == "png" || image_type_ == "PNG")) {
      RESULT_ERROR(&valid_,
                   "Invalid format type specified to "
                   "ImageEncoder: %s. Valid types are: png.",
                   image_type_.c_str());
    }
    valid_.set_success(true);
  }

  void validate(Result* result) {
    result->set_msg(valid_.msg());
    result->set_success(valid_.success());
  }

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override {
    auto& frame_col = input_columns[0];
    check_frame(CPU_DEVICE, frame_col[0]);

    i32 input_count = num_rows(frame_col);
    for (i32 i = 0; i < input_count; ++i) {
      const Frame* frame = frame_col[i].as_const_frame();
      std::vector<unsigned char> encoded_image;
      int error = encode_png(frame, encoded_image);
      if (error == 1) {
        RESULT_ERROR(&valid_, "Invalid frame type for ImageEncoder.");
      } else if (error == 2) {
        RESULT_ERROR(&valid_, "Invalid frame bitdepth for ImageEncoder.");
      } else if (error != 0) {
        RESULT_ERROR(&valid_, "Failed to encode image to PNG in ImageEncoder.");
      }
      u8* output_buf = new_buffer(CPU_DEVICE, encoded_image.size());
      std::memcpy(output_buf, encoded_image.data(), encoded_image.size());
      insert_element(output_columns[0], output_buf, encoded_image.size());
    }
  }

 private:
  Result valid_;
  std::string image_type_;
};

REGISTER_OP(ImageEncoder)
    .frame_input("frame")
    .output("img", ColumnType::Bytes, "Image")
    .protobuf_name("ImageEncoderArgs");

REGISTER_KERNEL(ImageEncoder, ImageEncoderKernel)
    .device(DeviceType::CPU)
    .num_devices(1);
}

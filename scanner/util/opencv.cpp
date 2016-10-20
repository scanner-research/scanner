#include "scanner/util/opencv.h"

namespace scanner {

cv::Mat bytesToImage(u8* buf, const VideoMetadata& metadata) {
  i64 frame_size = metadata.width() * metadata.height() * 3 * sizeof(u8);
  return cv::Mat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

cvc::GpuMat bytesToImage_gpu(u8* buf, const VideoMetadata& metadata) {
  i64 frame_size = metadata.width() * metadata.height() * 3 * sizeof(u8);
  return cvc::GpuMat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

}

#include "scanner/util/opencv.h"

namespace scanner {

cv::Mat bytesToImage(u8* buf, const VideoMetadata& metadata) {
  return cv::Mat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

#ifdef HAVE_CUDA

cvc::GpuMat bytesToImage_gpu(u8* buf, const VideoMetadata& metadata) {
  return cvc::GpuMat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

#endif
}

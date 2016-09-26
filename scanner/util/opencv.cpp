#include "scanner/util/opencv.h"

namespace scanner {

cv::Mat bytesToImage(u8* buf, i32 i, const DatasetItemMetadata& metadata) {
  i64 frame_size = metadata.width * metadata.height * 3 * sizeof(u8);
  u8* frame_buffer = buf + frame_size * i;
  return cv::Mat(metadata.height, metadata.width, CV_8UC3, frame_buffer);
}
}

#include "scanner/util/opencv.h"

#include "scanner/api/kernel.h"
#include "scanner/engine/db.h"
#include "scanner/util/image.h"

#ifdef HAVE_CUDA
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace scanner {

cv::Mat bytesToImage(u8 *buf, const FrameInfo &metadata) {
  return cv::Mat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

#ifdef HAVE_CUDA

cvc::GpuMat bytesToImage_gpu(u8 *buf, const FrameInfo &metadata) {
  return cvc::GpuMat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

cudaError_t convertNV12toRGBA(const cv::cuda::GpuMat &in,
                              cv::cuda::GpuMat &outFrame, int width, int height,
                              cv::cuda::Stream &stream) {
  cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
  return convertNV12toRGBA(in.ptr<uchar>(), in.step, outFrame.ptr<uchar>(),
                           outFrame.step, width, height, s);
}

cudaError_t convertRGBInterleavedToPlanar(const cv::cuda::GpuMat &in,
                                          cv::cuda::GpuMat &outFrame, int width,
                                          int height,
                                          cv::cuda::Stream &stream) {
  cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
  return convertRGBInterleavedToPlanar(in.ptr<uchar>(), in.step,
                                       outFrame.ptr<uchar>(), outFrame.step,
                                       width, height, s);
}

#endif
}

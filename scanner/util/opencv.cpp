#include "scanner/util/opencv.h"

#include "scanner/api/kernel.h"
#include "scanner/engine/metadata.h"
#include "scanner/util/image.h"

#ifdef HAVE_CUDA
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace scanner {

cv::DataType frame_to_cv_type(FrameType type, int channels) {
  cv::DataType cv_type;
  switch (type) {
    case U8: {
      cv_type = CV_8U;
      break;
    }
    case F32: {
      cv_type = CV_32F;
      break;
    }
    case F64: {
      cv_type = CV_64F;
      break
    }
  }
  return CV_MAKETYPE(cv_type, channels);
}

FrameType cv_to_frame_type(int type) {
  FrameType type;
  switch (type) {
    case CV_8U: {
      type = FrameType::U8;
      break;
    }
    case CV_32F: {
      type = FrameType::F32;
      break;
    }
    case CV_64F: {
      type = FrameType::F64;
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported OpenCV type: " << type;
    }
  }
  return type;
}

FrameInfo mat_to_frame_info(const cv::Mat& mat) {
  return FrameInfo(mat.channels(), mat.cols, mat.rows,
                   cv_to_frame_type(mat.depth()));
}


cv::Mat frame_to_mat(const Frame* frame) {
  return frame_to_mat((Frame*)frame);
}

cv::Mat frame_to_mat(Frame* frame) {
  return cv::Mat(frame->shape[1], frame->shape[2],
                 frame_to_cv_type(frame->type, frame->shape[0]), frame->data);
}

cv::Mat bytesToImage(u8* buf, const FrameInfo& metadata) {
  return cv::Mat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

#ifdef HAVE_CUDA

cv::GpuMat frame_to_gpu_mat(const Frame* frame) {
  return frame_to_gpu_mat((Frame*)frame);
}

cv::GpuMat frame_to_gpu_mat(Frame* frame) {
  return cv::GpuMat(frame->shape[1], frame->shape[2],
                    frame_to_cv_type(frame->type, frame->shape[0]),
                    frame->data);
}

cvc::GpuMat bytesToImage_gpu(u8* buf, const FrameInfo& metadata) {
  return cvc::GpuMat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

cudaError_t convertNV12toRGBA(const cv::cuda::GpuMat& in,
                              cv::cuda::GpuMat& outFrame, int width, int height,
                              cv::cuda::Stream& stream) {
  cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
  return convertNV12toRGBA(in.ptr<uchar>(), in.step, outFrame.ptr<uchar>(),
                           outFrame.step, width, height, s);
}

cudaError_t convertRGBInterleavedToPlanar(const cv::cuda::GpuMat& in,
                                          cv::cuda::GpuMat& outFrame, int width,
                                          int height,
                                          cv::cuda::Stream& stream) {
  cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
  return convertRGBInterleavedToPlanar(in.ptr<uchar>(), in.step,
                                       outFrame.ptr<uchar>(), outFrame.step,
                                       width, height, s);
}

#endif
}

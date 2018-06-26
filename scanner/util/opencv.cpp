#include "scanner/util/opencv.h"

#include "scanner/api/kernel.h"
#include "scanner/engine/metadata.h"
#include "scanner/util/image.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace scanner {

int frame_to_cv_type(FrameType type, int channels) {
  int cv_type;
  switch (type) {
    case FrameType::U8: {
      cv_type = CV_8U;
      break;
    }
    case FrameType::U16: {
      cv_type = CV_16U;
      break;
    }
    case FrameType::F32: {
      cv_type = CV_32F;
      break;
    }
    case FrameType::F64: {
      cv_type = CV_64F;
      break;
    }
  }
  return CV_MAKETYPE(cv_type, channels);
}

FrameType cv_to_frame_type(int t) {
  FrameType type;
  switch (t) {
    case CV_8U: {
      type = FrameType::U8;
      break;
    }
    case CV_16U: {
      type = FrameType::U16;
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
    default: { LOG(FATAL) << "Unsupported OpenCV type: " << t; }
  }
  return type;
}

FrameInfo mat_to_frame_info(const cv::Mat& mat) {
  return FrameInfo(mat.rows, mat.cols, mat.channels(),
                   cv_to_frame_type(mat.depth()));
}

Frame* mat_to_frame(const cv::Mat& mat) {
  Frame* frame = new_frame(CPU_DEVICE, mat_to_frame_info(mat));
  if (mat.isContinuous()) {
    memcpy(frame->data, mat.data, frame->size());
  } else {
    u64 offset = 0;
    u64 row_size = mat.cols * mat.elemSize();
    for (int i = 0; i < mat.rows; ++i) {
      memcpy(frame->data + offset, mat.data + i * mat.step, row_size);
      offset += row_size;
    }
  }
  return frame;
}

cv::Mat frame_to_mat(const Frame* frame) { return frame_to_mat((Frame*)frame); }

cv::Mat frame_to_mat(Frame* frame) {
  return cv::Mat(frame->height(), frame->width(),
                 frame_to_cv_type(frame->type, frame->channels()), frame->data);
}

cv::Mat bytesToImage(u8* buf, const FrameInfo& metadata) {
  return cv::Mat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

#ifdef HAVE_CUDA

cvc::GpuMat frame_to_gpu_mat(const Frame* frame) {
  return frame_to_gpu_mat((Frame*)frame);
}

cvc::GpuMat frame_to_gpu_mat(Frame* frame) {
  return cvc::GpuMat(frame->height(), frame->width(),
                     frame_to_cv_type(frame->type, frame->channels()),
                     frame->data);
}

cvc::GpuMat bytesToImage_gpu(u8* buf, const FrameInfo& metadata) {
  return cvc::GpuMat(metadata.height(), metadata.width(), CV_8UC3, buf);
}


FrameInfo gpu_mat_to_frame_info(const cv::cuda::GpuMat& mat) {
  return FrameInfo(mat.channels(), mat.cols, mat.rows,
                   cv_to_frame_type(mat.depth()));
}

Frame* gpu_mat_to_frame(const cv::cuda::GpuMat& mat) {
  int device;
  CU_CHECK(cudaGetDevice(&device));
  Frame* frame = new_frame(DeviceHandle(DeviceType::GPU, device),
                           gpu_mat_to_frame_info(mat));
  if (mat.isContinuous()) {
    cudaMemcpy(frame->data, mat.data, frame->size(), cudaMemcpyDefault);
  } else {
    size_t frame_pitch =
        frame->width() * frame->channels() * size_of_frame_type(frame->type);
    cudaMemcpy2D(frame->data, frame_pitch, mat.data, mat.step, mat.cols,
                 mat.rows, cudaMemcpyDefault);
  }
  return frame;
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

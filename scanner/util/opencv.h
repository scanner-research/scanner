/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "scanner/api/frame.h"
#include "scanner/util/common.h"
#include "scanner/api/kernel.h"
#include "scanner/engine/metadata.h"
#include "scanner/util/image.h"

#include <opencv2/opencv.hpp>

namespace scanner {

inline int frame_to_cv_type(FrameType type, int channels = 1) {
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

inline FrameType cv_to_frame_type(int t) {
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

inline FrameInfo mat_to_frame_info(const cv::Mat& mat) {
  return FrameInfo(mat.rows, mat.cols, mat.channels(),
                   cv_to_frame_type(mat.depth()));
}

inline Frame* mat_to_frame(const cv::Mat& mat) {
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

inline cv::Mat frame_to_mat(const Frame* frame) {
  return cv::Mat(frame->height(), frame->width(),
                 frame_to_cv_type(frame->type, frame->channels()), frame->data);
}

inline cv::Mat frame_to_mat(Frame* frame) {
  return cv::Mat(frame->height(), frame->width(),
                 frame_to_cv_type(frame->type, frame->channels()), frame->data);
}

inline cv::Mat bytesToImage(u8* buf, const FrameInfo& metadata) {
  return cv::Mat(metadata.height(), metadata.width(), CV_8UC3, buf);
}

}

#ifdef HAVE_CUDA

#include "scanner/util/cuda.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cvc = cv::cuda;

namespace scanner {

class InputFormat;

inline cvc::GpuMat frame_to_gpu_mat(const Frame* frame) {
  return cvc::GpuMat(frame->height(), frame->width(),
                     frame_to_cv_type(frame->type, frame->channels()),
                     frame->data);
}

inline cvc::GpuMat frame_to_gpu_mat(Frame* frame) {
  return cvc::GpuMat(frame->height(), frame->width(),
                     frame_to_cv_type(frame->type, frame->channels()),
                     frame->data);
}

inline cvc::GpuMat bytesToImage_gpu(u8* buf, const FrameInfo& metadata) {
  return cvc::GpuMat(metadata.height(), metadata.width(), CV_8UC3, buf);
}


inline FrameInfo gpu_mat_to_frame_info(const cv::cuda::GpuMat& mat) {
  return FrameInfo(mat.channels(), mat.cols, mat.rows,
                   cv_to_frame_type(mat.depth()));
}

inline Frame* gpu_mat_to_frame(const cv::cuda::GpuMat& mat) {
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

}
#endif

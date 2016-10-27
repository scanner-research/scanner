#pragma once

#include "scanner/util/common.h"
#include "scanner/util/cycle_timer.h"
#include "scanner/util/opencv.h"
#include "scanner/util/profiler.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

// #define DEBUG_FACE_DETECTOR
#define DEBUG_OPTICAL_FLOW

namespace scanner {

#ifdef HAVE_CUDA
typedef cvc::GpuMat Mat;
#else
typedef cv::Mat Mat;
#endif

class MovieFeatureEvaluator {
 public:
  MovieFeatureEvaluator(DeviceType device_type) : device_type_(device_type) {}

  virtual ~MovieFeatureEvaluator() {}

  void configure(const VideoMetadata& metadata) { this->metadata_ = metadata; }

  void set_profiler(Profiler* profiler) { this->profiler_ = profiler; }

  void reset_wrapper() {
#ifdef HAVE_CUDA
    initial_frame_ = cvc::GpuMat();
#else
    initial_frame_ = cv::Mat();
#endif
    reset();
  }

  virtual void reset() {}

  void evaluate_wrapper(std::vector<Mat>& inputs,
                        std::vector<u8*>& output_buffers,
                        std::vector<size_t>& output_sizes) {
    evaluate(inputs, output_buffers, output_sizes);
    inputs[inputs.size() - 1].copyTo(initial_frame_);
  }

  virtual void evaluate(std::vector<Mat>& inputs,
                        std::vector<u8*>& output_buffers,
                        std::vector<size_t>& output_sizes) = 0;

 protected:
  VideoMetadata metadata_;
  Mat initial_frame_;
  Profiler* profiler_;
  DeviceType device_type_;
};
}

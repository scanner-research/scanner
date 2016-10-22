#pragma once

#include "scanner/util/common.h"
#include "scanner/util/opencv.h"
#include "scanner/util/cycle_timer.h"
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

  virtual ~MovieFeatureEvaluator(){}

  void configure(const VideoMetadata& metadata) {
    this->metadata = metadata;
  }

  void set_profiler(Profiler* profiler) {
    this->profiler = profiler;
  }

  void reset_wrapper() {
#ifdef HAVE_CUDA
    initial_frame = cvc::GpuMat();
#else
    initial_frame = cv::Mat();
#endif
    reset();
  }

  virtual void reset() {}

  void evaluate_wrapper(
    std::vector<Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) {
    evaluate(inputs, output_buffers, output_sizes);
    inputs[inputs.size()-1].copyTo(initial_frame);
  }

  virtual void evaluate(
    std::vector<Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) = 0;

protected:
  VideoMetadata metadata;
  Mat initial_frame;
  Profiler* profiler;
};

}

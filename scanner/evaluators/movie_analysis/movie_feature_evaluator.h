#pragma once

#include "scanner/util/common.h"
#include "scanner/util/opencv.h"
#include "scanner/util/cycle_timer.h"

// #define DEBUG_FACE_DETECTOR
#define DEBUG_OPTICAL_FLOW

namespace scanner {

class MovieFeatureEvaluator {
public:
  virtual ~MovieFeatureEvaluator(){};

  void configure(const VideoMetadata& metadata) {
    this->metadata = metadata;
  }

  void reset_wrapper() {
    initial_frame = cv::Mat();
    reset();
  }

  virtual void reset() {}

  void evaluate_wrapper(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) {
    evaluate(inputs, output_buffers, output_sizes);
    inputs[inputs.size()-1].copyTo(initial_frame);
  }

  virtual void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) = 0;

protected:
  VideoMetadata metadata;
  cv::Mat initial_frame;
};

}

#pragma once

#include "scanner/util/common.h"
#include "scanner/util/opencv.h"
#include "scanner/util/cycle_timer.h"

// #define DEBUG_FACE_DETECTOR
// #define DEBUG_OPTICAL_FLOW

namespace scanner {

class MovieFeatureEvaluator {
public:
  virtual ~MovieFeatureEvaluator(){};

  void configure(const VideoMetadata& metadata) {
    this->metadata = metadata;
  }

  virtual void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) = 0;

protected:
  VideoMetadata metadata;
};

}

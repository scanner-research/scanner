#pragma once

#include "movie_feature_evaluator.h"

namespace scanner {

class HistogramEvaluator : public MovieFeatureEvaluator {
public:
  void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) override;
};

}

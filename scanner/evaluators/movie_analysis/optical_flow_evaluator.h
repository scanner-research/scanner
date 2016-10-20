#pragma once

#include "movie_feature_evaluator.h"

namespace scanner {

class OpticalFlowEvaluator : public MovieFeatureEvaluator {
public:
  void evaluate(
    std::vector<Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) override;
};

}

#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/opencv.h"

#include "movie_feature_evaluator.h"

namespace scanner {

class MovieEvaluator : public Evaluator {
 public:
  MovieEvaluator(EvaluatorConfig config);

  void configure(const VideoMetadata& metadata) override;

  void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
                const std::vector<std::vector<size_t>>& input_sizes,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 private:
  VideoMetadata metadata;
  std::map<std::string, std::unique_ptr<MovieFeatureEvaluator>> evaluators;
};

class MovieEvaluatorFactory : public EvaluatorFactory {
public:
  MovieEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};

}

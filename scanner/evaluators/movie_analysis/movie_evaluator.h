#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/opencv.h"

#include "movie_feature_evaluator.h"

namespace scanner {

class MovieEvaluator : public Evaluator {
 public:
  MovieEvaluator(EvaluatorConfig config, DeviceType device_type,
                 std::vector<std::string> outputs);

  void configure(const InputFormat& metadata) override;

  void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
                const std::vector<std::vector<size_t>>& input_sizes,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

  void reset() override;

 private:
  InputFormat metadata;
  std::map<std::string, std::unique_ptr<MovieFeatureEvaluator>> evaluators;
  DeviceType device_type_;
  std::vector<std::string> outputs_;
};

class MovieEvaluatorFactory : public EvaluatorFactory {
 public:
  MovieEvaluatorFactory(DeviceType device_type,
                        std::vector<std::string> outputs);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  std::vector<std::string> outputs_;
};
}

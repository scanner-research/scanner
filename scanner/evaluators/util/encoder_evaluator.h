#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"

namespace scanner {

class EncoderEvaluator : public Evaluator {
 public:
  EncoderEvaluator(EvaluatorConfig config);

  void configure(const VideoMetadata& metadata) override;

  void evaluate(const std::vector<std::vector<u8*>>& input_buffers,
                const std::vector<std::vector<size_t>>& input_sizes,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 private:
  VideoMetadata metadata;
};

class EncoderEvaluatorFactory : public EvaluatorFactory {
public:
  EncoderEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};

}

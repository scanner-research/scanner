#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"

namespace scanner {

class EncoderEvaluator : public Evaluator {
 public:
  EncoderEvaluator(EvaluatorConfig config);

  void configure(const BatchConfig& config) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 private:
  i32 frame_width_;
  i32 frame_height_;
};

class EncoderEvaluatorFactory : public EvaluatorFactory {
 public:
  EncoderEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};
}

#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"

namespace scanner {

class EncoderEvaluator : public Evaluator {
 public:
  EncoderEvaluator(EvaluatorConfig config);

  void configure(const InputFormat& metadata) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 private:
  InputFormat metadata;
};

class EncoderEvaluatorFactory : public EvaluatorFactory {
 public:
  EncoderEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};
}

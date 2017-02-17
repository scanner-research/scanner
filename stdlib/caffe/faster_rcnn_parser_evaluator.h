#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"

namespace scanner {

class FasterRCNNParserEvaluator : public Evaluator {
 public:
  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;
};

class FasterRCNNParserEvaluatorFactory : public EvaluatorFactory {
 public:
  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};
}

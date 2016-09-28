#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"

namespace scanner {

class HistogramEvaluator : public Evaluator {
 public:
  HistogramEvaluator(EvaluatorConfig config);

  void configure(const DatasetItemMetadata& metadata) override;

  void evaluate(i32 input_count,
                u8* input_buffer,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 private:
  DatasetItemMetadata metadata;
};

class HistogramEvaluatorFactory : public EvaluatorFactory {
 public:
  HistogramEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  i32 get_number_of_outputs() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};
}

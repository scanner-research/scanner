#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/opencv.h"

namespace scanner {

class MovieItemEvaluator {
public:
  virtual ~MovieItemEvaluator(){};

  void configure(const DatasetItemMetadata& metadata) {
    this->metadata = metadata;
  }

  virtual void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) = 0;

protected:
  DatasetItemMetadata metadata;
};

class MovieEvaluator : public Evaluator {
 public:
  MovieEvaluator(EvaluatorConfig config);

  void configure(const DatasetItemMetadata& metadata) override;

  void evaluate(i32 input_count,
                u8* input_buffer,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 private:
  DatasetItemMetadata metadata;
  std::map<std::string, std::unique_ptr<MovieItemEvaluator>> evaluators;
};

class MovieEvaluatorFactory : public EvaluatorFactory {
public:
  MovieEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  i32 get_number_of_outputs() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};

}

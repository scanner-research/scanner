#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/opencv.h"

namespace scanner {

class FaceEvaluator : public Evaluator {
 public:
  FaceEvaluator(EvaluatorConfig config);

  void configure(const DatasetItemMetadata& metadata) override;

  void evaluate(i32 input_count,
                u8* input_buffer,
                std::vector<std::vector<u8*>>& output_buffers,
                std::vector<std::vector<size_t>>& output_sizes) override;

 private:
  DatasetItemMetadata metadata;
  cv::CascadeClassifier face_detector;
};


class FaceEvaluatorFactory : public EvaluatorFactory {
 public:
  FaceEvaluatorFactory();

  EvaluatorCapabilities get_capabilities() override;

  i32 get_number_of_outputs() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;
};

}

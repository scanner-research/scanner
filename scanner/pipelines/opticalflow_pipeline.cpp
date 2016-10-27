#include "scanner/evaluators/movie_analysis/optical_flow_evaluator.h"

using namespace scanner;

std::vector<std::unique_ptr<EvaluatorFactory>> setup_evaluator_pipeline() {
  std::vector<std::unique_ptr<EvaluatorFactory>> factories;

  factories.emplace_back(new OpticalFlowEvaluatorFactory(DeviceType::GPU));

  return factories;
}

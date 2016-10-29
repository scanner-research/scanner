#include "scanner/engine.h"
#include "scanner/evaluators/movie_analysis/optical_flow_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

using namespace scanner;

PipelineDescription get_pipeline_description() {
  PipelineDescription desc;
  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new OpticalFlowEvaluatorFactory(DeviceType::GPU));

  return desc;
}

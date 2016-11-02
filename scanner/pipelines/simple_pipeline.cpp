#include "scanner/engine.h"
#include "scanner/evaluators/image_processing/blur_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

using namespace scanner;

PipelineDescription get_pipeline_description() {
  PipelineDescription desc;

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new BlurEvaluatorFactory(3, 0.3));

  return desc;
}

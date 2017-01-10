#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include <cstdlib>

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  Sampler::all_frames(info, desc);

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::GPU, VideoDecoderType::NVIDIA));
  // factories.emplace_back(
  //     new DecoderEvaluatorFactory(DeviceType::GPU, VideoDecoderType::NVIDIA));
  // factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::GPU));
  // factories.emplace_back(
  //     new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::INTEL));
  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::GPU));

  return desc;
}
}

REGISTER_PIPELINE(effective_decode_rate, get_pipeline_description);
}

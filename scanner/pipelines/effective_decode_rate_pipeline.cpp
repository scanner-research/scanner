#include "scanner/engine.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description() {
  PipelineDescription desc;
  desc.sampling = Sampling::SequenceGather;
  for (i32 i = 0; i < 5; ++i) {
    desc.gather_sequences.push_back({0, {Interval{0, 1000}}});
  }

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::CPU));

  return desc;
}
}

REGISTER_PIPELINE(effective_decode_rate, get_pipeline_description);
}

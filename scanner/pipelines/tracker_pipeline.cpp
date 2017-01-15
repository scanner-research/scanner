#include "scanner/engine/runtime.h"
#include "scanner/evaluators/tracker/tracker_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"

namespace scanner {
namespace {

PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  Sampler::range(info, desc, "composite_job", 0, 100);
  Sampler::join_prepend(info, desc, "composite_bboxes", "frame");

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
    new DecoderEvaluatorFactory(DeviceType::GPU, VideoDecoderType::NVIDIA));
  factories.emplace_back(new TrackerEvaluatorFactory(DeviceType::CPU, 32, 20));
  factories.emplace_back(
    new SwizzleEvaluatorFactory(
      DeviceType::CPU, {2}, {"bboxes"}));

  return desc;
}
}

REGISTER_PIPELINE(tracker, get_pipeline_description);
}

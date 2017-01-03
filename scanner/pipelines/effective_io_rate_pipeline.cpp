#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  Sampler::all_frames(info, desc);

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::CPU));

  return desc;
}
}

REGISTER_PIPELINE(effective_io_rate, get_pipeline_description);
}

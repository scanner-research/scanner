#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include <cstdlib>

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  const char *NUM_VIDEOS = std::getenv("SC_NUM_VIDEOS");
  i32 num_videos = std::atoi(NUM_VIDEOS);

  PipelineDescription desc;
  Sampler::all_frames(info, desc);
  for (i32 i = 1; i < num_videos; ++i) {
    Sampler::join_prepend(info, desc, "frame", "frame");
  }

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::CPU));

  return desc;
}
}

REGISTER_PIPELINE(multi_decode, get_pipeline_description);
}

#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/image_processing/blur_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_meta,
    const std::vector<DatasetItemMetadata>& item_metas) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  factories.emplace_back(new BlurEvaluatorFactory(3, 0.3));

  return desc;
}
}

REGISTER_PIPELINE(simple, get_pipeline_description);
}

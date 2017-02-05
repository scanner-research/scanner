#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/image/image_decoder_evaluator.h"
#include "scanner/evaluators/util/discard_evaluator.h"

#include <cstdlib>

namespace scanner {
namespace {
PipelineDescription
get_pipeline_description(const DatasetMetadata &dataset_meta,
                         const std::vector<DatasetItemMetadata> &item_metas) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  factories.emplace_back(new ImageDecoderEvaluatorFactory(DeviceType::CPU));
  factories.emplace_back(new DiscardEvaluatorFactory(DeviceType::CPU));

  return desc;
}
}

REGISTER_PIPELINE(image_decode_rate, get_pipeline_description);
}

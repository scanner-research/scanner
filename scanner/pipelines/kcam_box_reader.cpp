#include "scanner/engine/runtime.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_desc,
    const std::vector<DatasetItemMetadata>& item_descriptors) {
  PipelineDescription desc;
  desc.input_columns = {"base_bboxes", "tracked_bboxes"};

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new SwizzleEvaluatorFactory(DeviceType::CPU, {0}, {"base_bboxes"}));

  return desc;
}
}

REGISTER_PIPELINE(kcam_box_reader, get_pipeline_description);
}

#include "scanner/engine.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_parser_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_person_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm/cpm_person_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_desc,
    const std::vector<DatasetItemMetadata>& item_descriptors) {
  PipelineDescription desc;
  desc.input_columns = {"frame", "centers"};

  NetDescriptor cpm_descriptor;
  {
    std::string net_descriptor_file = "features/cpm.toml";
    std::ifstream net_file{net_descriptor_file};
    cpm_descriptor = descriptor_from_net_file(net_file);
  }

  i32 batch_size = 4;

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  // factories.emplace_back(
  //     new DecoderEvaluatorFactory(DeviceType::GPU,
  //     VideoDecoderType::NVIDIA));
  factories.emplace_back(new DecoderEvaluatorFactory(
      DeviceType::CPU, VideoDecoderType::SOFTWARE, 1));
  factories.emplace_back(new CPMInputEvaluatorFactory(
      DeviceType::GPU, cpm_descriptor, batch_size));
  factories.emplace_back(new CaffeEvaluatorFactory(
      DeviceType::GPU, cpm_descriptor, batch_size, true));
  // factories.emplace_back(new CPMParserEvaluatorFactory(DeviceType::CPU,
  // true));
  factories.emplace_back(
      new SwizzleEvaluatorFactory(DeviceType::CPU, {1}, {"joint_maps"}));

  return desc;
}
}

REGISTER_PIPELINE(find_pose, get_pipeline_description);
}

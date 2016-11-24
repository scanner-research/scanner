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
  const char* START_FRAME = std::getenv("SC_START_FRAME");
  const char* END_FRAME = std::getenv("SC_END_FRAME");

  i32 start_frame = std::atoi(START_FRAME);
  i32 end_frame = std::atoi(END_FRAME);

  PipelineDescription desc;
  desc.input_columns = {"frame"};
  desc.sampling = Sampling::SequenceGather;
  for (size_t i = 0; i < item_descriptors.size(); ++i) {
    const DatasetItemMetadata& meta = item_descriptors[i];
    desc.gather_sequences.push_back({i, {Interval{start_frame, end_frame}}});
  }

  NetDescriptor cpm_person_descriptor;
  {
    std::string net_descriptor_file = "features/cpm_person.toml";
    std::ifstream net_file{net_descriptor_file};
    cpm_person_descriptor = descriptor_from_net_file(net_file);
  }

  i32 batch_size = 8;
  DeviceType device_type;
  VideoDecoderType decoder_type;
#ifdef HAVE_CUDA
  device_type = DeviceType::GPU;
  decoder_type = VideoDecoderType::NVIDIA;
#else
  device_type = DeviceType::CPU;
  decoder_type = VideoDecoderType::SOFTWARE;
#endif

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(new CPMPersonInputEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size));
  factories.emplace_back(new CaffeEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size, true));
  factories.emplace_back(
      new CPMPersonParserEvaluatorFactory(DeviceType::CPU, true));
  factories.emplace_back(
      new SwizzleEvaluatorFactory(DeviceType::CPU, {1}, {"centers"}));

  return desc;
}
}

REGISTER_PIPELINE(find_person, get_pipeline_description);
}

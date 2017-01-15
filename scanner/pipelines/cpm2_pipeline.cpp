#include "scanner/engine/runtime.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/cpm2/cpm2_input_evaluator.h"
#include "scanner/evaluators/caffe/cpm2/cpm2_parser_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include <map>

namespace scanner {
namespace {

PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  Sampler::all_frames(info, desc);
  // Sampler::strided_frames(info, desc, 24);

  NetDescriptor cpm_person_descriptor;
  {
    std::string net_descriptor_file = "features/cpm2.toml";
    std::ifstream net_file{net_descriptor_file};
    cpm_person_descriptor = descriptor_from_net_file(net_file);
  }

  // CPM2 uses batch size for multiple scales
  i32 batch_size = 1;
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
  factories.emplace_back(new CPM2InputEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size));
  factories.emplace_back(new CaffeEvaluatorFactory(
      device_type, cpm_person_descriptor, batch_size, cpm2_net_config));
  factories.emplace_back(
      new CPM2ParserEvaluatorFactory(DeviceType::CPU));
  factories.emplace_back(
      new SwizzleEvaluatorFactory(DeviceType::CPU, {1}, {"joint_centers"}));

  return desc;
}
}

REGISTER_PIPELINE(cpm2, get_pipeline_description);
}

#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/caffe/caffe_evaluator.h"
#include "scanner/evaluators/caffe/default/default_input_evaluator.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"

#include "scanner/pipelines/benchmarks/sampling.h"

namespace scanner {
namespace {

const i32 BATCH_SIZE = 96;
PipelineDescription get_pipeline_description(const DatasetInformation& info) {
  PipelineDescription desc;
  benchmark_sampling(info, desc, false);

  NetDescriptor net_descriptor;
  {
    std::string net_descriptor_file = "features/googlenet.toml";
    std::ifstream net_file{net_descriptor_file};
    net_descriptor = descriptor_from_net_file(net_file);
  }

  DeviceType device_type;
  VideoDecoderType decoder_type;

  const char *DEVICE = std::getenv("SC_DEVICE");
  std::string device{DEVICE};
  if (device == "CPU") {
    device_type = DeviceType::CPU;
    decoder_type = VideoDecoderType::SOFTWARE;
  } else if (device == "GPU") {
    device_type = DeviceType::GPU;
    decoder_type = VideoDecoderType::NVIDIA;
  } else {
    LOG(FATAL) << "Invalid SC_DEVICE type `" << device << "`";
  }

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
    new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(
    new DefaultInputEvaluatorFactory(
      device_type, net_descriptor, BATCH_SIZE));
  factories.emplace_back(
    new CaffeEvaluatorFactory(DeviceType::GPU, net_descriptor, BATCH_SIZE));
  factories.emplace_back(new SwizzleEvaluatorFactory(DeviceType::GPU, {1}, {"feature"}));

  return desc;
}

REGISTER_PIPELINE(caffe_benchmark, get_pipeline_description);
}
}

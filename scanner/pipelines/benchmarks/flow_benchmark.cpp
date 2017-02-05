#include "scanner/eval/pipeline_description.h"
#include "scanner/evaluators/movie_analysis/optical_flow_evaluator.h"
#include "scanner/evaluators/util/discard_evaluator.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

#include "scanner/pipelines/benchmarks/sampling.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  PipelineDescription desc;
  benchmark_sampling(info, desc, true);

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

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));
  factories.emplace_back(new OpticalFlowEvaluatorFactory(device_type));
  factories.emplace_back(new DiscardEvaluatorFactory(device_type));

  return desc;
}

REGISTER_PIPELINE(flow_benchmark, get_pipeline_description);
}
}

#include "scanner/engine.h"
#include "scanner/evaluators/movie_analysis/optical_flow_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {
namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_meta,
    const std::vector<DatasetItemMetadata>& item_metas) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};

  DeviceType device_type;
  VideoDecoderType decoder_type;

//#ifdef HAVE_CUDA
#if 0
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
  factories.emplace_back(new OpticalFlowEvaluatorFactory(DeviceType::GPU));

  return desc;
}

REGISTER_PIPELINE(opticalflow, get_pipeline_description);
}
}

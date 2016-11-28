#include "scanner/engine.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {

namespace {
PipelineDescription get_pipeline_description(
    const DatasetMetadata& dataset_meta,
    const std::vector<DatasetItemMetadata>& item_metas) {
  PipelineDescription desc;
  desc.input_columns = {"frame"};
  desc.sampling = Sampling::Gather;

  std::ifstream infile("indices.txt");
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    PointSamples samples;
    iss >> samples.video_index;
    int frame;
    while (iss >> frame) {
      samples.frames.push_back(frame);
    }
    desc.gather_points.push_back(samples);
  }

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

  return desc;
}
}

REGISTER_PIPELINE(frame_extract, get_pipeline_description);
}

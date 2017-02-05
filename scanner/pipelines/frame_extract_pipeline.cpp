#include "scanner/engine/runtime.h"
#include "scanner/evaluators/video/decoder_evaluator.h"

namespace scanner {

namespace {
PipelineDescription get_pipeline_description(const DatasetInformation &info) {
  PipelineDescription desc;

  std::ifstream infile("indices.txt");
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    Task task;
    i32 video_index;
    iss >> video_index;
    task.table_name = std::to_string(video_index);
    TableSample sample;
    sample.job_name = "base";
    sample.table_name = task.table_name;
    sample.columns = {"frame"};
    int frame;
    while (iss >> frame) {
      sample.rows.push_back(frame);
    }
    task.samples.push_back(sample);
    desc.tasks.push_back(task);
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

  std::vector<std::unique_ptr<EvaluatorFactory>> &factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(device_type, decoder_type));

  return desc;
}
}

REGISTER_PIPELINE(frame_extract, get_pipeline_description);
}

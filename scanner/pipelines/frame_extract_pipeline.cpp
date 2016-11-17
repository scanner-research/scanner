#include "scanner/engine.h"
#include "scanner/evaluators/util/swizzle_evaluator.h"
#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/util/opencv.h"

namespace scanner {

class JPEGEvaluator : public Evaluator {
  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) {
    for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
      cv::Mat img = bytesToImage(input_columns[0].rows[i].buffer, metadata_);
      cv::imwrite("test.jpg", img);
      output_columns[0].rows.push_back(Row{new u8[1], 1});
    }
  }
};

class JPEGEvaluatorFactory : public EvaluatorFactory {
  EvaluatorCapabilities get_capabilities() {
    EvaluatorCapabilities caps;
    caps.device_type = DeviceType::CPU;
    caps.max_devices = 1;
    caps.warmup_size = 0;
    return caps;
  }

  std::vector<std::string> get_output_names() { return {"frame"}; }

  Evaluator* new_evaluator(const EvaluatorConfig& config) {
    return new JPEGEvaluator();
  }
};

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

  std::vector<std::unique_ptr<EvaluatorFactory>>& factories =
      desc.evaluator_factories;

  factories.emplace_back(
      new DecoderEvaluatorFactory(DeviceType::CPU, VideoDecoderType::SOFTWARE));
  // factories.emplace_back(
  //   new JPEGEvaluatorFactory);

  return desc;
}
}

REGISTER_PIPELINE(frame_extract, get_pipeline_description);
}

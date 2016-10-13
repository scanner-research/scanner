#include "scanner/evaluators/util/encoder_evaluator.h"

#include "scanner/util/opencv.h"

namespace scanner {

EncoderEvaluator::EncoderEvaluator(EvaluatorConfig config) {}

void EncoderEvaluator::configure(const VideoMetadata& metadata) {
  this->metadata = metadata;
}

void EncoderEvaluator::evaluate(
  const std::vector<std::vector<u8*>>& input_buffers,
  const std::vector<std::vector<size_t>>& input_sizes,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes)
{
  std::string ext(".mkv");
  std::string path = std::tmpnam(nullptr) + ext;
  {
    cv::VideoWriter writer(
      path,
      CV_FOURCC('H','2','6','4'),
      24.0, // TODO: get this from metadata
      cv::Size(metadata.width(), metadata.height()));

    for (auto& buf : input_buffers[0]) {
      cv::Mat img = bytesToImage(buf, metadata);
      cv::cvtColor(img, img, CV_BGR2RGB);
      writer.write(img);
    }
  }

  FILE *f = fopen(path.c_str(), "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  u8* buf = new u8[fsize];
  fread(buf, fsize, 1, f);
  fclose(f);

  output_buffers[0].push_back(buf);
  output_sizes[0].push_back(fsize);

  for (i32 i = 0; i < input_buffers[0].size() - 1; ++i) {
    output_buffers[0].push_back(new u8[4]);
    output_sizes[0].push_back(0);
  }
}

EncoderEvaluatorFactory::EncoderEvaluatorFactory() {}

EvaluatorCapabilities EncoderEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> EncoderEvaluatorFactory::get_output_names() {
  return {"video"};
}

Evaluator* EncoderEvaluatorFactory::new_evaluator(
  const EvaluatorConfig& config) {
  return new EncoderEvaluator(config);
}

}

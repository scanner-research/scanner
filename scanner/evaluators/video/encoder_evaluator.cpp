#include "scanner/evaluators/video/encoder_evaluator.h"

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
  auto start = now();
  // OpenCV 2.4.x apparently can't encode H.264 videos
#if CV_MAJOR_VERSION >= 3
  std::string ext(".mkv");
  int fourcc = CV_FOURCC('H','2','6','4');
#else
  std::string ext(".avi");
  int fourcc = CV_FOURCC('D','I','V','X');
#endif
  std::string path = std::tmpnam(nullptr) + ext;
  {
    cv::VideoWriter writer(
      path,
      fourcc,
      24.0, // TODO: get this from metadata
      cv::Size(metadata.width(), metadata.height()));

    for (auto& buf : input_buffers[0]) {
      cv::Mat img = bytesToImage(buf, metadata);
      cv::cvtColor(img, img, CV_BGR2RGB);
      writer.write(img);
    }
  }

  FILE *f = fopen(path.c_str(), "rb");
  if (f == NULL) {
    LOG(FATAL) << path << " could not be found";
  }
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  u8* buf = new u8[fsize];
  fread(buf, fsize, 1, f);
  fclose(f);

  for (i32 i = 0; i < input_buffers[0].size() - 1; ++i) {
    output_buffers[0].push_back(new u8[1]);
    output_sizes[0].push_back(0);
  }

  output_buffers[0].push_back(buf);
  output_sizes[0].push_back(fsize);

  if (profiler_) {
    profiler_->add_interval("encode", start, now());
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

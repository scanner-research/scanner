#include "scanner/evaluators/video/encoder_evaluator.h"

#include "scanner/util/opencv.h"

namespace scanner {

EncoderEvaluator::EncoderEvaluator(EvaluatorConfig config) {}

void EncoderEvaluator::configure(const InputFormat& metadata) {
  this->metadata = metadata;
}

void EncoderEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  auto start = now();
  // OpenCV 2.4.x apparently can't encode H.264 videos
  std::string ext;
  int fourcc;
  if (CV_MAJOR_VERSION >= 3) {
    ext = std::string(".mkv");
    fourcc = CV_FOURCC('H', '2', '6', '4');
  } else {
    ext = std::string(".avi");
    fourcc = CV_FOURCC('D', 'I', 'V', 'X');
  }
  char templt[] = "/tmp/videoXXXXXX";
  if (mkstemp(templt) == -1) {
    LOG(FATAL) << "Encoder failed to make temp file";
  }
  std::string path = std::string(templt) + ext;
  {
    cv::VideoWriter writer(path, fourcc,
                           24.0,  // TODO: get this from metadata
                           cv::Size(metadata.width(), metadata.height()));

    for (auto& buf : input_buffers[0]) {
      cv::Mat img = bytesToImage(buf, metadata);
      cv::cvtColor(img, img, CV_BGR2RGB);
      writer.write(img);
    }
  }

  FILE* f = fopen(path.c_str(), "rb");
  if (f == NULL) {
    LOG(FATAL) << "Encoder could not find " << path;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    LOG(FATAL) << "Encoder seek failed";
  }
  long fsize = ftell(f);
  if (fseek(f, 0, SEEK_SET) != 0) {
    LOG(FATAL) << "Encoder seek failed";
  }

  u8* buf = new u8[fsize];
  if (fread(buf, fsize, 1, f) != fsize) {
    LOG(FATAL) << "Encoder failed to read " << fsize << " bytes from " << path;
  }
  if (fclose(f) != 0) {
    LOG(FATAL) << "Encoder failed to close file " << path;
  }

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

#include "scanner/evaluators/video/encoder_evaluator.h"

#include "scanner/util/opencv.h"

namespace scanner {

EncoderEvaluator::EncoderEvaluator(EvaluatorConfig config) {}

void EncoderEvaluator::configure(const BatchConfig& config) {
  config_ = config;

  assert(config.formats.size() == 1);
  frame_width_ = config.formats[0].width();
  frame_height_ = config.formats[0].height();
}

void EncoderEvaluator::evaluate(const BatchedColumns& input_columns,
                                BatchedColumns& output_columns) {
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
                           cv::Size(frame_width_, frame_height_));

    for (const Row& r : input_columns[0].rows) {
      auto& buf = r.buffer;
      cv::Mat img = bytesToImage(buf, config_.formats[0]);
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

  for (i32 i = 0; i < input_columns[0].rows.size() - 1; ++i) {
    output_columns[0].rows.push_back(Row{new u8[1], 1});
  }

  output_columns[0].rows.push_back(Row{buf, static_cast<size_t>(fsize)});

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

std::vector<std::string> EncoderEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  return {"video"};
}

Evaluator* EncoderEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new EncoderEvaluator(config);
}
}

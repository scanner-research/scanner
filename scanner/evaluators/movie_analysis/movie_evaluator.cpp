#include <memory>

#include "scanner/evaluators/movie_analysis/movie_evaluator.h"
#include "scanner/util/opencv.h"

#include "histogram_evaluator.h"
#include "face_evaluator.h"
#include "optical_flow_evaluator.h"
#include "camera_motion_evaluator.h"

namespace scanner {

MovieEvaluator::MovieEvaluator(EvaluatorConfig config) {
  std::unique_ptr<MovieFeatureEvaluator> faces(new FaceEvaluator());
  std::unique_ptr<MovieFeatureEvaluator> histogram(new HistogramEvaluator());
  std::unique_ptr<MovieFeatureEvaluator> opticalflow(new OpticalFlowEvaluator());
  std::unique_ptr<MovieFeatureEvaluator> cameramotion(new CameraMotionEvaluator());
  evaluators["faces"] = std::move(faces);
  evaluators["histogram"] = std::move(histogram);
  evaluators["opticalflow"] = std::move(opticalflow);
  evaluators["cameramotion"] = std::move(cameramotion);
}

void MovieEvaluator::configure(const VideoMetadata& metadata) {
  this->metadata = metadata;
  for (auto& entry : evaluators) {
    entry.second->configure(metadata);
  }
}

void MovieEvaluator::reset() {
  for (auto& entry : evaluators) {
    entry.second->reset_wrapper();
  }
}

void MovieEvaluator::evaluate(
  const std::vector<std::vector<u8*>>& input_buffers,
  const std::vector<std::vector<size_t>>& input_sizes,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes)
{
  std::vector<cv::Mat> imgs;
  for (i32 i = 0; i < input_buffers[0].size(); ++i) {
    cv::Mat img = bytesToImage(input_buffers[0][i], metadata);
    imgs.push_back(img);
  }

  evaluators["opticalflow"]->evaluate_wrapper(imgs, output_buffers[0], output_sizes[0]);
// #if defined DEBUG_FACE_DETECTOR
//   evaluators["faces"]->evaluate(imgs, output_buffers[0], output_sizes[0]);
// #elif defined DEBUG_OPTICAL_FLOW
//   evaluators["opticalflow"]->evaluate(imgs, output_buffers[0], output_sizes[0]);
// #else
//   evaluators["faces"]->evaluate(imgs, output_buffers[0], output_sizes[0]);
//   evaluators["histogram"]->evaluate(imgs, output_buffers[1], output_sizes[1]);
//   evaluators["opticalflow"]->evaluate(imgs, output_buffers[2], output_sizes[2]);
// #endif
}

MovieEvaluatorFactory::MovieEvaluatorFactory() {
#ifdef USE_HEADHUNTER
  boost::filesystem::path config_path
    ("/homes/wcrichto/lightscan/eccv2014_face_detection_fefdw_HeadHunter_baseline.config.ini");
  objects_detection::init_objects_detection(config_path);
#endif
}

EvaluatorCapabilities MovieEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 1;
  return caps;
}

std::vector<std::string> MovieEvaluatorFactory::get_output_names() {
#ifdef DEBUG_FACE_DETECTOR
  return {"faces"};
#else
  //return {"faces", "histogram", "opticalflow"};
  return {"cameramotion"};
#endif
}

Evaluator* MovieEvaluatorFactory::new_evaluator(
  const EvaluatorConfig& config) {
  return new MovieEvaluator(config);
}

}

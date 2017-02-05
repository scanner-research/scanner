#include <memory>

#include "scanner/evaluators/movie_analysis/movie_evaluator.h"
#include "scanner/util/opencv.h"

#include "camera_motion_evaluator.h"
#include "face_evaluator.h"
#include "histogram_evaluator.h"
#include "optical_flow_evaluator.h"

namespace scanner {

MovieEvaluator::MovieEvaluator(EvaluatorConfig config, DeviceType device_type,
                               std::vector<std::string> outputs)
    : device_type_(device_type), outputs_(outputs) {
  if (device_type == DeviceType::GPU) {
#ifdef HAVE_CUDA
    assert(config.device_ids.size() == 1);
    cvc::setDevice(config.device_ids[0]);
#endif
  }

  std::unique_ptr<MovieFeatureEvaluator> faces(new FaceEvaluator(device_type_));
  std::unique_ptr<MovieFeatureEvaluator> histogram(new HistogramEvaluator(device_type_);
  std::unique_ptr<MovieFeatureEvaluator> opticalflow(new OpticalFlowEvaluator(device_type_));
  std::unique_ptr<MovieFeatureEvaluator> cameramotion(new CameraMotionEvaluator(device_type_));
  evaluators["faces"] = std::move(faces);
  evaluators["histogram"] = std::move(histogram);
  evaluators["opticalflow"] = std::move(opticalflow);
  evaluators["cameramotion"] = std::move(cameramotion);

  for (auto& entry : evaluators) {
    entry.second->set_profiler(profiler_);
  }
}

void MovieEvaluator::configure(const InputFormat &metadata) {
  this->metadata = metadata;
  for (auto &entry : evaluators) {
    entry.second->configure(metadata);
  }
}

void MovieEvaluator::reset() {
  for (auto &entry : evaluators) {
    entry.second->reset_wrapper();
  }
}

void MovieEvaluator::evaluate(
    const std::vector<std::vector<u8 *>> &input_buffers,
    const std::vector<std::vector<size_t>> &input_sizes,
    std::vector<std::vector<u8 *>> &output_buffers,
    std::vector<std::vector<size_t>> &output_sizes) {
  std::vector<Mat> imgs;
  for (i32 i = 0; i < input_buffers[0].size(); ++i) {
#ifdef HAVE_CUDA
    Mat img = bytesToImage_gpu(input_buffers[0][i], metadata);
#else
    Mat img = bytesToImage(input_buffers[0][i], metadata);
#endif
    imgs.push_back(img);
  }

  auto start = now();
  for (i32 i = 0; i < outputs_.size(); i++) {
    evaluators[outputs_[i]]->evaluate_wrapper(imgs, output_buffers[i],
                                              output_sizes[i]);
  }

  if (profiler_) {
    profiler_->add_interval("movie", start, now());
  }
}

MovieEvaluatorFactory::MovieEvaluatorFactory(DeviceType device_type,
                                             std::vector<std::string> outputs)
    : outputs_(outputs), device_type_(device_type) {
#ifdef USE_HEADHUNTER
  boost::filesystem::path config_path(
      "/homes/wcrichto/lightscan/"
      "eccv2014_face_detection_fefdw_HeadHunter_baseline.config.ini");
  objects_detection::init_objects_detection(config_path);
#endif
}

EvaluatorCapabilities MovieEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = 1;
  return caps;
}

std::vector<std::string> MovieEvaluatorFactory::get_output_names() {
  return outputs_;
}

Evaluator *MovieEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new MovieEvaluator(config, device_type_, outputs_);
}
}

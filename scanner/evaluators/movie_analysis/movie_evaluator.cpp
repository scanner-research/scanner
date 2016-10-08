#include <memory>

#include "scanner/evaluators/movie_analysis/movie_evaluator.h"
#include "scanner/util/opencv.h"
#include "scanner/util/cycle_timer.h"

#include <boost/gil/image.hpp>
#include "applications/objects_detection_lib/objects_detection_lib.hpp"

namespace scanner {

class FaceEvaluator : public MovieItemEvaluator {
public:
  FaceEvaluator() {
    // if (!face_detector.load("/opt/stanford/wcrichto/haarcascade_frontalface_alt.xml")) {
    // // if (!face_detector.load("/export/data1/stanford/lightscan/haarcascade_frontalface_alt.xml")) {
    //   LOG(FATAL) << "Failed to load face cascade";
    // }
  }

  void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) {
    for (auto& img : inputs) {
      std::vector<cv::Rect> faces;
      //cv_find_faces(img, faces);
      hh_find_faces(img, faces);

      i32 size = faces.size() * sizeof(i32) * 4;
      i32* output_buffer = (i32*) (new u8[size]);
      for (i32 j = 0; j < faces.size(); ++j) {
        cv::Rect& face = faces[j];
        i32* offset = output_buffer + j * 4;
        *(offset + 0) = face.x;
        *(offset + 1) = face.y;
        *(offset + 2) = face.width;
        *(offset + 3) = face.height;
      }

      output_buffers.push_back((u8*) output_buffer);
      output_sizes.push_back(size);
    }
  }

  void cv_find_faces(cv::Mat& img, std::vector<cv::Rect>& faces) {
    face_detector.detectMultiScale(
      img, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
  }

  void hh_find_faces(cv::Mat& img, std::vector<cv::Rect>& faces) {
    double start = CycleTimer::currentSeconds();

    LOG(INFO) << "Computing...";
    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB);

    boost::gil::rgb8c_view_t view =
      boost::gil::interleaved_view
      (rgb.cols, rgb.rows,
       reinterpret_cast<boost::gil::rgb8c_pixel_t*>(rgb.data),
       static_cast<size_t>(rgb.step));
    objects_detection::set_monocular_image(view);
    objects_detection::compute();
    objects_detection::detections_t detections = objects_detection::get_detections();

    LOG(INFO) << CycleTimer::currentSeconds() - start;

    for (auto& det : detections) {
      auto box = det.bounding_box;
      auto min_corner = box.min_corner();
      auto max_corner = box.max_corner();

      cv::Rect face = cv::Rect(
        min_corner.x(),
        min_corner.y(),
        max_corner.x() - min_corner.x(),
        max_corner.y() - min_corner.y());

      faces.push_back(face);
    }
  }


private:
  cv::CascadeClassifier face_detector;
};

const i32 BINS = 16;

class HistogramEvaluator : public MovieItemEvaluator {
public:
  void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) {
    i64 hist_size = BINS * 3 * sizeof(float);
    for (auto& img : inputs) {
      std::vector<cv::Mat> bgr_planes;
      cv::split(img, bgr_planes);

      cv::Mat r_hist, g_hist, b_hist;
      float range[] = {0, 256};
      const float* histRange = {range};

      cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &BINS, &histRange);
      cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &BINS, &histRange);
      cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &BINS, &histRange);

      std::vector<cv::Mat> hists = {r_hist, g_hist, b_hist};
      cv::Mat hist;
      cv::hconcat(hists, hist);

      u8* hist_buffer = new u8[hist_size];
      assert(hist_size == hist.total() * hist.elemSize());
      memcpy(hist_buffer, hist.data, hist_size);

      output_sizes.push_back(hist_size);
      output_buffers.push_back(hist_buffer);
    }
  }
};

MovieEvaluator::MovieEvaluator(EvaluatorConfig config) {
  std::unique_ptr<MovieItemEvaluator> faces(new FaceEvaluator());
  std::unique_ptr<MovieItemEvaluator> histogram(new HistogramEvaluator());
  evaluators["faces"] = std::move(faces);
  evaluators["histogram"] = std::move(histogram);
}

void MovieEvaluator::configure(const VideoMetadata& metadata) {
  this->metadata = metadata;
  for (auto& entry : evaluators) {
    entry.second->configure(metadata);
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

  LOG(ERROR) << "Calculating...";
  double start = CycleTimer::currentSeconds();
  evaluators["faces"]->evaluate(imgs, output_buffers[0], output_sizes[0]);
  evaluators["histogram"]->evaluate(imgs, output_buffers[1], output_sizes[1]);
  LOG(ERROR) << CycleTimer::currentSeconds() - start;
}

MovieEvaluatorFactory::MovieEvaluatorFactory() {
  boost::filesystem::path config_path
    ("/homes/wcrichto/lightscan/eccv2014_face_detection_fefdw_HeadHunter_baseline.config.ini");
  objects_detection::init_objects_detection(config_path);
}


EvaluatorCapabilities MovieEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> MovieEvaluatorFactory::get_output_names() {
  return {"faces", "histogram"};
}

Evaluator* MovieEvaluatorFactory::new_evaluator(
  const EvaluatorConfig& config) {
  return new MovieEvaluator(config);
}

}

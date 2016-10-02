#include <memory>
#include "scanner/evaluators/movie_analysis/movie_evaluator.h"
#include "scanner/util/opencv.h"

namespace scanner {

class FaceEvaluator : public MovieItemEvaluator {
public:
  FaceEvaluator() {
    // if (!face_detector.load("/opt/stanford/wcrichto/haarcascade_frontalface_alt.xml")) {
    if (!face_detector.load("/export/data1/stanford/lightscan/haarcascade_frontalface_alt.xml")) {
      LOG(FATAL) << "Failed to load face cascade";
    }
  }

  void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) {
    for (auto& img : inputs) {
      std::vector<cv::Rect> faces;
      face_detector.detectMultiScale(
        img, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

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

private:
  cv::CascadeClassifier face_detector;
};

const i32 BINS = 256;

class HistogramEvaluator : public MovieItemEvaluator {
public:
  void evaluate(
    std::vector<cv::Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) {
    i64 hist_size = BINS * 3 * sizeof(u8);
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
      cv::Mat hist, hist_u8;
      cv::hconcat(hists, hist);
      hist.convertTo(hist_u8, CV_8U);

      u8* hist_buffer = new u8[hist_size];
      memcpy(hist_buffer, hist_u8.data, hist_u8.total() * hist_u8.elemSize());

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

void MovieEvaluator::configure(const DatasetItemMetadata& metadata) {
  this->metadata = metadata;
  for (auto& entry : evaluators) {
    entry.second->configure(metadata);
  }
}

void MovieEvaluator::evaluate(
  i32 input_count,
  u8* input_buffer,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes)
{
  std::vector<cv::Mat> imgs;
  for (i32 i = 0; i < input_count; ++i) {
    imgs.push_back(bytesToImage(input_buffer, i, metadata));
  }

  evaluators["faces"]->evaluate(imgs, output_buffers[0], output_sizes[0]);
  evaluators["histogram"]->evaluate(imgs, output_buffers[1], output_sizes[1]);
}

MovieEvaluatorFactory::MovieEvaluatorFactory() {}

EvaluatorCapabilities MovieEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

i32 MovieEvaluatorFactory::get_number_of_outputs() { return 2; }

std::vector<std::string> MovieEvaluatorFactory::get_output_names() {
  return {"faces", "histogram"};
}

Evaluator* MovieEvaluatorFactory::new_evaluator(
  const EvaluatorConfig& config) {
  return new MovieEvaluator(config);
}

}

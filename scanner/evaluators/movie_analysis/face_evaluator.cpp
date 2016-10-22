#include "face_evaluator.h"

// #define USE_HEADHUNTER

#ifdef USE_HEADHUNTER
#include <boost/gil/image.hpp>
#include "applications/objects_detection_lib/objects_detection_lib.hpp"
#endif

namespace scanner {

FaceEvaluator::FaceEvaluator() {
#ifndef USE_HEADHUNTER
  // if (!face_detector.load("/opt/stanford/wcrichto/haarcascade_frontalface_alt.xml")) {
  if (!face_detector.load("/home/wcrichto/scanner/haarcascade_frontalface_alt.xml")) {
    LOG(FATAL) << "Failed to load face cascade";
  }
#endif
}

float iou(const cv::Rect& bx, const cv::Rect& by) {
  float x1 = std::max(bx.x, by.x);
  float y1 = std::max(bx.y, by.y);
  float x2 = std::min(bx.x + bx.width, by.x + by.width);
  float y2 = std::min(bx.y + bx.height, by.y + by.height);

  if (x1 >= x2 || y1 >= y2) { return 0.0; }
  float intersection = (y2 - y1) * (x2 - x1);
  float _union = (bx.width * bx.height) + (by.width * by.height) - intersection;
  float iou = intersection / _union;
  return std::isnan(iou) ? 0.0 : iou;
}

void FaceEvaluator::cv_find_faces(cv::Mat& img, std::vector<cv::Rect>& faces) {
  face_detector.detectMultiScale(
    img, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
}

#ifdef USE_HEADHUNTER
void hh_find_faces(cv::Mat& img, std::vector<cv::Rect>& faces) {
  double start = CycleTimer::currentSeconds();

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
#endif

void FaceEvaluator::evaluate(
  std::vector<Mat>& inputs,
  std::vector<u8*>& output_buffers,
  std::vector<size_t>& output_sizes) {
#ifdef HAVE_CUDA
  LOG(FATAL) << "GPU not supported for faces";
#else
  for (auto& img : inputs) {
    std::vector<cv::Rect> faces;

#ifdef USE_HEADHUNTER
    hh_find_faces(img, faces);
#else
    cv_find_faces(img, faces);
#endif // USE_HEADHUNTER

#ifdef DEBUG_FACE_DETECTOR
    for (auto& face : faces) {
      cv::rectangle(img, face, cv::Scalar(255, 0, 0));
    }
#endif // DEBUG_FACE_DETECTOR

#ifdef DEBUG_FACE_DETECTOR
    i32 size = img.total() * img.elemSize();
    u8* output_buffer = new u8[size];
    std::memcpy(output_buffer, img.data, size);
#else
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
#endif // DEBUG_FACE_DETECTOR

    output_buffers.push_back((u8*) output_buffer);
    output_sizes.push_back(size);
  }
#endif // HAVE_GPU
}

}

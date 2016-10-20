#pragma once

#include "movie_feature_evaluator.h"

namespace scanner {

class FaceEvaluator : public MovieFeatureEvaluator {
public:
  FaceEvaluator();

  void evaluate(
    std::vector<Mat>& inputs,
    std::vector<u8*>& output_buffers,
    std::vector<size_t>& output_sizes) override;

  void cv_find_faces(cv::Mat& img, std::vector<cv::Rect>& faces);

private:
  cv::CascadeClassifier face_detector;
};

}

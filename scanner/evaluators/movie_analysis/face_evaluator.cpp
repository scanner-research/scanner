/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/evaluators/movie_analysis/face_evaluator.h"

#include "scanner/util/opencv.h"

#include "Tracker.h"
#include "Config.h"

namespace scanner {

FaceEvaluator::FaceEvaluator(EvaluatorConfig) {
  // if (!face_detector.load("/opt/stanford/wcrichto/haarcascade_frontalface_alt.xml")) {
  if (!face_detector.load("/export/data1/stanford/lightscan/haarcascade_frontalface_alt.xml")) {
    LOG(FATAL) << "Failed to load face cascade";
  }
}

void FaceEvaluator::configure(const DatasetItemMetadata& metadata) {
  this->metadata = metadata;
}

Tracker* build_tracker() {
  Config conf;
  Config::FeatureKernelPair fkp = {
    Config::kFeatureTypeHaar,
    Config::kKernelTypeGaussian,
    {0.2}
  };
  conf.features.emplace_back(fkp);
  return new Tracker(conf);
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
  return isnan(iou) ? 0.0 : iou;
}

void FaceEvaluator::evaluate(
  i32 input_count,
  u8* input_buffer,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes)
{
  std::vector<Tracker*> trackers;
  for (i32 i = 0; i < input_count; ++i) {
    cv::Mat img = bytesToImage(input_buffer, i, metadata);
    cv::cvtColor(img, img, CV_RGB2GRAY);
    std::vector<cv::Rect> all_faces;
    std::vector<Tracker*> valid_trackers;
    // if (i > 0) {
    //   for (auto& tracker : trackers) {
    //     tracker->Track(img);
    //     float score = tracker->GetScore();
    //     // LOG(ERROR) << score;
    //     // if (score < 0.1) {
    //     //   delete tracker;
    //     //   continue;
    //     // }
    //     const FloatRect bb = tracker->GetBB();
    //     all_faces.push_back(cv::Rect(bb.XMin(), bb.YMin(), bb.Width(), bb.Height()));
    //     valid_trackers.push_back(tracker);
    //   }
    // }
    // trackers.swap(valid_trackers);

    std::vector<cv::Rect> detected_faces;
    face_detector.detectMultiScale(
      img, detected_faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (auto& new_face : detected_faces) {
      bool is_new = true;
      for (auto& old_face : all_faces) {
        if (iou(new_face, old_face) > 0.5) {
          is_new = false;
          break;
        }
      }
      if (is_new) {
        all_faces.push_back(new_face);
        // Tracker* tracker = build_tracker();
        // FloatRect bb(new_face.x, new_face.y, new_face.width, new_face.height);
        // tracker->Initialise(img, bb);
        // trackers.push_back(tracker);
      }
    }

    i32 size = all_faces.size() * sizeof(i32) * 4;
    i32* output_buffer = (i32*) (new u8[size]);
    for (i32 j = 0; j < all_faces.size(); ++j) {
      cv::Rect& face = all_faces[j];
      i32* offset = output_buffer + j * 4;
      *(offset + 0) = face.x;
      *(offset + 1) = face.y;
      *(offset + 2) = face.width;
      *(offset + 3) = face.height;
    }

    output_buffers[0].push_back((u8*)output_buffer);
    output_sizes[0].push_back(size);
  }

  for (auto& tracker : trackers) {
    delete tracker;
  }
}

FaceEvaluatorFactory::FaceEvaluatorFactory() {}

EvaluatorCapabilities FaceEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = DeviceType::CPU;
  caps.max_devices = 1;
  caps.warmup_size = 0;
  return caps;
}

i32 FaceEvaluatorFactory::get_number_of_outputs() { return 1; }

std::vector<std::string> FaceEvaluatorFactory::get_output_names() {
  return {"face"};
}

Evaluator* FaceEvaluatorFactory::new_evaluator(
  const EvaluatorConfig& config) {
  return new FaceEvaluator(config);
}
}

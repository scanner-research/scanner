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

#include "scanner/parsers/facenet_parser.h"

#include <fstream>
#include <queue>

namespace scanner {

FacenetParser::FacenetParser(double threshold)
    : num_templates_(25),
      net_input_width_(224),
      net_input_height_(224),
      cell_width_(8),
      cell_height_(8),
      grid_width_(net_input_width_ / cell_width_),
      grid_height_(net_input_height_ / cell_height_),
      threshold_(threshold) {
  {
    std::ifstream template_file{"features/caffe_facenet/facenet_templates.bin",
                                std::ifstream::binary};
    templates_.resize(num_templates_, std::vector<float>(4));
    for (i32 t = 0; t < 25; ++t) {
      for (i32 i = 0; i < 4; ++i) {
        assert(template_file.good());
        f64 d;
        template_file.read(reinterpret_cast<char*>(&d), sizeof(f64));
        templates_[t][i] = d;
      }
    }
  }

  feature_vector_lengths_ = {
      grid_width_ * grid_height_ * num_templates_,  // template probabilities
      grid_width_ * grid_height_ * num_templates_ * 4,  // template adjustments
  };
  feature_vector_sizes_ = {
      sizeof(f32) * feature_vector_lengths_[0],
      sizeof(f32) * feature_vector_lengths_[1],
  };
}

std::vector<std::string> FacenetParser::get_output_names() {
  return {"score_final"};
}

void FacenetParser::configure(const VideoMetadata& metadata) {
  metadata_ = metadata;

  net_input_width_ = metadata_.width();
  net_input_height_ = metadata_.height();
  grid_width_ = (net_input_width_ / cell_width_);
  grid_height_ = (net_input_height_ / cell_height_);

  feature_vector_lengths_ = {
      grid_width_ * grid_height_ * num_templates_,  // template probabilities
      grid_width_ * grid_height_ * num_templates_ * 4,  // template adjustments
  };
  feature_vector_sizes_ = {
      sizeof(f32) * feature_vector_lengths_[0],
      sizeof(f32) * feature_vector_lengths_[1],
  };
}

void FacenetParser::parse_output(const std::vector<u8*>& output,
                                 const std::vector<i64>& output_size,
                                 folly::dynamic& parsed_results) {
  // Track confidence per pixel for each category so we can calculate
  // uncertainty across the frame
  assert(output_size[0] ==
         (feature_vector_sizes_[0] + feature_vector_sizes_[1]));
  f32* template_confidences = reinterpret_cast<f32*>(output[0]);
  f32* template_adjustments = template_confidences + feature_vector_lengths_[0];

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  std::vector<Box> bboxes;
  for (i32 t = 0; t < num_templates_; ++t) {
    for (i32 xi = 0; xi < grid_width_; ++xi) {
      for (i32 yi = 0; yi < grid_height_; ++yi) {
        i32 vec_offset = xi * grid_height_ + yi;

        f32 confidence =
            template_confidences[t * grid_width_ * grid_height_ + vec_offset];

        if (confidence < threshold_) continue;

        f32 x = xi * cell_width_ - 2;
        f32 y = yi * cell_height_ - 2;

        f32 width = templates_[t][2] - templates_[t][0] + 1;
        f32 height = templates_[t][3] - templates_[t][1] + 1;

        f32 dcx = template_adjustments[(num_templates_ * 0 + t) * grid_width_ *
                                           grid_height_ +
                                       vec_offset];
        x += width * dcx;

        f32 dcy = template_adjustments[(num_templates_ * 1 + t) * grid_width_ *
                                           grid_height_ +
                                       vec_offset];
        y += height * dcy;

        f32 dcw = template_adjustments[(num_templates_ * 2 + t) * grid_width_ *
                                           grid_height_ +
                                       vec_offset];
        width *= std::exp(dcw);

        f32 dch = template_adjustments[(num_templates_ * 3 + t) * grid_width_ *
                                           grid_height_ +
                                       vec_offset];
        height *= std::exp(dch);

        x = (x / net_input_width_) * metadata_.width();
        y = (y / net_input_height_) * metadata_.height();

        width = (width / net_input_width_) * metadata_.width();
        height = (height / net_input_height_) * metadata_.height();

        if (width < 0 || height < 0) continue;

        Box bbox;
        bbox.x1 = x - width / 2;
        bbox.y1 = y - height / 2;
        bbox.x2 = x + width / 2;
        bbox.y2 = y + height / 2;
        bbox.confidence = confidence;
        bboxes.push_back(bbox);
      }
    }
  }

  std::vector<Box> best_boxes = nms(bboxes, 0.3);

  folly::dynamic out_bboxes = folly::dynamic::array();
  for (auto& b : best_boxes) {
    folly::dynamic bbox = folly::dynamic::object();
    f32 width = b.x2 - b.x1;
    f32 height = b.y2 - b.y1;
    bbox["category"] = 0;
    bbox["x"] = b.x1 + width / 2;
    bbox["y"] = b.y1 + height / 2;
    bbox["width"] = width;
    bbox["height"] = height;
    bbox["confidence"] = b.confidence;
    out_bboxes.push_back(bbox);
  }

  parsed_results["bboxes"] = out_bboxes;
  parsed_results["certainty"] = 0;
}

std::vector<FacenetParser::Box> FacenetParser::nms(std::vector<Box> boxes,
                                                   f32 overlap) {
  std::vector<bool> valid(boxes.size(), true);
  auto cmp = [](std::pair<f32, i32> left, std::pair<f32, i32> right) {
    return left.first < right.first;
  };
  std::priority_queue<std::pair<f32, i32>, std::vector<std::pair<f32, i32>>,
                      decltype(cmp)>
      q(cmp);
  for (i32 i = 0; i < (i32)boxes.size(); ++i) {
    q.emplace(boxes[i].confidence, i);
  }
  std::vector<i32> best;
  while (!q.empty()) {
    std::pair<f32, i32> entry = q.top();
    q.pop();
    i32 c_idx = entry.second;
    if (!valid[c_idx]) continue;

    best.push_back(c_idx);

    for (i32 i = 0; i < (i32)boxes.size(); ++i) {
      if (!valid[i]) continue;

      f32 x1 = std::max(boxes[c_idx].x1, boxes[i].x1);
      f32 y1 = std::max(boxes[c_idx].y1, boxes[i].y1);
      f32 x2 = std::min(boxes[c_idx].x2, boxes[i].x2);
      f32 y2 = std::min(boxes[c_idx].y2, boxes[i].y2);

      f32 o_w = std::max(0.0f, x2 - x1 + 1);
      f32 o_h = std::max(0.0f, y2 - y1 + 1);

      f32 box_overlap = o_w * o_h / ((boxes[i].x2 - boxes[i].x1 + 1) *
                                     (boxes[i].y2 - boxes[i].y1 + 1));

      valid[i] = box_overlap < overlap;
    }
  }

  std::vector<Box> out_boxes;
  for (i32 i : best) {
    out_boxes.push_back(boxes[i]);
  }
  return out_boxes;
}
}

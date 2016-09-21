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

#include "scanner/parsers/yolo_parser.h"

namespace scanner {

YoloParser::YoloParser(double threshold)
  : threshold_(threshold)
{
  categories_ = {
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
  };
  num_categories_ = static_cast<i32>(categories_.size());

  input_width_ = 448;
  input_height_ = 448;
  grid_width_ = 7;
  grid_height_ = 7;
  cell_width_ = input_width_ / grid_width_;
  cell_height_ = input_height_ / grid_height_;
  num_bboxes_ = 2;

  feature_vector_lengths_ = {
    grid_width_ * grid_height_ * num_categories_, // category confidences
    grid_width_ * grid_height_ * num_bboxes_,       // objectness
    grid_width_ * grid_height_ * num_bboxes_ * 4    // bbox attributes
  };
  feature_vector_sizes_ = {
    sizeof(f32) * feature_vector_lengths_[0],
    sizeof(f32) * feature_vector_lengths_[1],
    sizeof(f32) * feature_vector_lengths_[2],
  };
}

std::vector<std::string> YoloParser::get_output_names() {
  return {"result"};
}

void YoloParser::configure(const DatasetItemMetadata& metadata) {
}

void YoloParser::parse_output(
  const std::vector<u8*>& output,
  const std::vector<i64>& output_size,
  folly::dynamic& parsed_results)
{
  // Track confidence per pixel for each category so we can calculate
  // uncertainty across the frame
  assert(output_size[0] == (
           feature_vector_sizes_[0] +
           feature_vector_sizes_[1] +
           feature_vector_sizes_[2]));
  f32* category_confidences_vector =
    reinterpret_cast<f32*>(output[0]);
  f32* objectness_vector =
    category_confidences_vector + feature_vector_lengths_[0];
  f32* bbox_vector =
    objectness_vector += feature_vector_lengths_[1];

  std::vector<f32> pixel_confidences(
    input_height_ * input_width_ * num_categories_, 0.0f);

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  folly::dynamic bboxes = folly::dynamic::array();
  for (i32 yi = 0; yi < grid_height_; ++yi) {
    for (i32 xi = 0; xi < grid_width_; ++xi) {
      for (i32 bi = 0; bi < num_bboxes_; ++bi) {
        folly::dynamic bbox = folly::dynamic::object();
        i32 vec_offset = yi * grid_width_ + xi;

        f32 x =
          ((xi + bbox_vector[(vec_offset) * num_bboxes_ + bi * 4 + 0])
           / grid_width_) * input_width_;
        f32 y =
          ((yi + bbox_vector[(vec_offset) * num_bboxes_ + bi * 4 + 1])
           / grid_height_) * input_height_;

        f32 width =
          std::pow(bbox_vector[(vec_offset) * num_bboxes_ + bi * 4 + 3], 2)
          * input_width_;
        f32 height =
          std::pow(bbox_vector[(vec_offset) * num_bboxes_ + bi * 4 + 4], 2)
          * input_height_;

        std::vector<f32> category_probabilities(num_categories_);
        for (i32 c = 0; c < num_categories_; ++c) {
          f64 prob =
            objectness_vector[vec_offset * num_bboxes_ + bi] *
            category_confidences_vector[vec_offset + c];
          category_probabilities[c] = prob;

          if (prob < threshold_) continue;

          for (i32 bbox_y = std::max(y - height / 2, 0.0f);
               bbox_y < std::min(y + height / 2, (f32)input_height_);
               ++bbox_y)
          {
            for (i32 bbox_x = std::max(x - width / 2, 0.0f);
                 bbox_x < std::min(x + width / 2, (f32)input_width_);
                 ++bbox_x)
            {
              f32& max_confidence =
                pixel_confidences[bbox_y * input_width_ +
                                  bbox_x * num_categories_ +
                                  c];
              if (prob > max_confidence) {
                max_confidence = prob;
              }
            }
          }

          if (width < 0 || height < 0) continue;

          bbox["category"] = c;
          bbox["x"]      = x;
          bbox["y"]      = y;
          bbox["width"]  = width;
          bbox["height"] = height;
          bbox["confidence"] = prob;

          bboxes.push_back(bbox);
        }
      }
    }
  }

  i32 non_thresholded_pixels = 1;
  f64 certainty = 0.0f;
  for (i32 yi = 0; yi < input_height_; ++yi) {
    for (i32 xi = 0; xi < input_width_; ++xi) {
      // For each pixel, compute difference between two most
      // confident classes
      f32 max1 = std::numeric_limits<f32>::lowest();
      f32 max2 = std::numeric_limits<f32>::lowest();
      for (i32 c = 0; c < num_categories_; ++c) {
        const f32& confidence =
          pixel_confidences[yi * input_width_ +
                            xi * num_categories_ +
                            c];
        if (confidence > max1) {
          max2 = max1;
          max1 = confidence;
        } else if (confidence > max2) {
          max2 = confidence;
        }
      }
      certainty += (max1 - max2);
      if (max1 > threshold_ || max2 > threshold_) {
        non_thresholded_pixels++;
      }
    }
  }

  parsed_results["bboxes"] = bboxes;
  parsed_results["certainty"] =
    certainty / non_thresholded_pixels;
}

}

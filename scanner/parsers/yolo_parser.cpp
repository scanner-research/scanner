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

#include "scanner/parser/yolo_parser.h"

namespace scanner {

YoloParser::YoloParser() {
  categories_ = {
    "Car",
    "Pedestrian",
    "Cyclist",
  };
  input_width_ = 640;
  input_height_ = 480;
  grid_width_ = 40;
  grid_height_ = 30;
  cell_width_ = input_width_ / grid_width_;
  cell_height_ = input_height_ / grid_height_;

  num_categories_ = static_cast<i32>(categories_.size());
  feature_vector_lengths_ = {
    grid_width_ * grid_height_ * num_categories_ * 4, // bbox params
    grid_width_ * grid_height_ * num_categories_ * 1, // objectness
  };
  feature_vector_sizes_ = {
    sizeof(f32) * feature_vector_lengths_[0],
    sizeof(f32) * feature_vector_lengths_[1],
  };
}

std::vector<std::string> YoloParser::get_output_names() {
  return {"Layer19_bbox", "Layer19_cov"};
}

void YoloParser::parse_output(
  const std::vector<u8*>& output,
  const std::vector<i64>& output_size,
  folly::dynamic& parsed_results)
{
  // Track confidence per pixel for each category so we can calculate
// uncertainty across the frame
  std::vector<f32> pixel_confidences(
    input_height_ * input_width_ * num_categories_);

  memset(pixel_confidences.data(),
         0,
         sizeof(f32) * pixel_confidences.size());

// Get bounding box data from output feature vector and turn it
// into canonical center x, center y, width, height

// Confidence format is (confidence, x, y)
  std::vector<f32>& confidence_vector = feature_vectors[1];
// Bbox format is (bbox_values, x, y)
// bbox_values is 4 * num_categories_
  std::vector<f32>& bbox_vector = feature_vectors[0];
  i32 bbox_stride = grid_width_ * grid_height_;
  i32 category_stride = bbox_stride * 4;
  folly::dynamic bboxes = folly::dynamic::array();
  for (i32 yi = 0; yi < grid_height_; ++yi) {
    for (i32 xi = 0; xi < grid_width_; ++xi) {
      for (i32 category = 0; category < num_categories_; ++category) {
        if (filtered_category != -1 &&
            category != filtered_category)
        {
          continue;
        }
        folly::dynamic bbox = folly::dynamic::object();
        i32 vec_offset = yi * grid_width_ + xi;
        i32 category_offset = category_stride * category;

        f32 x = (xi * cell_width_ + 0.5) / input_width_;
        f32 y = (yi * cell_height_ + 0.5) / input_height_;

        f32 confidence = confidence_vector[vec_offset];
        if (confidence < threshold) continue;
        f32 abs_left =
          x - bbox_vector[category_offset +
                          bbox_stride * 0 +
                          vec_offset];
        f32 abs_top =
          y - bbox_vector[category_offset +
                          bbox_stride * 1 +
                          vec_offset];
        f32 abs_right =
          x + bbox_vector[category_offset +
                          bbox_stride * 2 +
                          vec_offset];
        f32 abs_bottom =
          y + bbox_vector[category_offset +
                          bbox_stride * 3 +
                          vec_offset];

        abs_left *= input_width_;
        abs_top *= input_height_;
        abs_right *= input_width_;
        abs_bottom *= input_height_;

        for (i32 bbox_y = std::max(abs_top, 0.0f);
             bbox_y < std::min(abs_bottom, (f32)input_height_);
             ++bbox_y)
        {
          for (i32 bbox_x = std::max(abs_left, 0.0f);
               bbox_x < std::min(abs_right, (f32)input_width_);
               ++bbox_x)
          {
            f32& max_confidence =
              pixel_confidences[bbox_y * input_width_ +
                                bbox_x * num_categories_ +
                                category];
            if (confidence > max_confidence) {
              max_confidence = confidence;
            }
          }
        }

        f32 width = (abs_right - abs_left);
        f32 height = (abs_bottom - abs_top);
        f32 center_x = width / 2 + abs_left;
        f32 center_y = height / 2 + abs_top;

        if (width < 0 || height < 0) continue;

        bbox["category"] = category;
        bbox["x"]      = center_x;
        bbox["y"]      = center_y;
        bbox["width"]  = width;
        bbox["height"] = height;
        bbox["confidence"] = confidence;

        bboxes.push_back(bbox);
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
      if (max1 > threshold || max2 > threshold) {
        non_thresholded_pixels++;
      }
    }
  }

  feature_data["data"]["bboxes"] = bboxes;
  feature_data["data"]["certainty"] =
    certainty / non_thresholded_pixels;
}

}

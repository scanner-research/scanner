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

#include "scanner/evaluators/caffe/facenet/facenet_parser_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

#include <queue>
#include <cassert>
#include <cmath>

namespace scanner {

FacenetParserEvaluator::FacenetParserEvaluator(const EvaluatorConfig &config,
                                               DeviceType device_type,
                                               i32 device_id, double threshold,
                                               bool forward_input)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      forward_input_(forward_input),
      num_templates_(25),
      net_input_width_(224),
      net_input_height_(224),
      cell_width_(8),
      cell_height_(8),
      grid_width_(net_input_width_ / cell_width_),
      grid_height_(net_input_height_ / cell_height_),
      threshold_(threshold) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU facenet parser support not implemented yet";
  }

  std::ifstream template_file{"features/caffe_facenet/facenet_templates.bin",
                              std::ifstream::binary};
  templates_.resize(num_templates_, std::vector<float>(4));
  for (i32 t = 0; t < 25; ++t) {
    for (i32 i = 0; i < 4; ++i) {
      assert(template_file.good());
      f64 d;
      template_file.read(reinterpret_cast<char *>(&d), sizeof(f64));
      templates_[t][i] = d;
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

void FacenetParserEvaluator::configure(const DatasetItemMetadata &metadata) {
  metadata_ = metadata;

  net_input_width_ = metadata_.width;
  net_input_height_ = metadata_.height;
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

void FacenetParserEvaluator::evaluate(
    const std::vector<std::vector<u8 *>> &input_buffers,
    const std::vector<std::vector<size_t>> &input_sizes,
    std::vector<std::vector<u8 *>> &output_buffers,
    std::vector<std::vector<size_t>> &output_sizes) {
  i32 input_count = (i32)input_buffers[0].size();

  i32 feature_idx;
  if (forward_input_) {
    assert(input_buffers.size() >= 2);
    feature_idx = 1;
  } else {
    assert(input_buffers.size() >= 1);
    feature_idx = 0;
  }

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  for (i32 b = 0; b < input_count; ++b) {
    assert(input_sizes[feature_idx][b] ==
           (feature_vector_sizes_[0] + feature_vector_sizes_[1]));

    std::vector<BoundingBox> bboxes;
    // Track confidence per pixel for each category so we can calculate
    // uncertainty across the frame
    f32 *template_confidences =
        reinterpret_cast<f32 *>(input_buffers[feature_idx][b]);
    f32 *template_adjustments =
        template_confidences + feature_vector_lengths_[0];

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

          f32 dcx = template_adjustments[(num_templates_ * 0 + t) *
                                             grid_width_ * grid_height_ +
                                         vec_offset];
          x += width * dcx;

          f32 dcy = template_adjustments[(num_templates_ * 1 + t) *
                                             grid_width_ * grid_height_ +
                                         vec_offset];
          y += height * dcy;

          f32 dcw = template_adjustments[(num_templates_ * 2 + t) *
                                             grid_width_ * grid_height_ +
                                         vec_offset];
          width *= std::exp(dcw);

          f32 dch = template_adjustments[(num_templates_ * 3 + t) *
                                             grid_width_ * grid_height_ +
                                         vec_offset];
          height *= std::exp(dch);

          x = (x / net_input_width_) * metadata_.width;
          y = (y / net_input_height_) * metadata_.height;

          width = (width / net_input_width_) * metadata_.width;
          height = (height / net_input_height_) * metadata_.height;

          if (width < 0 || height < 0) continue;

          BoundingBox bbox;
          bbox.x1 = x - width / 2;
          bbox.y1 = y - height / 2;
          bbox.x2 = x + width / 2;
          bbox.y2 = y + height / 2;
          bbox.confidence = confidence;
          bboxes.push_back(bbox);
        }
      }
    }

    std::vector<BoundingBox> best_bboxes = nms(bboxes, 0.3);

    size_t size = sizeof(size_t) + sizeof(BoundingBox) * best_bboxes.size();
    u8* buffer = new u8[size];
    output_buffers[feature_idx].push_back(buffer);
    output_sizes[feature_idx].push_back(size);

    *((size_t*)buffer) = best_bboxes.size();
    u8* buf = buffer + sizeof(size_t);
    for (size_t i = 0; i < best_bboxes.size(); ++i) {
      const BoundingBox& box = best_bboxes[i];
      memcpy(buf + i * sizeof(BoundingBox), &box, sizeof(BoundingBox));
    }
  }

  if (forward_input_) {
    u8 *buffer = nullptr;
    for (i32 b = 0; b < input_count; ++b) {
      size_t size = input_sizes[0][b];
      if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
        cudaMalloc((void**)&buffer, size);
        cudaMemcpy(buffer, input_buffers[0][b], size, cudaMemcpyDefault);
#else
        LOG(FATAL) << "Not built with CUDA support.";
#endif
      } else {
        buffer = new u8[size];
        memcpy(buffer, input_buffers[0][b], size);
      }
      output_buffers[0].push_back(buffer);
      output_sizes[0].push_back(size);
    }
  }
}

std::vector<BoundingBox> FacenetParserEvaluator::nms(
    const std::vector<BoundingBox>& boxes, f32 overlap) {
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

  std::vector<BoundingBox> out_boxes;
  for (i32 i : best) {
    out_boxes.push_back(boxes[i]);
  }
  return out_boxes;
}

FacenetParserEvaluatorFactory::FacenetParserEvaluatorFactory(
    DeviceType device_type, double threshold, bool forward_input)
    : device_type_(device_type),
      threshold_(threshold),
      forward_input_(forward_input) {}

EvaluatorCapabilities FacenetParserEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU facenet parser support not implemented yet";
    caps.max_devices = 1;
  } else {
    caps.max_devices = 1;
  }
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> FacenetParserEvaluatorFactory::get_output_names() {
  std::vector<std::string> output_names;
  if (forward_input_) {
    output_names.push_back("frame");
  }
  output_names.push_back("bboxes");
  return output_names;
}

Evaluator *FacenetParserEvaluatorFactory::new_evaluator(
    const EvaluatorConfig &config) {
  return new FacenetParserEvaluator(config, device_type_, 0, threshold_,
                                    forward_input_);
}
}

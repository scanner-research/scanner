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
#include "scanner/evaluators/serialize.h"

#include "scanner/util/bbox.h"
#include "scanner/util/common.h"
#include "scanner/util/util.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#include <opencv2/cudawarping.hpp>
#endif

#include <cassert>
#include <cmath>

namespace scanner {

FacenetParserEvaluator::FacenetParserEvaluator(const EvaluatorConfig& config,
                                               DeviceType device_type,
                                               i32 device_id, double threshold,
                                               NMSType nms_type,
                                               bool forward_input)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      nms_type_(nms_type),
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
      template_file.read(reinterpret_cast<char*>(&d), sizeof(f64));
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

void FacenetParserEvaluator::configure(const InputFormat& metadata) {
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

void FacenetParserEvaluator::evaluate(const BatchedColumns& input_columns,
                                      BatchedColumns& output_columns) {
  i32 input_count = (i32)input_columns[0].rows.size();

  i32 feature_idx;
  i32 frame_idx;
  if (forward_input_) {
    assert(input_columns.size() >= 2);
    feature_idx = 1;
    frame_idx = 0;
  } else {
    assert(input_columns.size() >= 1);
    feature_idx = 0;
    frame_idx = 0;
  }

  // Get bounding box data from output feature vector and turn it
  // into canonical center x, center y, width, height
  for (i32 b = 0; b < input_count; ++b) {
    assert(input_columns[feature_idx].rows[b].size ==
           (feature_vector_sizes_[0] + feature_vector_sizes_[1]));

    std::vector<BoundingBox> bboxes;
    // Track confidence per pixel for each category so we can calculate
    // uncertainty across the frame
    f32* template_confidences =
        reinterpret_cast<f32*>(input_columns[feature_idx].rows[b].buffer);
    f32* template_adjustments =
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

          x = (x / net_input_width_) * metadata_.width();
          y = (y / net_input_height_) * metadata_.height();

          width = (width / net_input_width_) * metadata_.width();
          height = (height / net_input_height_) * metadata_.height();

          if (width < 0 || height < 0 || std::isnan(width) ||
              std::isnan(height) || std::isnan(x) || std::isnan(y))
            continue;

          BoundingBox bbox;
          bbox.set_x1(x - width / 2);
          bbox.set_y1(y - height / 2);
          bbox.set_x2(x + width / 2);
          bbox.set_y2(y + height / 2);
          bbox.set_score(confidence);

          if (bbox.x1() < 0 || bbox.y1() < 0 || bbox.x2() > metadata_.width() ||
              bbox.y2() > metadata_.height())
            continue;

          bboxes.push_back(bbox);
        }
      }
    }

    std::vector<BoundingBox> best_bboxes;
    switch (nms_type_) {
      case NMSType::Best:
        best_bboxes = best_nms(bboxes, 0.3);
        break;
      case NMSType::Average:
        best_bboxes = average_nms(bboxes, 0.3);
        break;
      case NMSType::None:
        best_bboxes = bboxes;
        break;
    }

    // Assume size of a bounding box is the same size as all bounding boxes
    size_t size;
    u8* buffer;
    serialize_bbox_vector(best_bboxes, buffer, size);
    output_columns[feature_idx].rows.push_back(Row{buffer, size});
  }

  if (forward_input_) {
    u8* buffer = nullptr;
    for (i32 b = 0; b < input_count; ++b) {
      size_t size = input_columns[frame_idx].rows[b].size;
      if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
        cudaMalloc((void**)&buffer, size);
        cudaMemcpy(buffer, input_columns[frame_idx].rows[b].buffer, size,
                   cudaMemcpyDefault);
#else
        LOG(FATAL) << "Not built with CUDA support.";
#endif
      } else {
        buffer = new u8[size];
        memcpy(buffer, input_columns[frame_idx].rows[b].buffer, size);
      }
      output_columns[frame_idx].rows.push_back(Row{buffer, size});
    }
  }
}

FacenetParserEvaluatorFactory::FacenetParserEvaluatorFactory(
    DeviceType device_type, double threshold,
    FacenetParserEvaluator::NMSType nms_type, bool forward_input)
    : device_type_(device_type),
      threshold_(threshold),
      nms_type_(nms_type),
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

Evaluator* FacenetParserEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new FacenetParserEvaluator(config, device_type_, 0, threshold_,
                                    nms_type_, forward_input_);
}
}

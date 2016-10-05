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

#include "scanner/evaluators/tracker/tracker_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

#include "struck/Tracker.h"
#include "struck/Config.h"

namespace scanner {

TrackerEvaluator::TrackerEvaluator(const EvaluatorConfig& config,
                                   DeviceType device_type,
                                   i32 device_id,
                                   i32 warmup_count)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      warmup_count_(warmup_count)
{
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU tracker support not implemented yet";
  }
}

void TrackerEvaluator::configure(const DatasetItemMetadata& metadata) {
  LOG(INFO) << "Tracker configure";
  metadata_ = metadata;
}

void TrackerEvaluator::reset() {
  LOG(INFO) << "Tracker reset";
  trackers_.clear();
  tracker_configs_.clear();
}

void TrackerEvaluator::evaluate(
    const std::vector<std::vector<u8 *>> &input_buffers,
    const std::vector<std::vector<size_t>> &input_sizes,
    std::vector<std::vector<u8 *>> &output_buffers,
    std::vector<std::vector<size_t>> &output_sizes) {
  assert(input_buffers.size() >= 2);

  i32 input_count = input_buffers[0].size();
  LOG(INFO) << "Tracker evaluate on " << input_count << " inputs";

  for (i32 b = 0; b < input_count; ++b) {
    u8 *bbox_buffer = input_buffers[1][b];
    size_t num_bboxes = *((size_t *)bbox_buffer);
    bbox_buffer += sizeof(size_t);

    // Find all the boxes which overlap the existing tracked boxes and update
    // the tracked boxes confidence values to those as well as the time since
    // last being seen

    // For boxes which don't overlap existing ones, create a new track for them
    std::vector<BoundingBox> new_detected_bboxes;
    for (size_t i = 0; i < num_bboxes; ++i) {
      BoundingBox box;
      box.x1 = *((f32 *)bbox_buffer);
      bbox_buffer += sizeof(f32);
      box.y1 = *((f32 *)bbox_buffer);
      bbox_buffer += sizeof(f32);
      box.x2 = *((f32 *)bbox_buffer);
      bbox_buffer += sizeof(f32);
      box.y2 = *((f32 *)bbox_buffer);
      bbox_buffer += sizeof(f32);
      box.confidence = *((f32 *)bbox_buffer);

      i32 overlap_idx = -1;
      for (size_t j = 0; j < tracked_bboxes_.size(); ++j) {
        auto& tracked_bbox = tracked_bboxes_[j];
        if (iou(box, tracked_bbox) > 0.3) {
          overlap_idx = j;
          break;
        }
      }
      if (overlap_idx != -1) {
        // Overlap with existing box
        tracked_bboxes_[overlap_idx] = box;
        frames_since_last_detection_[overlap_idx] = 0;
      } else {
        // New box
        new_detected_bboxes.push_back(box);
      }
    }

    // Check if any tracks have been many frames without being redetected and
    // remove them
    for (i32 i = 0; i < (i32)trackers_.size(); ++i) {
      if (frames_since_last_detection_[i] > UNDETECTED_WINDOW) {
        tracked_bboxes_.erase(tracked_bboxes_.begin() + i);
        frames_since_last_detection_.erase(
            frames_since_last_detection_.begin() + i);
        trackers_.erase(trackers_.begin() + i);
        tracker_configs_.erase(tracker_configs_.begin() + i);
        i--;
      }
    }

    // Perform tracking for all existing tracks that we have
    std::vector<BoundingBox> generated_bboxes;
    {
      u8 *buffer = input_buffers[0][b];
      cv::Mat frame(metadata_.height, metadata_.width, CV_8UC3, buffer);
      for (i32 i = 0; i < (i32)trackers_.size(); ++i) {
        auto& tracker = trackers_[i];
        tracker->Track(frame);
        const struck::FloatRect &tracked_bbox = tracker->GetBB();
        BoundingBox box;
        box.x1 = tracked_bbox.XMin();
        box.y1 = tracked_bbox.YMin();
        box.x2 = tracked_bbox.XMax();
        box.y2 = tracked_bbox.YMax();
        generated_bboxes.push_back(box);

        frames_since_last_detection_[i]++;
      }
    }

    // Add new detected bounding boxes to the fold
    for (const BoundingBox &box : new_detected_bboxes) {
      struck::FloatRect r(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

      tracker_configs_.push_back(struck::Config{});
      struck::Config &config = tracker_configs_.back();
      config.frameWidth = metadata_.width;
      config.frameHeight = metadata_.height;
      struck::Config::FeatureKernelPair fkp;
      fkp.feature = struck::Config::kFeatureTypeHaar;
      fkp.kernel = struck::Config::kKernelTypeLinear;
      config.features.push_back(fkp);
      struck::Tracker *tracker = new struck::Tracker(config);

      u8 *buffer = input_buffers[0][0];
      cv::Mat frame(metadata_.height, metadata_.width, CV_8UC3, buffer);
      tracker->Initialise(frame, r);

      tracked_bboxes_.push_back(box);
      trackers_.emplace_back(tracker);
      frames_since_last_detection_.push_back(0);

      generated_bboxes.push_back(box);
    }

    {
      size_t size =
          sizeof(size_t) + sizeof(BoundingBox) * generated_bboxes.size();
      u8 *buffer = new u8[size];
      output_buffers[1].push_back(buffer);
      output_sizes[1].push_back(size);

      *((size_t *)buffer) = generated_bboxes.size();
      u8 *buf = buffer + sizeof(size_t);
      for (size_t i = 0; i < generated_bboxes.size(); ++i) {
        const BoundingBox &box = generated_bboxes[i];
        memcpy(buf + i * sizeof(BoundingBox), &box, sizeof(BoundingBox));
      }
    }
    {
      size_t size =
          sizeof(size_t) + sizeof(BoundingBox) * generated_bboxes.size();
      u8 *buffer = new u8[size];
      output_buffers[2].push_back(buffer);
      output_sizes[2].push_back(size);

      *((size_t *)buffer) = generated_bboxes.size();
      u8 *buf = buf + sizeof(size_t);
      for (size_t i = 0; i < generated_bboxes.size(); ++i) {
        const BoundingBox &box = generated_bboxes[i];
        memcpy(buf + i * sizeof(BoundingBox), &box, sizeof(BoundingBox));
      }
    }
  }

  u8 *buffer = nullptr;
  for (i32 b = 0; b < input_count; ++b) {
    size_t size = input_sizes[0][b];
    if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
      cudaMalloc((void **)&buffer, size);
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

float TrackerEvaluator::iou(const BoundingBox& bl, const BoundingBox& br) {
  float x1 = std::max(bl.x1, br.x1);
  float y1 = std::max(bl.y1, br.y1);
  float x2 = std::min(bl.x2, br.x2);
  float y2 = std::min(bl.y2, br.y2);

  float bl_width = bl.x2 - bl.x1;
  float bl_height = bl.y2 - bl.y1;
  float br_width = br.x2 - br.x1;
  float br_height= br.y2 - br.y1;
  if (x1 >= x2 || y1 >= y2) { return 0.0; }
  float intersection = (y2 - y1) * (x2 - x1);
  float _union = (bl_width * bl_height) + (br_width * br_height) - intersection;
  float iou = intersection / _union;
  return isnan(iou) ? 0.0 : iou;
}


TrackerEvaluatorFactory::TrackerEvaluatorFactory(DeviceType device_type,
                                                 i32 warmup_count)
    : device_type_(device_type), warmup_count_(warmup_count) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU tracker support not implemented yet";
  }
}

EvaluatorCapabilities TrackerEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = 1;
  caps.warmup_size = warmup_count_;
  return caps;
}

std::vector<std::string> TrackerEvaluatorFactory::get_output_names() {
  return {"image", "before_bboxes", "after_bboxes"};
}

Evaluator *
TrackerEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new TrackerEvaluator(config, device_type_, 0, warmup_count_);
}
}

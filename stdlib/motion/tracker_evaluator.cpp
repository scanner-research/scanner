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
#include "scanner/evaluators/serialize.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

#include "struck/Config.h"
#include "struck/Tracker.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#endif

#include <cmath>
#include <thread>

namespace scanner {

TrackerEvaluator::TrackerEvaluator(const EvaluatorConfig& config,
                                   DeviceType device_type, i32 device_id,
                                   i32 warmup_count, i32 max_tracks)
  : config_(config),
    device_type_(device_type),
    device_id_(device_id),
    warmup_count_(warmup_count),
    max_tracks_(max_tracks) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU tracker support not implemented yet";
  }
}

void TrackerEvaluator::configure(const BatchConfig& config) {
  VLOG(1) << "Tracker configure";
  assert(config.formats.size() == 1);
  metadata_ = config.formats[0];
}

void TrackerEvaluator::reset() {
  VLOG(1) << "Tracker reset";
  tracks_.clear();
}

void TrackerEvaluator::evaluate(const BatchedElements& input_columns,
                                BatchedElements& output_columns) {
  assert(input_columns.size() >= 2);

  i32 input_count = input_columns[0].rows.size();
  VLOG(1) << "Tracker evaluate on " << input_count << " inputs";

  i32 frame_idx = 0;
  i32 box_idx = 1;

  printf("num tracks %d\n", tracks_.size());
  for (i32 b = 0; b < input_count; ++b) {
    std::vector<BoundingBox> all_boxes = deserialize_proto_vector<BoundingBox>(
        input_columns[box_idx].rows[b].buffer,
        input_columns[box_idx].rows[b].size);

    // Find all the boxes which overlap the existing tracked boxes and update
    // the tracked boxes confidence values to those as well as the time since
    // last being seen

    // For boxes which don't overlap existing ones, create a new track for them
    std::vector<BoundingBox> detected_bboxes;
    std::vector<BoundingBox> new_detected_bboxes;
    for (const BoundingBox& box : all_boxes) {
      i32 overlap_idx = -1;
      for (size_t j = 0; j < tracks_.size(); ++j) {
        auto& tracked_bbox = tracks_[j].box;
        if (iou(box, tracked_bbox) > IOU_THRESHOLD) {
          overlap_idx = j;
          break;
        }
      }
      if (overlap_idx != -1) {
        // Overlap with existing box
        tracks_[overlap_idx].box = box;
        tracks_[overlap_idx].frames_since_last_detection = 0;
      } else {
        // New box
        new_detected_bboxes.push_back(box);
      }
      detected_bboxes.push_back(box);
    }

    // Perform tracking for all existing tracks that we have
    std::vector<BoundingBox> generated_bboxes;
    {
      u8* buffer = input_columns[frame_idx].rows[b].buffer;
      assert(input_columns[frame_idx].rows[b].size ==
             metadata_.height() * metadata_.width() * 3 * sizeof(u8));
      cv::Mat frame(metadata_.height(), metadata_.width(), CV_8UC3, buffer);
      std::vector<f64> scores(tracks_.size());
      std::vector<struck::FloatRect> tracked_bboxes(tracks_.size());
      std::vector<std::thread> tracker_threads(tracks_.size());
      auto track_fn = [](struck::Tracker* tracker, const cv::Mat& frame,
                         f64& score, struck::FloatRect& tracked_bbox) {
        tracker->Track(frame);
        score = tracker->GetScore();
        tracked_bbox = tracker->GetBB();
      };

      for (i32 i = 0; i < (i32)tracks_.size(); ++i) {
        auto& track = tracks_[i];
        auto& tracker = track.tracker;
        tracker_threads[i] =
            std::thread(track_fn, tracker.get(), std::ref(frame),
                        std::ref(scores[i]), std::ref(tracked_bboxes[i]));
      }
      for (i32 i = 0, jid = 0; i < (i32)tracks_.size(); ++i, ++jid) {
        auto& track = tracks_[i];
        auto& tracker = track.tracker;
        tracker_threads[jid].join();
        f64 score = scores[jid];
        struck::FloatRect tracked_bbox = tracked_bboxes[jid];
        if (score < TRACK_SCORE_THRESHOLD) {
          tracks_.erase(tracks_.begin() + i);
          i--;
        } else {
          BoundingBox box;
          box.set_x1(tracked_bbox.XMin());
          box.set_y1(tracked_bbox.YMin());
          box.set_x2(tracked_bbox.XMax());
          box.set_y2(tracked_bbox.YMax());
          box.set_score(track.box.score());
          box.set_track_id(track.id);
          box.set_track_score(score);
          generated_bboxes.push_back(box);

          track.frames_since_last_detection++;
        }
      }
    }

    // Add new detected bounding boxes to the fold
    if (tracks_.size() + new_detected_bboxes.size() > max_tracks_) {
      std::vector<std::tuple<f64, i32>> track_thresholds;
      for (i32 i = 0; i < tracks_.size(); ++i) {
        track_thresholds.push_back(
            std::make_tuple<f64, i32>(tracks_[i].tracker->GetScore(), (i32)i));
      }
      std::sort(track_thresholds.begin(), track_thresholds.end(),
                [](auto& left, auto& right) {
                  return std::get<0>(left) < std::get<0>(right);
                });
      i32 num_tracks_to_remove =
          std::min(tracks_.size(),
                   tracks_.size() + new_detected_bboxes.size() - max_tracks_);
      std::vector<i32> idx_to_remove;
      for (i32 i = 0; i < num_tracks_to_remove; ++i) {
        idx_to_remove.push_back(std::get<1>(track_thresholds[i]));
      }
      std::sort(idx_to_remove.begin(), idx_to_remove.end());
      for (i32 i = 0; i < num_tracks_to_remove; ++i) {
        i32 idx = idx_to_remove[num_tracks_to_remove - 1 - i];
        tracks_.erase(tracks_.begin() + idx);
      }
    }
    assert(tracks_.size() <= max_tracks_);
    for (BoundingBox& box : new_detected_bboxes) {
      tracks_.resize(tracks_.size() + 1);
      Track& track = tracks_.back();
      // i32 tracker_id = next_tracker_id_++;
      i32 tracker_id = unif(gen);
      track.id = tracker_id;
      track.config.reset(new struck::Config{});
      struck::Config& config = *track.config.get();
      config.frameWidth = metadata_.width();
      config.frameHeight = metadata_.height();
      struck::Config::FeatureKernelPair fkp;
      fkp.feature = struck::Config::kFeatureTypeHaar;
      fkp.kernel = struck::Config::kKernelTypeLinear;
      config.features.push_back(fkp);
      track.tracker.reset(new struck::Tracker(config));

      u8* buffer = input_columns[frame_idx].rows[b].buffer;
      assert(input_columns[frame_idx].rows[b].size ==
             metadata_.height() * metadata_.width() * 3);
      cv::Mat frame(metadata_.height(), metadata_.width(), CV_8UC3, buffer);

      // Clamp values
      float x1 = std::max(box.x1(), 0.0f);
      float y1 = std::max(box.y1(), 0.0f);
      float x2 = std::min(box.x2(), (f32)metadata_.width());
      float y2 = std::min(box.y2(), (f32)metadata_.height());
      struck::FloatRect r(x1, y1, x2 - x1, y2 - y1);
      track.tracker->Initialise(frame, r);

      box.set_track_id(track.id);
      box.set_track_score(0.0f);
      track.frames_since_last_detection = 0;
    }

    {
      size_t size;
      u8* buffer;

      serialize_bbox_vector(detected_bboxes, buffer, size);
      output_columns[1].rows.push_back(Row{buffer, size});

      serialize_bbox_vector(generated_bboxes, buffer, size);
      output_columns[2].rows.push_back(Row{buffer, size});
    }
  }

  for (i32 b = 0; b < input_count; ++b) {
    output_columns[0].rows.push_back(input_columns[frame_idx].rows[b]);
  }
}

float TrackerEvaluator::iou(const BoundingBox& bl, const BoundingBox& br) {
  float x1 = std::max(bl.x1(), br.x1());
  float y1 = std::max(bl.y1(), br.y1());
  float x2 = std::min(bl.x2(), br.x2());
  float y2 = std::min(bl.y2(), br.y2());

  float bl_width = bl.x2() - bl.x1();
  float bl_height = bl.y2() - bl.y1();
  float br_width = br.x2() - br.x1();
  float br_height = br.y2() - br.y1();
  if (x1 >= x2 || y1 >= y2) {
    return 0.0;
  }
  float intersection = (y2 - y1) * (x2 - x1);
  float _union = (bl_width * bl_height) + (br_width * br_height) - intersection;
  float iou = intersection / _union;
  return std::isnan(iou) ? 0.0 : iou;
}

TrackerEvaluatorFactory::TrackerEvaluatorFactory(DeviceType device_type,
                                                 i32 warmup_count,
                                                 i32 max_tracks)
  : device_type_(device_type),
    warmup_count_(warmup_count),
    max_tracks_(max_tracks) {
  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "GPU tracker support not implemented yet";
  }
}

EvaluatorCapabilities TrackerEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = EvaluatorCapabilities::UnlimitedDevices;
  caps.warmup_size = warmup_count_;
  return caps;
}

std::vector<std::string> TrackerEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  return {"frame", "before_bboxes", "after_bboxes"};
}

Evaluator* TrackerEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new TrackerEvaluator(config, device_type_, 0, warmup_count_,
                              max_tracks_);
}
}

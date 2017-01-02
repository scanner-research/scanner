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

#include "scanner/engine/sampling.h"

namespace scanner {

// Gets the list of work items for a sequence of rows in the job
RowIntervals slice_into_row_intervals(const JobMetadata& job,
                                      const std::vector<i64>& rows) {
  RowIntervals info;
  // Analyze rows and job to determine what item ids and offsets in them to
  // sample from
  i32 io_item_size = job.io_item_size();
  auto item_from_row = [io_item_size](i64 r) -> i32 {
    return r / io_item_size;
  };
  auto offset_from_row = [io_item_size](i64 r) -> i64 {
    return r % io_item_size;
  };

  assert(!rows.empty());
  i32 current_item = item_from_row(rows[0]);
  i64 item_start = offset_from_row(rows[0]);
  i64 item_end = item_start + 1;
  std::vector<i64> valid_offsets;
  for (i64 row : rows) {
    i32 item = item_from_row(row);
    i64 item_offset = offset_from_row(row);
    if (item != current_item) {
      // Start a new item and push the current one into the list
      info.item_ids.push_back(current_item);
      info.item_intervals.push_back(std::make_tuple(item_start, item_end));
      info.valid_offsets.push_back(valid_offsets);

      current_item = item;
      item_start = item_offset;
      item_end = item_offset + 1;
      valid_offsets.clear();
    }

    valid_offsets.push_back(item_offset);
    item_end = item_offset + 1;
  }
  info.item_ids.push_back(current_item);
  info.item_intervals.push_back(std::make_tuple(item_start, item_end));
  info.valid_offsets.push_back(valid_offsets);

  return info;
}

VideoIntervals slice_into_video_intervals(
    const std::vector<i64>& keyframe_positions,
    const std::vector<i64>& rows) {
  VideoIntervals info;
  assert(keyframe_positions.size() >= 2);
  size_t start_keyframe_index = 0;
  size_t end_keyframe_index = 1;
  i64 next_keyframe = keyframe_positions[end_keyframe_index];
  std::vector<i64> valid_frames;
  for (i64 row : rows) {
    if (row >= next_keyframe) {
      assert(end_keyframe_index < keyframe_positions.size() - 1);
      next_keyframe = keyframe_positions[end_keyframe_index + 1];
      if (row >= next_keyframe) {
        // Skipped a keyframe, so make a new interval
        info.keyframe_index_intervals.push_back(
            std::make_tuple(start_keyframe_index, end_keyframe_index));
        info.valid_frames.push_back(valid_frames);

        end_keyframe_index = end_keyframe_index + 1;
        while (row > keyframe_positions[end_keyframe_index]) {
          end_keyframe_index++;
        }
        valid_frames.clear();
        start_keyframe_index = end_keyframe_index - 1;
        next_keyframe = keyframe_positions[end_keyframe_index];
      } else {
        end_keyframe_index++;
      }
    }
    valid_frames.push_back(row);
  }
  info.keyframe_index_intervals.push_back(
      std::make_tuple(start_keyframe_index, end_keyframe_index));
  info.valid_frames.push_back(valid_frames);
  return info;
}

}

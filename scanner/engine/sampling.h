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

#pragma once

#include "scanner/engine/runtime.h"
#include "scanner/util/common.h"

namespace scanner {
namespace internal {

struct RowIntervals {
  std::vector<i32> item_ids;
  std::vector<std::tuple<i64, i64>> item_intervals;
  std::vector<std::vector<i64>> valid_offsets;
};

// Gets the list of work items for a sequence of rows in the job
RowIntervals slice_into_row_intervals(const JobMetadata& job,
                                      const std::vector<i64>& rows);

struct VideoIntervals {
  std::vector<std::tuple<size_t, size_t>> keyframe_index_intervals;
  std::vector<std::vector<i64>> valid_frames;
};

VideoIntervals slice_into_video_intervals(
    const std::vector<i64>& keyframe_positions, const std::vector<i64>& rows);

}
}

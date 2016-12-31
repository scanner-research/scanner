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

struct RowLocations {
  // For regular columns
  std::vector<i32> work_items;
  std::vector<Interval> work_item_intervals;
};

// Gets the list of work items for a sequence of rows in the job
RowLocations row_work_item_locations(Sampling sampling, i32 group_id,
                                     const LoadWorkEntry& entry) const;

struct FrameLocations {
  // For frame column
  std::vector<Interval> intervals;
  std::vector<DecodeArgs> video_args;
  std::vector<ImageDecodeArgs> image_args;
};

FrameLocations frame_locations(i32 video_index,
                               const LoadWorkEntry& entry) const;
}

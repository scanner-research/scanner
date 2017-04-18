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

#include "scanner/util/common.h"
#include "scanner/util/profiler.h"
#include "scanner/metadata.pb.h"

#include <vector>

namespace scanner {

using proto::FrameType;

size_t size_of_frame_type(FrameType type);

const i32 FRAME_DIMS = 3;

//! FrameInfo
struct FrameInfo {
  FrameInfo() = default;
  FrameInfo(const FrameInfo& info) = default;
  FrameInfo(FrameInfo&& info) = default;
  FrameInfo& operator=(const FrameInfo&) = default;

  FrameInfo(int shape0, int shape1, int shape2, FrameType type);

  bool operator==(const FrameInfo& other) const;
  bool operator!=(const FrameInfo& other) const;

  size_t size() const;

  //! Only valid when the dimensions are (height, width, channels)
  int width() const;

  //! Only valid when the dimensions are (height, width, channels)
  int height() const;

  //! Only valid when the dimensions are (height, width, channels)
  int channels() const;

  int shape[FRAME_DIMS];
  FrameType type;
};

//! Frame
class Frame {
public:
  Frame(FrameInfo info, u8* buffer);

  FrameInfo as_frame_info() const;

  size_t size() const;

  //! Only valid when the dimensions are (height, width, channels)
  int width() const;

  //! Only valid when the dimensions are (height, width, channels)
  int height() const;

  //! Only valid when the dimensions are (height, width, channels)
  int channels() const;

  int shape[FRAME_DIMS];
  FrameType type;
  u8* data;
};

Frame* new_frame(DeviceHandle device, FrameInfo info);

void delete_frame(DeviceHandle device, u8* buffer);

std::vector<Frame*> new_frames(DeviceHandle device, FrameInfo info, i32 num);

}

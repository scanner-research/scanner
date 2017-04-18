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

#include "scanner/api/frame.h"
#include "scanner/util/memory.h"

namespace scanner {

size_t size_of_frame_type(FrameType type) {
  size_t s;
  switch (type) {
    case FrameType::U8:
      s = sizeof(u8);
      break;
    case FrameType::F32:
      s = sizeof(f32);
      break;
    case FrameType::F64:
      s = sizeof(f64);
      break;
  }
  return s;
}

FrameInfo::FrameInfo(int shape0, int shape1, int shape2,
                     FrameType t) {
  assert(shape0 >= 0);
  assert(shape1 >= 0);
  assert(shape2 >= 0);

  shape[0] = shape0;
  shape[1] = shape1;
  shape[2] = shape2;
  type = t;
}

bool FrameInfo::operator==(const FrameInfo& other) const {
  bool same = (type == other.type);
  for (int i = 0; i < FRAME_DIMS; ++i) {
    same &= (shape[i] == other.shape[i]);
  }
  return same;
}

bool FrameInfo::operator!=(const FrameInfo& other) const {
  return !(*this == other);
}

size_t FrameInfo::size() const {
  size_t s = size_of_frame_type(type);
  for (int i = 0; i < FRAME_DIMS; ++i) {
    s *= shape[i];
  }
  return s;
}

Frame::Frame(FrameInfo info, u8* b) : data(b) {
  memcpy(shape, info.shape, sizeof(int) * FRAME_DIMS);
  type = info.type;
}


FrameInfo Frame::as_frame_info() const {
  return FrameInfo(shape[0], shape[1], shape[2], type);
}

size_t Frame::size() const {
  return as_frame_info().size();
}

Frame* new_frame(DeviceHandle device, FrameInfo info) {
  u8* buffer = new_buffer(device, info.size());
  return new Frame(info, buffer);
}

std::vector<Frame*> new_frames(DeviceHandle device, FrameInfo info, i32 num) {
  u8* buffer = new_block_buffer(device, info.size() * num, num);
  std::vector<Frame*> frames;
  for (i32 i = 0; i < num; ++i) {
    frames.push_back(new Frame(info, buffer + i * info.size()));
  }
  return frames;
}

}

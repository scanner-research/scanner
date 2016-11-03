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

#include "scanner/evaluators/types.pb.h"

#include <vector>

namespace scanner {

inline void serialize_bbox_vector(const std::vector<BoundingBox>& bboxes,
                                  u8*& buffer, size_t& size) {
  size = sizeof(size_t) + sizeof(i32);
  i32 bbox_size = 0;
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const BoundingBox& box = bboxes[i];
    bbox_size = std::max(bbox_size, box.ByteSize());
  }
  size += bbox_size * bboxes.size();
  buffer = new u8[size];

  u8* buf = buffer;
  *((size_t*)buf) = bboxes.size();
  buf += sizeof(size_t);
  *((i32*)buf) = bbox_size;
  buf += sizeof(i32);
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const BoundingBox& box = bboxes[i];
    assert(box.ByteSize() <= bbox_size);
    box.SerializeToArray(buf + i * bbox_size, bbox_size);
  }
}

inline void serialize_decode_args(const DecodeArgs& args, u8*& buffer,
                                  size_t& size) {
  size = args.ByteSize();
  buffer = new u8[size];
  args.SerializeToArray(buffer, size);
}

inline DecodeArgs deserialize_decode_args(const u8* buffer, size_t size) {
  DecodeArgs args;
  args.ParseFromArray(buffer, size);
  return args;
}
}

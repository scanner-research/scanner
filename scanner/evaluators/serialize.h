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
#include "scanner/util/memory.h"

#include <cstddef>
#include <vector>

namespace scanner {
namespace {
template <typename T>
inline T deser(const u8*& buffer, size_t& size_left) {
  assert(size_left >= sizeof(T));
  T e = *((T*)buffer);
  buffer += sizeof(T);
  size_left -= sizeof(T);
  return e;
}
}

template <typename T>
void serialize_proto(const T& element, u8*& buffer, size_t& size) {
  i32 element_size = element.ByteSize();
  buffer = new_buffer(CPU_DEVICE, size);
  size = element_size;
  element.SerializeToArray(buffer, element_size);
}

template <typename T>
T deserialize_proto(const u8* buffer, size_t size) {
  T e;
  e.ParseFromArray(buffer, size);
  return e;
}

template <typename T>
void serialize_proto_vector(const std::vector<T>& elements, u8*& buffer,
                            size_t& size) {
  size = sizeof(size_t);
  for (auto& e : elements) {
    size += e.ByteSize() + sizeof(i32);
  }
  buffer = new_buffer(CPU_DEVICE, size);

  u8* buf = buffer;
  *((size_t*)buf) = elements.size();
  buf += sizeof(size_t);
  for (size_t i = 0; i < elements.size(); ++i) {
    const T& e = elements[i];
    i32 element_size = e.ByteSize();
    *((i32*)buf) = element_size;
    buf += sizeof(i32);
    e.SerializeToArray(buf, element_size);
    buf += element_size;
  }
}

template <typename T>
void serialize_proto_vector_of_vectors(
    const std::vector<std::vector<T>>& elements, u8*& buffer, size_t& size) {
  size = sizeof(size_t);
  for (auto &vec_of_e : elements) {
    size += sizeof(size_t);
    for (auto &e : vec_of_e) {
      size += e.ByteSize() + sizeof(i32);
    }
  }
  buffer = new_buffer(CPU_DEVICE, size);

  u8* buf = buffer;
  *((size_t*)buf) = elements.size();
  buf += sizeof(size_t);
  for (size_t i = 0; i < elements.size(); ++i) {
    const std::vector<T>& vec_of_e = elements[i];
    *((size_t*)buf) = vec_of_e.size();
    buf += sizeof(size_t);
    for (size_t j = 0; j < vec_of_e.size(); ++j) {
      const T& e = vec_of_e[j];
      i32 element_size = e.ByteSize();
      *((i32*)buf) = element_size;
      buf += sizeof(i32);
      e.SerializeToArray(buf, element_size);
      buf += element_size;
    }
  }
}

template <typename T>
std::vector<T> deserialize_proto_vector(const u8* buffer, size_t size) {
  const u8* buf = buffer;
  size_t num_elements = deser<size_t>(buf, size);
  std::vector<T> elements;
  for (size_t i = 0; i < num_elements; ++i) {
    i32 element_size = deser<i32>(buf, size);
    assert(size >= element_size);
    T e;
    e.ParseFromArray(buf, element_size);
    size -= element_size;
    buf += element_size;
    elements.push_back(e);
  }
  return elements;
}

inline void serialize_bbox_vector(const std::vector<BoundingBox>& bboxes,
                                  u8*& buffer, size_t& size) {
  serialize_proto_vector(bboxes, buffer, size);
}

inline void serialize_decode_args(const DecodeArgs& args, u8*& buffer,
                                  size_t& size) {
  size = args.ByteSize();
  buffer = new_buffer(CPU_DEVICE, size);
  args.SerializeToArray(buffer, size);
}

inline DecodeArgs deserialize_decode_args(const u8* buffer, size_t size) {
  DecodeArgs args;
  args.ParseFromArray(buffer, size);
  return args;
}

inline void serialize_image_decode_args(const ImageDecodeArgs& args,
                                        u8*& buffer, size_t& size) {
  size = args.ByteSize();
  buffer = new_buffer(CPU_DEVICE, size);
  args.SerializeToArray(buffer, size);
}

inline ImageDecodeArgs deserialize_image_decode_args(const u8* buffer,
                                                     size_t size) {
  ImageDecodeArgs args;
  args.ParseFromArray(buffer, size);
  return args;
}
}

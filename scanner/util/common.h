/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "glog/logging.h"
#include "scanner/engine/rpc.pb.h"
#include "scanner/metadata.pb.h"
#include "scanner/types.pb.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Common data types
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

using proto::DeviceType;
using proto::ImageEncodingType;
using proto::ImageColorSpace;
using proto::ColumnType;
using proto::LoadWorkEntry;
using proto::Column;
using proto::MemoryPoolConfig;
using proto::BoundingBox;
using proto::Point;
using proto::Result;

struct DeviceHandle {
 public:
  bool operator==(const DeviceHandle& other) {
    return type == other.type && id == other.id;
  }

  bool operator!=(const DeviceHandle& other) { return !(*this == other); }

  bool can_copy_to(const DeviceHandle& other) {
    return !(this->type == DeviceType::GPU && other.type == DeviceType::GPU &&
             this->id != other.id);
  }

  bool is_same_address_space(const DeviceHandle& other) {
    return this->type == other.type &&
           ((this->type == DeviceType::CPU) ||
            (this->type == DeviceType::GPU && this->id == other.id));
  }

  DeviceType type;
  i32 id;
};

std::ostream& operator<<(std::ostream& os, const DeviceHandle& handle);

static const DeviceHandle CPU_DEVICE = {DeviceType::CPU, 0};

struct Interval {
  i32 start;
  i32 end;
};

struct StridedInterval {
 public:
  StridedInterval() = default;
  StridedInterval(i32 start, i32 end, i32 stride = 1);
  StridedInterval(const Interval&);

  i32 start;
  i32 end;
  i32 stride = 1;
};

bool string_to_image_encoding_type(const std::string& s,
                                   proto::ImageEncodingType& t);
std::string image_encoding_type_to_string(proto::ImageEncodingType d);

#define RESULT_ERROR(result__, str__, ...)          \
  {                                                 \
    char errstr__[1024];                            \
    snprintf(errstr__, 1024, str__, ##__VA_ARGS__); \
    LOG(ERROR) << errstr__;                         \
    (result__)->set_success(false);                 \
    (result__)->set_msg(errstr__);                  \
  }

///////////////////////////////////////////////////////////////////////////////
/// Global constants
extern i32 NUM_CUDA_STREAMS;       // # of cuda streams for image processing
}

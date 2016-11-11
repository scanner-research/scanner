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

#include "scanner/metadata.pb.h"

#include <cstdint>
#include <string>

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

enum class DeviceType {
  GPU,
  CPU,
};

enum class Sampling {
  All,
  Strided,
  Gather,
  SequenceGather,
};

struct Interval {
  i32 start;
  i32 end;
};

bool string_to_dataset_type(const std::string& s, DatasetType& t);
std::string dataset_type_to_string(DatasetType d);

bool string_to_image_encoding_type(const std::string& s, ImageEncodingType& t);
std::string image_encoding_type_to_string(ImageEncodingType d);

///////////////////////////////////////////////////////////////////////////////
/// Global constants
extern i32 PUS_PER_NODE;           // # of available processing units per node
extern i32 GPUS_PER_NODE;          // # of available GPUs per node
extern i32 WORK_ITEM_SIZE;         // Base size of a work item
extern i32 TASKS_IN_QUEUE_PER_PU;  // How many tasks per PU to allocate
extern i32 LOAD_WORKERS_PER_NODE;  // # of worker threads loading data
extern i32 SAVE_WORKERS_PER_NODE;  // # of worker threads loading data
extern i32 NUM_CUDA_STREAMS;       // # of cuda streams for image processing
}

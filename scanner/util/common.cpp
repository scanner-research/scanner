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

#include "scanner/util/common.h"

namespace scanner {

std::ostream &operator<<(std::ostream &os, DeviceHandle const& handle) {
  std::string name;
  if (handle.type == DeviceType::CPU) {
    name = "CPU";
  } else if (handle.type == DeviceType::GPU) {
    name = "GPU";
  } else {
    LOG(FATAL) << "Invalid device type";
  }
  return os << "{" << name << ", " << handle.id << "}";
}

StridedInterval::StridedInterval(i32 start, i32 end, i32 stride)
    : start(start), end(end), stride(stride) {}

StridedInterval::StridedInterval(const Interval& i)
    : start(i.start), end(i.end), stride(1) {}

bool string_to_image_encoding_type(const std::string& s,
                                   ImageEncodingType& type) {
  bool success = true;
  if (s == "png" || s == "PNG") {
    type = ImageEncodingType::PNG;
  } else if (s == "jpeg" || s == "JPEG" || s == "jpg" || s == "JPG") {
    type = ImageEncodingType::JPEG;
  } else if (s == "bmp" || s == "BMP") {
    type = ImageEncodingType::BMP;
  } else if (s == "raw" || s == "RAW") {
    type = ImageEncodingType::RAW;
  } else {
    success = false;
  }
  return success;
}

std::string image_encoding_type_to_string(ImageEncodingType t) {
  std::string s;
  switch (t) {
    case ImageEncodingType::JPEG:
      s = "jpeg";
      break;
    case ImageEncodingType::PNG:
      s = "png";
      break;
    case ImageEncodingType::BMP:
      s = "bmp";
      break;
    case ImageEncodingType::RAW:
      s = "raw";
      break;
    default:
      assert(false);
  }
  return s;
}

i32 KERNEL_INSTANCES_PER_NODE = 1;// Number of available PUs per node
i32 CPUS_PER_NODE = -1;            // Number of available CPUs per node
std::vector<i32> GPU_DEVICE_IDS;  // GPU device ids to use
i64 IO_ITEM_SIZE = 64;            // Number of rows to load and save at a time
i64 WORK_ITEM_SIZE = 8;           // Max size of a work item
i32 TASKS_IN_QUEUE_PER_PU = 4;    // How many tasks per PU to allocate to a node
i32 LOAD_WORKERS_PER_NODE = 2;    // Number of worker threads loading data
i32 SAVE_WORKERS_PER_NODE = 2;    // Number of worker threads loading data
i32 NUM_CUDA_STREAMS = 32;        // Number of cuda streams for image processing

}

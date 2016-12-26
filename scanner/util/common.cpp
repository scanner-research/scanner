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

StridedInterval::StridedInterval(i32 start, i32 end, i32 stride)
    : start(start), end(end), stride(stride) {}

StridedInterval::StridedInterval(const Interval& i)
    : start(i.start), end(i.end), stride(1) {}

bool string_to_dataset_type(const std::string& s, DatasetType& type) {
  bool success = true;
  if (s == "video") {
    type = DatasetType_Video;
  } else if (s == "image") {
    type = DatasetType_Image;
  } else {
    success = false;
  }
  return success;
}

std::string dataset_type_to_string(DatasetType d) {
  std::string s;
  switch (d) {
    case DatasetType_Video:
      s = "video";
      break;
    case DatasetType_Image:
      s = "image";
      break;
    default:
      assert(false);
  }
  return s;
}

bool string_to_image_encoding_type(const std::string& s,
                                   ImageEncodingType& type) {
  bool success = true;
  if (s == "png" || s == "PNG") {
    type = ImageEncodingType::PNG;
  } else if (s == "jpeg" || s == "JPEG" || s == "jpg" || s == "JPG") {
    type = ImageEncodingType::JPEG;
  } else if (s == "bmp") {
    type = ImageEncodingType::BMP;
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
    default:
      assert(false);
  }
  return s;
}

int PUS_PER_NODE = 1;             // Number of available PUs per node
int GPUS_PER_NODE = 2;            // Number of available GPUs per node
int WORK_ITEM_SIZE = 8;           // Base size of a work item
int TASKS_IN_QUEUE_PER_PU = 4;    // How many tasks per PU to allocate to a node
int LOAD_WORKERS_PER_NODE = 2;    // Number of worker threads loading data
int SAVE_WORKERS_PER_NODE = 2;    // Number of worker threads loading data
std::vector<i32> GPU_DEVICE_IDS;  // GPU device ids to use
int NUM_CUDA_STREAMS = 32;        // Number of cuda streams for image processing
}

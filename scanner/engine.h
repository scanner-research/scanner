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

#include "scanner/video/video_decoder.h"

#include "storehouse/storage_backend.h"

#include <string>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Work structs - structs used to exchange data between workers during 
///   execution of the run command.
struct VideoWorkItem {
  int video_index;
  int start_frame;
  int end_frame;
};

struct LoadWorkEntry {
  int work_item_index;
};

struct DecodeWorkEntry {
  int work_item_index;
  int start_keyframe;
  int end_keyframe;
  size_t encoded_data_size;
  char* buffer;
};

struct DecodeBufferEntry {
  size_t buffer_size;
  char* buffer;
};

struct EvalWorkEntry {
  int work_item_index;
  size_t decoded_frames_size;
  char* buffer;
};

struct SaveWorkEntry {
  int work_item_index;
  std::vector<size_t> output_buffer_sizes;
  std::vector<char*> output_buffers;
};

///////////////////////////////////////////////////////////////////////////////

enum class NetFrameworkType {
  GPU,
  CPU,
};

std::vector<NetFrameworkType> get_supported_net_framework_types();

bool has_net_framework_type(NetFrameworkType type);

void run_job(
  storehouse::StorageConfig* config,
  VideoDecoderType decoder_type,
  NetFrameworkType net_framework_type,
  const std::string& job_name,
  const std::string& dataset_name,
  const std::string& net_descriptor_file);

}

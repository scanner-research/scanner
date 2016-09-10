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

#include "storehouse/storage_backend.h"

#include <nvcuvid.h>

#include <string>
#include <vector>
#include <map>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Global constants
extern int PUS_PER_NODE;           // # of available processing units per node
extern int GLOBAL_BATCH_SIZE;       // Batch size for network
extern int BATCHES_PER_WORK_ITEM;   // How many batches per work item
extern int TASKS_IN_QUEUE_PER_PU;  // How many tasks per PU to allocate
extern int LOAD_WORKERS_PER_NODE;   // # of worker threads loading data
extern int SAVE_WORKERS_PER_NODE;   // # of worker threads loading data
extern int NUM_CUDA_STREAMS;        // # of cuda streams for image processing

///////////////////////////////////////////////////////////////////////////////
/// Path functions
inline std::string dataset_descriptor_path(const std::string& dataset_name)
{
  return dataset_name + "_dataset_descriptor.bin";
}

inline std::string dataset_item_data_path(const std::string& dataset_name,
                                          const std::string& item_name)
{
  return dataset_name + "_dataset/" + item_name + "_data.bin";
}

inline std::string dataset_item_video_path(const std::string& dataset_name,
                                           const std::string& item_name)
{
  return dataset_name + "_dataset/" + item_name + ".mp4";
}

inline std::string dataset_item_video_timestamps_path(
  const std::string& dataset_name,
  const std::string& item_name)
{
  return dataset_name + "_dataset/" + item_name + "_web_timestamps.bin";
}

inline std::string dataset_item_metadata_path(const std::string& dataset_name,
                                              const std::string& item_name)
{
  return dataset_name + "_dataset/" + item_name + "_metadata.bin";
}

inline std::string job_item_output_path(const std::string& job_name,
                                        const std::string& item_name,
                                        const std::string& layer_name,
                                        int start,
                                        int end)
{
  return job_name + "_job/" + item_name + "_" +
    layer_name + "_" +
    std::to_string(start) + "-" +
    std::to_string(end) + ".bin";
}

inline std::string job_descriptor_path(const std::string& job_name) {
  return job_name + "_job_descriptor.bin";
}

inline std::string job_profiler_path(const std::string& job_name, int node) {
  return job_name + "_job_profiler_" + std::to_string(node) + ".bin";
}

inline int frames_per_work_item() {
  return GLOBAL_BATCH_SIZE * BATCHES_PER_WORK_ITEM;
}

///////////////////////////////////////////////////////////////////////////////
/// Common persistent data structs and their serialization helpers

struct DatasetDescriptor {
  std::vector<std::string> original_video_paths;
  std::vector<std::string> item_names;
};

struct DatasetItemMetadata {
  int32_t frames;
  int32_t width;
  int32_t height;
  cudaVideoCodec codec_type;
  cudaVideoChromaFormat chroma_format;
  std::vector<char> metadata_packets;
  std::vector<int64_t> keyframe_positions;
  std::vector<int64_t> keyframe_timestamps;
  std::vector<int64_t> keyframe_byte_offsets;
};

struct DatasetItemWebTimestamps {
  int time_base_numerator;
  int time_base_denominator;
  std::vector<int64_t> dts_timestamps;
  std::vector<int64_t> pts_timestamps;
};

struct JobDescriptor {
  std::string dataset_name;
  std::map<std::string, std::vector<std::tuple<int, int>>> intervals;
};

void serialize_dataset_descriptor(
  storehouse::WriteFile* file,
  const DatasetDescriptor& descriptor);

DatasetDescriptor deserialize_dataset_descriptor(
  storehouse::RandomReadFile* file,
  uint64_t& file_pos);

void serialize_dataset_item_metadata(
  storehouse::WriteFile* file,
  const DatasetItemMetadata& metadata);

DatasetItemMetadata deserialize_dataset_item_metadata(
  storehouse::RandomReadFile* file,
  uint64_t& file_pos);

void serialize_dataset_item_web_timestamps(
  storehouse::WriteFile* file,
  const DatasetItemWebTimestamps& metadata);

DatasetItemWebTimestamps deserialize_dataset_item_web_timestamps(
  storehouse::RandomReadFile* file,
  uint64_t& file_pos);

void serialize_job_descriptor(
  storehouse::WriteFile* file,
  const JobDescriptor& descriptor);

JobDescriptor deserialize_job_descriptor(
  storehouse::RandomReadFile* file,
  uint64_t& file_pos);

}

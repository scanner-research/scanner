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

#include "scanner/metadata.pb.h"
#include "scanner/util/common.h"
#include "storehouse/storage_backend.h"

#include <set>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Common persistent data structs and their serialization helpers
struct DatabaseMetadata {
 public:
  DatabaseMetadata();
  DatabaseMetadata(const DatabaseDescriptor& descriptor);

  const DatabaseDescriptor& get_descriptor() const;

  bool has_dataset(const std::string& dataset) const;
  bool has_dataset(i32 dataset_id) const;
  i32 get_dataset_id(const std::string& dataset) const;
  const std::string& get_dataset_name(i32 dataset_id) const;
  i32 add_dataset(const std::string& dataset);
  void remove_dataset(i32 dataset_id);

  bool has_job(i32 dataset_id, const std::string& job) const;
  bool has_job(i32 job_id) const;
  i32 get_job_id(i32 dataset_id, const std::string& job_name) const;
  const std::string& get_job_name(i32 job_id) const;
  i32 add_job(i32 dataset_id, const std::string& job_name);
  void remove_job(i32 job_id);

  // private:
  i32 next_dataset_id;
  i32 next_job_id;
  std::map<i32, std::string> dataset_names;
  std::map<i32, std::set<i32>> dataset_job_ids;
  std::map<i32, std::string> job_names;

 private:
  mutable DatabaseDescriptor descriptor;
};

struct DatasetMetadata {
 public:
  DatasetMetadata();
  DatasetMetadata(const DatasetDescriptor& descriptor);

  const DatasetDescriptor& get_descriptor() const;

  i32 id() const;
  std::string name() const;
  DatasetType type() const;
  i32 total_rows() const;
  i32 max_width() const;
  i32 max_height() const;
  std::vector<std::string> original_paths() const;
  std::vector<std::string> item_names() const;

 private:
  mutable DatasetDescriptor descriptor;
};

struct VideoMetadata {
 public:
  VideoMetadata();
  VideoMetadata(const VideoDescriptor& descriptor);

  const VideoDescriptor& get_descriptor() const;

  i32 frames() const;
  i32 width() const;
  i32 height() const;
  std::vector<i64> keyframe_positions() const;
  std::vector<i64> keyframe_byte_offsets() const;

 private:
  mutable VideoDescriptor descriptor;
};

struct ImageFormatGroupMetadata {
 public:
  ImageFormatGroupMetadata();
  ImageFormatGroupMetadata(const ImageFormatGroupDescriptor& descriptor);

  const ImageFormatGroupDescriptor& get_descriptor() const;

  i32 num_images() const;
  i32 width() const;
  i32 height() const;
  ImageEncodingType encoding_type() const;
  ImageColorSpace color_space() const;
  std::vector<i64> compressed_sizes() const;

 private:
  mutable ImageFormatGroupDescriptor descriptor;
};

struct JobMetadata {
 public:
  JobMetadata();
  JobMetadata(const JobDescriptor& job);

  const JobDescriptor& get_job_descriptor() const;

  i32 id() const;

  std::string name() const;

  i32 io_item_size() const;

  i32 work_item_size() const;

  i32 num_nodes() const;

  const std::vector<std::string>& columns() const;

  i32 column_id(const std::string& column_name) const;

  const std::vector<std::string>& table_names() const;

  bool has_table(const std::string& table_name) const;

  i32 table_id(const std::string& table_name) const;

  i64 rows_in_table(i32 table_id) const;

  i64 total_rows() const;

 private:
  mutable JobDescriptor job_descriptor_;
  std::vector<std::string> columns_;
  std::map<std::string, i32> column_id_;
  std::vector<std::string> table_names_;
  std::map<std::string, i32> table_id_;
  std::map<i32, i64> rows_in_table_;
};

void serialize_database_metadata(storehouse::WriteFile* file,
                                 const DatabaseMetadata& metadata);

DatabaseMetadata deserialize_database_metadata(storehouse::RandomReadFile* file,
                                               u64& file_pos);

void serialize_dataset_descriptor(storehouse::WriteFile* file,
                                  const DatasetDescriptor& descriptor);

DatasetDescriptor deserialize_dataset_descriptor(
    storehouse::RandomReadFile* file, u64& file_pos);

void serialize_video_metadata(storehouse::WriteFile* file,
                              const VideoMetadata& metadata);

VideoMetadata deserialize_video_metadata(storehouse::RandomReadFile* file,
                                         u64& file_pos);

void serialize_image_format_group_metadata(
    storehouse::WriteFile* file, const ImageFormatGroupMetadata& metadata);

ImageFormatGroupMetadata deserialize_image_format_group_metadata(
    storehouse::RandomReadFile* file, u64& file_pos);

void serialize_web_timestamps(storehouse::WriteFile* file,
                              const WebTimestamps& metadata);

WebTimestamps deserialize_web_timestamps(storehouse::RandomReadFile* file,
                                         u64& file_pos);

void serialize_job_descriptor(storehouse::WriteFile* file,
                              const JobDescriptor& descriptor);

JobDescriptor deserialize_job_descriptor(storehouse::RandomReadFile* file,
                                         u64& file_pos);

///////////////////////////////////////////////////////////////////////////////
/// Path functions

extern std::string PREFIX;

void set_database_path(std::string path);

inline std::string database_metadata_path() {
  return PREFIX + "db_metadata.bin";
}

inline std::string dataset_directory(const std::string& dataset_name) {
  return PREFIX + "datasets/" + dataset_name;
}

inline std::string dataset_descriptor_path(const std::string& dataset_name) {
  return dataset_directory(dataset_name) + "/descriptor.bin";
}

inline std::string dataset_data_directory(const std::string& dataset_name) {
  return dataset_directory(dataset_name) + "/data";
}

inline std::string dataset_item_data_path(const std::string& dataset_name,
                                          const std::string& item_name) {
  return dataset_data_directory(dataset_name) + "/" + item_name + "_data.bin";
}

inline std::string dataset_item_video_path(const std::string& dataset_name,
                                           const std::string& item_name) {
  return dataset_data_directory(dataset_name) + "/" + item_name + ".mp4";
}

inline std::string dataset_item_video_timestamps_path(
    const std::string& dataset_name, const std::string& item_name) {
  return dataset_data_directory(dataset_name) + "/" + item_name +
         "_web_timestamps.bin";
}

inline std::string dataset_item_metadata_path(const std::string& dataset_name,
                                              const std::string& item_name) {
  return dataset_data_directory(dataset_name) + "/" + item_name +
         "_metadata.bin";
}

inline std::string job_directory(const std::string& dataset_name,
                                 const std::string& job_name) {
  return dataset_directory(dataset_name) + "/jobs/" + job_name;
}

inline std::string job_item_output_path(const std::string& dataset_name,
                                        const std::string& job_name,
                                        const std::string& video_name,
                                        const std::string& column_name,
                                        i32 work_item_index) {
  return job_directory(dataset_name, job_name) + "/" + video_name + "_" +
         column_name + "_" + std::to_string(work_item_index) + ".bin";
}

inline std::string job_descriptor_path(const std::string& dataset_name,
                                       const std::string& job_name) {
  return job_directory(dataset_name, job_name) + "/descriptor.bin";
}

inline std::string job_profiler_path(const std::string& dataset_name,
                                     const std::string& job_name, i32 node) {
  return job_directory(dataset_name, job_name) + "/profile_" +
         std::to_string(node) + ".bin";
}

inline i32 rows_per_io_item() { return IO_ITEM_SIZE; }

inline i32 rows_per_work_item() { return WORK_ITEM_SIZE; }

inline i32 base_job_id() { return 0; }

inline std::string base_job_name() { return "base"; }

inline i32 base_column_id() { return 0; }

inline std::string base_column_name() { return "frame"; }

inline std::string base_column_args_name() { return "frame_args"; }
}

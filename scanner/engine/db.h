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

#include "scanner/util/common.h"
#include "scanner/util/storehouse.h"
#include "storehouse/storage_backend.h"

#include <set>

namespace scanner {
namespace internal {

///////////////////////////////////////////////////////////////////////////////
/// Common persistent data structs and their serialization helpers

template<typename T>
struct Metadata {
public:
  using Descriptor = T;
  const Descriptor& get_descriptor() const;
  const std::string& descriptor_path() const;

protected:
  mutable Descriptor descriptor;
};

struct DatabaseMetadata : Metadata<proto::DatabaseDescriptor> {
 public:
  DatabaseMetadata();
  DatabaseMetadata(const Descriptor& descriptor);

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
};

struct VideoMetadata : Metadata<proto::VideoDescriptor> {
 public:
  VideoMetadata();
  VideoMetadata(const Descriptor& descriptor);

  i32 id() const;
  i32 frames() const;
  i32 width() const;
  i32 height() const;
  std::vector<i64> keyframe_positions() const;
  std::vector<i64> keyframe_byte_offsets() const;
};

struct ImageFormatGroupMetadata : Metadata<proto::VideoDescriptor> {
 public:
  ImageFormatGroupMetadata();
  ImageFormatGroupMetadata(const Descriptor& descriptor);

  i32 num_images() const;
  i32 width() const;
  i32 height() const;
  ImageEncodingType encoding_type() const;
  ImageColorSpace color_space() const;
  std::vector<i64> compressed_sizes() const;
};

struct JobMetadata : Metadata<proto::JobDescriptor> {
 public:
  JobMetadata();
  JobMetadata(const Descriptor& job);

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
  std::vector<std::string> columns_;
  std::map<std::string, i32> column_ids_;
  std::vector<std::string> table_names_;
  std::map<std::string, i32> table_ids_;
  mutable std::map<i32, i64> rows_in_table_;
};

struct TableMetadata : Metadata<proto::TableDescriptor> {
public:
  TableMetadata();
  TableMetadata(const Descriptor& table);
};

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
                                        i32 table_id,
                                        const std::string& column_name,
                                        i32 work_item_index) {
  return job_directory(dataset_name, job_name) + "/" +
         std::to_string(table_id) + "_" + column_name + "_" +
         std::to_string(work_item_index) + ".bin";
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

///////////////////////////////////////////////////////////////////////////////
/// Helpers

template <typename T>
void serialize_db_proto(storehouse::WriteFile* file, const T& descriptor) {
  int size = descriptor.ByteSize();
  std::vector<u8> data(size);
  descriptor.SerializeToArray(data.data(), size);
  s_write(file, data.data(), size);
}

template <typename T>
T deserialize_db_proto(storehouse::RandomReadFile* file, u64& pos) {
  T descriptor;
  std::vector<u8> data = storehouse::read_entire_file(file, pos);
  descriptor.ParseFromArray(data.data(), data.size());
  return descriptor;
}

template <typename T>
void write_db_proto(storehouse::StorageBackend *storage, T db_proto) {
  std::unique_ptr<storehouse::WriteFile> output_file;
  BACKOFF_FAIL(make_unique_write_file(storage, db_proto.descriptor_path(), output_file));
  serialize_db_proto<typename T::Descriptor>(output_file.get(), db_proto.get_descriptor());
  BACKOFF_FAIL(output_file->save());
}

template <typename T>
T read_db_proto(storehouse::StorageBackend* storage, const std::string& path) {
  std::unique_ptr<storehouse::RandomReadFile> db_in_file;
  BACKOFF_FAIL(
    make_unique_random_read_file(storage, path, db_in_file));
  u64 pos = 0;
  return T(deserialize_db_proto<typename T::Descriptor>(db_in_file.get(), pos));
}

template <typename T>
using WriteFn = void(*)(storehouse::StorageBackend *storage, T db_proto);

template <typename T>
using ReadFn = T(*)(storehouse::StorageBackend *storage, const std::string& path);

constexpr WriteFn<DatabaseMetadata> write_database_metadata =
  write_db_proto<DatabaseMetadata>;
constexpr ReadFn<DatabaseMetadata> read_database_metadata =
  read_db_proto<DatabaseMetadata>;

constexpr WriteFn<JobMetadata> write_job_metadata =
  write_db_proto<JobMetadata>;
constexpr ReadFn<JobMetadata> read_job_metadata =
  read_db_proto<JobMetadata>;

constexpr WriteFn<TableMetadata> write_table_metadata =
  write_db_proto<TableMetadata>;
constexpr ReadFn<TableMetadata> read_table_metadata =
  read_db_proto<TableMetadata>;

}
}

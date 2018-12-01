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
/// Path functions
const std::string& get_database_path();

void set_database_path(std::string path);

const std::string get_scanner_path();

void set_scanner_path(std::string path);

inline std::string database_metadata_path() {
  return get_database_path() + "db_metadata.bin";
}

inline std::string table_megafile_path() {
  return get_database_path() + "table_megafile.bin";
}

inline std::string table_directory(i32 table_id) {
  return get_database_path() + "tables/" + std::to_string(table_id);
}

inline std::string table_descriptor_path(i32 table_id) {
  return table_directory(table_id) + "/descriptor.bin";
}

inline std::string table_item_output_path(i32 table_id, i32 column_id,
                                          i32 item_id) {
  return table_directory(table_id) + "/" + std::to_string(column_id) + "_" +
         std::to_string(item_id) + ".bin";
}

inline std::string table_item_video_metadata_path(i32 table_id, i32 column_id,
                                                  i32 item_id) {
  return table_directory(table_id) + "/" + std::to_string(column_id) + "_" +
         std::to_string(item_id) + "_video_metadata.bin";
}

inline std::string table_item_metadata_path(i32 table_id, i32 column_id,
                                            i32 item_id) {
  return table_directory(table_id) + "/" + std::to_string(column_id) + "_" +
         std::to_string(item_id) + "_metadata.bin";
}

inline std::string bulk_job_directory(i32 bulk_job_id) {
  return get_database_path() + "jobs/" + std::to_string(bulk_job_id);
}

inline std::string bulk_job_descriptor_path(i32 bulk_job_id) {
  return bulk_job_directory(bulk_job_id) + "/descriptor.bin";
}

inline std::string bulk_job_master_profiler_path(i32 bulk_job_id) {
  return bulk_job_directory(bulk_job_id) + "/profile_master.bin";
}

inline std::string bulk_job_worker_profiler_path(i32 bulk_job_id, i32 node) {
  return bulk_job_directory(bulk_job_id) + "/profile_" + std::to_string(node) +
         ".bin";
}

inline std::string eval_map_path(i32 bulk_job_id, i64 task_id) {
  return bulk_job_directory(bulk_job_id) + "/eval_map_" + std::to_string(task_id) +
         ".bin";
}

///////////////////////////////////////////////////////////////////////////////
/// Common persistent data structs and their serialization helpers

template <typename T>
class Metadata {
 public:
  using Descriptor = T;
  Metadata() {}
  Metadata(const Descriptor& d) : descriptor_(d) {}

  Descriptor& get_descriptor() const { return descriptor_; }

  std::string descriptor_path() const;

 protected:
  mutable Descriptor descriptor_;
};

class DatabaseMetadata : public Metadata<proto::DatabaseDescriptor> {
 public:
  DatabaseMetadata();
  DatabaseMetadata(const Descriptor& descriptor);

  const Descriptor& get_descriptor() const;

  static std::string descriptor_path();

  std::vector<std::string> table_names() const;

  bool has_table(const std::string& table) const;
  bool has_table(i32 table_id) const;
  i32 get_table_id(const std::string& table) const;
  const std::string& get_table_name(i32 table_id) const;
  i32 add_table(const std::string& table);
  void commit_table(i32 table_id);
  bool table_is_committed(i32 table_id) const;
  void remove_table(i32 table_id);

  const std::vector<std::string>& bulk_job_names() const;

  bool has_bulk_job(const std::string& job) const;
  bool has_bulk_job(i32 job_id) const;
  i32 get_bulk_job_id(const std::string& job_name) const;
  const std::string& get_bulk_job_name(i32 job_id) const;
  i32 add_bulk_job(const std::string& job_name);
  void commit_bulk_job(i32 job_id);
  bool bulk_job_is_committed(i32 job_id) const;
  void remove_bulk_job(i32 job_id);

 private:
  i32 next_table_id_;
  i32 next_bulk_job_id_;
  std::unordered_map<i32, std::string> table_id_names_;
  std::unordered_map<std::string, i32> table_name_ids_;
  std::unordered_map<i32, bool> table_committed_;

  std::unordered_map<i32, std::string> bulk_job_id_names_;
  std::unordered_map<i32, bool> bulk_job_committed_;
};

class VideoMetadata : public Metadata<proto::VideoDescriptor> {
 public:
  VideoMetadata();
  VideoMetadata(const Descriptor& descriptor);

  static std::string descriptor_path(i32 table_id, i32 column_id, i32 item_id);

  i32 table_id() const;
  i32 column_id() const;
  i32 item_id() const;
  i32 frames() const;
  i32 width() const;
  i32 height() const;
  i32 channels() const;
  proto::FrameType frame_type() const;
  proto::VideoDescriptor::VideoCodecType codec_type() const;
  i64 num_encoded_videos() const;
  std::vector<i64> frames_per_video() const;
  std::vector<i64> keyframes_per_video() const;
  std::vector<i64> size_per_video() const;

  std::vector<u64> keyframe_indices() const;
  std::vector<u64> sample_offsets() const;
  std::vector<u64> sample_sizes() const;
  std::vector<u8> metadata() const;
  std::string data_path() const;
  bool inplace() const;
};

class ImageFormatGroupMetadata
    : public Metadata<proto::ImageFormatGroupDescriptor> {
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

class BulkJobMetadata : public Metadata<proto::BulkJobDescriptor> {
 public:
  BulkJobMetadata();
  BulkJobMetadata(const Descriptor& job);

  static std::string descriptor_path(i32 job_id);

  i32 id() const;

  std::string name() const;

  i32 io_packet_size() const;

  i32 work_packet_size() const;

  i32 num_nodes() const;

  const std::vector<proto::Column>& columns() const;

  i32 column_id(const std::string& column_name) const;

  const std::vector<std::string>& table_names() const;

  bool has_table(const std::string& name) const;

  // i64 rows_in_table(const std::string& name) const;

  // i64 total_rows() const;

 private:
  std::vector<Column> columns_;
  std::unordered_map<std::string, i32> column_ids_;
  std::vector<std::string> table_names_;
  mutable std::unordered_map<std::string, i64> rows_in_table_;
};

class TableMetadata : public Metadata<proto::TableDescriptor> {
 public:
  TableMetadata();
  TableMetadata(const Descriptor& table);

  static std::string descriptor_path(i32 table_id);

  i32 id() const;

  std::string name() const;

  i64 num_rows() const;

  std::vector<i64> end_rows() const;

  const std::vector<proto::Column>& columns() const;

  bool has_column(const std::string& name) const;

  std::string column_name(i32 column_id) const;

  i32 column_id(const std::string& name) const;

  ColumnType column_type(i32 column_id) const;

 private:
  std::vector<proto::Column> columns_;
};

///////////////////////////////////////////////////////////////////////////////
/// Constants

inline std::string index_column_name() { return "index"; }

inline std::string frame_column_name() { return "frame"; }

inline std::string frame_info_column_name() { return "frame_info"; }

///////////////////////////////////////////////////////////////////////////////
/// Helpers

template <typename T>
void serialize_db_proto(storehouse::WriteFile* file, const T& descriptor) {
  size_t size = descriptor.ByteSizeLong();
  std::vector<u8> data(size);
  descriptor.SerializeToArray(data.data(), size);
  s_write(file, data.data(), size);
}

template <typename T>
T deserialize_db_proto(storehouse::RandomReadFile* file, u64& pos) {
  T descriptor;
  uint64_t size;
  BACKOFF_FAIL(file->get_size(size),
               "while trying to get size for " + file->path());
  std::vector<u8> data = storehouse::read_entire_file(
      file, pos, std::max((size_t)size, (size_t)1024 * 1024));
  descriptor.ParseFromArray(data.data(), data.size());
  return descriptor;
}

template <typename T>
void write_db_proto(storehouse::StorageBackend* storage, T db_proto) {
  std::unique_ptr<storehouse::WriteFile> output_file;
  const std::string& desc_path =
      db_proto.Metadata<typename T::Descriptor>::descriptor_path();
  BACKOFF_FAIL(make_unique_write_file(storage, desc_path, output_file),
               "while trying to make write file for " + desc_path);
  serialize_db_proto<typename T::Descriptor>(output_file.get(),
                                             db_proto.get_descriptor());
  BACKOFF_FAIL(output_file->save(),
               "while trying to save " + output_file->path());
}

template <typename T>
T read_db_proto(storehouse::StorageBackend* storage, const std::string& path) {
  std::unique_ptr<storehouse::RandomReadFile> db_in_file;
  BACKOFF_FAIL(make_unique_random_read_file(storage, path, db_in_file),
               "while trying to make read file for " + path);
  u64 pos = 0;
  return T(deserialize_db_proto<typename T::Descriptor>(db_in_file.get(), pos));
}

template <typename T>
using WriteFn = void (*)(storehouse::StorageBackend* storage, T db_proto);

template <typename T>
using ReadFn = T (*)(storehouse::StorageBackend* storage,
                     const std::string& path);

constexpr WriteFn<DatabaseMetadata> write_database_metadata =
    write_db_proto<DatabaseMetadata>;
constexpr ReadFn<DatabaseMetadata> read_database_metadata =
    read_db_proto<DatabaseMetadata>;

void write_table_megafile(
    storehouse::StorageBackend* storage,
    const std::unordered_map<i32, TableMetadata>& table_metadata);

void read_table_megafile(
    storehouse::StorageBackend* storage,
    std::unordered_map<i32, TableMetadata>& table_metadata);

constexpr WriteFn<BulkJobMetadata> write_bulk_job_metadata =
    write_db_proto<BulkJobMetadata>;
constexpr ReadFn<BulkJobMetadata> read_bulk_job_metadata =
    read_db_proto<BulkJobMetadata>;

constexpr WriteFn<TableMetadata> write_table_metadata =
    write_db_proto<TableMetadata>;
constexpr ReadFn<TableMetadata> read_table_metadata =
    read_db_proto<TableMetadata>;

constexpr WriteFn<VideoMetadata> write_video_metadata =
    write_db_proto<VideoMetadata>;
constexpr ReadFn<VideoMetadata> read_video_metadata =
    read_db_proto<VideoMetadata>;
}
}

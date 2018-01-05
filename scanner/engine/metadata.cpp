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

#include "scanner/engine/metadata.h"
#include "scanner/engine/runtime.h"
#include "scanner/util/storehouse.h"
#include "scanner/util/util.h"
#include "storehouse/storage_backend.h"

#include <errno.h>
#include <libgen.h>
#include <limits.h> /* PATH_MAX */
#include <string.h>
#include <sys/stat.h> /* mkdir(2) */
#include <cassert>
#include <cstdarg>
#include <iostream>
#include <sstream>

using storehouse::WriteFile;
using storehouse::RandomReadFile;
using storehouse::StoreResult;

namespace scanner {
using namespace proto;

namespace internal {

template <>
std::string Metadata<DatabaseDescriptor>::descriptor_path() const {
  const DatabaseMetadata* meta = (const DatabaseMetadata*)this;
  return database_metadata_path();
}

template <>
std::string Metadata<VideoDescriptor>::descriptor_path() const {
  const VideoMetadata* meta = (const VideoMetadata*)this;
  return table_item_video_metadata_path(meta->table_id(), meta->column_id(),
                                        meta->item_id());
}

template <>
std::string Metadata<BulkJobDescriptor>::descriptor_path() const {
  const BulkJobMetadata* meta = (const BulkJobMetadata*)this;
  return bulk_job_descriptor_path(meta->id());
}

template <>
std::string Metadata<TableDescriptor>::descriptor_path() const {
  const TableMetadata* meta = (const TableMetadata*)this;
  return table_descriptor_path(meta->id());
}

DatabaseMetadata::DatabaseMetadata() : next_table_id_(0), next_bulk_job_id_(0) {}

DatabaseMetadata::DatabaseMetadata(const DatabaseDescriptor& d)
  : Metadata(d),
    next_table_id_(d.next_table_id()),
    next_bulk_job_id_(d.next_bulk_job_id()) {
  for (int i = 0; i < descriptor_.tables_size(); ++i) {
    const DatabaseDescriptor::Table& table = descriptor_.tables(i);
    table_id_names_.insert({table.id(), table.name()});
    table_committed_.insert({table.id(), table.committed()});
  }
  for (int i = 0; i < descriptor_.bulk_jobs_size(); ++i) {
    const DatabaseDescriptor_BulkJob& bulk_job = descriptor_.bulk_jobs(i);
    bulk_job_id_names_.insert({bulk_job.id(), bulk_job.name()});
    bulk_job_committed_.insert({bulk_job.id(), bulk_job.committed()});
  }
}

const DatabaseDescriptor& DatabaseMetadata::get_descriptor() const {
  descriptor_.set_next_table_id(next_table_id_);
  descriptor_.set_next_bulk_job_id(next_bulk_job_id_);
  descriptor_.clear_tables();
  descriptor_.clear_bulk_jobs();

  for (auto& kv : table_id_names_) {
    auto table = descriptor_.add_tables();
    table->set_id(kv.first);
    table->set_name(kv.second);
    table->set_committed(table_committed_.at(kv.first));
  }

  for (auto& kv : bulk_job_id_names_) {
    auto bulk_job = descriptor_.add_bulk_jobs();
    bulk_job->set_id(kv.first);
    bulk_job->set_name(kv.second);
    bulk_job->set_committed(bulk_job_committed_.at(kv.first));
  }

  return descriptor_;
}

std::string DatabaseMetadata::descriptor_path() {
  return database_metadata_path();
}

std::vector<std::string> DatabaseMetadata::table_names() const {
  std::vector<std::string> names;
  for (auto& entry : table_id_names_) {
    names.push_back(entry.second);
  }
  return names;
}

bool DatabaseMetadata::has_table(const std::string& table) const {
  for (const auto& kv : table_id_names_) {
    if (kv.second == table) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_table(i32 table_id) const {
  return table_id_names_.count(table_id) > 0;
}

i32 DatabaseMetadata::get_table_id(const std::string& table) const {
  i32 id = -1;
  for (const auto& kv : table_id_names_) {
    if (kv.second == table) {
      id = kv.first;
      break;
    }
  }
  LOG_IF(WARNING, id == -1) << "Table " << table << " does not exist.";
  return id;
}

const std::string& DatabaseMetadata::get_table_name(i32 table_id) const {
  return table_id_names_.at(table_id);
}

i32 DatabaseMetadata::add_table(const std::string& table) {
  i32 table_id = -1;
  if (!has_table(table)) {
    table_id = next_table_id_++;
    table_id_names_[table_id] = table;
    table_committed_[table_id] = false;
  }
  return table_id;
}

void DatabaseMetadata::commit_table(i32 table_id) {
  assert(table_id_names_.count(table_id) > 0);
  table_committed_[table_id] = true;
}

bool DatabaseMetadata::table_is_committed(i32 table_id) const {
  assert(table_id_names_.count(table_id) > 0);
  return table_committed_.at(table_id);
}

void DatabaseMetadata::remove_table(i32 table_id) {
  assert(table_id_names_.count(table_id) > 0);
  table_id_names_.erase(table_id);
}

const std::vector<std::string>& DatabaseMetadata::bulk_job_names() const {
  std::vector<std::string> names;
  for (auto& entry : bulk_job_id_names_) {
    names.push_back(entry.second);
  }
  return names;
}

bool DatabaseMetadata::has_bulk_job(const std::string& bulk_job) const {
  for (const auto& kv : bulk_job_id_names_) {
    if (kv.second == bulk_job) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_bulk_job(i32 bulk_job_id) const {
  return bulk_job_id_names_.count(bulk_job_id) > 0;
}

i32 DatabaseMetadata::get_bulk_job_id(const std::string& bulk_job) const {
  i32 bulk_job_id = -1;
  for (const auto& kv : bulk_job_id_names_) {
    if (kv.second == bulk_job) {
      bulk_job_id = kv.first;
      break;
    }
  }
  assert(bulk_job_id != -1);
  return bulk_job_id;
}

const std::string& DatabaseMetadata::get_bulk_job_name(i32 bulk_job_id) const {
  return bulk_job_id_names_.at(bulk_job_id);
}

i32 DatabaseMetadata::add_bulk_job(const std::string& bulk_job_name) {
  i32 bulk_job_id = next_bulk_job_id_++;
  bulk_job_id_names_[bulk_job_id] = bulk_job_name;
  bulk_job_committed_[bulk_job_id] = false;
  return bulk_job_id;
}

void DatabaseMetadata::commit_bulk_job(i32 bulk_job_id) {
  assert(bulk_job_id_names_.count(bulk_job_id) > 0);
  bulk_job_committed_[bulk_job_id] = true;

}

bool DatabaseMetadata::bulk_job_is_committed(i32 bulk_job_id) const {
  assert(bulk_job_id_names_.count(bulk_job_id) > 0);
  return bulk_job_committed_.at(bulk_job_id);
}

void DatabaseMetadata::remove_bulk_job(i32 bulk_job_id) {
  assert(bulk_job_id_names_.count(bulk_job_id) > 0);
  bulk_job_id_names_.erase(bulk_job_id);
}

///////////////////////////////////////////////////////////////////////////////
/// VideoMetdata
VideoMetadata::VideoMetadata() {}

VideoMetadata::VideoMetadata(const VideoDescriptor& descriptor)
  : Metadata(descriptor) {}

std::string VideoMetadata::descriptor_path(i32 table_id, i32 column_id,
                                           i32 item_id) {
  return table_item_video_metadata_path(table_id, column_id, item_id);
}

i32 VideoMetadata::table_id() const { return descriptor_.table_id(); }

i32 VideoMetadata::column_id() const { return descriptor_.column_id(); }

i32 VideoMetadata::item_id() const { return descriptor_.item_id(); }

i32 VideoMetadata::frames() const { return descriptor_.frames(); }

i32 VideoMetadata::width() const { return descriptor_.width(); }

i32 VideoMetadata::height() const { return descriptor_.height(); }

i32 VideoMetadata::channels() const { return descriptor_.channels(); }

FrameType VideoMetadata::frame_type() const { return descriptor_.frame_type(); }

VideoDescriptor::VideoCodecType VideoMetadata::codec_type() const {
  return descriptor_.codec_type();
}

i64 VideoMetadata::num_encoded_videos() const {
  return descriptor_.num_encoded_videos();
}

std::vector<i64> VideoMetadata::frames_per_video() const {
  return std::vector<i64>(descriptor_.frames_per_video().begin(),
                          descriptor_.frames_per_video().end());
}

std::vector<i64> VideoMetadata::keyframes_per_video() const {
  return std::vector<i64>(descriptor_.keyframes_per_video().begin(),
                          descriptor_.keyframes_per_video().end());
}

std::vector<i64> VideoMetadata::size_per_video() const {
  return std::vector<i64>(descriptor_.size_per_video().begin(),
                          descriptor_.size_per_video().end());
}

std::vector<u64> VideoMetadata::keyframe_indices() const {
  return std::vector<u64>(descriptor_.keyframe_indices().begin(),
                          descriptor_.keyframe_indices().end());
}

std::vector<u64> VideoMetadata::sample_offsets() const {
  return std::vector<u64>(descriptor_.sample_offsets().begin(),
                          descriptor_.sample_offsets().end());
}

std::vector<u64> VideoMetadata::sample_sizes() const {
  return std::vector<u64>(descriptor_.sample_sizes().begin(),
                          descriptor_.sample_sizes().end());
}

std::vector<u8> VideoMetadata::metadata() const {
  return std::vector<u8>(descriptor_.metadata_packets().begin(),
                         descriptor_.metadata_packets().end());
}

std::string VideoMetadata::data_path() const {
  return descriptor_.data_path();
}

bool VideoMetadata::inplace() const {
  return descriptor_.inplace();
}

///////////////////////////////////////////////////////////////////////////////
/// ImageFormatGroupMetadata
ImageFormatGroupMetadata::ImageFormatGroupMetadata() {}

ImageFormatGroupMetadata::ImageFormatGroupMetadata(
    const ImageFormatGroupDescriptor& descriptor)
  : Metadata(descriptor) {}

i32 ImageFormatGroupMetadata::num_images() const {
  return descriptor_.num_images();
}

i32 ImageFormatGroupMetadata::width() const { return descriptor_.width(); }

i32 ImageFormatGroupMetadata::height() const { return descriptor_.height(); }

ImageEncodingType ImageFormatGroupMetadata::encoding_type() const {
  return descriptor_.encoding_type();
}

ImageColorSpace ImageFormatGroupMetadata::color_space() const {
  return descriptor_.color_space();
}

std::vector<i64> ImageFormatGroupMetadata::compressed_sizes() const {
  return std::vector<i64>(descriptor_.compressed_sizes().begin(),
                          descriptor_.compressed_sizes().end());
}

///////////////////////////////////////////////////////////////////////////////
/// BulkJobMetadata
BulkJobMetadata::BulkJobMetadata() {}
BulkJobMetadata::BulkJobMetadata(const BulkJobDescriptor& job) : Metadata(job) {
  for (auto& t : descriptor_.jobs()) {
    table_names_.push_back(t.output_table_name());
  }
}

std::string BulkJobMetadata::descriptor_path(i32 job_id) {
  return bulk_job_descriptor_path(job_id);
}

i32 BulkJobMetadata::id() const { return descriptor_.id(); }

std::string BulkJobMetadata::name() const { return descriptor_.name(); }

i32 BulkJobMetadata::io_packet_size() const {
  return descriptor_.io_packet_size();
}

i32 BulkJobMetadata::work_packet_size() const {
  return descriptor_.work_packet_size();
}

i32 BulkJobMetadata::num_nodes() const { return descriptor_.num_nodes(); }

const std::vector<std::string>& BulkJobMetadata::table_names() const {
  return table_names_;
}

bool BulkJobMetadata::has_table(const std::string& name) const {
  for (const std::string& n : table_names_) {
    if (n == name) {
      return true;
    }
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////
/// TableMetadata
TableMetadata::TableMetadata() {}
TableMetadata::TableMetadata(const TableDescriptor& table) : Metadata(table) {
  for (auto& c : descriptor_.columns()) {
    columns_.push_back(c);
  }
}

std::string TableMetadata::descriptor_path(i32 table_id) {
  return table_descriptor_path(table_id);
}

i32 TableMetadata::id() const { return descriptor_.id(); }

std::string TableMetadata::name() const { return descriptor_.name(); }

i64 TableMetadata::num_rows() const {
  return descriptor_.end_rows(descriptor_.end_rows_size() - 1);
}

std::vector<i64> TableMetadata::end_rows() const {
  return std::vector<i64>(descriptor_.end_rows().begin(),
                          descriptor_.end_rows().end());
}

const std::vector<Column>& TableMetadata::columns() const { return columns_; }

bool TableMetadata::has_column(const std::string& name) const {
  for (auto& c : descriptor_.columns()) {
    if (c.name() == name) {
      return true;
    }
  }
  return false;
}

std::string TableMetadata::column_name(i32 column_id) const {
  for (auto& c : descriptor_.columns()) {
    if (c.id() == column_id) {
      return c.name();
    }
  }
  LOG(FATAL) << "Column id " << column_id << " not found!";
}

i32 TableMetadata::column_id(const std::string& column_name) const {
  for (auto& c : descriptor_.columns()) {
    if (c.name() == column_name) {
      return c.id();
    }
  }
  LOG(FATAL) << "Column name " << column_name << " not found!";
}

ColumnType TableMetadata::column_type(i32 column_id) const {
  for (auto& c : descriptor_.columns()) {
    if (c.id() == column_id) {
      return c.type();
    }
  }
  LOG(FATAL) << "Column id " << column_id << " not found!";
}

namespace {
std::string& get_database_path_ref() {
  static std::string prefix = "";
  return prefix;
}
}

const std::string& get_database_path() {
  std::atomic_thread_fence(std::memory_order_acquire);
  return get_database_path_ref();
}

void set_database_path(std::string path) {
  VLOG(1) << "Setting DB path to " << path;
  get_database_path_ref() = path + "/";
  std::atomic_thread_fence(std::memory_order_release);
}

void write_table_megafile(
    storehouse::StorageBackend* storage,
    const std::map<i32, TableMetadata>& table_metadata) {
  std::unique_ptr<WriteFile> output_file;
  BACKOFF_FAIL(make_unique_write_file(storage, table_megafile_path(),
                                      output_file));
  // Get all table descriptor sizes and write them
  std::vector<i32> ids;
  for (const auto& kv : table_metadata) {
    ids.push_back(kv.first);
  }
  std::sort(ids.begin(), ids.end());
  std::vector<size_t> sizes;
  for (i32 id : ids) {
    const auto& td = table_metadata.at(id).get_descriptor();
    size_t size = td.ByteSizeLong();
    sizes.push_back(size);
  }
  // Write out # table entries
  s_write(output_file.get(), (size_t)ids.size());

  // Write out ids and sizes
  s_write(output_file.get(), (u8*)ids.data(), ids.size() * sizeof(i32));
  s_write(output_file.get(), (u8*)sizes.data(), sizes.size() * sizeof(size_t));
  // Write all table descriptors
  size_t BATCH_SIZE = 10000;
  for (size_t b = 0; b < ids.size(); b += BATCH_SIZE) {
    size_t max_i = std::min(b + BATCH_SIZE, ids.size());

    size_t total_size = 0;
    for (size_t i = b; i < max_i; ++i) {
      total_size += sizes[i];
    }
    std::vector<u8> data(total_size);
    size_t offset = 0;
    for (size_t i = b; i < max_i; ++i) {
      i32 id = ids[i];
      const auto& td = table_metadata.at(id).get_descriptor();
      td.SerializeToArray(data.data() + offset, sizes[i]);
      offset += sizes[i];
    }
    s_write(output_file.get(), data.data(), total_size);
  }
}

void read_table_megafile(storehouse::StorageBackend* storage,
                         std::map<i32, TableMetadata>& table_metadata) {
  std::unique_ptr<RandomReadFile> file;
  BACKOFF_FAIL(make_unique_random_read_file(storage,
                                            table_megafile_path(), file));

  u64 file_size = 0;
  BACKOFF_FAIL(file->get_size(file_size));
  u64 pos = 0;

  // Read # entires
  size_t num_entries = s_read<size_t>(file.get(), pos);
  // Read ids
  std::vector<i32> ids(num_entries);
  s_read(file.get(), (u8*)ids.data(), num_entries * sizeof(i32), pos);
  // Read sizes
  std::vector<size_t> sizes(num_entries);
  s_read(file.get(), (u8*)sizes.data(), num_entries * sizeof(size_t), pos);

  // Read all table descriptors
  size_t BATCH_SIZE = 10000;
  for (size_t b = 0; b < ids.size(); b += BATCH_SIZE) {
    size_t max_i = std::min(b + BATCH_SIZE, ids.size());

    size_t total_size = 0;
    for (size_t i = b; i < max_i; ++i) {
      total_size += sizes[i];
    }
    std::vector<u8> data(total_size);
    s_read(file.get(), data.data(), total_size, pos);

    size_t offset = 0;
    for (size_t i = b; i < max_i; ++i) {
      i32 id = ids[i];
      proto::TableDescriptor td;
      td.ParseFromArray(data.data() + offset, sizes[i]);
      table_metadata[id] = TableMetadata(td);
      offset += sizes[i];
    }
  }
}

}
}

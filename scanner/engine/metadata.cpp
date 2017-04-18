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
std::string Metadata<JobDescriptor>::descriptor_path() const {
  const JobMetadata* meta = (const JobMetadata*)this;
  return job_descriptor_path(meta->id());
}

template <>
std::string Metadata<TableDescriptor>::descriptor_path() const {
  const TableMetadata* meta = (const TableMetadata*)this;
  return table_descriptor_path(meta->id());
}

DatabaseMetadata::DatabaseMetadata() : next_table_id_(0), next_job_id_(0) {}

DatabaseMetadata::DatabaseMetadata(const DatabaseDescriptor& d)
  : Metadata(d),
    next_table_id_(d.next_table_id()),
    next_job_id_(d.next_job_id()) {
  for (int i = 0; i < descriptor_.tables_size(); ++i) {
    const DatabaseDescriptor::Table& table = descriptor_.tables(i);
    table_id_names_.insert({table.id(), table.name()});
  }
  for (int i = 0; i < descriptor_.jobs_size(); ++i) {
    const DatabaseDescriptor_Job& job = descriptor_.jobs(i);
    job_id_names_.insert({job.id(), job.name()});
    job_names_.push_back(job.name());
  }
}

const DatabaseDescriptor& DatabaseMetadata::get_descriptor() const {
  descriptor_.set_next_table_id(next_table_id_);
  descriptor_.set_next_job_id(next_job_id_);
  descriptor_.clear_tables();
  descriptor_.clear_jobs();

  for (auto& kv : table_id_names_) {
    auto table = descriptor_.add_tables();
    table->set_id(kv.first);
    table->set_name(kv.second);
  }

  for (auto& kv : job_id_names_) {
    auto job = descriptor_.add_jobs();
    job->set_id(kv.first);
    job->set_name(kv.second);
  }

  return descriptor_;
}

std::string DatabaseMetadata::descriptor_path() {
  return database_metadata_path();
}

const std::vector<std::string> DatabaseMetadata::table_names() const {
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
  }
  return table_id;
}

void DatabaseMetadata::remove_table(i32 table_id) {
  assert(table_id_names_.count(table_id) > 0);
  table_id_names_.erase(table_id);
}

const std::vector<std::string>& DatabaseMetadata::job_names() const {
  return job_names_;
}

bool DatabaseMetadata::has_job(const std::string& job) const {
  for (const auto& kv : job_id_names_) {
    if (kv.second == job) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_job(i32 job_id) const {
  return job_id_names_.count(job_id) > 0;
}

i32 DatabaseMetadata::get_job_id(const std::string& job) const {
  i32 job_id = -1;
  for (const auto& kv : job_id_names_) {
    if (kv.second == job) {
      job_id = kv.first;
      break;
    }
  }
  assert(job_id != -1);
  return job_id;
}

const std::string& DatabaseMetadata::get_job_name(i32 job_id) const {
  return job_id_names_.at(job_id);
}

i32 DatabaseMetadata::add_job(const std::string& job_name) {
  i32 job_id = next_job_id_++;
  job_id_names_[job_id] = job_name;
  return job_id;
}

void DatabaseMetadata::remove_job(i32 job_id) {
  assert(job_id_names_.count(job_id) > 0);
  job_id_names_.erase(job_id);
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

std::vector<i64> VideoMetadata::keyframe_positions() const {
  return std::vector<i64>(descriptor_.keyframe_positions().begin(),
                          descriptor_.keyframe_positions().end());
}

std::vector<i64> VideoMetadata::keyframe_byte_offsets() const {
  return std::vector<i64>(descriptor_.keyframe_byte_offsets().begin(),
                          descriptor_.keyframe_byte_offsets().end());
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
/// JobMetadata
JobMetadata::JobMetadata() {}
JobMetadata::JobMetadata(const JobDescriptor& job) : Metadata(job) {
  for (auto& c : descriptor_.columns()) {
    columns_.push_back(c);
    column_ids_.insert({c.name(), c.id()});
  }
  for (auto& t : descriptor_.tasks()) {
    table_names_.push_back(t.output_table_name());
  }
}

std::string JobMetadata::descriptor_path(i32 job_id) {
  return job_descriptor_path(job_id);
}

i32 JobMetadata::id() const { return descriptor_.id(); }

std::string JobMetadata::name() const { return descriptor_.name(); }

i32 JobMetadata::work_item_size() const { return descriptor_.work_item_size(); }

i32 JobMetadata::num_nodes() const { return descriptor_.num_nodes(); }

const std::vector<Column>& JobMetadata::columns() const { return columns_; }

i32 JobMetadata::column_id(const std::string& column_name) const {
  column_ids_.at(column_name);
}

const std::vector<std::string>& JobMetadata::table_names() const {
  return table_names_;
}

bool JobMetadata::has_table(const std::string& name) const {
  for (const std::string& n : table_names_) {
    if (n == name) {
      return true;
    }
  }
  return false;
}

// i64 JobMetadata::rows_in_table(const std::string &name) const {
//   i64 rows = -1;
//   auto it = rows_in_table_.find(name);
//   if (it == rows_in_table_.end()) {
//     for (const proto::Task &task : descriptor_.tasks()) {
//       assert(task.samples_size() > 0);
//       const proto::TableSample &sample = task.samples(0);
//       rows = sample.rows_size();
//       rows_in_table_.insert(std::make_pair(name, rows));
//     }
//   } else {
//     rows = it->second;
//   }
//   assert(rows != -1);
//   return rows;
// }

// i64 JobMetadata::total_rows() const {
//   i64 rows = 0;
//   for (const proto::Task &task : descriptor_.tasks()) {
//     assert(task.samples_size() > 0);
//     const proto::TableSample &sample = task.samples(0);
//     rows += sample.rows_size();
//   }
//   return rows;
// }

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
}
}

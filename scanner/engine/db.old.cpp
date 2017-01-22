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

#include "scanner/engine/db.h"
#include "scanner/engine/runtime.h"
#include "scanner/util/jsoncpp.h"
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
namespace internal {

using namespace proto;

DatabaseMetadata::DatabaseMetadata() : next_job_id(0) {}

DatabaseMetadata::DatabaseMetadata(const DatabaseDescriptor& d)
    : descriptor_(d),
      next_dataset_id_(d.next_dataset_id()),
      next_job_id_(d.next_job_id()) {
  for (int i = 0; i < descriptor.tables_size(); ++i) {
    const DatabaseDescriptor::Table& table = descriptor.tables(i);
    table_names.insert({dataset.id(), dataset.name()});
  }
  for (int i = 0; i < descriptor.jobs_size(); ++i) {
    const DatabaseDescriptor_Job& job = descriptor.jobs(i);
    job_names.insert({job.id(), job.name()});
  }
}

const DatabaseDescriptor& DatabaseMetadata::get_descriptor() const {
  descriptor.set_next_table_id(next_table_id);
  descriptor.set_next_job_id(next_job_id);
  descriptor.clear_tables();
  descriptor.clear_jobs();

  for (auto& kv : table_names) {
    auto table = descriptor.add_table();
    dataset->set_id(kv.first);
    dataset->set_name(kv.second);
  }

  for (auto& kv : job_names) {
    auto job = descriptor.add_jobs();
    job->set_id(kv.first);
    job->set_name(kv.second);
  }

  return descriptor;
}

std::string DatabaseMetadata::descriptor_path() const {
  return database_metadata_path();
}

bool DatabaseMetadata::has_table(const std::string& table) const {
  for (const auto& kv : table_names) {
    if (kv.second == table) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_table(i32 table_id) const {
  return table_names.count(table_id) > 0;
}

i32 DatabaseMetadata::get_table_id(const std::string& table) const {
  i32 id = -1;
  for (const auto& kv : table_names) {
    if (kv.second == table) {
      id = kv.first;
      break;
    }
  }
  assert(id != -1);
  return id;
}

const std::string& DatabaseMetadata::get_table_name(i32 table_id) const {
  return table_names.at(table_id);
}

i32 DatabaseMetadata::add_table(const std::string& table) {
  i32 table_id = next_table_id++;
  table_names[table_id] = table;
  return table_id;
}

void DatabaseMetadata::remove_table(i32 table_id) {
  assert(table_names.count(table_id) > 0);
  table_names.erase(table_id);
}

bool DatabaseMetadata::has_job(const std::string& job) const {
  for (const auto& kv : job_names) {
    if (kv.second == job) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_job(i32 job_id) const {
  return job_names.count(job_id) > 0;
}

i32 DatabaseMetadata::get_job_id(const std::string& job) const {
  i32 job_id = -1;
  for (const auto& kv : job_names) {
    if (kv.second == job) {
      job_id = kv.first;
      break;
    }
  }
  assert(job_id != -1);
  return job_id;
}

const std::string& DatabaseMetadata::get_job_name(i32 job_id) const {
  return job_names.at(job_id);
}

i32 DatabaseMetadata::add_job(const std::string& job_name) {
  i32 job_id = next_job_id++;
  job_names[job_id] = job_name;
  return job_id;
}

void DatabaseMetadata::remove_job(i32 job_id) {
  assert(job_names.count(job_id) > 0);
  job_names.erase(job_id);
}

///////////////////////////////////////////////////////////////////////////////
/// VideoMetdata
VideoMetadata::VideoMetadata() {}
<<<<<<< Updated upstream

VideoMetadata::VideoMetadata(const VideoDescriptor& descriptor)
    : descriptor(descriptor) {}

std::string VideoDescriptor::descriptor_path(i32 table_id, i32 column_id,
                                             i32 item_id) {
  return table_item_video_metadata_path(table_id, column_id, item_id);
}

const VideoDescriptor& VideoMetadata::get_descriptor() const {
  return descriptor;
}

const VideoDescriptor& VideoMetadata::descriptor_path() const {
  return table_item_video_metadata_path(table_id(), column_id(), item_id());
}

i32 VideoMetadata::table_id() const { return descriptor.table_id(); }

i32 VideoMetadata::column_id() const { return descriptor.column_id(); }
=======
>>>>>>> Stashed changes

i32 VideoMetadata::item_id() const { return descriptor.item_id(); }

i32 VideoMetadata::frames() const { return descriptor.frames(); }

i32 VideoMetadata::width() const { return descriptor.width(); }

i32 VideoMetadata::height() const { return descriptor.height(); }

std::vector<i64> VideoMetadata::keyframe_positions() const {
  return std::vector<i64>(descriptor.keyframe_positions().begin(),
                          descriptor.keyframe_positions().end());
}

std::vector<i64> VideoMetadata::keyframe_byte_offsets() const {
  return std::vector<i64>(descriptor.keyframe_byte_offsets().begin(),
                          descriptor.keyframe_byte_offsets().end());
}

///////////////////////////////////////////////////////////////////////////////
/// ImageFormatGroupMetadata
ImageFormatGroupMetadata::ImageFormatGroupMetadata() {}

ImageFormatGroupMetadata::ImageFormatGroupMetadata(
    const ImageFormatGroupDescriptor& descriptor)
    : descriptor(descriptor) {}

const ImageFormatGroupDescriptor& ImageFormatGroupMetadata::get_descriptor()
    const {
  return descriptor;
}

i32 ImageFormatGroupMetadata::num_images() const {
  return descriptor.num_images();
}

i32 ImageFormatGroupMetadata::width() const { return descriptor.width(); }

i32 ImageFormatGroupMetadata::height() const { return descriptor.height(); }

ImageEncodingType ImageFormatGroupMetadata::encoding_type() const {
  return descriptor.encoding_type();
}

ImageColorSpace ImageFormatGroupMetadata::color_space() const {
  return descriptor.color_space();
}

std::vector<i64> ImageFormatGroupMetadata::compressed_sizes() const {
  return std::vector<i64>(descriptor.compressed_sizes().begin(),
                          descriptor.compressed_sizes().end());
}

///////////////////////////////////////////////////////////////////////////////
/// JobMetadata
JobMetadata::JobMetadata() {}
JobMetadata::JobMetadata(const JobDescriptor &job) : descriptor_(job) {
  for (auto &c : descriptor_.columns()) {
    columns_.push_back(c);
    column_ids_.insert({c.name(), c.id()});
  }
  for (auto &t : descriptor_.tasks()) {
    table_names_.push_back(t.table_name());
    table_ids_.insert({t.table_name(), 0});
  }
}

std::string JobMetadata::descriptor_path(i32 job_id) const {
  return job_descriptor_path(job_id);
}

const JobDescriptor& JobMetadata::get_job_descriptor() const {
  return descriptor_;
}

std::string JobMetadata::descriptor_path() const {
  return job_descriptor_path(id());
}

i32 JobMetadata::id() const { return descriptor_.id(); }

std::string JobMetadata::name() const { return descriptor_.name(); }

i32 JobMetadata::io_item_size() const {
  return descriptor_.io_item_size();
}

i32 JobMetadata::work_item_size() const {
  return descriptor_.work_item_size();
}

i32 JobMetadata::num_nodes() const {
  return descriptor_.num_nodes();
}

const std::vector<Column>& JobMetadata::columns() const {
  return columns_;
}

i32 JobMetadata::column_id(const std::string& column_name) const {
  column_ids_.at(column_name);
}

const std::vector<std::string>& JobMetadata::table_names() const {
  return table_names_;
}

bool JobMetadata::has_table(const std::string& table_name) const {
  return table_ids_.count(table_name) > 0;
}

i32 JobMetadata::table_id(const std::string& table_name) const {
  return table_ids_.at(table_name);
}

i64 JobMetadata::rows_in_table(i32 table_id) const {
  i64 rows = -1;
  auto it = rows_in_table_.find(table_id);
  if (it == rows_in_table_.end()) {
    for (const Task& task : job_descriptor_.tasks()) {
      assert(task.samples_size() > 0);
      const TableSample& sample = task.samples(0);
      rows = sample.rows_size();
      rows_in_table_.insert(std::make_pair(table_id, rows));
    }
  } else {
    rows = it->second;
  }
  assert(rows != -1);
  return rows;
}

i64 JobMetadata::total_rows() const {
  i64 rows = 0;
  for (const Task& task : job_descriptor_.tasks()) {
    assert(task.samples_size() > 0);
    const TableSample& sample = task.samples(0);
    rows += sample.rows_size();
  }
  return rows;
}

///////////////////////////////////////////////////////////////////////////////
/// TableMetadata
TableMetadata::TableMetadata() {}
TableMetadata::TableMetadata(const TableDescriptor &table) : descriptor_(table) {
  for (auto &c : descriptor_.columns()) {
    columns_.push_back(c);
  }
}

std::string TableDescriptor::descriptor_path(i32 table_id) {
  return table_descriptor(table_id);
}

const TableDescriptor& TableMetadata::get_job_descriptor() const {
  return job_descriptor_;
}

std::string TableDescriptor::descriptor_path() {
  return table_descriptor(id());
}

i32 TableMetadata::id() const { return descriptor_.id(); }

std::string TableMetadata::name() const { return descriptor_.name(); }

i32 TableMetadata::num_rows() const {
  return descriptor_.num_rows();
}

i32 TableMetadata::rows_per_item() const {
  return descriptor_.rows_per_item();
}

const std::vector<Column>& TableMetadata::columns() const {
  return columns_;
}

std::string TableMetadata::column_name(i32 column_id) const {
  for (auto &c : descriptor_.columns()) {
    if (c.id() == column_id) {
      return c.name();
    }
  }
  LOG(FATAL) << "Column id " << column_id << " not found!";
}

i32 TableMetadata::column_id(const std::string& column_name) const {
  for (auto &c : descriptor_.columns()) {
    if (c.name() == column_name) {
      return c.id();
    }
  }
  LOG(FATAL) << "Column name " << column_name << " not found!";
}

ColumnType TableMetadata::column_type(i32 column_id) const {
  for (auto &c : descriptor_.columns()) {
    if (c.id() == column_id) {
      return c.type();
    }
  }
  LOG(FATAL) << "Column id " << column_id << " not found!";
}

std::string PREFIX = "";

void set_database_path(std::string path) { PREFIX = path + "/"; }

void write_new_table(storehouse::StorageBackend *storage,
                     DatabaseMetadata &meta,
                     TableMetadata &table) {
  LOG(INFO) << "Writing new table " << table.name() << "..." << std::endl;
  TableDescriptor& table_desc = table.get_descriptor();
  i32 table_id = meta.add_table(table.name());
  table_desc.set_id(table_id);

  write_table_metadata(table);
  write_database_metadata(meta);
  LOG(INFO) << "Finished writing new table " << table.name() << "."
            << std::endl;
}
}
}

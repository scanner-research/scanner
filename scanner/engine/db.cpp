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
#include "scanner/engine.h"
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

DatabaseMetadata::DatabaseMetadata() : next_dataset_id(0), next_job_id(0) {}

DatabaseMetadata::DatabaseMetadata(const DatabaseDescriptor& d)
    : descriptor(d),
      next_dataset_id(d.next_dataset_id()),
      next_job_id(d.next_job_id()) {
  for (int i = 0; i < descriptor.datasets_size(); ++i) {
    const DatabaseDescriptor_Dataset& dataset = descriptor.datasets(i);
    dataset_names.insert({dataset.id(), dataset.name()});
    dataset_job_ids[dataset.id()] = {};
  }
  for (int i = 0; i < descriptor.jobs_size(); ++i) {
    const DatabaseDescriptor_Job& job = descriptor.jobs(i);
    job_names.insert({job.id(), job.name()});
  }
  for (int i = 0; i < descriptor.job_to_datasets_size(); ++i) {
    const DatabaseDescriptor_JobToDataset& job_to_dataset =
        descriptor.job_to_datasets(i);
    dataset_job_ids[job_to_dataset.dataset_id()].insert(
        job_to_dataset.job_id());
  }
}

const DatabaseDescriptor& DatabaseMetadata::get_descriptor() const {
  descriptor.set_next_dataset_id(next_dataset_id);
  descriptor.set_next_job_id(next_job_id);
  descriptor.clear_datasets();
  descriptor.clear_jobs();
  descriptor.clear_job_to_datasets();

  for (auto& kv : dataset_names) {
    auto dataset = descriptor.add_datasets();
    dataset->set_id(kv.first);
    dataset->set_name(kv.second);
  }

  for (auto& kv : job_names) {
    auto job = descriptor.add_jobs();
    job->set_id(kv.first);
    job->set_name(kv.second);
  }

  for (auto& kv : dataset_job_ids) {
    for (i32 job_id : kv.second) {
      auto job = descriptor.add_job_to_datasets();
      job->set_dataset_id(kv.first);
      job->set_job_id(job_id);
    }
  }

  return descriptor;
}

bool DatabaseMetadata::has_dataset(const std::string& dataset) const {
  for (const auto& kv : dataset_names) {
    if (kv.second == dataset) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_dataset(i32 dataset_id) const {
  return dataset_names.count(dataset_id) > 0;
}

i32 DatabaseMetadata::get_dataset_id(const std::string& dataset) const {
  i32 id = -1;
  for (const auto& kv : dataset_names) {
    if (kv.second == dataset) {
      id = kv.first;
      break;
    }
  }
  assert(id != -1);
  return id;
}

const std::string& DatabaseMetadata::get_dataset_name(i32 dataset_id) const {
  return dataset_names.at(dataset_id);
}

i32 DatabaseMetadata::add_dataset(const std::string& dataset) {
  i32 dataset_id = next_dataset_id++;
  dataset_names[dataset_id] = dataset;
  dataset_job_ids[dataset_id] = {};
  return dataset_id;
}

void DatabaseMetadata::remove_dataset(i32 dataset_id) {
  for (i32 job_id : dataset_job_ids.at(dataset_id)) {
    job_names.erase(job_id);
  }
  assert(dataset_job_ids.count(dataset_id) > 0);
  assert(dataset_names.count(dataset_id) > 0);
  dataset_job_ids.erase(dataset_id);
  dataset_names.erase(dataset_id);
}

bool DatabaseMetadata::has_job(i32 dataset_id, const std::string& job) const {
  for (i32 job_id : dataset_job_ids.at(dataset_id)) {
    if (job == job_names.at(job_id)) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_job(i32 job_id) const {
  return job_names.count(job_id) > 0;
}

i32 DatabaseMetadata::get_job_id(i32 dataset_id, const std::string& job) const {
  i32 job_id = -1;
  for (i32 j_id : dataset_job_ids.at(dataset_id)) {
    if (job == job_names.at(j_id)) {
      job_id = j_id;
      break;
    }
  }
  assert(job_id != -1);
  return job_id;
}

const std::string& DatabaseMetadata::get_job_name(i32 job_id) const {
  return job_names.at(job_id);
}

i32 DatabaseMetadata::add_job(i32 dataset_id, const std::string& job_name) {
  i32 job_id = next_job_id++;
  dataset_job_ids.at(dataset_id).insert(job_id);
  job_names[job_id] = job_name;
  return job_id;
}

void DatabaseMetadata::remove_job(i32 job_id) {
  bool found = false;
  for (auto& kv : dataset_job_ids) {
    if (kv.second.count(job_id) > 0) {
      kv.second.erase(job_id);
      found = true;
      break;
    }
  }
  assert(found);
  assert(job_names.count(job_id) > 0);
  job_names.erase(job_id);
}

///////////////////////////////////////////////////////////////////////////////
/// DatasetMetadata
DatasetMetadata::DatasetMetadata() {}

DatasetMetadata::DatasetMetadata(const DatasetDescriptor& descriptor)
    : descriptor(descriptor) {}

const DatasetDescriptor& DatasetMetadata::get_descriptor() const {
  return descriptor;
}

i32 DatasetMetadata::id() const { return descriptor.id(); }

std::string DatasetMetadata::name() const { return descriptor.name(); }

DatasetType DatasetMetadata::type() const { return descriptor.type(); }

i32 DatasetMetadata::total_frames() const {
  if (this->type() == DatasetType_Video) {
    return descriptor.video_data().total_frames();
  } else if (this->type() == DatasetType_Image) {
    return descriptor.image_data().total_images();
  } else {
    assert(false);
    return {};
  }
}

i32 DatasetMetadata::max_width() const {
  if (this->type() == DatasetType_Video) {
    return descriptor.video_data().max_width();
  } else if (this->type() == DatasetType_Image) {
    return descriptor.image_data().max_width();
  } else {
    assert(false);
    return {};
  }
}

i32 DatasetMetadata::max_height() const {
  if (this->type() == DatasetType_Video) {
    return descriptor.video_data().max_height();
  } else if (this->type() == DatasetType_Image) {
    return descriptor.image_data().max_height();
  } else {
    assert(false);
    return {};
  }
}

std::vector<std::string> DatasetMetadata::original_paths() const {
  if (this->type() == DatasetType_Video) {
    return std::vector<std::string>(
        descriptor.video_data().original_video_paths().begin(),
        descriptor.video_data().original_video_paths().end());
  } else if (this->type() == DatasetType_Image) {
    return std::vector<std::string>(
        descriptor.image_data().original_image_paths().begin(),
        descriptor.image_data().original_image_paths().end());
  } else {
    assert(false);
    return {};
  }
}

std::vector<std::string> DatasetMetadata::item_names() const {
  if (this->type() == DatasetType_Video) {
    return std::vector<std::string>(
        descriptor.video_data().video_names().begin(),
        descriptor.video_data().video_names().end());
  } else if (this->type() == DatasetType_Image) {
    std::vector<std::string> item_names;
    for (i32 i = 0; i < descriptor.image_data().format_groups_size(); ++i) {
      item_names.push_back(std::to_string(i));
    }
    return item_names;
  } else {
    assert(false);
    return {};
  }
}

///////////////////////////////////////////////////////////////////////////////
/// VideoMetdata
VideoMetadata::VideoMetadata() {}

VideoMetadata::VideoMetadata(const VideoDescriptor& descriptor)
    : descriptor(descriptor) {}

const VideoDescriptor& VideoMetadata::get_descriptor() const {
  return descriptor;
}

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
JobMetadata::JobMetadata(const JobDescriptor &job) : job_descriptor_(job) {
  for (auto &c : job_descriptor_.columns()) {
    columns_.push_back(c.name());
    column_ids_.insert({c.name(), c.id()});
  }
  for (auto &t : job_descriptor_.tasks()) {
    table_names_.push_back(t.table_name());
    table_ids_.insert({t.table_name(), t.table_id()});
  }
}

const JobDescriptor& JobMetadata::get_job_descriptor() const {
  return job_descriptor_;
}

i32 JobMetadata::id() const { return job_descriptor_.id(); }

std::string JobMetadata::name() const { return job_descriptor_.name(); }

i32 JobMetadata::io_item_size() const {
  return job_descriptor_.io_item_size();
}

i32 JobMetadata::work_item_size() const {
  return job_descriptor_.work_item_size();
}

i32 JobMetadata::num_nodes() const {
  return job_descriptor_.num_nodes();
}

const std::vector<std::string>& JobMetadata::columns() const {
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
    for (const JobDescriptor::Task& task : job_descriptor_.tasks()) {
      assert(task.samples_size() > 0);
      JobDescriptor::Task::TableSample& sample = task.samples(0);
      rows = sample.rows_size();
      rows_in_table_.insert({table_id, rows});
    }
  } else {
    rows = it->second;
  }
  assert(rows != -1);
  return rows;
}

i64 JobMetadata::total_rows() const {
  i64 rows = 0;
  for (const JobDescriptor::Task& task : job_descriptor_.tasks()) {
    assert(task.samples_size() > 0);
    JobDescriptor::Task::TableSample& sample = task.samples(0);
    rows += sample.rows_size();
  }
  return rows;
}

namespace {

template <typename T>
void serialize(storehouse::WriteFile* file, const T& descriptor) {
  int size = descriptor.ByteSize();
  std::vector<u8> data(size);
  descriptor.SerializeToArray(data.data(), size);
  write(file, data.data(), size);
}

template <typename T>
T deserialize(storehouse::RandomReadFile* file, u64& pos) {
  T descriptor;
  std::vector<u8> data = storehouse::read_entire_file(file, pos);
  descriptor.ParseFromArray(data.data(), data.size());
  return descriptor;
}
}

void serialize_database_metadata(storehouse::WriteFile* file,
                                 const DatabaseMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

DatabaseMetadata deserialize_database_metadata(storehouse::RandomReadFile* file,
                                               u64& pos) {
  return DatabaseMetadata(deserialize<DatabaseDescriptor>(file, pos));
}

void serialize_dataset_descriptor(WriteFile* file,
                                  const DatasetDescriptor& descriptor) {
  serialize(file, descriptor);
}

DatasetDescriptor deserialize_dataset_descriptor(RandomReadFile* file,
                                                 uint64_t& pos) {
  return deserialize<DatasetDescriptor>(file, pos);
}

void serialize_video_metadata(WriteFile* file, const VideoMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

VideoMetadata deserialize_video_metadata(RandomReadFile* file, uint64_t& pos) {
  return VideoMetadata{deserialize<VideoDescriptor>(file, pos)};
}

void serialize_image_format_group_metadata(
    WriteFile* file, const ImageFormatGroupMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

ImageFormatGroupMetadata deserialize_image_format_group_metadata(
    RandomReadFile* file, uint64_t& pos) {
  return ImageFormatGroupMetadata{
      deserialize<ImageFormatGroupDescriptor>(file, pos)};
}

void serialize_web_timestamps(WriteFile* file, const WebTimestamps& ts) {
  serialize(file, ts);
}

WebTimestamps deserialize_web_timestamps(RandomReadFile* file, uint64_t& pos) {
  return deserialize<WebTimestamps>(file, pos);
}

void serialize_job_descriptor(WriteFile* file,
                              const JobDescriptor& descriptor) {
  serialize(file, descriptor);
}

JobDescriptor deserialize_job_descriptor(RandomReadFile* file, uint64_t& pos) {
  return deserialize<JobDescriptor>(file, pos);
}

std::string PREFIX = "";

void set_database_path(std::string path) { PREFIX = path + "/"; }
}

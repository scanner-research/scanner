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
#include <sstream>
#include <iostream>

using storehouse::WriteFile;
using storehouse::RandomReadFile;
using storehouse::StoreResult;

namespace scanner {

int PUS_PER_NODE = 1;           // Number of available GPUs per node
int WORK_ITEM_SIZE = 8;         // Base size of a work item
int TASKS_IN_QUEUE_PER_PU = 4;  // How many tasks per GPU to allocate to a node
int LOAD_WORKERS_PER_NODE = 2;  // Number of worker threads loading data
int SAVE_WORKERS_PER_NODE = 2;  // Number of worker threads loading data
int NUM_CUDA_STREAMS = 32;      // Number of cuda streams for image processing

DatabaseMetadata::DatabaseMetadata() : next_dataset_id(0), next_job_id(0) {}

DatabaseMetadata::DatabaseMetadata(const DatabaseDescriptor& d)
    : descriptor(d), next_dataset_id(d.next_dataset_id()),
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
    const DatabaseDescriptor_JobToDataset &job_to_dataset =
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
      job->set_job_id(kv.first);
      job->set_dataset_id(job_id);
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

i32 DatabaseMetadata::get_dataset_id(const std::string &dataset) const {
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

i32 DatabaseMetadata::add_dataset(const std::string &dataset) {
  i32 dataset_id = next_dataset_id++;
  dataset_names[dataset_id] = dataset;
  dataset_job_ids[dataset_id] = {};
  return dataset_id;
}

void DatabaseMetadata::remove_dataset(i32 dataset_id) {
  for (i32 job_id : dataset_job_ids.at(dataset_id)) {
    job_names.erase(job_id);
  }
  dataset_job_ids.erase(dataset_id);
  dataset_names.erase(dataset_id);
}

bool DatabaseMetadata::has_job(const std::string &job) const {
  for (const auto& kv : job_names) {
    if (job == kv.second) {
      return true;
    }
  }
  return false;
}

bool DatabaseMetadata::has_job(i32 job_id) const {
  return job_names.count(job_id) > 0;
}

i32 DatabaseMetadata::get_job_id(const std::string &job) const {
  i32 job_id = -1;
  for (const auto& kv : job_names) {
    if (job == kv.second) {
      job_id = kv.first;
      break;
    }
  }
  assert(job_id != -1);
  return job_id;
}

const std::string &DatabaseMetadata::get_job_name(i32 job_id) const {
  return job_names.at(job_id);
}

i32 DatabaseMetadata::add_job(i32 dataset_id, const std::string &job_name) {
  i32 job_id = next_job_id++;
  dataset_job_ids.at(dataset_id).insert(job_id);
  job_names[job_id] = job_name;
  return job_id;
}

void DatabaseMetadata::remove_job(i32 job_id) {
  for (auto& kv : dataset_job_ids) {
    if (kv.second.count(job_id) > 0) {
      kv.second.erase(job_id);
    }
  }
  job_names.erase(job_id);
}

VideoMetadata::VideoMetadata() {
}

VideoMetadata::VideoMetadata(const VideoDescriptor &descriptor)
    : descriptor(descriptor) {}

const VideoDescriptor& VideoMetadata::get_descriptor() const {
  return descriptor;
}

i32 VideoMetadata::frames() const {
  return descriptor.frames();
}

i32 VideoMetadata::width() const {
  return descriptor.width();
}

i32 VideoMetadata::height() const {
  return descriptor.height();
}

std::vector<i64> VideoMetadata::keyframe_positions() const {
  return std::vector<i64>(descriptor.keyframe_positions().begin(),
                          descriptor.keyframe_positions().end());
}

std::vector<i64> VideoMetadata::keyframe_byte_offsets() const {
  return std::vector<i64>(descriptor.keyframe_byte_offsets().begin(),
                          descriptor.keyframe_byte_offsets().end());
}

namespace {

template <typename T>
void serialize(storehouse::WriteFile *file, const T &descriptor) {
  int size = descriptor.ByteSize();
  std::vector<u8> data(size);
  descriptor.SerializeToArray(data.data(), size);
  write(file, data.data(), size);
}

template <typename T>
T deserialize(storehouse::RandomReadFile *file, u64 &pos) {
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

void serialize_video_metadata(WriteFile* file,
                              const VideoMetadata& metadata) {
  serialize(file, metadata.get_descriptor());
}

VideoMetadata deserialize_video_metadata(RandomReadFile *file, uint64_t &pos) {
  return VideoMetadata{deserialize<VideoDescriptor>(file, pos)};
}

void serialize_web_timestamps(WriteFile* file, const WebTimestamps& ts) {
  serialize(file, ts);
}

WebTimestamps deserialize_web_timestamps(RandomReadFile *file, uint64_t &pos) {
  return deserialize<WebTimestamps>(file, pos);
}

void serialize_job_descriptor(WriteFile* file,
                              const JobDescriptor& descriptor) {
  serialize(file, descriptor);
}

JobDescriptor deserialize_job_descriptor(RandomReadFile* file,
                                         uint64_t& pos) {
  return deserialize<JobDescriptor>(file, pos);
}
}

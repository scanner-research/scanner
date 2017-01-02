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

#include "scanner/eval/pipeline_description.h"

#include "scanner/metadata.pb.h"
#include "scanner/engine/db.h"

#include "storehouse/storage_backend.h"

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {

namespace {
std::map<std::string, PipelineGeneratorFn>& pipeline_fns() {
  static std::map<std::string, PipelineGeneratorFn> m;
  return m;
}
}

TableInformation::TableInformation(
    i64 rows, const std::vector<std::string>& sample_job_names,
    const std::vector<std::string>& sample_table_names,
    const std::vector<std::vector<std::string>>& sample_columns,
    const std::vector<std::vector<i64>>& sample_rows)
    : rows_(rows),
      sample_job_names_(sample_job_names),
      sample_table_names_(sample_table_names),
      sample_columns_(sample_columns),
      sample_rows_(sample_rows_) {}

i64 TableInformation::num_rows() const { return rows_; }

JobInformation::JobInformation(const std::string& dataset_name,
                               const std::string& job_name,
                               storehouse::StorageBackend* storage)
    : dataset_name_(dataset_name), job_name_(job_name), storage_(storage) {
  // Load database metadata
  DatabaseMetadata db_meta{};
  {
    std::string db_meta_path = database_metadata_path();
    std::unique_ptr<RandomReadFile> meta_in_file;
    BACKOFF_FAIL(
        make_unique_random_read_file(storage, db_meta_path, meta_in_file));
    u64 pos = 0;
    db_meta = deserialize_database_metadata(meta_in_file.get(), pos);
  }
  i32 dataset_id = db_meta.get_dataset_id(dataset_name);

  JobDescriptor descriptor;
  {
    std::unique_ptr<RandomReadFile> file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage, job_descriptor_path(dataset_name, job_name), file));
    u64 pos = 0;
    descriptor = deserialize_job_descriptor(file.get(), pos);
  }
  JobMetadata meta(descriptor);
  table_names_ = meta.table_names();
  column_names_ = meta.columns();

  for (auto& task : descriptor.tasks()) {
    std::vector<std::string> sample_job_names;
    std::vector<std::string> sample_table_names;
    std::vector<std::vector<std::string>> sample_column_names;
    std::vector<std::vector<i64>> sample_rows;
    for (auto& sample : task.samples()) {
      std::string sampled_job_name =
          db_meta.get_job_name(sample.job_id());
      JobDescriptor sampled_desc;
      {
        std::unique_ptr<RandomReadFile> file;
        BACKOFF_FAIL(make_unique_random_read_file(
            storage, job_descriptor_path(dataset_name, sampled_job_name),
            file));
        u64 pos = 0;
        descriptor = deserialize_job_descriptor(file.get(), pos);
      }
      JobMetadata sampled_meta(sampled_desc);

      sample_job_names.push_back(sampled_job_name);
      sample_table_names.push_back(
          sampled_meta.table_names()[sample.table_id()]);
      std::vector<std::string> column_names;
      for (const auto& col : sample.columns()) {
        column_names.push_back(col.name());
      }
      sample_column_names.push_back(column_names);
      sample_rows.push_back(
          std::vector<i64>(sample.rows().begin(), sample.rows().end()));
    }
    tables_.insert(
        std::make_pair(task.table_name(),
                       TableInformation(task.samples(0).rows_size(),
                                        sample_job_names, sample_table_names,
                                        sample_column_names, sample_rows)));
  }
}

const std::vector<std::string>& JobInformation::table_names() {
  return table_names_;
}

const std::vector<std::string>& JobInformation::column_names() {
  return column_names_;
}

const TableInformation& JobInformation::table(const std::string& name) {
  auto it = tables_.find(name);
  LOG_IF(FATAL, it == tables_.end())
      << "Could not find table " << name << " in job " << job_name_
      << " under dataset " << dataset_name_ << "!";
  return it->second;
}

DatasetInformation::DatasetInformation(
    const std::string &dataset_name, const std::vector<std::string> &job_names,
    storehouse::StorageBackend* storage)
    : dataset_name_(dataset_name), job_names_(job_names), storage_(storage) {}

const std::vector<std::string>& DatasetInformation::job_names() {
  return job_names_;
}

const JobInformation& DatasetInformation::job(const std::string& name) {
  auto it = job_.find(name);
  if (it == job_.end()) {
    bool found = false;
    for (auto& j : job_names_) {
      if (name == j) {
        found = true;
        break;
      }
    }
    LOG_IF(FATAL, !found) << "Could not find job " << name << " in dataset "
                          << dataset_name_ << "!";
    job_.insert({name, JobInformation(dataset_name_, name, storage_)});
    return job_.at(name);
  } else {
    return it->second;
  }
}

bool add_pipeline(std::string name, PipelineGeneratorFn fn) {
  LOG_IF(FATAL, pipeline_fns().count(name) > 0)
      << "Pipeline with name " << name << " has already been registered!";
  pipeline_fns().insert({name, fn});
  return true;
}

PipelineGeneratorFn get_pipeline(const std::string& name) {
  if (pipeline_fns().count(name) == 0) {
    std::string current_names;
    for (auto& entry : pipeline_fns()) {
      current_names += entry.first + " ";
    }

    LOG(FATAL) << "Pipeline with name " << name << " has not been registered. "
               << "Valid pipelines are: " << current_names;
  }

  return pipeline_fns().at(name);
}
}

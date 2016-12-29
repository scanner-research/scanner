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

namespace scanner {

namespace {
std::map<std::string, PipelineGeneratorFn>& pipeline_fns() {
  static std::map<std::string, PipelineGeneratorFn> m;
  return m;
}
}

TableInformation::TableInformation(
    i64 rows, const std::vector<std::string> &sample_job_names,
    const std::vector<std::string> &sample_table_names,
    const std::vector<std::vector<std::string>> &sample_columns,
    const std::vector<std::vector<i64>> &sample_rows)
    : rows_(rows), sample_job_names_(sample_job_names),
      sample_table_names_(sample_table_names), sample_columns_(sample_columns),
      sample_rows_(sample_rows_) {}

i64 TableMetadata::num_rows() const { return rows_; }

JobMetadata::JobMetadata(const std::string& dataset_name)
    : dataset_name_(dataset_name) {
  for (auto& kv : tables) {
    tables_names_.push_back(kv.first);
  }
}

const std::vector<std::string>& JobMetadata::table_names() {
  return table_names_;
}

const TableMetadata& JobMetadata::table(const std::string& name) {
  return tables_.at(name);
}

JobInformation::JobInformation(const std::string& dataset_name,
                               const std::string& job_name)
    : dataset_name_(dataset_name), job_name_(job_name) {
  JobDescriptor descriptor;
  {
    std::string descriptor_path = ;
    std::unique_ptr<RandomReadFile> file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage, job_descriptor_path(dataset_name, job_name), file));
    u64 pos = 0;
    descriptor = deserialize_job_descriptor(file.get(), pos);
  }
  JobMetadata meta(descriptor);
  table_names_ = meta.table_names();
  column_names_ = meta.columns();

  for (JobDescriptor::Task& task : descriptor.tasks) {
    std::vector<std::string> sample_job_names;
    std::vector<std::string> sample_table_names;
    std::vector<std::vector<std::string>> sample_column_names;
    std::vector<std::vector<i64>> sample_rows;
    for (JobDescriptor::Task::TableSample& sample : task.samples) {
      sample_job_names.push_back(sample.job_name());
      sample_table_names.push_back(sample.table_name());
      sample_columns_names.push_back(std::vector<std::string>(
          sample.columns().begin(), sample.columns().end()));
      sample_rows.push_back(
          std::vector<i64>(sample.rows().begin(), sample.rows().end()));
    }
    tables_.insert({task.table_name(),
                    TableInformation(task.samples(0).rows(), sample_job_names,
                                     sample_table_names, sample_column_names,
                                     sample_rows)});
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
    const std::string &dataset_name, const std::vector<std::string> &job_names)
    : dataset_name_(dataset_name), job_names_(job_names) {}

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
    job_.insert({name, JobInformation(dataset_name_, job_name_)});
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

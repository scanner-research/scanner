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
    : rows_(rows) {
  for (size_t i = 0; i < sample_job_names.size(); ++i) {
    samples_.emplace_back();
    TableSample& sample = samples_.back();
    sample.job_name = sample_job_names[i];
    sample.table_name = sample_table_names[i];
    sample.columns = sample_columns[i];
    sample.rows = sample_rows[i];
  }
}

i64 TableInformation::num_rows() const { return rows_; }

const std::vector<TableSample>& TableInformation::samples() const {
  return samples_;
}

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
      // Skip the base job dummy sample
      if (sample.job_id() == -1) continue;

      std::string sampled_job_name =
          db_meta.get_job_name(sample.job_id());
      JobDescriptor sampled_desc;
      {
        std::unique_ptr<RandomReadFile> file;
        BACKOFF_FAIL(make_unique_random_read_file(
            storage, job_descriptor_path(dataset_name, sampled_job_name),
            file));
        u64 pos = 0;
        sampled_desc = deserialize_job_descriptor(file.get(), pos);
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

const std::vector<std::string>& JobInformation::table_names() const {
  return table_names_;
}

const std::vector<std::string>& JobInformation::column_names() const {
  return column_names_;
}

const TableInformation& JobInformation::table(const std::string& name) const {
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

const std::vector<std::string>& DatasetInformation::job_names() const {
  return job_names_;
}

const JobInformation& DatasetInformation::job(const std::string& name) const {
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

void Sampler::all_frames(const DatasetInformation& info,
                         PipelineDescription& desc) {
  strided_frames(info, desc, 1, 0);
}

void Sampler::strided_frames(const DatasetInformation& info,
                             PipelineDescription& desc, i64 stride,
                             i64 offset) {
  strided(info, desc, base_job_name(), {base_column_name()}, stride, offset);
}

void Sampler::all(const DatasetInformation& info,
                         PipelineDescription& desc,
                         const std::string& job_name) {
  const JobInformation& job = info.job(job_name);
  all(info, desc, job_name, job.column_names());
}

void Sampler::all(const DatasetInformation& info, PipelineDescription& desc,
                  const std::string& job_name,
                  const std::vector<std::string>& columns) {
  strided(info, desc, job_name, columns, 1, 0);
}

void Sampler::strided(const DatasetInformation& info, PipelineDescription& desc,
                      const std::string& job_name, i64 stride, i64 offset) {
  const JobInformation& job = info.job(job_name);
  strided(info, desc, job_name, job.column_names(), stride, offset);
}

void Sampler::strided(const DatasetInformation& info, PipelineDescription& desc,
                      const std::string& job_name,
                      const std::vector<std::string>& columns, i64 stride,
                      i64 offset) {
  const JobInformation& job = info.job(job_name);
  for (const std::string& table_name : job.table_names()) {
    Task task;
    task.table_name = table_name;
    TableSample sample;
    sample.job_name = job_name;
    sample.table_name = table_name;
    sample.columns = columns;
    const TableInformation& table = job.table(table_name);
    for (i64 r = offset; r < table.num_rows(); r += stride) {
      sample.rows.push_back(r);
    }
    task.samples.push_back(sample);
    desc.tasks.push_back(task);
  }
}

void Sampler::range_frames(const DatasetInformation& info,
                           PipelineDescription& desc, i64 start, i64 end)
{
  range(info, desc, base_job_name(), start, end);
}

void Sampler::range(const DatasetInformation& info, PipelineDescription& desc,
                    const std::string& job_name,
                    i64 start, i64 end)
{
  const JobInformation& job = info.job(job_name);
  range(info, desc, job_name, job.column_names(), start, end);
}

void Sampler::range(const DatasetInformation& info, PipelineDescription& desc,
                    const std::string& job_name,
                    const std::vector<std::string>& columns, i64 start,
                    i64 end) {
  strided_range(info, desc, job_name, columns, 1, start, end);
}

void Sampler::strided_range_frames(const DatasetInformation& info,
                                   PipelineDescription& desc, i64 stride,
                                   i64 start, i64 end) {
  strided_range(info, desc, base_job_name(), {base_column_name()}, stride,
                start, end);
}

void Sampler::strided_range(const DatasetInformation& info,
                            PipelineDescription& desc,
                            const std::string& job_name,
                            const std::vector<std::string>& columns, i64 stride,
                            i64 start, i64 end) {
  const JobInformation& job = info.job(job_name);
  for (const std::string& table_name : job.table_names()) {
    Task task;
    task.table_name = table_name;
    TableSample sample;
    sample.job_name = job_name;
    sample.table_name = table_name;
    sample.columns = columns;
    const TableInformation& table = job.table(table_name);
    for (i64 r = start; r < end; r += stride) {
      sample.rows.push_back(r);
    }
    task.samples.push_back(sample);
    desc.tasks.push_back(task);
  }
}

void Sampler::join_prepend(const DatasetInformation& info,
                           PipelineDescription& desc,
                           const std::string& existing_column,
                           const std::string& to_join_column) {
  LOG_IF(FATAL, desc.tasks.empty())
      << "Can not join when no tasks are specified!";
  size_t i;
  for (i = 0; i < desc.tasks[0].samples.size(); ++i) {
    bool found = false;
    for (const std::string& c : desc.tasks[0].samples[i].columns) {
      if (c == existing_column) {
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }
  LOG_IF(FATAL, i == desc.tasks[0].samples.size())
      << "Join requested between column " << existing_column << " and column "
      << to_join_column << " but column " << existing_column << " "
      << "is not in any table samples.";

  const std::string& existing_job_name = desc.tasks[0].samples[i].job_name;
  const JobInformation& existing_job = info.job(existing_job_name);

  for (Task& task : desc.tasks) {
    const std::string& existing_table_name = task.samples[i].table_name;
    // Find ancestor
    std::vector<std::vector<i64>> queue_rows = {task.samples[i].rows};
    std::vector<std::string> queue_job = {existing_job_name};
    std::vector<std::string> queue_table = {existing_table_name};

    std::vector<i64> parent_rows;
    std::string parent_job_name;
    std::string parent_table_name;
    bool found = false;
    while (!found && !queue_rows.empty()) {
      parent_rows = queue_rows.back();
      parent_job_name = queue_job.back();
      parent_table_name = queue_table.back();
      queue_rows.pop_back();
      queue_job.pop_back();
      queue_table.pop_back();

      const JobInformation& parent_job = info.job(parent_job_name);
      // Check if column we are looking for is in this ancestor
      for (const std::string& c : parent_job.column_names()) {
        if (c == to_join_column) {
          found = true;
          break;
        }
      }
      for (const TableSample& sample :
           parent_job.table(parent_table_name).samples()) {
        // Figure out which rows of current job were sampled from parent
        std::vector<i64> new_rows;
        for (i64 r : parent_rows) {
          new_rows.push_back(sample.rows.at(r));
        }
        queue_rows.push_back(new_rows);
        queue_job.push_back(sample.job_name);
        queue_table.push_back(sample.table_name);
      }
    }
    LOG_IF(FATAL, !found) << "Could not find column " << to_join_column
                          << " in any ancestor jobs!";

    TableSample sample;
    sample.job_name = parent_job_name;
    sample.table_name = parent_table_name;
    sample.columns = {to_join_column};
    sample.rows = parent_rows;
    task.samples.insert(task.samples.begin() + i, sample);
  }
}


  // job: base, cols: {frame} -> job: base, cols: {frame}

void Sampler::join_append(const DatasetInformation& info,
                          PipelineDescription& desc,
                          const std::string& existing_column,
                          const std::string& to_join_column)
{
}

}

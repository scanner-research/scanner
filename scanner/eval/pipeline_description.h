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

#pragma once

#include "scanner/eval/evaluator_factory.h"
#include "scanner/util/common.h"

#include <functional>
#include <vector>

namespace scanner {

struct TableSample {
  /**
   * @brief Name of the job to select from
   */
  std::string job_name;

  /**
   * @brief Name of table to sample from
   */
  std::string table_name;

  /**
   * @brief Columns to grab from the input table
   */
  std::vector<std::string> columns;

  /**
   * @brief Indices of the rows to sample
   */
  std::vector<i64> rows;
};

struct Task {
  /**
   * @brief Name of the table to write the results of this task to
   */
  std::string table_name;

  /**
   * @brief Specifies a list of tables to sample from.
   *
   * If multiple tables are given, their columns will be joined together so
   * that the first evaluator receives all of the columns from the first row
   * specified in each TableSample, all of the columns from the second row,
   * and so on. All TableSamples must have the same number of rows selected.
   *
   */
  std::vector<TableSample> samples;

  i32 resolution_downsample_factor;
};

/**
 * @brief Defines evaluators and a sampling pattern to run over a dataset.
 *
 * A pipeline is a sequence, or chain, of evaluators which execute over a stream
 * of video data. A sampling pattern can be specified that selects a subset of
 * frames from the videos in a given dataset. The chain of evaluators is
 * specified by the "evaluator_factories" variable.
 */
struct PipelineDescription {
  std::vector<Task> tasks;

  /**
   * @brief The chain of evaluators which will be executed over the input
   */
  std::vector<std::unique_ptr<EvaluatorFactory>> evaluator_factories;
};

struct TableInformation {
 public:
  TableInformation(i64 rows,
                   const std::vector<std::string>& sample_job_names,
                   const std::vector<std::string>& sample_table_names,
                   const std::vector<std::vector<std::string>>& sample_columns,
                   const std::vector<i64>& sample_rows);

  i64 num_rows() const;

 private:
  i64 rows_;
  std::vector<std::string> sample_job_names_;
  std::vector<std::string> sample_table_names_;
  std::vector<std::vector<std::string>> sample_columns_;
  std::vector<std::vector<i64>> sample_rows_;
};

struct JobInformation {
 public:
  JobInformation(const std::string& dataset_name,
                 const std::string& job_name);

  const std::vector<std::string>& table_names();

  const std::vector<std::string>& column_names();

  const TableInformation& table(const std::string& name);

 private:
  std::string dataset_name_;
  std::string job_name_;
  std::vector<std::string> table_names_;
  std::vector<std::string> column_names_;
  std::map<std::string, TableInformation> tables_;
};

struct DatasetInformation {
 public:
  DatasetInformation(const std::string& dataset_name,
                     const std::vector<std::string>& job_names);

  const std::vector<std::string>& job_names();

  const JobInformation& job(const std::string& name);

 private:
  std::string dataset_name_;
  std::vector<std::string> job_names_;
  std::map<std::string, JobInformation> job_;
};

using PipelineGeneratorFn = std::function<PipelineDescription(
    const DatasetInformation&)>;

bool add_pipeline(std::string name, PipelineGeneratorFn fn);

PipelineGeneratorFn get_pipeline(const std::string& name);

#define REGISTER_PIPELINE(name, fn) \
  static bool dummy_##name = add_pipeline(#name, fn);
}

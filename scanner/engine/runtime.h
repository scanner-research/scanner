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

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/eval/pipeline_description.h"
#include "scanner/video/video_decoder.h"
#include "scanner/util/memory.h"

#include "storehouse/storage_backend.h"

#include <string>

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Work structs - structs used to exchange data between workers during
///   execution of the run command.
struct IOItem {
  // @brief the output table id
  i32 table_id;
  // @brief the unique id for this item in the table
  i64 item_id;
  // @brief the first row in this item
  i64 start_row;
  // @brief the row after the last row in this item
  i64 end_row;
};

struct WorkItem {
  // @brief the index in the IOItem list of the parent io item
  i64 io_item_index;
};

struct LoadWorkEntry {
  struct Sample {
    // @brief which job to select tables from
    i32 job_id;
    // @brief which table to select rows from
    i32 table_id;
    // @brief the columns to read from
    std::vector<std::string> columns;
    // @brief the rows to read from the sampled table
    std::vector<i64> rows;
  };

  i32 io_item_index;
  std::vector<Sample> samples;
};

struct EvalWorkEntry {
  i32 io_item_index;
  std::vector<std::string> column_names;
  BatchedColumns columns;
  DeviceHandle buffer_handle;
  bool video_decode_item;
};

///////////////////////////////////////////////////////////////////////////////
struct JobParameters {
  storehouse::StorageConfig* storage_config;
  MemoryPoolConfig memory_pool_config;
  std::string dataset_name;
  PipelineGeneratorFn pipeline_gen_fn;
  std::string out_job_name;
};

void run_job(JobParameters& params);
}

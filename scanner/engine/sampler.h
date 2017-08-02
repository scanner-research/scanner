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

#pragma once

#include "scanner/engine/metadata.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/util/common.h"
#include "scanner/util/profiler.h"

#include <vector>

namespace scanner {
namespace internal {

/* Types of sampling
   - All: selects all rows from the table
   - Stride: selects every Nth row (with optional offset)
   - Range: select all rows within [start, end)
   - Strided Range: select every Nth row within [start, end)
   - Gather: select arbitrary set of rows

   Requiring access to more than metadata:
   - Filter: select all rows where some predicate holds on one of the columns
 */

struct RowSample {
  std::vector<i64> warmup_rows;
  std::vector<i64> rows;
};

class Sampler {
 public:
  Sampler(const std::string& name, const TableMetadata& table)
    : name_(name), table_(table) {}

  virtual ~Sampler() {}

  const std::string& name() const { return name_; }

  virtual Result validate() = 0;

  virtual i64 total_rows() const = 0;

  virtual i64 total_samples() const = 0;

  virtual RowSample next_sample() = 0;

  virtual void reset() = 0;

  virtual RowSample sample_at(i64 sample_idx) = 0;

  virtual i64 offset_at_sample(i64 sample_idx) const = 0;

 protected:
  std::string name_;
  TableMetadata table_;
};

Result make_sampler_instance(const std::string& sampler_type,
                             const std::vector<u8>& sampler_args,
                             const TableMetadata& sampled_table,
                             Sampler*& sampler);

class TaskSampler {
 public:
  TaskSampler(const TableMetaCache& table_metas,
              const proto::Task& task);

  Result validate();

  i64 total_rows();

  i64 total_samples();

  Result next_work(proto::NewWork& new_work);

  void reset();

  Result sample_at(i64 sample_idx, proto::NewWork& new_work);

 private:
  const TableMetaCache& table_metas_;
  proto::Task task_;
  Result valid_;
  std::vector<std::unique_ptr<Sampler>> samplers_;
  i64 total_rows_ = 0;
  i32 table_id_;
  i64 total_samples_ = 0;
  i64 samples_pos_ = 0;
  i64 allocated_rows_ = 0;
};
}
}

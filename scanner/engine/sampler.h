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

class DomainSampler {
 public:
  DomainSampler(const std::string& name)
    : name_(name) {}

  virtual ~DomainSampler() {}

  const std::string& name() const { return name_; }

  virtual Result validate() = 0;

  virtual Result get_upstream_rows(const std::vector<i64>& downstream_rows,
                                   std::vector<i64>& upstream_rows) const = 0;

  virtual Result get_num_downstream_rows(
      i64 num_upstream_rows,
      i64& num_downstream_rows) const = 0;

 protected:
  std::string name_;
};

Result
make_domain_sampler_instance(const std::string& sampler_type,
                             const std::vector<u8>& sampler_args,
                             DomainSampler*& sampler);

struct PartitionGroup {
  std::vector<i64> row;
};

class Partitioner {
 public:
  Partitioner(const std::string& name, i64 num_rows)
    : name_(name), num_rows_(num_rows) {}

  virtual ~Partitioner() {}

  const std::string& name() const { return name_; }

  virtual Result validate() = 0;

  virtual i64 total_rows() const = 0;

  virtual i64 total_groups() const = 0;

  virtual std::vector<i64> total_rows_per_group() const = 0;

  virtual PartitionGroup next_group() = 0;

  virtual void reset() = 0;

  virtual PartitionGroup group_at(i64 group_idx) = 0;

  virtual i64 offset_at_group(i64 group_idx) const = 0;

 protected:
  std::string name_;
  i64 num_rows_;
};

Result make_partitioner_instance(const std::string& sampler_type,
                                 const std::vector<u8>& sampler_args,
                                 i64 num_rows, Partitioner*& partitioner);
}
}

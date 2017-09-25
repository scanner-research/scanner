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

#include "scanner/engine/sampler.h"
#include "scanner/metadata.pb.h"

#include <cmath>
#include <vector>

namespace scanner {
namespace internal {

namespace {

using DomainSamplerFactory =
    std::function<DomainSampler*(const std::vector<u8>&)>;

// 1 to 1 mapping
class DefaultDomainSampler : public DomainSampler {
 public:
  DefaultDomainSampler(const std::vector<u8>& args)
    : DomainSampler("Default") {
    valid_.set_success(true);
  }

  Result validate() override {
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_upstream_rows(const std::vector<i64>& input_rows,
                           std::vector<i64>& output_rows) const {
    output_rows = input_rows;
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_num_downstream_rows(i64 num_upstream_rows,
                                 i64& num_downstream_rows) const {
    num_downstream_rows = num_upstream_rows;
    Result result;
    result.set_success(true);
    return result;
  }

 private:
  Result valid_;
};

class StridedDomainSampler : public DomainSampler {
 public:
  StridedDomainSampler(const std::vector<u8>& args)
    : DomainSampler("Strided") {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "StridedSampler provided with invalid protobuf args");
      return;
    }
    if (args_.stride() <= 0) {
      RESULT_ERROR(&valid_,
                   "Strided sampler stride (%ld) must be greater than zero",
                   args_.stride());
      return;
    }
  }

  Result validate() override {
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_upstream_rows(const std::vector<i64>& downstream_rows,
                           std::vector<i64>& upstream_rows) const {
    for (i64 in : downstream_rows) {
      upstream_rows.push_back(in * args_.stride());
    }
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_num_downstream_rows(i64 num_upstream_rows,
                                 i64& num_downstream_rows) const {
    num_downstream_rows = ceil(num_upstream_rows / float(args_.stride()));
    Result result;
    result.set_success(true);
    return result;
  }

 private:
  Result valid_;
  proto::StridedSamplerArgs args_;
}

class StridedRangesDomainSampler : public DomainSampler {
 public:
  StridedRangeSampler(const std::vector<u8>& args)
    : DomainSampler("StridedRange") {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "StridedRange sampler provided with invalid protobuf args");
      return;
    }
    if (args_.stride() <= 0) {
      RESULT_ERROR(&valid_,
                   "StridedRange stride (%ld) must be greater than zero",
                   args_.stride());
      return;
    }
    if (args_.starts_size() != args_.ends_size()) {
      RESULT_ERROR(&valid_,
                   "StridedRange starts and ends not the same size");
      return;
    }
    i64 offset = 0;
    for (i64 i = 0; i < args_.starts_size(); ++i) {
      if (args_.starts(i) > args_.ends(i)) {
        RESULT_ERROR(&valid_,
                     "StridedRange start (%ld) should not be after end (%ld)",
                     args_.starts(i), args_.ends(i));
        return;
      }
      i64 rows =
          ceil((args_.ends(i) - args_.starts(i)) / (float)args_.stride());
      offset_at_range_starts_.push_back(offset);
      offset += rows;
    }
    offset_at_range_starts_.push_back(offset);
  }

  Result validate() override { return valid_; }

  Result get_upstream_rows(const std::vector<i64>& downstream_rows,
                           std::vector<i64>& upstream_rows) const override {
    Result valid;
    valid.set_success(true);
    for (i64 in_row : downstream_rows) {
      i64 range_idx = -1;
      for (i64 i = 1; i < offset_at_range_starts_.size(); ++i) {
        i64 start_offset = offset_at_range_starts_[i];
        if (in_row < start_offset) {
          range_idx = i - 1;
          break;
        }
      }
      if (range_idx == -1) {
        RESULT_ERROR(&valid,
                     "StridedRange received out of bounds request for row %ld "
                     "(max requestable row is %ld).",
                     in_row,
                     offset_at_range_starts_.back());
        return valid;
      }
      i64 normed_in = in_row - offset_at_range_starts_[range_idx];
      i64 out_row = args_.starts(range_idx) + normed_in * args_.stride();
      upstream_rows.push_back(out_row);
    }
    return valid;
  }

  Result get_num_downstream_rows(i64 num_upstream_rows,
                                 i64& num_downstream_rows) const {
    i64 i = 0;
    for (; i < args_.ends_size(); ++i) {
      i64 start_offset = offset_at_range_starts_[i];
      if (num_upstream_rows < args_.ends(i)) {
        break;
      }
    }
    num_downstream_rows = 0;
    for (i64 se = 0; se < i; ++se) {
      num_downstream_rows +=
          ceil((args_.ends(se) - args_.starts(se)) / float(args_.stride()));
    }
    if (i != args_.ends_size()) {
      num_downstream_rows +=
          ceil((num_upstream_rows - args_.starts(i)) / float(args_.stride()));
    }
    Result valid;
    valid.set_success(true);
    return valid;
  }

 private:
  Result valid_;
  proto::StridedRangeSamplerArgs args_;
  std::vector<i64> offset_at_range_starts_;
};

class GatherDomainSampler : public DomainSampler {
 public:
  GatherDomainSampler(const std::vector<u8>& args)
    : DomainSampler("Gather") {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "Gather sampler provided with invalid protobuf args");
      return;
    }
  }

  Result validate() override { return valid_; }

  Result get_upstream_rows(const std::vector<i64>& upstream_rows,
                           std::vector<i64>& downstream_rows) const override {
    Result valid;
    valid.set_success(true);
    for (i64 in_row : upstream_rows) {
      if (in_row >= args_.rows_size()) {
        RESULT_ERROR(&valid,
                     "Gather sampler received out of bounds request for "
                     "row %ld (max requestable row is %ld).",
                     in_row,
                     args_.rows_size());
        return valid;
      }
      downstream_rows.push_back(args_.rows(in_row));
    }
    return valid;
  }

  Result get_num_downstream_rows(i64 num_upstream_rows,
                                 i64& num_downstream_rows) const {
    num_downstream_rows = 0;
    for (i64 r : args_.rows()) {
      if (r >= num_upstream_rows) {
        break;
      }
      num_downstream_rows++;
    }
    Result valid;
    valid.set_success(true);
    return valid;
  }

 private:
  Result valid_;
  proto::GatherSamplerArgs args_;
};


class SpaceNullDomainSampler : public DomainSampler {
 public:
  SpaceNullDomainSampler(const std::vector<u8>& args)
    : DomainSampler("SpaceNull") {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "SpaceNull sampler provided with invalid protobuf args");
      return;
    }
  }

  Result validate() override {
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_upstream_rows(const std::vector<i64>& downstream_rows,
                           std::vector<i64>& upstream_rows) const {
    std::set<i64> required_rows;
    for (i64 r : downstream_rows) {
      required_rows.insert(r / args_.spacing());
    }
    for (i64 r : required_rows) {
      upstream_rows.push_back(r);
    }
    std::sort(upstream_rows.begin(), upstream_rows.end());
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_num_downstream_rows(i64 num_upstream_rows,
                                 i64& num_downstream_rows) const {
    num_downstream_rows = num_upstream_rows * args_.spacing();
    Result result;
    result.set_success(true);
    return result;
  }

 private:
  Result valid_;
  proto::SpaceNullSamplerArgs args_;
};


class SpaceRepeatDomainSampler : public DomainSampler {
 public:
  SpaceRepeatDomainSampler(const std::vector<u8>& args)
    : DomainSampler("SpaceRepeat") {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "SpaceRepeat sampler provided with invalid protobuf args");
      return;
    }
  }

  Result validate() override {
    std::set<i64> required_rows;
    for (i64 r : downstream_rows) {
      required_rows.insert(r / args_.spacing());
    }
    for (i64 r : required_rows) {
      upstream_rows.push_back(r);
    }
    std::sort(upstream_rows.begin(), upstream_rows.end());
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_upstream_rows(const std::vector<i64>& input_rows,
                           std::vector<i64>& output_rows) const {
    output_rows = input_rows;
    Result result;
    result.set_success(true);
    return result;
  }

  Result get_num_downstream_rows(i64 num_upstream_rows,
                                 i64& num_downstream_rows) const {
    num_downstream_rows = num_upstream_rows * args_.spacing();
    Result result;
    result.set_success(true);
    return result;
  }

 private:
  Result valid_;
  proto::SpaceRepeatSamplerArgs args_;
};

template <typename T>
DomainSamplerFactory make_domain_factory() {
  return [](const std::vector<u8>& args) {
    return new T(args);
  };
}
}

Result make_domain_sampler_instance(const std::string& sampler_type,
                                    const std::vector<u8>& sampler_args,
                                    DomainSampler*& sampler) {
  static std::map<std::string, TaskSamplerFactory> samplers = {
      {"All", make_domain_factory<DefaultDomainSampler>()},
      {"Strided", make_domain_factory<StridedDomainSampler>()},
      {"StridedRanges", make_domain_factory<StridedRangesDomainSampler>()},
      {"Gather", make_domain_factory<GatherDomainSampler>()}};

  Result result;
  result.set_success(true);

  // Check if sampler type exists
  auto it = samplers.find(sampler_type);
  if (it == samplers.end()) {
    RESULT_ERROR(&result, "DomainSampler type not found: %s",
                 sampler_type.c_str());
    return result;
  }

  // Validate sampler args
  DomainSamplerFactory factory = it->second;
  DomainTaskSampler* potential_sampler = factory(sampler_args);
  result = potential_sampler->validate();
  if (!result.success()) {
    delete potential_sampler;
  } else {
    sampler = potential_sampler;
  }

  return result;
}


using TaskSamplerFactory =
    std::function<TaskSampler*(const std::vector<u8>&, i64 num_rows)>;

class AllTaskSampler : public TaskSampler {
 public:
  AllTaskSampler(const std::vector<u8>& args, int64 num_rows)
    : TaskSampler("All", num_rows) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_, "All sampler provided with invalid protobuf args");
      return;
    }
    if (args_.sample_size() <= 0) {
      RESULT_ERROR(&valid_,
                   "All sampler sample size (%ld) must be greater than 0",
                   args_.sample_size());
      return;
    }
    if (args_.warmup_size() < 0) {
      RESULT_ERROR(&valid_,
                   "All sampler warmup size (%ld) must be non-negative",
                   args_.warmup_size());
      return;
    }
    total_samples_ = (i64)std::ceil((float)num_rows_ / args_.sample_size());
    for (i64 i = 0; i < num_rows_; i += args_.sample_size()) {
      offset_at_sample_.push_back(i);
    }
  }

  Result validate() override {
    Result result;
    result.set_success(true);
    return result;
  }

  i64 total_rows() const override { return num_rows_; }

  i64 total_tasks() const override {
    return total_tasks_;
  }

  TaskRows next_task() override {
    assert(curr_task_idx_ < total_tasks_);
    return task_at(curr_task_idx_++);
  }

  void reset() override { curr_task_idx_ = 0; }

  TaskRows task_at(i64 task_idx) override {
    i64 pos = args_.sample_size() * task_idx;
    i64 ws = std::max(0l, pos - args_.warmup_size());
    i64 s = pos;
    i64 e = std::min(total_rows(), pos + args_.sample_size());
    assert(ws >= 0);
    assert(s >= 0);
    assert(e <= total_rows());
    RowSample sample;
    for (i64 i = ws; i < s; ++i) {
      sample.warmup_rows.push_back(i);
    }
    for (i64 i = s; i < e; ++i) {
      sample.rows.push_back(i);
    }
    return task;
  }

  i64 offset_at_task(i64 task_idx) const override {
    return offset_at_task_.at(task_idx);
  }

 private:
  Result valid_;
  proto::AllTaskSamplerArgs args_;
  i64 curr_task_idx_ = 0;
  i64 total_tasks_;
  std::vector<i64> offset_at_task_;
};

class StridedRangeTaskSampler : public TaskSampler {
 public:
  StridedRangeTaskSampler(const std::vector<u8>& args, i64 num_rows)
    : TaskSampler("StridedRange", num_rows) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "StridedRange sampler provided with invalid protobuf args");
      return;
    }
    if (args_.stride() <= 0) {
      RESULT_ERROR(&valid_,
                   "StridedRange stride (%ld) must be greater than zero",
                   args_.stride());
      return;
    }
    if (args_.warmup_starts_size() != args_.starts_size() ||
        args_.starts_size() != args_.ends_size()) {
      RESULT_ERROR(&valid_,
                   "StridedRange warmups, starts, and ends not the same size");
      return;
    }
    for (i64 i = 0; i < args_.warmup_starts_size(); ++i) {
      if (args_.warmup_starts(i) > args_.starts(i)) {
        RESULT_ERROR(
            &valid_,
            "StridedRange warmup start (%ld) should not be after start (%ld)",
            args_.warmup_starts(i), args_.starts(i));
        return;
      }
      if (args_.starts(i) > args_.ends(i)) {
        RESULT_ERROR(&valid_,
                     "StridedRange start (%ld) should not be after end (%ld)",
                     args_.starts(i), args_.ends(i));
        return;
      }
      if (args_.ends(i) > num_rows_) {
        RESULT_ERROR(
            &valid_,
            "StridedRange end (%ld) should be less than table num rows (%ld)",
            args_.ends(i), num_rows_);
        return;
      }
      i64 rows =
          ceil((args_.ends(i) - args_.starts(i)) / (float)args_.stride());
      offset_at_task_.push_back(total_rows_);
      total_rows_ += rows;
    }
    total_tasks_ = args_.warmup_starts_size();
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_tasks() const override { return total_tasks_; }

  RowSample next_task() override {
    assert(curr_task_idx_ < total_tasks_);
    return task_at(curr_task_idx_++);
  }

  void reset() override { curr_task_idx_ = 0; }

  RowSample task_at(i64 task_idx) override {
    i64 stride = args_.stride();
    i64 ws = args_.warmup_starts(task_idx);
    i64 s = args_.starts(task_idx);
    i64 e = args_.ends(task_idx);
    RowSample task;
    for (i64 i = ws; i < s; i += stride) {
      task.warmup_rows.push_back(i);
    }
    for (i64 i = s; i < e; i += stride) {
      task.rows.push_back(i);
    }
    return task;
  }

  i64 offset_at_task(i64 task_idx) const override {
    return offset_at_task_.at(task_idx);
  }

 private:
  Result valid_;
  proto::StridedRangeTaskSamplerArgs args_;
  i64 total_rows_ = 0;
  i64 total_tasks_ = 0;
  std::vector<i64> offset_at_task_;
  i64 curr_task_idx_ = 0;
};

class GatherTaskSampler : public TaskSampler {
 public:
  GatherTaskSampler(const std::vector<u8>& args, i64 num_rows)
    : TaskSampler("Gather", num_rows) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "Gather sampler provided with invalid protobuf args");
      return;
    }
    for (i32 i = 0; i < args_.samples_size(); ++i) {
      auto& s = args_.samples(i);
      i64 rows = args_.samples(i).rows_size();
      offset_at_task_.push_back(total_rows_);
      total_rows_ += rows;
    }
    total_tasks_ = args_.tasks_size();
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_tasks() const override { return total_tasks_; }

  RowSample next_task() override {
    assert(curr_task_idx_ < total_tasks_);
    return task_at(curr_task_idx_++);
  }

  void reset() override { curr_task_idx_ = 0; }

  RowTask task_at(i64 task_idx) override {
    RowSample task;
    auto& s = args_.samples(curr_task_idx_);
    task.warmup_rows =
        std::vector<i64>(s.warmup_rows().begin(), s.warmup_rows().end());
    task.rows = std::vector<i64>(s.rows().begin(), s.rows().end());
    return task;
  }

  i64 offset_at_task(i64 task_idx) const override {
    return offset_at_task_.at(task_idx);
  }

 private:
  Result valid_;
  proto::GatherTaskSamplerArgs args_;
  i64 total_rows_ = 0;
  i64 total_tasks_ = 0;
  std::vector<i64> offset_at_task_;
  i64 curr_task_idx_ = 0;
};

template <typename T>
TaskSamplerFactory make_factory() {
  return [](const std::vector<u8>& args, i64 num_rows) {
    return new T(args, num_rows);
  };
}
}

Result make_task_sampler_instance(const std::string& sampler_type,
                                  const std::vector<u8>& sampler_args,
                                  i64 num_rows,
                                  TaskSampler*& sampler) {
  static std::map<std::string, TaskSamplerFactory> samplers = {
      {"All", make_factory<AllTaskSampler>()},
      {"StridedRange", make_factory<StridedRangeTaskSampler>()},
      {"Gather", make_factory<GatherTaskSampler>()}};

  Result result;
  result.set_success(true);

  // Check if sampler type exists
  auto it = samplers.find(sampler_type);
  if (it == samplers.end()) {
    RESULT_ERROR(&result, "TaskSampler type not found: %s", sampler_type.c_str());
    return result;
  }

  // Validate sampler args
  TaskSamplerFactory factory = it->second;
  TaskSampler* potential_sampler = factory(sampler_args, num_rows);
  result = potential_sampler->validate();
  if (!result.success()) {
    delete potential_sampler;
  } else {
    sampler = potential_sampler;
  }

  return result;
}

}
}

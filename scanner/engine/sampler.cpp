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
#include "scanner/sampler_args.pb.h"

#include <cmath>
#include <vector>
#include <algorithm>

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

  Result get_downstream_rows(
      const std::vector<i64>& upstream_rows, std::vector<i64>& downstream_rows,
      std::vector<i64>& downstream_upstream_mapping) const {
    downstream_rows = upstream_rows;
    for (i64 i = 0; i < upstream_rows.size(); ++i) {
      downstream_upstream_mapping.push_back(i);
    }
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

  Result get_downstream_rows(
      const std::vector<i64>& upstream_rows, std::vector<i64>& downstream_rows,
      std::vector<i64>& downstream_upstream_mapping) const {
    for (i64 i = 0; i < upstream_rows.size(); ++i) {
      i64 in = upstream_rows[i];
      if (in % args_.stride() == 0) {
        downstream_rows.push_back(in / args_.stride());
        downstream_upstream_mapping.push_back(i);
      }
    }
    Result result;
    result.set_success(true);
    return result;
  }

 private:
  Result valid_;
  proto::StridedSamplerArgs args_;
};

class StridedRangesDomainSampler : public DomainSampler {
 public:
  StridedRangesDomainSampler(const std::vector<u8>& args)
    : DomainSampler("StridedRanges") {
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

  Result get_downstream_rows(
      const std::vector<i64>& upstream_rows, std::vector<i64>& downstream_rows,
      std::vector<i64>& downstream_upstream_mapping) const {
    i64 offset = 0;
    i64 range_idx = 0;
    for (i64 i = 0; i < upstream_rows.size(); ++i) {
      i64 r = upstream_rows[i];
      while (range_idx < args_.ends_size() &&
             !(r >= args_.starts(range_idx) && r < args_.ends(range_idx))) {
        // Add number of valid rows in this range sequence to offset
        offset += (args_.ends(range_idx) - args_.starts(range_idx) +
                   args_.stride() - 1) /
                  args_.stride();
        range_idx++;
      }
      if (range_idx == args_.ends_size()) {
        break;
      }
      i64 relative_r = (r - args_.starts(range_idx));
      if (relative_r % args_.stride() == 0) {
        downstream_rows.push_back(offset + relative_r / args_.stride());
        downstream_upstream_mapping.push_back(i);
      }
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
    i64 offset = 0;
    for (i64 r : args_.rows()) {
      gather_rows_[r] = offset++;
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
                     "row %ld (max requestable row is %d).",
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

  Result get_downstream_rows(
      const std::vector<i64>& upstream_rows, std::vector<i64>& downstream_rows,
      std::vector<i64>& downstream_upstream_mapping) const {
    for (i64 i = 0; i < upstream_rows.size(); ++i) {
      i64 r = upstream_rows[i];
      if (gather_rows_.count(r) > 0) {
        downstream_rows.push_back(gather_rows_.at(r));
        downstream_upstream_mapping.push_back(i);
      }
    }
    Result valid;
    valid.set_success(true);
    return valid;
  }

 private:
  Result valid_;
  proto::GatherSamplerArgs args_;
  std::map<i64, i64> gather_rows_;
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
    return valid_;
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

  Result get_downstream_rows(
      const std::vector<i64>& upstream_rows, std::vector<i64>& downstream_rows,
      std::vector<i64>& downstream_upstream_mapping) const {
    for (i64 i = 0; i < upstream_rows.size(); ++i) {
      i64 r = upstream_rows[i];
      i64 base = r * args_.spacing();
      downstream_rows.push_back(base);
      downstream_upstream_mapping.push_back(i);
      for (i64 offset = base + 1; offset < base + args_.spacing(); ++offset) {
        downstream_rows.push_back(offset);
        downstream_upstream_mapping.push_back(-1);
      }
    }
    Result valid;
    valid.set_success(true);
    return valid;
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

  Result validate() override { return valid_; }

  Result get_upstream_rows(const std::vector<i64>& input_rows,
                           std::vector<i64>& output_rows) const {
    std::unordered_set<i64> required_rows;
    for (i64 r : input_rows) {
      required_rows.insert(r / args_.spacing());
    }
    output_rows = std::vector<i64>(required_rows.begin(), required_rows.end());
    std::sort(output_rows.begin(), output_rows.end());
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

  Result get_downstream_rows(
      const std::vector<i64>& upstream_rows, std::vector<i64>& downstream_rows,
      std::vector<i64>& downstream_upstream_mapping) const {
    for (i64 i = 0; i < upstream_rows.size(); ++i) {
      i64 r = upstream_rows[i];
      i64 base = r * args_.spacing();
      for (i64 offset = base; offset < base + args_.spacing(); ++offset) {
        downstream_rows.push_back(offset);
        downstream_upstream_mapping.push_back(i);
      }
    }
    Result valid;
    valid.set_success(true);
    return valid;
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
  static std::map<std::string, DomainSamplerFactory> samplers = {
      {"All", make_domain_factory<DefaultDomainSampler>()},
      {"Strided", make_domain_factory<StridedDomainSampler>()},
      {"StridedRanges", make_domain_factory<StridedRangesDomainSampler>()},
      {"Gather", make_domain_factory<GatherDomainSampler>()},
      {"SpaceNull", make_domain_factory<SpaceNullDomainSampler>()},
      {"SpaceRepeat", make_domain_factory<SpaceRepeatDomainSampler>()},
  };

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
  DomainSampler* potential_sampler = factory(sampler_args);
  result = potential_sampler->validate();
  if (!result.success()) {
    delete potential_sampler;
  } else {
    sampler = potential_sampler;
  }

  return result;
}

namespace {

using PartitionerFactory =
    std::function<Partitioner*(const std::vector<u8>&, i64 num_rows)>;

class StridedPartitioner : public Partitioner {
 public:
  StridedPartitioner(const std::vector<u8>& args, i64 num_rows)
    : Partitioner("Strided", num_rows) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_, "All sampler provided with invalid protobuf args");
      return;
    }
    if (args_.stride() <= 0) {
      RESULT_ERROR(&valid_,
                   "Strided partitioner stride (%ld) must be greater than 0",
                   args_.stride());
      return;
    }
    if (args_.group_size() <= 0) {
      RESULT_ERROR(
          &valid_,
          "Strided partitioner group size (%ld) must be greater than 0",
          args_.group_size());
      return;
    }
    i64 num_strided_rows = (num_rows_ + args_.stride() - 1) / args_.stride();
    total_groups_ =
        (i64)std::ceil(num_strided_rows / (float)args_.group_size());
    for (i64 i = 0; i < num_strided_rows; i += args_.group_size()) {
      offset_at_group_.push_back(i);
    }
    offset_at_group_.push_back(num_strided_rows);
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override {
    return (num_rows_ + args_.stride() - 1) / args_.stride();
  }

  i64 total_groups() const override { return total_groups_; }

  std::vector<i64> total_rows_per_group() const override {
    std::vector<i64> rows;
    for (i64 i = 0; i < total_groups_; ++i) {
      rows.push_back(offset_at_group_[i + 1] - offset_at_group_[i]);
    }
    return rows;
  }

  PartitionGroup next_group() override {
    assert(curr_group_idx_ < total_groups_);
    return group_at(curr_group_idx_++);
  }

  void reset() override { curr_group_idx_ = 0; }

  PartitionGroup group_at(i64 group_idx) override {
    i64 pos = args_.group_size() * group_idx;
    i64 s = pos;
    i64 e = std::min(total_rows(), pos + args_.group_size());
    assert(s >= 0);
    assert(e <= total_rows());
    PartitionGroup group;
    for (i64 i = s; i < e; ++i) {
      group.rows.push_back(i * args_.stride());
    }
    return group;
  }

  i64 offset_at_group(i64 group_idx) const override {
    return offset_at_group_.at(group_idx);
  }

 private:
  Result valid_;
  proto::StridedPartitionerArgs args_;
  i64 curr_group_idx_ = 0;
  i64 total_groups_;
  std::vector<i64> offset_at_group_;
};

class StridedRangePartitioner : public Partitioner {
 public:
  StridedRangePartitioner(const std::vector<u8>& args, i64 num_rows)
    : Partitioner("StridedRange", num_rows) {
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
                   "StridedRange tarts and ends not the same size");
      return;
    }
    for (i64 i = 0; i < args_.starts_size(); ++i) {
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
      offset_at_group_.push_back(total_rows_);
      total_rows_ += rows;
    }
    offset_at_group_.push_back(total_rows_);
    total_groups_ = args_.starts_size();
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_groups() const override { return total_groups_; }

  std::vector<i64> total_rows_per_group() const override {
    std::vector<i64> rows;
    for (i64 i = 0; i < total_groups_; ++i) {
      rows.push_back(offset_at_group_[i + 1] - offset_at_group_[i]);
    }
    return rows;
  }

  PartitionGroup next_group() override {
    assert(curr_group_idx_ < total_groups_);
    return group_at(curr_group_idx_++);
  }

  void reset() override { curr_group_idx_ = 0; }

  PartitionGroup group_at(i64 group_idx) override {
    i64 stride = args_.stride();
    i64 s = args_.starts(group_idx);
    i64 e = args_.ends(group_idx);
    PartitionGroup group;
    for (i64 i = s; i < e; i += stride) {
      group.rows.push_back(i);
    }
    return group;
  }

  i64 offset_at_group(i64 group_idx) const override {
    return offset_at_group_.at(group_idx);
  }

 private:
  Result valid_;
  proto::StridedRangePartitionerArgs args_;
  i64 total_rows_ = 0;
  i64 total_groups_ = 0;
  std::vector<i64> offset_at_group_;
  i64 curr_group_idx_ = 0;
};

class GatherPartitioner : public Partitioner {
 public:
  GatherPartitioner(const std::vector<u8>& args, i64 num_rows)
    : Partitioner("Gather", num_rows) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "Gather sampler provided with invalid protobuf args");
      return;
    }
    for (i32 i = 0; i < args_.groups_size(); ++i) {
      auto& s = args_.groups(i);
      i64 rows = s.rows_size();
      offset_at_group_.push_back(total_rows_);
      total_rows_ += rows;
    }
    offset_at_group_.push_back(total_rows_);
    total_groups_ = args_.groups_size();
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_groups() const override { return total_groups_; }

  std::vector<i64> total_rows_per_group() const override {
    std::vector<i64> rows;
    for (i64 i = 0; i < total_groups_; ++i) {
      rows.push_back(offset_at_group_[i + 1] - offset_at_group_[i]);
    }
    return rows;
  }

  PartitionGroup next_group() override {
    assert(curr_group_idx_ < total_groups_);
    return group_at(curr_group_idx_++);
  }

  void reset() override { curr_group_idx_ = 0; }

  PartitionGroup group_at(i64 group_idx) override {
    PartitionGroup group;
    auto& s = args_.groups(curr_group_idx_);
    group.rows = std::vector<i64>(s.rows().begin(), s.rows().end());
    return group;
  }

  i64 offset_at_group(i64 group_idx) const override {
    return offset_at_group_.at(group_idx);
  }

 private:
  Result valid_;
  proto::GatherPartitionerArgs args_;
  i64 total_rows_ = 0;
  i64 total_groups_ = 0;
  std::vector<i64> offset_at_group_;
  i64 curr_group_idx_ = 0;
};

template <typename T>
PartitionerFactory make_factory() {
  return [](const std::vector<u8>& args, i64 num_rows) {
    return new T(args, num_rows);
  };
}
}

Result make_partitioner_instance(const std::string& sampler_type,
                                 const std::vector<u8>& sampler_args,
                                 i64 num_rows, Partitioner*& sampler) {
  static std::map<std::string, PartitionerFactory> samplers = {
      {"Strided", make_factory<StridedPartitioner>()},
      {"StridedRange", make_factory<StridedRangePartitioner>()},
      {"Gather", make_factory<GatherPartitioner>()}};

  Result result;
  result.set_success(true);

  // Check if sampler type exists
  auto it = samplers.find(sampler_type);
  if (it == samplers.end()) {
    RESULT_ERROR(&result, "Partitioner type not found: %s", sampler_type.c_str());
    return result;
  }

  // Validate sampler args
  PartitionerFactory factory = it->second;
  Partitioner* potential_sampler = factory(sampler_args, num_rows);
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

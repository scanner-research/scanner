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

using SamplerFactory =
    std::function<Sampler*(const std::vector<u8>&, const TableMetadata&)>;

class AllSampler : public Sampler {
 public:
  AllSampler(const std::vector<u8>& args, const TableMetadata& table)
    : Sampler("All", table) {
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
  }

  Result validate() override {
    Result result;
    result.set_success(true);
    return result;
  }

  i64 total_rows() const override { return table_.num_rows(); }

  i64 total_samples() const override {
    return (int)std::ceil((float)table_.num_rows() / args_.sample_size());
  }

  RowSample next_sample() override {
    RowSample sample;
    i64 ws = std::max(0l, rows_pos_ - args_.warmup_size());
    i64 s = rows_pos_;
    i64 e = std::min(total_rows(), rows_pos_ + args_.sample_size());
    rows_pos_ = e;
    assert(rows_pos_ <= total_rows());
    for (i64 i = ws; i < s; ++i) {
      sample.warmup_rows.push_back(i);
    }
    for (i64 i = s; i < e; ++i) {
      sample.rows.push_back(i);
    }
    return sample;
  }

  void reset() override { rows_pos_ = 0; }

 private:
  Result valid_;
  proto::AllSamplerArgs args_;
  i64 rows_pos_ = 0;
};

class StridedRangeSampler : public Sampler {
 public:
  StridedRangeSampler(const std::vector<u8>& args, const TableMetadata& table)
    : Sampler("StridedRange", table) {
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
      if (args_.ends(i) > table.num_rows()) {
        RESULT_ERROR(
            &valid_,
            "StridedRange end (%ld) should be less than table num rows (%ld)",
            args_.ends(i), table.num_rows());
        return;
      }
      total_rows_ +=
          ceil((args_.ends(i) - args_.starts(i)) / (float)args_.stride());
    }
    total_samples_ = args_.warmup_starts_size();
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_samples() const override { return total_samples_; }

  RowSample next_sample() override {
    RowSample sample;
    i64 stride = args_.stride();
    i64 ws = args_.warmup_starts(samples_pos_);
    i64 s = args_.starts(samples_pos_);
    i64 e = args_.ends(samples_pos_);
    for (i64 i = ws; i < s; i += stride) {
      sample.warmup_rows.push_back(i);
    }
    for (i64 i = s; i < e; i += stride) {
      sample.rows.push_back(i);
    }
    samples_pos_++;
    assert(samples_pos_ <= args_.warmup_starts_size());
    return sample;
  }

  void reset() override { samples_pos_ = 0; }

 private:
  Result valid_;
  proto::StridedRangeSamplerArgs args_;
  i64 total_rows_ = 0;
  i64 total_samples_ = 0;
  size_t samples_pos_ = 0;
};

class StencilSampler : public Sampler {
 public:
  StencilSampler(const std::vector<u8>& args, const TableMetadata& table)
    : Sampler("Stencil", table) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "Stencil sampler provided with invalid protobuf args");
      return;
    }
    if (args_.stride() <= 0) {
      RESULT_ERROR(&valid_, "Stencil stride (%ld) must be greater than zero",
                   args_.stride());
      return;
    }
    for (i64 i = 0; i < args_.stencil_size(); ++i) {
      if (args_.stencil(i) >= 0) {
        RESULT_ERROR(&valid_, "Stencil elements (%ld) must be less than zero",
                     args_.stencil(i));
        return;
      }
    }
    for (i64 i = 0; i < args_.starts_size(); ++i) {
      if (args_.starts(i) > args_.ends(i)) {
        RESULT_ERROR(&valid_,
                     "Stencil start (%ld) should not be after end (%ld)",
                     args_.starts(i), args_.ends(i));
        return;
      }
      for (i64 j = 0; j < args_.stencil_size(); ++j) {
        if (args_.starts(i) + args_.stencil(j) < 0) {
          RESULT_ERROR(
              &valid_,
              "Stencil start (%ld) with stencil offset %ld should be greater "
              "than zero",
              args_.starts(i), args_.stencil(j));
          return;
        }
      }
      if (args_.ends(i) > table.num_rows()) {
        RESULT_ERROR(&valid_,
                     "Stencil end (%ld) should be less "
                     "than table num rows (%ld)",
                     args_.ends(i), table.num_rows());
        return;
      }
      total_rows_ += (args_.ends(i) - args_.starts(i)) / args_.stride();
    }
    total_samples_ = args_.starts_size();
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_samples() const override {
    // NOTE(apoms): not a mistake, stencil sampler returns 1 row each time
    return total_rows_;
  }

  RowSample next_sample() override {
    RowSample sample;
    i64 stride = args_.stride();
    i64 s = args_.starts(samples_pos_);
    int curr_start = s + stride * rows_pos_;
    for (i64 off : args_.stencil()) {
      sample.warmup_rows.push_back(curr_start + off);
    }
    sample.rows.push_back(curr_start);

    rows_pos_++;
    i64 e = args_.ends(samples_pos_);
    if (curr_start + stride >= e) {
      rows_pos_ = 0;
      samples_pos_++;
    }
    assert(samples_pos_ <= args_.starts_size());
    return sample;
  }

  void reset() override {
    samples_pos_ = 0;
    rows_pos_ = 0;
  }

 private:
  Result valid_;
  proto::StencilSamplerArgs args_;
  i64 total_rows_ = 0;
  i64 total_samples_ = 0;
  size_t samples_pos_ = 0;
  size_t rows_pos_ = 0;
};

class GatherSampler : public Sampler {
 public:
  GatherSampler(const std::vector<u8>& args, const TableMetadata& table)
    : Sampler("Gather", table) {
    valid_.set_success(true);
    if (!args_.ParseFromArray(args.data(), args.size())) {
      RESULT_ERROR(&valid_,
                   "Gather sampler provided with invalid protobuf args");
      return;
    }
    for (i32 i = 0; i < args_.samples_size(); ++i) {
      auto& s = args_.samples(i);
      total_rows_ += args_.samples(i).rows_size();
    }
  }

  Result validate() override { return valid_; }

  i64 total_rows() const override { return total_rows_; }

  i64 total_samples() const override { return args_.samples_size(); }

  RowSample next_sample() override {
    RowSample sample;
    auto& s = args_.samples(samples_pos_);
    sample.warmup_rows =
        std::vector<i64>(s.warmup_rows().begin(), s.warmup_rows().end());
    sample.rows = std::vector<i64>(s.rows().begin(), s.rows().end());
    samples_pos_++;
    assert(samples_pos_ <= args_.samples_size());
    return sample;
  }

  void reset() override { samples_pos_ = 0; }

 private:
  Result valid_;
  proto::GatherSamplerArgs args_;
  i64 total_rows_ = 0;
  size_t samples_pos_ = 0;
};

template <typename T>
SamplerFactory make_factory() {
  return [](const std::vector<u8>& args, const TableMetadata& table) {
    return new T(args, table);
  };
}
}

Result make_sampler_instance(const std::string& sampler_type,
                             const std::vector<u8>& sampler_args,
                             const TableMetadata& sampled_table,
                             Sampler*& sampler) {
  static std::map<std::string, SamplerFactory> samplers = {
      {"All", make_factory<AllSampler>()},
      {"StridedRange", make_factory<StridedRangeSampler>()},
      {"Stencil", make_factory<StencilSampler>()},
      {"Gather", make_factory<GatherSampler>()}};

  Result result;
  result.set_success(true);

  // Check if sampler type exists
  auto it = samplers.find(sampler_type);
  if (it == samplers.end()) {
    RESULT_ERROR(&result, "Sampler type not found: %s", sampler_type.c_str());
    return result;
  }

  // Validate sampler args
  SamplerFactory factory = it->second;
  Sampler* potential_sampler = factory(sampler_args, sampled_table);
  result = potential_sampler->validate();
  if (!result.success()) {
    delete potential_sampler;
  } else {
    sampler = potential_sampler;
  }

  return result;
}

TaskSampler::TaskSampler(
    const std::map<std::string, TableMetadata>& table_metas,
    const proto::Task& task)
  : table_metas_(table_metas), task_(task) {
  valid_.set_success(true);
  if (table_metas.count(task.output_table_name()) == 0) {
    RESULT_ERROR(&valid_, "Output table %s does not exist.",
                 task.output_table_name().c_str());
    return;
  }
  // Create samplers for this task
  for (auto& sample : task.samples()) {
    if (table_metas.count(sample.table_name()) == 0) {
      RESULT_ERROR(&valid_, "Requested table %s does not exist.",
                   sample.table_name().c_str());
      return;
    }
    const TableMetadata& t_meta = table_metas.at(sample.table_name());
    std::vector<u8> sampler_args(sample.sampling_args().begin(),
                                 sample.sampling_args().end());
    Sampler* sampler = nullptr;
    valid_ = make_sampler_instance(sample.sampling_function(), sampler_args,
                                   t_meta, sampler);
    if (!valid_.success()) {
      return;
    }
    samplers_.emplace_back(sampler);
  }
  total_rows_ = samplers_[0]->total_rows();
  total_samples_ = samplers_[0]->total_samples();
  for (auto& sampler : samplers_) {
    if (sampler->total_rows() != total_rows_) {
      RESULT_ERROR(&valid_,
                   "Samplers for task %s output a different number "
                   "of rows (%ld vs. %ld)",
                   task.output_table_name().c_str(), sampler->total_rows(),
                   total_rows_);
      return;
    }
    if (sampler->total_samples() != total_samples_) {
      RESULT_ERROR(&valid_,
                   "Samplers for task %s output a different number "
                   "of samples (%ld vs. %ld)",
                   task.output_table_name().c_str(), sampler->total_samples(),
                   total_samples_);
      return;
    }
  }
  table_id_ = table_metas.at(task.output_table_name()).id();
}

Result TaskSampler::validate() { return valid_; }

i64 TaskSampler::total_rows() { return total_rows_; }

i64 TaskSampler::total_samples() { return total_samples_; }

Result TaskSampler::next_work(proto::NewWork& new_work) {
  if (!valid_.success()) {
    return valid_;
  }

  i64 sample_num = samples_pos_;
  samples_pos_++;
  assert(samples_pos_ <= total_samples_);

  i64 item_id = sample_num;

  proto::LoadWorkEntry& load_item = *new_work.mutable_load_work();
  load_item.set_io_item_index(item_id);
  i64 warmup_rows = 0;
  i64 rows = 0;
  for (i32 i = 0; i < task_.samples_size(); ++i) {
    auto& sample = task_.samples(i);
    const TableMetadata& t_meta = table_metas_.at(sample.table_name());
    i32 sample_table_id = t_meta.id();

    auto& sampler = samplers_[i];
    RowSample row_sample = sampler->next_sample();

    proto::LoadSample* load_sample = load_item.add_samples();
    load_sample->set_table_id(sample_table_id);
    for (auto col_name : sample.column_names()) {
      load_sample->add_column_ids(t_meta.column_id(col_name));
    }
    load_sample->set_warmup_size(row_sample.warmup_rows.size());
    for (i64 r : row_sample.warmup_rows) {
      load_sample->add_rows(r);
    }
    for (i64 r : row_sample.rows) {
      load_sample->add_rows(r);
    }
    if (i == 0) {
      warmup_rows = row_sample.warmup_rows.size();
      rows = row_sample.rows.size();
    } else {
      if (row_sample.warmup_rows.size() != warmup_rows) {
        RESULT_ERROR(&valid_,
                     "Samplers for task %s output a different number "
                     "of warmup rows per sample (%ld vs. %ld)",
                     task_.output_table_name().c_str(),
                     row_sample.warmup_rows.size(), warmup_rows);
        return valid_;
      }
      if (row_sample.rows.size() != rows) {
        RESULT_ERROR(&valid_,
                     "Samplers for task %s output a different number "
                     "of rows per sample (%ld vs. %ld)",
                     task_.output_table_name().c_str(), row_sample.rows.size(),
                     rows);
        return valid_;
      }
    }
  }

  proto::IOItem& item = *new_work.mutable_io_item();
  item.set_table_id(table_id_);
  item.set_item_id(item_id);
  item.set_start_row(allocated_rows_);
  item.set_end_row(allocated_rows_ + rows);

  allocated_rows_ += rows;

  return valid_;
}
}
}

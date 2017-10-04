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

#include "scanner/engine/dag_analysis.h"
#include "scanner/api/op.h"
#include "scanner/api/kernel.h"

namespace scanner {
namespace internal {

bool is_builtin_op(const std::string& name) {
  for (const auto& n : BUILTIN_OP_NAMES) {
    if (n == name) {
      return true;
    }
  }
  return false;
}

Result validate_jobs_and_ops(
    DatabaseMetadata& meta, TableMetaCache& table_metas,
    const std::vector<proto::Job>& jobs,
    const std::vector<proto::Op>& ops,
    DAGAnalysisInfo& info) {
  std::vector<i32>& op_slice_level = info.op_slice_level;
  std::map<i64, i64>& input_ops = info.input_ops;
  std::map<i64, i64>& slice_ops = info.slice_ops;
  std::map<i64, i64>& unslice_ops = info.unslice_ops;
  std::map<i64, i64>& sampling_ops = info.sampling_ops;
  std::map<i64, std::vector<i64>>& op_children = info.op_children;
  {
    // Validate ops
    OpRegistry* op_registry = get_op_registry();
    KernelRegistry* kernel_registry = get_kernel_registry();

    i32 op_idx = 0;
    // Keep track of op names and outputs for verifying that requested
    // edges between ops are valid
    std::vector<std::vector<std::string>> op_outputs;
    // Slices are currently restricted to not nest and there to only exist
    // a single slice grouping from start to finish currently.
    for (auto& op : ops) {
      op_names.push_back(op.name());

      // Input Op's output is defined by the input table column they sample
      if (op.name() == INPUT_OP_NAME) {
        if (op.inputs().size() == 0) {
          RESULT_ERROR(result, "Input op at %d did not specify any inputs.",
                       op_idx);
          return;
        }
        if (op.inputs().size() > 1) {
          RESULT_ERROR(result, "Input op at %d specified more than one input.",
                       op_idx);
          return;
        }
        op_outputs.emplace_back();
        op_outputs.back().push_back(op.inputs(0).column());
        size_t input_ops_size = input_ops.size();
        input_ops[op_idx] = input_ops_size;
        op_slice_level.push_back(0);
        op_idx++;
        continue;
      }

      // Verify the inputs for this Op
      i32 input_count = op.inputs().size();
      i32 input_slice_level = 0;
      if (input_count > 0) {
        input_slice_level = op_slice_level.at(op.inputs(0).op_index());
      }
      for (auto& input : op.inputs()) {
        // Verify inputs are topologically sorted so we can traverse linearly
        if (input.op_index() >= op_idx) {
          RESULT_ERROR(result,
                       "Op %s (%d) referenced input index %d."
                       "Ops must be specified in topo sort order.",
                       op.name().c_str(), op_idx, input.op_index());
          return;
        }

        // Verify the requested input is provided by the Op the input is being
        // requested from.
        const std::string& input_op_name = op_names.at(input.op_index());
        const std::vector<std::string>& inputs =
            op_outputs.at(input.op_index());
        const std::string requested_input_column = input.column();
        bool found = false;
        for (auto& out_col : inputs) {
          if (requested_input_column == out_col) {
            found = true;
            break;
          }
        }
        if (!found) {
          RESULT_ERROR(result,
                       "Op %s at index %d requested column %s from input "
                       "Op %s at index %d but that Op does not have the "
                       "requsted column.",
                       op.name().c_str(), op_idx,
                       requested_input_column.c_str(), input_op_name.c_str(),
                       input.op_index());
          return;
        }

        // Verify inputs are from the same slice level
        if (op_slice_level.at(input.op_index()) != input_slice_level) {
          RESULT_ERROR(result,
                       "Input Op %s (%d) specified inputs at "
                       "different slice levels (%d vs %d). Ops within at "
                       "a slice level should only receive inputs from other "
                       "Ops at the same slice level.",
                       op.name(), op_idx, input_slice_level,
                       op_slice_level.at(input.op_index()));
          return;
        }

        // HACK(apoms): we currently restrict all unslice outputs to only
        // be consumed by an output Op to make it easy to schedule each
        // slice like an independent task.
        if (input_op_name == UNSLICE_OP_NAME &&
            op.name() != OUTPUT_TABLE_NAME) {
          RESULT_ERROR(result,
                       "Unslice Op specified as input to %s Op. Scanner "
                       "currently only supports Output Ops consuming "
                       "the results of an Unslice Op.",
                       op.name());
        }

        // Keep op children info for later analysis
        op_children[input.op_idx()].push_back(op_idx);
      }

      // Slice
      int output_silce_level = input_slice_level;
      if (op.name() == SLICE_OP_NAME) {
        if (output_slice_level > 0) {
          RESULT_ERROR(result, "Nested slicing not currently supported.");
          return;
        }
        size_t slice_ops_size = slice_ops.size();
        slice_ops[op_idx] = slice_ops_size;
        op_outputs.emplace_back();
        for (auto& input : op.inputs()) {
          op_outputs.back().push_back(input.column());
        }
        output_slice_level += 1;
      }
      // Unslice
      else if (op.name() == UNSLICE_OP_NAME) {
        if (input_slice_level == 0) {
          RESULT_ERROR(result,
                       "Unslice received inputs that have not been "
                       "sliced.");
          return;
        }
        size_t unslice_ops_size = unslice_ops.size();
        unslice_ops[op_idx] = unslice_ops_size;
        op_outputs.emplace_back();
        for (auto& input : op.inputs()) {
          op_outputs.back().push_back(input.column());
        }
        output_slice_level -= 1;
      }
      // Sample & Space
      else if (op.name() == "Sample" || op.name() == "SampleFrame" ||
               op.name() == "Space" || op.name() == "SpaceFrame") {
        size_t sampling_ops_size = sampling_ops.size();
        sampling_ops[op_idx] = sampling_ops_size;
        op_outputs.emplace_back();
        for (auto& input : op.inputs()) {
          op_outputs.back().push_back(input.column());
        }
      }
      // Output
      else if (op.name() == OUTPUT_OP_NAME) {
        if (input_slice_level != 0) {
          RESULT_ERROR(result,
                       "Final output columns are sliced. Final outputs must "
                       "be unsliced.");
          return;
        }
      }
      // Verify op exists and record outputs
      else {
        op_outputs.emplace_back();
        if (!op_registry->has_op(op.name())) {
          RESULT_ERROR(result, "Op %s is not registered.", op.name().c_str());
          return;
        } else {
          // Keep track of op outputs for verifying dependent ops
          for (auto& col :
               op_registry->get_op_info(op.name())->output_columns()) {
            op_outputs.back().push_back(col.name());
          }
        }
        if (!kernel_registry->has_kernel(op.name(), op.device_type())) {
          RESULT_ERROR(result,
                       "Op %s at index %d requested kernel with device type "
                       "%s but no such kernel exists.",
                       op.name().c_str(), op_idx,
                       (op.device_type() == DeviceType::CPU ? "CPU" : "GPU"));
          return;
        }
      }
      op_slice_level.push_back(output_slice_level);
      // Perform Op parameter verification (stenciling, batching, # inputs)
      if (!is_builtin_op(op.name())) {
        OpInfo* info = op_registry->get_op_info(op.name());
        KernelFactory* factory =
            kernel_registry->get_kernel(op.name(), op.device_type());
        // Check that the # of inputs match up
        // TODO(apoms): type check for frame
        if (!info->variadic_inputs()) {
          i32 expected_inputs = info->input_columns().size();
          if (expected_inputs != input_count) {
            RESULT_ERROR(
                result,
                "Op %s at index %d expects %d input columns, but received %d",
                op.name().c_str(), op_idx, expected_inputs, input_count);
            return;
          }
        }

        // Check that a stencil is not set on a non-stenciling kernel
        // If can't stencil, then should have a zero size stencil or a size 1
        // stencil with the element 0
        if (!info->can_stencil() &&
            !((op.stencil_size() == 0) ||
              (op.stencil_size() == 1 && op.stencil(0) == 0))) {
          RESULT_ERROR(
              result,
              "Op %s at index %d specified stencil but that Op was not "
              "declared to support stenciling. Add .stencil() to the Op "
              "declaration to support stenciling.",
              op.name().c_str(), op_idx);
          return;
        }
        // Check that a stencil is not set on a non-stenciling kernel
        if (!factory->can_batch() && op.batch() > 1) {
          RESULT_ERROR(
              result,
              "Op %s at index %d specified a batch size but the Kernel for "
              "that Op was not declared to support batching. Add .batch() to "
              "the Kernel declaration to support batching.",
              op.name().c_str(), op_idx);
          return;
        }
      }
      op_idx++;
    }
    if (op_names.size() < 2) {
      RESULT_ERROR(result,
                   "Must specify at least two Ops: "
                   "an Input Op, and an Output Op. "
                   "However, %lu Op(s) were specified.",
                   op_names.size());
      return;
    } else {
      if (op_names.back() != OUTPUT_OP_NAME) {
        RESULT_ERROR(result, "Last Op is %s but must be %s",
                     op_names.back().c_str(),
                     OUTPUT_OP_NAME.c_str());
        return;
      }
    }
  }

  // Validate table tasks
  std::set<std::string> job_output_table_names;
  for (auto& job : jobs) {
    if (job.output_table_name() == "") {
      RESULT_ERROR(result,
                   "Job specified with empty output table name. Output "
                   "tables can not have empty names")
      return;
    }
    if (meta.has_table(job.output_table_name())) {
      RESULT_ERROR(result,
                   "Job specified with duplicate output table name. "
                   "A table with name %s already exists.",
                   job.output_table_name().c_str());
      return;
    }
    if (job_output_table_names.count(job.output_table_name()) >
        0) {
      RESULT_ERROR(result,
                   "Multiple table tasks specified with output table name %s. "
                   "Table names must be unique.",
                   job.output_table_name().c_str());
      return;
    }
    job_output_table_names.insert(job.output_table_name());

    // Verify table task column inputs
    if (job.inputs().size() == 0) {
      RESULT_ERROR(
          result,
          "Job %s did not specify any table inputs. Jobs "
          "must specify at least one table to sample from.",
          job.output_table_name().c_str());
      return;
    } else {
      std::set<i32> used_input_ops;
      for (auto& column_input : job.inputs()) {
        // Verify input is specified on an Input Op
        if (used_input_ops.count(column_input.op_index()) > 0) {
          RESULT_ERROR(result,
                       "Job %s tried to set input column for Input Op "
                       "at %d twice.",
                       job.output_table_name().c_str(),
                       column_input.op_index());
          return;
        }
        if (input_ops.count(column_input.op_index()) == 0) {
          RESULT_ERROR(result,
                       "Job %s tried to set input column for Input Op "
                       "at %d, but this Op is not an Input Op.",
                       job.output_table_name().c_str(),
                       column_input.op_index());
          return;
        }
        used_input_ops.insert(column_input.op_index());
        // Verify column input table exists
        if (!meta.has_table(column_input.table_name())) {
          RESULT_ERROR(result,
                       "Job %s tried to sample from non-existent table "
                       "%s.",
                       job.output_table_name().c_str(),
                       column_input.table_name().c_str());
          return;
        }
        // Verify column input column exists in the requested table
        if (!table_metas.at(column_input.table_name())
                 .has_column(column_input.column_name())) {
          RESULT_ERROR(result,
                       "Job %s tried to sample column %s from table %s, "
                       "but that column is not in that table.",
                       job.output_table_name().c_str(),
                       column_input.column_name().c_str(),
                       column_input.table_name().c_str());
          return;
        }
      }
    }

    // Verify sampling args for table task
    {
      std::set<i32> used_sampling_ops;
      for (auto& sampling_args_assignment : job.sampling_args_assignment()) {
        if (used_sampling_ops.count(sampling_arg.op_index()) > 0) {
          RESULT_ERROR(result,
                       "Job %s tried to set sampling args for Op at %d "
                       "twice.",
                       job.output_table_name().c_str(),
                       sampling_arg.op_index());
          return;
        }
        if (sampling_ops.count(sampling_arg.op_index()) == 0) {
          RESULT_ERROR(result,
                       "Job %s tried to set sampling args for Op at %d, "
                       "but this Op is not a sampling Op.",
                       job.output_table_name().c_str(),
                       sampling_arg.op_index());
          return;
        }
        used_sampling_ops.insert(sampling_arg.op_index());
        // TODO(apoms): verify sampling args are valid
        if (sampling_args_assignment.size() == 0) {
          RESULT_ERROR(result,
                       "Job %s tried to set empty sampling args for Op at %d.",
                       job.output_table_name().c_str(),
                       sampling_arg.op_index());
          return;
        }
        i32 slice_level = op_slice_level.at(sampling_arg.op_index());
        if (slice_level == 0 &&
            sampling_args_assignment.size() > 1) {
          RESULT_ERROR(result,
                       "Job %s tried to set multiple sampling args for "
                       "Op at %d that has not been sliced.",
                       job.output_table_name().c_str(),
                       sampling_arg.op_index());
          return;
        }
      }
    }
  }
}

Result determine_input_rows_to_slices(
    DatabaseMetadata& meta, TableMetaCache& table_metas,
    const std::vector<proto::Job>& jobs,
    const std::vector<proto::Op>& ops,
    DAGAnalysisInfo& info,
    std::vector<std::map<i64, i64>>& slice_input_rows,
    std::vector<i64>& total_output_rows) {
  Result result;
  result.set_success(true);
  const std::vector<i32>& op_slice_level = info.op_slice_level;
  const std::map<i64, i64>& input_ops = info.input_ops;
  const std::map<i64, i64>& slice_ops = info.slice_ops;
  const std::map<i64, i64>& sampling_ops = info.sampling_ops;
  const std::map<i64, std::vector<i64>>& op_children = info.op_children;
  // For each job, use table rows to determine number of total possible outputs
  // by propagating downward through Op DAG
  for (const proto::Job& job : jobs) {
    slice_input_rows.emplace_back();
    const std::map<i64, i64>& job_slice_input_rows = slice_input_rows.back();
    // Create domain samplers using sampling args
    // Op idx -> samplers for each slice group
    std::map<i64, std::vector<std::unique_ptr<DomainSampler>>> domain_samplers;
    for (const proto::SamplingArgsAssignment& saa :
         job.sampling_args_assignment()) {
      std::vector<std::unique_ptr<DomainSampler>>& samplers =
          domain_samplers[saa.op_index()];
      // Assign number of rows to correct op
      for (auto& sa : saa.sampling_args()) {
        DomainSampler* sampler;
        result = make_domain_sampler_instance(
            sa.sampling_function(), std::vector<u8>(sa.sampling_args().begin(),
                                                    sa.sampling_args().end()),
            sampler);
        if (!result.success()) {
          return result;
        }
        domain_samplers.emplace_back(sampler);
      }
    }
    // Each Op can have a vector of outputs because of one level slicing
    // Op idx -> input columns -> slice groups
    std::vector<std::vector<std::vector<i64>>> op_num_inputs(ops.size());
    // Currently, we constrain there to only be a single number of slice groups
    // per job (no slicing in different ways) to make it easy to schedule
    // as tasks
    i64 number_of_slice_groups = -1;
    // First populate num rows from table inputs
    for (const proto::ColumnInput& ci : job.inputs()) {
      // Determine number of rows for the requested table
      i64 num_rows = table_metas.at(ci.table_name()).num_rows();
      // Assign number of rows to correct op
      op_num_inputs[ci.op_index()] = {{num_rows}};
    }
    bool success = false;
    std::vector<i64> ready_ops;
    for (auto& kv : input_ops) {
      ready_ops.push_back(kv.first);
    }
    while (!ready_ops.empty()) {
      i64 op_idx = ready_ops.back();
      ready_ops.pop_back();

      // Verify inputs are rate matched
      std::vector<i64> slice_group_outputs;
      {
        const std::vector<i64>& first_input_column_slice_groups =
            op_num_inputs.at(op_idx).at(0);
        // Check all columns match the first column
        for (const auto& input_column_slice_groups : op_num_inputs.at(op_idx)) {
          // Verify there are the same number of slice groups
          if (input_column_slice_groups.size() !=
              first_input_column_slice_groups.size()) {
            RESULT_ERROR(
                &result,
                "Job %s specified multiple inputs with a differing "
                "number of slice groups for %s Op at %ld (%ld vs %ld).",
                job.output_table_name().c_str(), ops[op_idx].name().c_str(),
                op_idx, first_input_column_slice_groups.size(),
                input_column_slice_groups.size());
            return result;
          }
          // Verify the number of rows for each slice group matches
          for (size_t i = 0; i < first_input_column_slice_groups.size(); ++i) {
            if (input_column_slice_groups.at(i) !=
                first_input_column_slice_groups.at(i)) {
              RESULT_ERROR(&result,
                           "Job %s specified multiple inputs with a differing "
                           "number of rows for slice group %d for %s Op at %ld "
                           "(%ld vs %ld).",
                           job.output_table_name().c_str(),
                           i, ops[op_idx].name().c_str(), op_idx,
                           input_column_slice_groups.at(i),
                           first_input_column_slice_groups.at(i));
              return result;
            }
          }
        }
        slice_group_outputs = first_input_column_slice_groups;
      }
      // Check if we are done
      if (ops[op_idx].name() == OUTPUT_OP_NAME) {
        // Should always be at slice level 0
        assert(slice_group_outputs.size() == 1);
        total_output_rows.push_back(slice_group_outputs.at(0));
        success = true;
        break;
      }
      // Check if this is a sampling Op
      if (sampling_ops.count(op_idx) > 0) {
        i64 sampling_op_idx = samping_ops.at(op_idx);
        // Verify number of samplers is equal to number of slice groups
        if (domain_samplers.at(op_idx).size() != slice_group_outputs.size()) {
          RESULT_ERROR(&result,
                       "Job %s specified %d samplers but there are %d slice "
                       "groups for %s Op at %ld.",
                       job.output_table_name().c_str(),
                       domain_samplers.at(op_idx).size(),
                       slice_group_outputs.size(), ops[op_idx].name().c_str(),
                       op_idx);
          return result;
        }
        // Apply domain samplers to determine downstream row count
        std::vector<i64> new_slice_group_outputs;
        for (size_t i = 0; i < slice_group_outputs.size(); ++i) {
          auto& sampler = domain_samplers.at(op_idx).at(i);
          i64 new_outputs = 0;
          result = sampler->get_num_downstream_rows(slice_group_outputs.at(i),
                                                    new_outputs);
          if (!result.success()) {
            return result;
          }
          new_slice_group_outputs.push_back(new_outputs);
        }
        slice_group_outputs = new_slice_group_outputs;
      }

      // Check if this is a slice op
      if (slice_ops.count(op_idx) > 0) {
        assert(op_slice_level.at(op_idx) == 1);
        assert(slice_group_outputs.size() == 0);
        // Create Partitioner to enumerate slices
        Partitioner* partitioner = nullptr;
        result = make_partitioner(
            ops.at(op_idx).partitioner_args().sampling_function(),
            ops.at(op_idx).partitioner_args().sampling_args(),
            slice_group_outputs.at(0),
            partitioner);
        if (!result.success()) {
          return result;
        }

        // Track job slice inputs so we can determine number of groups later
        job_slice_input_rows.at(op_idx) = slice_group_outputs.at(0);
        // Update outputs with the new slice group outputs for this partition
        slice_group_outputs = partitioner->total_rows_per_group();
        delete partitioner;

        if (number_of_slice_groups == -1) {
          number_of_slice_groups == slice_group_outputs.size();
        } else if (slice_group_outputs.size() != number_of_slice_groups) {
          RESULT_ERROR(
              &result,
              "Job %s specified one slice with %d groups and another "
              "slice with %d groups. Scanner currently does not "
              "support multiple slices with different numbers of groups "
              "in the same job.",
              job.output_table_name().c_str(),
              slice_group_output.size(), number_of_slice_groups);
          return result;
        }
      }

      // Check if this is an unslice op
      if (unslice_ops.count(op_idx) > 0) {
        assert(op_slice_level.at(op_idx) == 0);

        // Concatenate all slice group outputs
        i64 new_outputs = 0;
        for (i64 group_outputs : slice_group_outputs) {
          new_outputs += group_outputs;
        }
        slice_group_outputs = {new_outputs};
      }

      for (i64 child_op_idx : op_children.at(op_idx)) {
        op_num_inputs.at(child_op_idx).push_back(slice_group_outputs);
        // Check if Op has all of its inputs. If so, add to ready stack
        if (op_num_inputs.at(child_op_idx).size() ==
            ops[child_op_idx].inputs_size()) {
          ready_ops.push_back(child_op_idx);
        }
      }
    }
    if (!success) {
      // This should never happen...
      assert(false);
    }
  }
  return result;
}

std::tuple<i64, i64> determine_stencil_bounds(
    const std::vector<proto::Op>& ops) {
  i64 min = std::numeric_limits<i64>::max();
  i64 max = std::numeric_limits<i64>::min();

  OpRegistry* op_registry = get_op_registry();
  // Skip input and output table ops
  for (size_t i = 0; i < ops.size() - 1; ++i) {
    auto& op = ops[i];
    const auto& op_info = op_registry->get_op_info(op.name());

    std::vector<i32> stencil;
    if (op.stencil_size() > 0) {
      stencil = std::vector<i32>(op.stencil().begin(), op.stencil().end());
    } else {
      stencil = op_info->preferred_stencil();
    }

    min = std::min((i64)stencil[0], min);
    max = std::max((i64)stencil[stencil.size() - 1], max);
  }

  return std::make_tuple(min, max);
}

Result derive_slice_final_output_rows(
    const proto::Job& job, const std::vector<proto::Op>& ops, i64 slice_op_idx,
    i64 slice_input_rows, DAGAnalysisInfo& info,
    std::vector<i64>& slice_output_partition) {
  Result result;
  result.set_success(true);
  // First create partitioner to determine slice groups
  Partitioner* partitioner = nullptr;
  result = make_partitioner(
      ops.at(slice_op_idx).partitioner_args().sampling_function(),
      ops.at(slice_op_idx).partitioner_args().sampling_args(), slice_input_rows,
      partitioner);
  if (!result.success()) {
    return result;
  }
  // Traverse down graph from each slice group and count the number of rows
  // produced. This is the partition offset that we will use to split the graph
  std::vector<i64>& slice_rows = slice_output_partition;
  i64 current_offset = 0;
  for (size_t i = 0; i < partitioner->total_groups(); ++i) {
    const PartitionGroup& g = partitioner->group_at(i);
    // Traverse down all children until reaching the output
    std::vector<i64> input_row_count(ops.size());
    std::vector<i64> next_queue;
    input_row_count[slice_op_idx] = g.rows.size();
    next_queue.push_back(slice_op_idx);
    while (!next_queue.empty()) {
      i64 op_idx = next_queue.back();
      next_queue.pop_back();

      auto& op = ops.at(op_idx);
      // Check if sampling Op or unslice Op
      i64 input_row_count = input_row_count.at(op_idx);
      i64 output_row_count = input_row_count;
      if (op.name() == "Space" || op.name() == "SpaceFrame" ||
          op.name() == "Sample" || op.name() == "SampleFrame") {
        auto& sa = saa.sampling_args(i);
        DomainSampler* sampler;
        result = make_domain_sampler_instance(
            sa.sampling_function(), std::vector<u8>(sa.sampling_args().begin(),
                                                    sa.sampling_args().end()),
            sampler);
        if (!result.success()) {
          return result;
        }
        // Perform row count modification
        result =
            sampler->get_num_downstream_rows(input_row_count, output_row_count);
        if (!result.success()) {
          return result;
        }
        delete sampler;
      }
      else if (op.name() == UNSLICE_OP_NAME) {
      }
      else if (op.name() == OUTPUT_OP_NAME) {
        // We are done
        current_offset += input_row_count;
        slice_rows.push_back(current_offset);
        break;
      }

      for (i64 cid : info.op_children.at(op_idx)) {
        input_row_count.at(cid) = g.rows.size();
        next_queue.push_back(cid);
      }
    }
  }
  assert(slice_rows.size() == partitioner->total_groups());
  return result;
}

}
}

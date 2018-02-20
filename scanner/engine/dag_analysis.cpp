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
#include "scanner/engine/sampler.h"
#include "scanner/engine/source_registry.h"
#include "scanner/engine/enumerator_registry.h"
#include "scanner/engine/column_enumerator.h"
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
  std::map<i64, i64>& input_ops = info.source_ops;
  std::map<i64, i64>& slice_ops = info.slice_ops;
  std::map<i64, i64>& unslice_ops = info.unslice_ops;
  std::map<i64, i64>& sampling_ops = info.sampling_ops;
  std::map<i64, std::vector<i64>>& op_children = info.op_children;

  Result result;
  result.set_success(true);
  {
    // Validate ops
    OpRegistry* op_registry = get_op_registry();
    KernelRegistry* kernel_registry = get_kernel_registry();
    SourceRegistry* source_registry = get_source_registry();
    EnumeratorRegistry* enumerator_registry = get_enumerator_registry();

    i32 op_idx = 0;
    // Keep track of op names and outputs for verifying that requested
    // edges between ops are valid
    std::vector<std::vector<std::string>> op_outputs;
    // Slices are currently restricted to not nest and there to only exist
    // a single slice grouping from start to finish currently.
    std::vector<std::string> op_names;
    for (auto& op : ops) {
      op_names.push_back(op.name());

      if (op.is_source()) {
        if (op.inputs().size() == 0) {
          RESULT_ERROR(&result, "Input op at %d did not specify any inputs.",
                       op_idx);
          return result;
        }
        if (op.inputs().size() > 1) {
          RESULT_ERROR(&result, "Input op at %d specified more than one input.",
                       op_idx);
          return result;
        }
        op_outputs.emplace_back();
        for (auto& col :
             source_registry->get_source(op.name())->output_columns()) {
          op_outputs.back().push_back(col.name());
        }
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
          RESULT_ERROR(&result,
                       "Op %s (%d) referenced input index %d."
                       "Ops must be specified in topo sort order.",
                       op.name().c_str(), op_idx, input.op_index());
          return result;
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
          RESULT_ERROR(&result,
                       "Op %s at index %d requested column %s from input "
                       "Op %s at index %d but that Op does not have the "
                       "requested column.",
                       op.name().c_str(), op_idx,
                       requested_input_column.c_str(), input_op_name.c_str(),
                       input.op_index());
          return result;
        }

        // Verify inputs are from the same slice level
        if (op_slice_level.at(input.op_index()) != input_slice_level) {
          RESULT_ERROR(&result,
                       "Input Op %s (%d) specified inputs at "
                       "different slice levels (%d vs %d). Ops within at "
                       "a slice level should only receive inputs from other "
                       "Ops at the same slice level.",
                       op.name().c_str(), op_idx, input_slice_level,
                       op_slice_level.at(input.op_index()));
          return result;
        }

        // HACK(apoms): we currently restrict all unslice outputs to only
        // be consumed by an output Op to make it easy to schedule each
        // slice like an independent task.
        if (input_op_name == UNSLICE_OP_NAME &&
            op.name() != OUTPUT_OP_NAME) {
          RESULT_ERROR(&result,
                       "Unslice Op specified as input to %s Op. Scanner "
                       "currently only supports Output Ops consuming "
                       "the results of an Unslice Op.",
                       op.name().c_str());
          return result;
        }

        // Keep op children info for later analysis
        op_children[input.op_index()].push_back(op_idx);
      }

      // Slice
      int output_slice_level = input_slice_level;
      if (op.name() == SLICE_OP_NAME) {
        if (output_slice_level > 0) {
          RESULT_ERROR(&result, "Nested slicing not currently supported.");
          return result;
        }
        size_t slice_ops_size = slice_ops.size();
        slice_ops[op_idx] = slice_ops_size;
        output_slice_level += 1;
        op_outputs.emplace_back();
        for (auto& input : op.inputs()) {
          op_outputs.back().push_back(input.column());
        }
      }
      // Unslice
      else if (op.name() == UNSLICE_OP_NAME) {
        if (input_slice_level == 0) {
          RESULT_ERROR(&result,
                       "Unslice received inputs that have not been "
                       "sliced.");
          return result;
        }
        size_t unslice_ops_size = unslice_ops.size();
        unslice_ops[op_idx] = unslice_ops_size;
        output_slice_level -= 1;
        op_outputs.emplace_back();
        for (auto& input : op.inputs()) {
          op_outputs.back().push_back(input.column());
        }
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
          RESULT_ERROR(&result,
                       "Final output columns are sliced. Final outputs must "
                       "be unsliced.");
          return result;
        }
      }
      // Verify source exists and record outputs
      else if (op.is_source()) {
        op_outputs.emplace_back();
        if (!source_registry->has_source(op.name())) {
          RESULT_ERROR(&result, "Source %s is not registered.",
                       op.name().c_str());
          return result;
        } else {
          // Keep track of source outputs for verifying dependent ops
          for (auto& col :
               source_registry->get_source(op.name())->output_columns()) {
            op_outputs.back().push_back(col.name());
          }
        }
        if (!enumerator_registry->has_enumerator(op.name())) {
          RESULT_ERROR(&result,
                       "Source %s at index %d does not have a corresponding "
                       "enumerator with name %s.",
                       op.name().c_str(), op_idx,
                       op.name().c_str());
          return result;
        }
      }
      // Verify op exists and record outputs
      else {
        op_outputs.emplace_back();
        if (!op_registry->has_op(op.name())) {
          RESULT_ERROR(&result, "Op %s is not registered.", op.name().c_str());
          return result;
        } else {
          // Keep track of op outputs for verifying dependent ops
          for (auto& col :
               op_registry->get_op_info(op.name())->output_columns()) {
            op_outputs.back().push_back(col.name());
          }
        }
        if (!kernel_registry->has_kernel(op.name(), op.device_type())) {
          RESULT_ERROR(&result,
                       "Op %s at index %d requested kernel with device type "
                       "%s but no such kernel exists.",
                       op.name().c_str(), op_idx,
                       (op.device_type() == DeviceType::CPU ? "CPU" : "GPU"));
          return result;
        }
      }
      op_slice_level.push_back(output_slice_level);
      // Perform Op parameter verification (stenciling, batching, # inputs)
      if (!op.is_source() && !is_builtin_op(op.name())) {
        OpInfo* info = op_registry->get_op_info(op.name());
        KernelFactory* factory =
            kernel_registry->get_kernel(op.name(), op.device_type());
        // Check that the # of inputs match up
        // TODO(apoms): type check for frame
        if (!info->variadic_inputs()) {
          i32 expected_inputs = info->input_columns().size();
          if (expected_inputs != input_count) {
            RESULT_ERROR(
                &result,
                "Op %s at index %d expects %d input columns, but received %d",
                op.name().c_str(), op_idx, expected_inputs, input_count);
            return result;
          }
        }

        // Check that a stencil is not set on a non-stenciling kernel
        // If can't stencil, then should have a zero size stencil or a size 1
        // stencil with the element 0
        if (!info->can_stencil() &&
            !((op.stencil_size() == 0) ||
              (op.stencil_size() == 1 && op.stencil(0) == 0))) {
          RESULT_ERROR(
              &result,
              "Op %s at index %d specified stencil but that Op was not "
              "declared to support stenciling. Add .stencil() to the Op "
              "declaration to support stenciling.",
              op.name().c_str(), op_idx);
          return result;
        }
        // Check that a stencil is not set on a non-stenciling kernel
        if (!factory->can_batch() && op.batch() > 1) {
          RESULT_ERROR(
              &result,
              "Op %s at index %d specified a batch size but the Kernel for "
              "that Op was not declared to support batching. Add .batch() to "
              "the Kernel declaration to support batching.",
              op.name().c_str(), op_idx);
          return result;
        }
      }
      op_idx++;
    }
    if (op_names.size() < 2) {
      RESULT_ERROR(&result,
                   "Must specify at least two Ops: "
                   "an Input Op, and an Output Op. "
                   "However, %lu Op(s) were specified.",
                   op_names.size());
      return result;
    } else {
      if (op_names.back() != OUTPUT_OP_NAME) {
        RESULT_ERROR(&result, "Last Op is %s but must be %s",
                     op_names.back().c_str(),
                     OUTPUT_OP_NAME.c_str());
        return result;
      }
    }
  }

  // Validate table tasks
  std::set<std::string> job_output_table_names;
  for (auto& job : jobs) {
    if (job.output_table_name() == "") {
      RESULT_ERROR(&result,
                   "Job specified with empty output table name. Output "
                   "tables can not have empty names")
      return result;
    }
    if (meta.has_table(job.output_table_name())) {
      RESULT_ERROR(&result,
                   "Job specified with duplicate output table name. "
                   "A table with name %s already exists.",
                   job.output_table_name().c_str());
      return result;
    }
    if (job_output_table_names.count(job.output_table_name()) > 0) {
      RESULT_ERROR(&result,
                   "Multiple table tasks specified with output table name %s. "
                   "Table names must be unique.",
                   job.output_table_name().c_str());
      return result;
    }
    job_output_table_names.insert(job.output_table_name());

    // Verify table task column inputs
    if (job.inputs().size() == 0) {
      RESULT_ERROR(
          &result,
          "Job %s did not specify any table inputs. Jobs "
          "must specify at least one table to sample from.",
          job.output_table_name().c_str());
      return result;
    } else {
      std::set<i32> used_input_ops;
      for (auto& column_input : job.inputs()) {
        // Verify input is specified on an Input Op
        if (used_input_ops.count(column_input.op_index()) > 0) {
          RESULT_ERROR(&result,
                       "Job %s tried to set input args for Source Op "
                       "at %d twice.",
                       job.output_table_name().c_str(),
                       column_input.op_index());
          return result;
        }
        if (input_ops.count(column_input.op_index()) == 0) {
          RESULT_ERROR(&result,
                       "Job %s tried to set input args for Source Op "
                       "at %d, but this Op is not a Source Op.",
                       job.output_table_name().c_str(),
                       column_input.op_index());
          return result;
        }
        used_input_ops.insert(column_input.op_index());
      }
    }

    // Verify sampling args for table task
    {
      std::set<i32> used_sampling_ops;
      for (auto& sampling_args_assignment : job.sampling_args_assignment()) {
        if (used_sampling_ops.count(sampling_args_assignment.op_index()) > 0) {
          RESULT_ERROR(&result,
                       "Job %s tried to set sampling args for Op at %d "
                       "twice.",
                       job.output_table_name().c_str(),
                       sampling_args_assignment.op_index());
          return result;
        }
        if (sampling_ops.count(sampling_args_assignment.op_index()) == 0 &&
            slice_ops.count(sampling_args_assignment.op_index()) == 0) {
          RESULT_ERROR(&result,
                       "Job %s tried to set sampling args for Op at %d, "
                       "but this Op is not a sampling or slicing Op.",
                       job.output_table_name().c_str(),
                       sampling_args_assignment.op_index());
          return result;
        }
        used_sampling_ops.insert(sampling_args_assignment.op_index());
        // TODO(apoms): verify sampling args are valid
        if (sampling_args_assignment.sampling_args().size() == 0) {
          RESULT_ERROR(&result,
                       "Job %s tried to set empty sampling args for Op at %d.",
                       job.output_table_name().c_str(),
                       sampling_args_assignment.op_index());
          return result;
        }
        i32 slice_level =
            op_slice_level.at(sampling_args_assignment.op_index());
        if (slice_level == 0 &&
            sampling_args_assignment.sampling_args().size() > 1) {
          RESULT_ERROR(&result,
                       "Job %s tried to set multiple sampling args for "
                       "Op at %d that has not been sliced.",
                       job.output_table_name().c_str(),
                       sampling_args_assignment.op_index());
          return result;
        }
      }
    }
  }
  return result;
}

Result determine_input_rows_to_slices(
    DatabaseMetadata& meta, TableMetaCache& table_metas,
    const std::vector<proto::Job>& jobs,
    const std::vector<proto::Op>& ops,
    DAGAnalysisInfo& info) {
  Result result;
  result.set_success(true);
  const std::vector<i32>& op_slice_level = info.op_slice_level;
  const std::map<i64, i64>& input_ops = info.source_ops;
  const std::map<i64, i64>& slice_ops = info.slice_ops;
  const std::map<i64, i64>& unslice_ops = info.unslice_ops;
  const std::map<i64, i64>& sampling_ops = info.sampling_ops;
  const std::map<i64, std::vector<i64>>& op_children = info.op_children;
  std::vector<std::map<i64, i64>>& slice_input_rows = info.slice_input_rows;
  std::vector<std::map<i64, std::vector<i64>>>& slice_output_rows =
      info.slice_output_rows;
  std::vector<std::map<i64, std::vector<i64>>>& unslice_input_rows =
      info.unslice_input_rows;
  std::vector<std::map<i64, std::vector<i64>>>& total_rows_per_op =
      info.total_rows_per_op;
  std::vector<i64>& total_output_rows = info.total_output_rows;
  // For each job, use table rows to determine number of total possible outputs
  // by propagating downward through Op DAG
  for (const proto::Job& job : jobs) {
    slice_input_rows.emplace_back();
    std::map<i64, i64>& job_slice_input_rows = slice_input_rows.back();
    slice_output_rows.emplace_back();
    std::map<i64, std::vector<i64>>& job_slice_output_rows =
        slice_output_rows.back();
    unslice_input_rows.emplace_back();
    std::map<i64, std::vector<i64>>& job_unslice_input_rows =
        unslice_input_rows.back();
    total_rows_per_op.emplace_back();
    std::map<i64, std::vector<i64>>& job_total_rows_per_op =
        total_rows_per_op.back();
    // Create domain samplers using sampling args
    // Op idx -> samplers for each slice group
    std::map<i64, proto::SamplingArgsAssignment> args_assignment;
    std::map<i64, std::vector<std::unique_ptr<DomainSampler>>> domain_samplers;
    for (const proto::SamplingArgsAssignment& saa :
         job.sampling_args_assignment()) {
      if (ops.at(saa.op_index()).name() == SLICE_OP_NAME) {
        args_assignment[saa.op_index()] = saa;
      } else {
        std::vector<std::unique_ptr<DomainSampler>>& samplers =
            domain_samplers[saa.op_index()];
        // Assign number of rows to correct op
        for (auto& sa : saa.sampling_args()) {
          DomainSampler* sampler;
          result = make_domain_sampler_instance(
              sa.sampling_function(),
              std::vector<u8>(sa.sampling_args().begin(),
                              sa.sampling_args().end()),
              sampler);
          if (!result.success()) {
            return result;
          }
          samplers.emplace_back(sampler);
        }
      }
    }
    // Create enumerators using enumerator args
    std::map<i64, i64> source_input_rows;
    {
      // Instantiate enumerators to determine number of rows produced from each
      // Source op
      auto registry = get_enumerator_registry();
      for (auto& source_input : job.inputs()) {
        const std::string& source_name = ops.at(source_input.op_index()).name();
        EnumeratorFactory* factory = registry->get_enumerator(source_name);
        EnumeratorConfig config;
        size_t size = source_input.enumerator_args().size();
        config.args =
            std::vector<u8>(source_input.enumerator_args().begin(),
                            source_input.enumerator_args().end());
        std::unique_ptr<Enumerator> e(factory->new_instance(config));
        // If this is a source enumerator, we must provide table meta
        if (auto column_enumerator = dynamic_cast<ColumnEnumerator*>(e.get())) {
          column_enumerator->set_table_meta(&table_metas);
        }
        // Get actual num row count
        source_input_rows[source_input.op_index()] = e->total_elements();
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
    for (const proto::SourceInput& ci : job.inputs()) {
      // Determine number of rows for the requested table
      i64 num_rows = source_input_rows.at(ci.op_index());
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
                "number of slice groups for %s Op at %ld (%lu vs %lu).",
                job.output_table_name().c_str(), ops[op_idx].name().c_str(),
                op_idx, first_input_column_slice_groups.size(),
                input_column_slice_groups.size());
            return result;
          }
          // Verify the number of rows for each slice group matches
          for (size_t i = 0; i < first_input_column_slice_groups.size(); ++i) {
            if (input_column_slice_groups.at(i) !=
                first_input_column_slice_groups.at(i)) {
              RESULT_ERROR(
                  &result,
                  "Job %s specified multiple inputs with a differing "
                  "number of rows for slice group %lu for %s Op at %ld "
                  "(%lu vs %lu).",
                  job.output_table_name().c_str(), i,
                  ops[op_idx].name().c_str(), op_idx,
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
        i64 sampling_op_idx = sampling_ops.at(op_idx);
        // Verify number of samplers is equal to number of slice groups
        if (domain_samplers.at(op_idx).size() != slice_group_outputs.size()) {
          RESULT_ERROR(&result,
                       "Job %s specified %lu samplers but there are %lu slice "
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
        assert(slice_group_outputs.size() == 1);
        // Create Partitioner to enumerate slices
        Partitioner* partitioner = nullptr;
        auto& args = args_assignment[op_idx].sampling_args(0);
        result = make_partitioner_instance(
            args.sampling_function(),
            std::vector<u8>(
                args.sampling_args().begin(),
                args.sampling_args().end()),
            slice_group_outputs.at(0),
            partitioner);
        if (!result.success()) {
          return result;
        }

        // Track job slice inputs so we can determine number of groups later
        job_slice_input_rows.insert({op_idx, slice_group_outputs.at(0)});
        // Update outputs with the new slice group outputs for this partition
        slice_group_outputs = partitioner->total_rows_per_group();
        delete partitioner;

        if (number_of_slice_groups == -1) {
          number_of_slice_groups == slice_group_outputs.size();
        } else if (slice_group_outputs.size() != number_of_slice_groups) {
          RESULT_ERROR(
              &result,
              "Job %s specified one slice with %lu groups and another "
              "slice with %lu groups. Scanner currently does not "
              "support multiple slices with different numbers of groups "
              "in the same job.",
              job.output_table_name().c_str(),
              slice_group_outputs.size(), number_of_slice_groups);
          return result;
        }
        job_slice_output_rows.insert({op_idx, slice_group_outputs});
      }

      // Check if this is an unslice op
      if (unslice_ops.count(op_idx) > 0) {
        assert(op_slice_level.at(op_idx) == 0);

        job_unslice_input_rows.insert({op_idx, slice_group_outputs});
        // Concatenate all slice group outputs
        i64 new_outputs = 0;
        for (i64 group_outputs : slice_group_outputs) {
          new_outputs += group_outputs;
        }
        slice_group_outputs = {new_outputs};
      }
      // Track size of output domain for this Op for use in boundary condition
      // check
      job_total_rows_per_op[op_idx] = slice_group_outputs;

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
  proto::SamplingArgs args;
  {
    bool found = false;
    for (auto& saa : job.sampling_args_assignment()) {
      if (saa.op_index() == slice_op_idx) {
        args = saa.sampling_args(0);
        found = true;
      }
    }
    assert(found);
  }
  result = make_partitioner_instance(
      args.sampling_function(),
      std::vector<u8>(args.sampling_args().begin(), args.sampling_args().end()),
      slice_input_rows, partitioner);
  if (!result.success()) {
    return result;
  }
  // Traverse down graph from each slice group and count the number of rows
  // produced. This is the partition offset that we will use to split the graph
  std::vector<i64>& slice_rows = slice_output_partition;
  slice_rows.push_back(0);
  i64 current_offset = 0;
  for (size_t i = 0; i < partitioner->total_groups(); ++i) {
    const PartitionGroup& g = partitioner->group_at(i);
    // Traverse down all children until reaching the output
    std::vector<i64> input_row_counts(ops.size());
    std::vector<i64> next_queue;
    input_row_counts[slice_op_idx] = g.rows.size();
    next_queue.push_back(slice_op_idx);
    while (!next_queue.empty()) {
      i64 op_idx = next_queue.back();
      next_queue.pop_back();

      auto& op = ops.at(op_idx);
      // Check if sampling Op or unslice Op
      i64 input_row_count = input_row_counts.at(op_idx);
      i64 output_row_count = input_row_count;
      if (op.name() == "Space" || op.name() == "SpaceFrame" ||
          op.name() == "Sample" || op.name() == "SampleFrame") {
        proto::SamplingArgs sa;
        for (auto& saa : job.sampling_args_assignment()) {
          if (saa.op_index() == op_idx) {
            sa = saa.sampling_args(i);
          }
        }
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
        delete sampler;
        if (!result.success()) {
          return result;
        }
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
        input_row_counts.at(cid) = output_row_count;
        next_queue.push_back(cid);
      }
    }
  }
  assert(slice_rows.size() == partitioner->total_groups() + 1);
  return result;
}

void populate_analysis_info(const std::vector<proto::Op>& ops,
                            DAGAnalysisInfo& info) {
  std::vector<i32>& op_slice_level = info.op_slice_level;
  std::map<i64, i64>& input_ops = info.source_ops;
  std::map<i64, i64>& slice_ops = info.slice_ops;
  std::map<i64, i64>& unslice_ops = info.unslice_ops;
  std::map<i64, i64>& sampling_ops = info.sampling_ops;
  std::map<i64, std::vector<i64>>& op_children = info.op_children;
  std::map<i64, bool>& bounded_state_ops = info.bounded_state_ops;
  std::map<i64, bool>& unbounded_state_ops = info.unbounded_state_ops;

  std::map<i64, i32>& warmup_sizes = info.warmup_sizes;
  std::map<i64, i32>& batch_sizes = info.batch_sizes;
  std::map<i64, std::vector<i32>>& stencils = info.stencils;

  // Validate ops
  OpRegistry* op_registry = get_op_registry();
  KernelRegistry* kernel_registry = get_kernel_registry();

  i32 op_idx = 0;
  // Keep track of op names and outputs for verifying that requested
  // edges between ops are valid
  std::vector<std::vector<std::string>> op_outputs;
  // Slices are currently restricted to not nest and there to only exist
  // a single slice grouping from start to finish currently.
  std::vector<std::string> op_names;
  for (auto& op : ops) {
    op_names.push_back(op.name());

    // Input Op's output is defined by the input columns they sample
    if (op.is_source()) {
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
      // Keep op children info for later analysis
      op_children[input.op_index()].push_back(op_idx);
    }

    // Slice
    int output_slice_level = input_slice_level;
    if (op.name() == SLICE_OP_NAME) {
      assert(output_slice_level == 0);
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
      assert(input_slice_level > 0);
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
      assert(input_slice_level == 0);
    }
    // Verify op exists and record outputs
    else {
      op_outputs.emplace_back();
      assert(op_registry->has_op(op.name()));

      // Keep track of op outputs for verifying dependent ops
      for (auto& col : op_registry->get_op_info(op.name())->output_columns()) {
        op_outputs.back().push_back(col.name());
      }

      assert(kernel_registry->has_kernel(op.name(), op.device_type()));
    }
    op_slice_level.push_back(output_slice_level);

    // Perform Op parameter verification (stenciling, batching, # inputs)
    if (!is_builtin_op(op.name())) {
      OpInfo* info = op_registry->get_op_info(op.name());
      KernelFactory* factory =
          kernel_registry->get_kernel(op.name(), op.device_type());

      // Use default batch if not specified
      i32 batch_size = op.batch() != -1
                           ? op.batch()
                           : factory->preferred_batch_size();
      batch_sizes[op_idx] = batch_size;
      // Use default stencil if not specified
      std::vector<i32> stencil;
      if (op.stencil_size() > 0) {
        stencil = std::vector<i32>(op.stencil().begin(), op.stencil().end());
      } else {
        stencil = info->preferred_stencil();
      }
      stencils[op_idx] = stencil;
      if (info->has_bounded_state()) {
        bounded_state_ops[op_idx] = true;
        warmup_sizes[op_idx] = op.warmup() != -1 ? op.warmup() : info->warmup();
      }
      else if (info->has_unbounded_state()) {
        unbounded_state_ops[op_idx] = true;
      }
    }
    op_idx++;
  }
}

void remap_input_op_edges(std::vector<proto::Op>& ops,
                          DAGAnalysisInfo& info) {
  auto rename_col = [](i32 op_idx, const std::string& n) {
    return std::to_string(op_idx) + "_" + n;
  };
  SourceRegistry* source_registry = get_source_registry();
  auto get_source_output = [&](i32 op_idx) {
    return source_registry->get_source(ops.at(op_idx).name())
        ->output_columns()[0]
        .name();
  };
  auto& remap_map = info.input_ops_to_first_op_columns;
  {
    auto first_op_input = ops.at(0).mutable_inputs(0);
    auto first_op_output = get_source_output(0);
    first_op_input->set_column(rename_col(0, first_op_output));
    remap_map[0] = 0;
  }
  for (size_t op_idx = 1; op_idx < ops.size(); ++op_idx) {
    auto& op = ops.at(op_idx);
    // If input Op, add column to original input Op and get rid of existing
    // column
    if (op.is_source()) {
      remap_map[op_idx] = ops.at(0).inputs_size();

      std::string new_column_name =
          rename_col(op_idx, get_source_output(op_idx));
      proto::OpInput* new_input = ops.at(0).add_inputs();
      new_input->set_op_index(-1);
      new_input->set_column(new_column_name);

      ops.at(op_idx).clear_inputs();
    }
    // Remap all inputs to input Ops to the first Op
    for (size_t i = 0; i < op.inputs_size(); ++i) {
      auto input = op.mutable_inputs(i);
      i32 input_op_idx = input->op_index();
      if (remap_map.count(input_op_idx) > 0) {
        input->set_op_index(0);
        input->set_column(rename_col(input_op_idx, input->column()));
      }
    }
  }
}

void perform_liveness_analysis(const std::vector<proto::Op>& ops,
                               DAGAnalysisInfo& results) {
  const std::map<i64, bool>& bounded_state_ops = results.bounded_state_ops;
  const std::map<i64, bool>& unbounded_state_ops = results.unbounded_state_ops;
  const std::map<i64, i32>& warmup_sizes = results.warmup_sizes;
  const std::map<i64, i32>& batch_sizes = results.batch_sizes;
  const std::map<i64, std::vector<i32>>& stencils = results.stencils;

  std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns =
      results.live_columns;
  std::vector<std::vector<i32>>& dead_columns = results.dead_columns;
  std::vector<std::vector<i32>>& unused_outputs = results.unused_outputs;
  std::vector<std::vector<i32>>& column_mapping = results.column_mapping;

  // Start off with the columns from the gathered tables
  OpRegistry* op_registry = get_op_registry();
  KernelRegistry* kernel_registry = get_kernel_registry();
  // Active intermediates
  std::map<i32, std::vector<std::tuple<std::string, i32>>> intermediates;
  {
    auto& input_op = ops.at(0);
    for (const auto& col : input_op.inputs()) {
      const std::string& input_col = col.column();
      // Set last used to first op so that all input ops are live to start
      // with. We could eliminate input columns which aren't used, but this
      // also requires modifying the samples.
      intermediates[0].push_back(std::make_tuple(input_col, 1));
    }
  }
  for (size_t i = 1; i < ops.size(); ++i) {
    auto& op = ops.at(i);
    // For each input, update the intermediate last used index to the
    // current index
    for (auto& eval_input : op.inputs()) {
      i32 parent_index = eval_input.op_index();
      const std::string& parent_col = eval_input.column();
      bool found = false;
      for (auto& kv : intermediates.at(parent_index)) {
        if (std::get<0>(kv) == parent_col) {
          found = true;
          std::get<1>(kv) = i;
          break;
        }
      }
      assert(found);
    }
    if (op.name() == OUTPUT_OP_NAME) {
      continue;
    }
    // Add this op's outputs to the intermediate list
    if (is_builtin_op(op.name())) {
      // Make sure it is initialized even if no inputs
      intermediates[i] = {};
      for (auto& input : op.inputs()) {
        std::string col = input.column();
        // HACK(apoms): we remap input column names but don't update
        // the downstream column. A better solution would be to
        // explicitly enumerate the output column names during the initial
        // dag analysis and keep it around.
        if (ops.at(input.op_index()).is_source()) {
          col = col.substr(col.find("_") + 1);
        }
        intermediates[i].push_back(std::make_tuple(col, i));
      }
    } else {
      const auto& op_info = op_registry->get_op_info(op.name());
      for (const auto& output_column : op_info->output_columns()) {
        intermediates[i].push_back(std::make_tuple(output_column.name(), i));
      }
    }
  }

  // The live columns at each op index
  live_columns.resize(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    i32 op_index = i;
    auto& columns = live_columns[i];
    size_t max_i = std::min((size_t)(ops.size() - 2), i);
    for (size_t j = 0; j <= max_i; ++j) {
      for (auto& kv : intermediates.at(j)) {
        i32 last_used_index = std::get<1>(kv);
        if (last_used_index > op_index) {
          // Last used index is greater than current index, so still live
          columns.push_back(std::make_tuple((i32)j, std::get<0>(kv)));
        }
      }
    }
  }

  // The columns to remove for the current kernel
  dead_columns.resize(ops.size());
  // Outputs from the current kernel that are not used
  unused_outputs.resize(ops.size());
  // Indices in the live columns list that are the inputs to the current
  // kernel.
  column_mapping.resize(ops.size());
  for (size_t i = 1; i < ops.size(); ++i) {
    i32 op_index = i;
    auto& prev_columns = live_columns[i - 1];
    auto& op = ops.at(op_index);
    // Determine which columns are no longer live
    {
      auto& unused = unused_outputs[i];
      auto& dead = dead_columns[i];
      // For all parent Ops, check if we are the last Op to use
      // their output column
      size_t max_i = std::min((size_t)(ops.size() - 2), (size_t)i);
      for (size_t j = 0; j <= max_i; ++j) {
        i32 parent_index = j;
        // For the current parent Op, check if we are the last to use
        // any of its outputs
        for (auto& kv : intermediates.at(j)) {
          i32 last_used_index = std::get<1>(kv);
          if (last_used_index == op_index) {
            // We are the last to use the Op column.
            // Column is no longer live, so remove it.
            const std::string& col_name = std::get<0>(kv);
            if (j == i) {
              // This column was produced by the current Op but not used
              i32 col_index = -1;
              const std::vector<Column>& op_cols =
                  op_registry->get_op_info(op.name())->output_columns();
              for (size_t k = 0; k < op_cols.size(); k++) {
                if (col_name == op_cols[k].name()) {
                  col_index = k;
                  break;
                }
              }
              assert(col_index != -1);
              unused.push_back(col_index);
            } else {
              // This column was produced by a previous Op
              // Determine where in the previous live columns list this
              // column existed
              i32 col_index = -1;
              for (i32 k = 0; k < (i32)prev_columns.size(); ++k) {
                const std::tuple<i32, std::string>& live_input =
                    prev_columns[k];
                if (parent_index == std::get<0>(live_input) &&
                    col_name == std::get<1>(live_input)) {
                  col_index = k;
                  break;
                }
              }
              assert(col_index != -1);
              dead.push_back(col_index);
            }
          }
        }
      }
    }
    // For each input to the Op, determine where in the live column list
    // that input is
    auto& mapping = column_mapping.at(op_index);
    for (const auto& eval_input : op.inputs()) {
      i32 parent_index = eval_input.op_index();
      const std::string& col = eval_input.column();
      i32 col_index = -1;
      for (i32 k = 0; k < (i32)prev_columns.size(); ++k) {
        const std::tuple<i32, std::string>& live_input = prev_columns[k];
        if (parent_index == std::get<0>(live_input) &&
            col == std::get<1>(live_input)) {
          col_index = k;
          break;
        }
      }
      assert(col_index != -1);
      mapping.push_back(col_index);
    }
  }
}

Result derive_stencil_requirements(
    const DatabaseMetadata& meta, TableMetaCache& table_meta,
    const proto::Job& job, const std::vector<proto::Op>& ops,
    const DAGAnalysisInfo& analysis_results,
    proto::BulkJobParameters::BoundaryCondition boundary_condition,
    i64 table_id, i64 job_idx, i64 task_idx,
    const std::vector<i64>& output_rows, LoadWorkEntry& output_entry,
    std::deque<TaskStream>& task_streams) {
  const std::map<i64, std::vector<i32>>& stencils = analysis_results.stencils;
  const std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns =
      analysis_results.live_columns;

  output_entry.set_table_id(table_id);
  output_entry.set_job_index(job_idx);
  output_entry.set_task_index(task_idx);

  i64 num_ops = ops.size();

  const std::map<i64, std::vector<i64>>& job_slice_output_rows =
      analysis_results.slice_output_rows.at(job_idx);
  const std::map<i64, std::vector<i64>>& job_unslice_input_rows =
      analysis_results.unslice_input_rows.at(job_idx);
  const std::map<i64, bool>& bounded_state_ops =
      analysis_results.bounded_state_ops;
  const std::map<i64, bool>& unbounded_state_ops =
      analysis_results.unbounded_state_ops;
  const std::map<i64, i32>& warmup_sizes = analysis_results.warmup_sizes;
  // Create domain samplers
  // Op -> slice
  std::map<i64, std::vector<std::unique_ptr<DomainSampler>>> domain_samplers;
  for (const proto::SamplingArgsAssignment& saa :
       job.sampling_args_assignment()) {
    std::vector<std::unique_ptr<DomainSampler>>& samplers =
        domain_samplers[saa.op_index()];
    // Assign number of rows to correct op
    if (ops.at(saa.op_index()).name() != SLICE_OP_NAME) {
      for (auto& sa : saa.sampling_args()) {
        DomainSampler* sampler;
        Result result = make_domain_sampler_instance(
            sa.sampling_function(),
            std::vector<u8>(sa.sampling_args().begin(),
                            sa.sampling_args().end()),
            sampler);
        if (!result.success()) {
          return result;
        }
        samplers.emplace_back(sampler);
      }
    }
  }

  std::vector<std::unique_ptr<Enumerator>> enumerators(job.inputs_size());
  {
    // Instantiate enumerators to determine number of rows produced from each
    // Source op
    auto registry = get_enumerator_registry();
    for (auto& source_input : job.inputs()) {
      const std::string& source_name = ops.at(source_input.op_index()).name();
      i32 col_idx = analysis_results.input_ops_to_first_op_columns.at(
          source_input.op_index());
      EnumeratorFactory* factory = registry->get_enumerator(source_name);
      EnumeratorConfig config;
      size_t size = source_input.enumerator_args().size();
      config.args = std::vector<u8>(source_input.enumerator_args().begin(),
                                    source_input.enumerator_args().end());
      Enumerator* e = factory->new_instance(config);
      enumerators[col_idx].reset(e);
      // If this is a source enumerator, we must provide table meta
      if (auto column_enumerator = dynamic_cast<ColumnEnumerator*>(e)) {
        column_enumerator->set_table_meta(&table_meta);
      }
    }
  }

  // Compute the required rows for each kernel based on the stencil, sampling
  // operations, and slice operations.
  // For each Op, determine the set of rows needed in the live columns list
  // and the set of rows to feed to the Op at the current column mapping
  // Op -> Rows
  std::vector<std::set<i64>> required_output_rows_at_op(ops.size());
  std::vector<std::vector<i64>> required_input_rows_at_op(ops.size());
  // Track inputs for ecah column of the input Op since different rnput Op
  // colums may correspond to different tables and conservatively requesting
  // all rows could cause an invalid access
  std::vector<std::set<i64>> required_input_op_output_rows;
  required_input_op_output_rows.resize(ops.at(0).inputs_size());
  std::vector<std::vector<i64>> required_input_op_input_rows;
  required_input_op_input_rows.resize(ops.at(0).inputs_size());
  assert(ops.at(0).inputs_size() == job.inputs_size());
  std::vector<std::vector<ElementArgs>> required_input_op_element_args;
  required_input_op_element_args.resize(ops.at(0).inputs_size());
  // HACK(apoms): we currently propagate this boundary condition upward,
  // but that would technically cause the upstream sequence to have more
  // elements than required. Should we stop the boundary condition at the Op
  // by deduplication?
  auto handle_boundary = [boundary_condition](
      const std::vector<i64>& downstream_rows, i64 max_rows,
      std::vector<i64>& bounded_rows) {
    // Handle rows which touch boundaries
    for (size_t i = 0; i < downstream_rows.size(); ++i) {
      i64 r = downstream_rows[i];
      if (r < 0 || r >= max_rows) {
        switch (boundary_condition) {
          case proto::BulkJobParameters::REPEAT_EDGE: {
            r = (r < 0) ? 0 : max_rows - 1;
            break;
          }
          case proto::BulkJobParameters::REPEAT_NULL: {
            r = -1;
            break;
          }
          case proto::BulkJobParameters::ERROR: {
            Result result;
            RESULT_ERROR(&result, "Boundary error.");
            return result;
          }
        }
      }
      bounded_rows.push_back(r);
    }
    Result result;
    result.set_success(true);
    return result;
  };
  // Walk up the Ops to derive upstream rows
  i32 slice_group = 0;
  {
    // Initialize output rows
    required_output_rows_at_op.at(num_ops - 1) =
        std::set<i64>(output_rows.begin(), output_rows.end());
    // For each kernel, derive the minimal required upstream elements
    for (i64 op_idx = num_ops - 1; op_idx >= 0; --op_idx) {
      auto& op = ops.at(op_idx);
      std::vector<i64> downstream_rows(
          required_output_rows_at_op.at(op_idx).begin(),
          required_output_rows_at_op.at(op_idx).end());
      std::sort(downstream_rows.begin(), downstream_rows.end());
      std::vector<i64> compute_rows;
      // Determine which upstream rows are needed for the requested output rows
      std::vector<i64> new_rows;
      // Input Op
      if (op.is_source()) {
        // Ignore if it is not the first input
        if (op_idx == 0) {
          for (size_t i = 0; i < enumerators.size(); ++i) {
            std::vector<i64> output_rows(
                required_input_op_output_rows.at(i).begin(),
                required_input_op_output_rows.at(i).end());
            std::sort(output_rows.begin(), output_rows.end());
            std::vector<i64>& input_rows = required_input_op_input_rows.at(i);
            i64 num_rows = enumerators[i]->total_elements();

            // Perform boundary restriction
            Result result = handle_boundary(output_rows, num_rows, input_rows);
            if (!result.success()) {
              return result;
            }
            // Generate all the args for the requested input rows
            std::vector<ElementArgs>& element_args =
                required_input_op_element_args.at(i);
            element_args.reserve(input_rows.size());
            for (i64 input_row : input_rows) {
              element_args.push_back(enumerators[i]->element_args_at(input_row));
            }
          }
        }
      }
      // Sample or Space Op
      else if (op.name() == SAMPLE_OP_NAME) {
        // Use domain sampler
        i32 slice = 0;
        if (analysis_results.op_slice_level.at(op_idx) > 0) {
          assert(slice_group != -1);
          slice = slice_group;
        }
        Result result = domain_samplers.at(op_idx)
                            .at(slice)
                            ->get_upstream_rows(downstream_rows, new_rows);
        if (!result.success()) {
          return result;
        }
      }
      // Space Op
      else if (op.name() == SPACE_OP_NAME) {
        // Use domain sampler
        i32 slice = 0;
        if (analysis_results.op_slice_level.at(op_idx) > 0) {
          assert(slice_group != -1);
          slice = slice_group;
        }
        Result result = domain_samplers.at(op_idx).at(slice)->get_upstream_rows(
            downstream_rows, new_rows);
        if (!result.success()) {
          return result;
        }
      }
      // Slice Op
      else if (op.name() == SLICE_OP_NAME) {
        // We know which slice group we are in already from the unslice
        // HACK(apoms): we currently restrict pipelines such that slices
        // can be computed entirely independently and choose output rows
        // that do not cross state boundaries to make it possible to assume
        // that all rows are in the same slice
        assert(slice_group != -1);

        const auto& slice_output_counts = job_slice_output_rows.at(op_idx);
        i64 offset = 0;
        for (i64 i = 0; i < slice_group; ++i) {
          offset += slice_output_counts.at(i);
        }

        i64 rows_in_group = slice_output_counts.at(slice_group);
        // Perform boundary restriction
        std::vector<i64> bounded_rows;
        Result result =
            handle_boundary(downstream_rows, rows_in_group, bounded_rows);
        if (!result.success()) {
          return result;
        }

        // Remap row indices
        for (i64 r : bounded_rows) {
          new_rows.push_back(r + offset);
        }
      }
      // Unslice Op
      else if (op.name() == UNSLICE_OP_NAME) {
        // Determine which slices we are in and propagate those rows upwards
        // HACK(apoms): we currently restrict pipelines such that slices
        // can be computed entirely independently and choose output rows
        // that do not cross state boundaries to make it possible to assume
        // that all rows are in the same slice
        i64 downstream_min = downstream_rows[0];
        i64 downstream_max = downstream_rows[downstream_rows.size() - 1];
        const auto& unslice_input_counts = job_unslice_input_rows.at(op_idx);
        i64 offset = 0;
        slice_group = 0;
        bool found = false;
        for (; slice_group < unslice_input_counts.size(); ++slice_group) {
          if (downstream_min >= offset &&
              downstream_max < offset + unslice_input_counts.at(slice_group)) {
            found = true;
            break;
          }
          offset += unslice_input_counts.at(slice_group);
        }
        assert(found);
        // Remap row indices
        for (i64 r : downstream_rows) {
          new_rows.push_back(r - offset);
        }
      }
      // Output Op
      else if (op.name() == OUTPUT_OP_NAME) {
        new_rows = downstream_rows;
      }
      // Regular Op
      else {
        assert(!is_builtin_op(op.name()));
        std::unordered_set<i64> current_rows;
        current_rows.reserve(downstream_rows.size());
        // If bounded state, we need to handle warmup
        if (bounded_state_ops.count(op_idx) > 0) {
          i32 warmup = warmup_sizes.at(op_idx);
          for (i64 r : downstream_rows) {
            // Check that we have all warmup rows
            for (i64 i = 0; i <= warmup; ++i) {
              i64 req_row = r - i;
              if (req_row < 0) {
                continue;
              }
              current_rows.insert(req_row);
            }
          }
        }
        // If unbounded state, we need all upstream inputs from 0
        else if (unbounded_state_ops.count(op_idx) > 0) {
          i32 max_required_row = downstream_rows.back();
          for (i64 i = 0; i <= max_required_row; ++i) {
            current_rows.insert(i);
          }
        } else {
          current_rows.insert(downstream_rows.begin(), downstream_rows.end());
        }
        compute_rows = std::vector<i64>(current_rows.begin(),
                                        current_rows.end());
        std::sort(compute_rows.begin(), compute_rows.end());

        // Ensure we have inputs for stenciling kernels
        std::unordered_set<i64> stencil_rows;
        const std::vector<i32>& stencil = stencils.at(op_idx);
        for (i64 r : current_rows) {
          for (i64 s : stencil) {
            stencil_rows.insert(r + s);
          }
        }
        new_rows = std::vector<i64>(stencil_rows.begin(), stencil_rows.end());
        std::sort(new_rows.begin(), new_rows.end());
      }

      required_input_rows_at_op.at(op_idx) = new_rows;
      // Input Op inputs do not connect to any other Ops
      if (!op.is_source()) {
        for (auto& input : op.inputs()) {
          if (input.op_index() == 0) {
            // For the input Op, we track each input column separately since
            // they may come from different tables
            i64 col_id = -1;
            for (size_t i = 0; i < ops.at(0).inputs_size(); ++i) {
              const auto& col = ops.at(0).inputs(i);
              if (col.column() == input.column()) {
                col_id = i;
                break;
              }
            }
            assert(col_id != -1);
            required_input_op_output_rows.at(col_id).insert(new_rows.begin(),
                                                            new_rows.end());
          }
          auto& input_outputs = required_output_rows_at_op.at(input.op_index());
          input_outputs.insert(new_rows.begin(), new_rows.end());
        }
      }

      if (compute_rows.empty()) {
        compute_rows = new_rows;
      }

      TaskStream s;
      s.slice_group = slice_group;
      s.valid_input_rows = new_rows;
      s.compute_input_rows = compute_rows;
      s.valid_output_rows = downstream_rows;
      task_streams.push_front(s);
    }
  }

  // Get rid of input stream since this is already captured by the load samples
  task_streams.pop_front();

  for (size_t i = 0; i < enumerators.size(); ++i) {
    auto out_arg = output_entry.add_source_args();
    google::protobuf::RepeatedField<i64> input_data(
        required_input_op_input_rows.at(i).begin(),
        required_input_op_input_rows.at(i).end());
    out_arg->mutable_input_row_ids()->Swap(&input_data);
    google::protobuf::RepeatedField<i64> output_data(
        required_input_op_output_rows.at(i).begin(),
        required_input_op_output_rows.at(i).end());
    out_arg->mutable_output_row_ids()->Swap(&output_data);

    const auto& ele = required_input_op_element_args.at(i);
    for (size_t j = 0; j < ele.size(); ++j) {
      out_arg->add_args(ele[j].args.data(), ele[j].args.size());
    }
  }
  Result result;
  result.set_success(true);
  return result;
}

}
}

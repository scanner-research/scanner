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

#include "scanner/engine/worker.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/save_worker.h"
#include "scanner/util/cuda.h"

#include <arpa/inet.h>
#include <grpc/grpc_posix.h>
#include <grpc/support/log.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <omp.h>

// For avcodec_register_all()... should go in software video with global mutex
extern "C" {
#include "libavcodec/avcodec.h"
}

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {

namespace {
inline bool operator==(const MemoryPoolConfig& lhs,
                       const MemoryPoolConfig& rhs) {
  return (lhs.cpu().use_pool() == rhs.cpu().use_pool()) &&
         (lhs.cpu().free_space() == rhs.cpu().free_space()) &&
         (lhs.gpu().use_pool() == rhs.gpu().use_pool()) &&
         (lhs.gpu().free_space() == rhs.gpu().free_space());
}
inline bool operator!=(const MemoryPoolConfig& lhs,
                       const MemoryPoolConfig& rhs) {
  return !(lhs == rhs);
}

struct AnalysisResults {
  std::vector<std::vector<std::tuple<i32, std::string>>> live_columns;
  std::vector<std::vector<i32>> dead_columns;
  std::vector<std::vector<i32>> unused_outputs;
  std::vector<std::vector<i32>> column_mapping;

  std::vector<i32> warmup_sizes;
  std::vector<i32> batch_sizes;
  std::vector<std::vector<i32>> stencils;
};

AnalysisResults analyze_dag(const proto::TaskSet& task_set) {
  AnalysisResults results;

  std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns =
      results.live_columns;
  std::vector<std::vector<i32>>& dead_columns = results.dead_columns;
  std::vector<std::vector<i32>>& unused_outputs = results.unused_outputs;
  std::vector<std::vector<i32>>& column_mapping = results.column_mapping;

  std::vector<i32>& warmup_sizes = results.warmup_sizes;
  std::vector<i32>& batch_sizes = results.batch_sizes;
  std::vector<std::vector<i32>>& stencils = results.stencils;

  // Start off with the columns from the gathered tables
  OpRegistry* op_registry = get_op_registry();
  KernelRegistry* kernel_registry = get_kernel_registry();
  auto& ops = task_set.ops();
  std::map<i32, std::vector<std::tuple<std::string, i32>>> intermediates;
  {
    auto& input_op = ops.Get(0);
    for (const std::string& input_col : input_op.inputs(0).columns()) {
      // Set last used to first op so that all input ops are live to start
      // with. We could eliminate input columns which aren't used, but this
      // also requires modifying the samples.
      intermediates[0].push_back(std::make_tuple(input_col, 1));
    }
  }
  for (size_t i = 1; i < ops.size(); ++i) {
    auto& op = ops.Get(i);
    // For each input, update the intermediate last used index to the
    // current index
    for (auto& eval_input : op.inputs()) {
      i32 parent_index = eval_input.op_index();
      for (const std::string& parent_col : eval_input.columns()) {
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
    }
    if (i == ops.size() - 1) {
      continue;
    }
    // Add this op's outputs to the intermediate list
    const auto& op_info = op_registry->get_op_info(op.name());
    for (const auto& output_column : op_info->output_columns()) {
      intermediates[i].push_back(std::make_tuple(output_column.name(), i));
    }
    const auto& kernel_factory =
        kernel_registry->get_kernel(op.name(), op.device_type());
    // Use default batch if not specified
    i32 batch_size =
        op.batch() != -1 ? op.batch() : kernel_factory->preferred_batch_size();
    batch_sizes.push_back(batch_size);
    // Use default stencil if not specified
    std::vector<i32> stencil;
    if (op.stencil_size() > 0) {
      stencil = std::vector<i32>(op.stencil().begin(), op.stencil().end());
    } else {
      stencil = op_info->preferred_stencil();
    }
    stencils.push_back(stencil);
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
  dead_columns.resize(ops.size() - 1);
  // Outputs from the current kernel that are not used
  unused_outputs.resize(ops.size() - 1);
  // Indices in the live columns list that are the inputs to the current
  // kernel. Starts from the second evalutor (index 1)
  column_mapping.resize(ops.size() - 1);
  for (size_t i = 1; i < ops.size(); ++i) {
    i32 op_index = i;
    auto& prev_columns = live_columns[i - 1];
    auto& op = ops.Get(op_index);
    // Determine which columns are no longer live
    {
      auto& unused = unused_outputs[i - 1];
      auto& dead = dead_columns[i - 1];
      size_t max_i = std::min((size_t)(ops.size() - 2), (size_t)i);
      for (size_t j = 0; j <= max_i; ++j) {
        i32 parent_index = j;
        for (auto& kv : intermediates.at(j)) {
          i32 last_used_index = std::get<1>(kv);
          if (last_used_index == op_index) {
            // Column is no longer live, so remove it.
            const std::string& col_name = std::get<0>(kv);
            if (j == i) {
              // This op has an unused output
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
    auto& mapping = column_mapping[op_index - 1];
    for (const auto& eval_input : op.inputs()) {
      i32 parent_index = eval_input.op_index();
      for (const std::string& col : eval_input.columns()) {
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
  return results;
}

void derive_stencil_requirements(storehouse::StorageBackend* storage,
                                 const AnalysisResults& analysis_results,
                                 const LoadWorkEntry& load_work_entry,
                                 const std::vector<std::vector<i32>>& stencils,
                                 i64 initial_work_item_size,
                                 LoadWorkEntry& output_entry,
                                 std::deque<TaskStream>& task_streams) {
  output_entry.set_io_item_index(load_work_entry.io_item_index());

  i64 num_kernels = stencils.size();

  // Compute the required rows for each kernel based on the stencil
  // HACK(apoms): this will only really work for linear DAGs. For DAGs with
  //   non-linear topologies, this might break. Supporting proper DAGs would
  //   require tracking stencils up each branch individually.
  std::vector<i64> current_rows;
  const proto::LoadSample& sample = load_work_entry.samples(0);
  i64 last_row = sample.rows(sample.rows_size() - 1);
  std::string table_path = TableMetadata::descriptor_path(sample.table_id());
  TableMetadata meta = read_table_metadata(storage, table_path);
  {
    current_rows = std::vector<i64>(sample.rows().begin(), sample.rows().end());
    TaskStream s;
    s.valid_output_rows = current_rows;
    task_streams.push_front(s);
    // For each kernel, derive the required elements via its stencil
    for (i64 i = 0; i < num_kernels; ++i) {
      const std::vector<i32>& stencil = stencils[num_kernels - 1 - i];
      std::unordered_set<i64> new_rows;
      new_rows.reserve(current_rows.size());
      for (i64 r : current_rows) {
        // Ignore rows which can not achieve their stencil
        if (r - stencil[0] < 0 ||
            r + stencil[stencil.size() - 1] >= meta.num_rows()) continue;
        for (i64 s : stencil) {
          new_rows.insert(r + s);
        }
      }
      current_rows = std::vector<i64>(new_rows.begin(), new_rows.end());
      std::sort(current_rows.begin(), current_rows.end());
      TaskStream s;
      s.valid_output_rows = current_rows;
      task_streams.push_front(s);
    }
  }
  // Compute the required work item sizes to produce the minimal amount of
  // output for each invocation of the kernels
  std::vector<i64> work_item_sizes;
  {
    // Lists the last row that the stencil cache would have seen
    // at this point
    std::vector<i64> last_stencil_cache_row(num_kernels + 1, -1);
    std::vector<size_t> produced_rows(num_kernels + 1);

    size_t num_input_rows = task_streams.front().valid_output_rows.size();
    size_t num_output_rows = task_streams.back().valid_output_rows.size();
    while (produced_rows.front() < num_input_rows) {
      i64 work_item_size = initial_work_item_size;
      // For each kernel, determine which rows of input it needs given the
      // current stencil cache and position in required rows
      for (i64 k = num_kernels; k >= 1; k--) {
        const TaskStream& prev_s = task_streams[k - 1];
        const TaskStream& s = task_streams[k];
        size_t pos = produced_rows[k];
        const std::vector<i32> stencil = analysis_results.stencils[k - 1];
        i64 batch_size = analysis_results.batch_sizes[k - 1];

        // If the kernel is batched, we need to make sure we round up to
        // request a batch of input.
        if (work_item_size % batch_size != 0) {
          work_item_size += (batch_size - work_item_size % batch_size);
        }

        // If we are at the end of the task, then we can not provide
        // a full batch and must provide a partial one
        if (pos + work_item_size > prev_s.valid_output_rows.size()) {
          work_item_size = s.valid_output_rows.size() - pos;
        }

        // Compute which input rows are needed for the batch of outputs
        std::set<i64> required_input_rows;
        for (i64 i = 0; i < work_item_size; ++i) {
          i64 row = s.valid_output_rows[pos + i];
          for (i32 s : stencil) {
            required_input_rows.insert(row + s);
          }
        }
        std::vector<i64> sorted_input_rows(required_input_rows.begin(),
                                           required_input_rows.end());
        std::sort(sorted_input_rows.begin(), sorted_input_rows.end());

        // For all the rows not in the stencil cache, we will request them
        // from the upstream kernel by setting the work item size
        i64 rows_to_request = 0;
        {
          i64 i = 0;
          for (; i < sorted_input_rows.size(); ++i) {
            // If the requested row is not in the stencil cache, we are done
            // with this work item
            if (sorted_input_rows[i] > last_stencil_cache_row[k - 1]) {
              break;
            }
          }
          rows_to_request = (sorted_input_rows.size() - i);
        }
        assert(rows_to_request > 0);
        work_item_size = rows_to_request;
      }
      produced_rows[0] += work_item_size;
      last_stencil_cache_row[0] =
          task_streams[0].valid_output_rows[produced_rows[0] - 1];
      assert(produced_rows[0] > 0);
      work_item_sizes.push_back(work_item_size);

      // Propagate downward what rows will be in the stencil cache due to the
      // computed number of rows of input
      for (i64 k = 1; k < num_kernels + 1; k++) {
        const TaskStream& ts = task_streams[k];
        size_t& pos = produced_rows[k];
        const std::vector<i32>& stencil = analysis_results.stencils[k - 1];
        i64 batch_size = analysis_results.batch_sizes[k - 1];

        // Figure out how many rows will be produced given work_item_size
        // inputs
        i64 rows = 0;
        for (; pos + rows < ts.valid_output_rows.size(); ++rows) {
          if (ts.valid_output_rows[pos + rows] + stencil.back() >
              last_stencil_cache_row[k - 1]) {
            break;
          }
        }
        assert(pos + rows - 1 < ts.valid_output_rows.size());
        // Round down if we don't have enough for a batch unless this is
        // the end of the task
        if (rows % batch_size != 0 &&
            ts.valid_output_rows[pos + rows - 1] != last_row) {
          rows -= (rows % batch_size);
        }
        assert(rows > 0);

        // Update how many rows we have produced
        pos += rows;
        assert(pos > 0);
        last_stencil_cache_row[k] = ts.valid_output_rows[pos - 1];

        // Send the rows to the next kernel
        work_item_size = rows;
      }
    }
  }

  // Get rid of input stream since this is already captured by the load samples
  task_streams.pop_front();

  for (i64 r : work_item_sizes) {
    output_entry.add_work_item_sizes(r);
  }

  for (const proto::LoadSample& sample : load_work_entry.samples()) {
    auto out_sample = output_entry.add_samples();
    out_sample->set_table_id(sample.table_id());
    out_sample->mutable_column_ids()->CopyFrom(sample.column_ids());
    out_sample->set_warmup_size(sample.warmup_size());
    google::protobuf::RepeatedField<i64> data(
        current_rows.begin(), current_rows.end());
    out_sample->mutable_rows()->Swap(&data);
  }
}

void load_driver(LoadInputQueue& load_work,
                 std::vector<EvalQueue>& initial_eval_work,
                 LoadWorkerArgs args) {
  Profiler& profiler = args.profiler;
  LoadWorker worker(args);
  while (true) {
    auto idle_start = now();

    std::tuple<i32, std::deque<TaskStream>, IOItem, LoadWorkEntry> entry;
    load_work.pop(entry);
    i32& output_queue_idx = std::get<0>(entry);
    auto& task_streams = std::get<1>(entry);
    IOItem& io_item = std::get<2>(entry);
    LoadWorkEntry& load_work_entry = std::get<3>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (load_work_entry.io_item_index() == -1) {
      break;
    }

    VLOG(2) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
            << "): processing item " << load_work_entry.io_item_index();

    auto work_start = now();

    auto input_entry = std::make_tuple(io_item, load_work_entry);
    worker.feed(input_entry);

    while (true) {
      std::tuple<IOItem, EvalWorkEntry> output_entry;
      i32 io_item_size = 0; // dummy for now
      if (worker.yield(io_item_size, output_entry)) {
        auto work_entry = std::get<1>(output_entry);
        work_entry.first = !task_streams.empty();
        work_entry.last = worker.done();
        initial_eval_work[output_queue_idx].push(
            std::make_tuple(task_streams, std::get<0>(output_entry),
                            work_entry));
        // We use the task streams being empty to indicate that this is
        // a new task, so clear it here to show that this is from the same task
        task_streams.clear();
      } else {
        break;
      }
    }
    profiler.add_interval("task", work_start, now());
  }
  VLOG(1) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
          << "): thread finished";
}

std::mutex no_pipelining_lock;
std::condition_variable no_pipelining_cvar;

void pre_evaluate_driver(EvalQueue& input_work, EvalQueue& output_work,
                         PreEvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  PreEvaluateWorker worker(args);
  std::map<i32, Queue<
                  std::tuple<std::deque<TaskStream>, IOItem, EvalWorkEntry>>>
      task_work_queue;

  i32 active_task = -1;
  while (true) {
    auto idle_start = now();

    if (task_work_queue.empty() ||
        (active_task != -1 && task_work_queue.at(active_task).size() <= 0)) {
      std::tuple<std::deque<TaskStream>, IOItem, EvalWorkEntry> entry;
      input_work.pop(entry);

      auto& task_streams = std::get<0>(entry);
      IOItem& io_item = std::get<1>(entry);
      EvalWorkEntry& work_entry = std::get<2>(entry);
      if (work_entry.io_item_index == -1) {
        break;
      }

      task_work_queue[work_entry.io_item_index].push(entry);
    }

    args.profiler.add_interval("idle", idle_start, now());

    if (active_task == -1) {
      // Choose the next active task
      active_task = task_work_queue.begin()->first;
    }

    // Wait until we have the next io item
    if (task_work_queue.at(active_task).size() <= 0) {
      continue;
    }

    // Grab next entry for active task
    std::tuple<std::deque<TaskStream>, IOItem, EvalWorkEntry> entry;
    task_work_queue.at(active_task).pop(entry);

    auto& task_streams = std::get<0>(entry);
    IOItem& io_item = std::get<1>(entry);
    EvalWorkEntry& work_entry = std::get<2>(entry);

    VLOG(2) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
            << "): "
            << "processing item " << work_entry.io_item_index;

    auto work_start = now();

    i32 total_rows = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      total_rows = std::max(total_rows, (i32)work_entry.columns[i].size());
    }

    bool first = work_entry.first;
    bool last = work_entry.last;

    auto input_entry = std::make_tuple(io_item, work_entry);
    worker.feed(input_entry, first);
    i32 work_item_index = 0;
    while (work_item_index < work_entry.work_item_sizes.size()) {
      i32 work_item_size = work_entry.work_item_sizes.at(work_item_index++);
      total_rows -= work_item_size;

      std::tuple<IOItem, EvalWorkEntry> output_entry;
      if (!worker.yield(work_item_size, output_entry)) {
        break;
      }

      if (first) {
        output_work.push(std::make_tuple(task_streams,
                                         std::get<0>(output_entry),
                                         std::get<1>(output_entry)));
        first = false;
      } else {
        output_work.push(std::make_tuple(std::deque<TaskStream>(),
                                         std::get<0>(output_entry),
                                         std::get<1>(output_entry)));
      }

      if (std::getenv("NO_PIPELINING")) {
        std::unique_lock<std::mutex> lk(no_pipelining_lock);
        no_pipelining_cvar.wait(lk);
      }
    }

    if (last) {
      task_work_queue.erase(active_task);
      active_task = -1;
    }

    profiler.add_interval("task", work_start, now());
  }

  VLOG(1) << "Pre-evaluate (N/PU: " << args.node_id << "/" << args.worker_id
          << "): thread finished ";
}

void evaluate_driver(EvalQueue& input_work, EvalQueue& output_work,
                     EvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  EvaluateWorker worker(args);
  while (true) {
    auto idle_pull_start = now();

    std::tuple<std::deque<TaskStream>, IOItem, EvalWorkEntry> entry;
    input_work.pop(entry);

    auto& task_streams = std::get<0>(entry);
    IOItem& io_item = std::get<1>(entry);
    EvalWorkEntry& work_entry = std::get<2>(entry);

    args.profiler.add_interval("idle_pull", idle_pull_start, now());

    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
            << args.kg << "): processing item " << work_entry.io_item_index;

    auto work_start = now();

    if (task_streams.size() > 0) {
      // Start of a new task. Tell kernels what outputs they should produce.
      std::vector<TaskStream> streams;
      for (i32 i = 0; i < args.kernel_factories.size(); ++i) {
        assert(!task_streams.empty());
        streams.push_back(task_streams.front());
        task_streams.pop_front();
      }
      worker.new_task(streams);
    }

    i32 work_item_size = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      work_item_size =
          std::max(work_item_size, (i32)work_entry.columns[i].size());
    }

    auto input_entry = std::make_tuple(io_item, work_entry);
    worker.feed(input_entry);
    std::tuple<IOItem, EvalWorkEntry> output_entry;
    bool result = worker.yield(work_item_size, output_entry);
    (void)result;
    assert(result);

    profiler.add_interval("task", work_start, now());

    auto idle_push_start = now();
    output_work.push(std::make_tuple(task_streams, std::get<0>(output_entry),
                                     std::get<1>(output_entry)));
    args.profiler.add_interval("idle_push", idle_push_start, now());

  }
  VLOG(1) << "Evaluate (N/KI: " << args.node_id << "/" << args.ki
          << "): thread finished";
}

void post_evaluate_driver(EvalQueue& input_work, OutputEvalQueue& output_work,
                          PostEvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  PostEvaluateWorker worker(args);
  while (true) {
    auto idle_start = now();

    std::tuple<std::deque<TaskStream>, IOItem, EvalWorkEntry> entry;
    input_work.pop(entry);
    IOItem& io_item = std::get<1>(entry);
    EvalWorkEntry& work_entry = std::get<2>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
            << "): processing item " << work_entry.io_item_index;

    auto work_start = now();

    auto input_entry = std::make_tuple(io_item, work_entry);
    worker.feed(input_entry);
    std::tuple<IOItem, EvalWorkEntry> output_entry;
    bool result = worker.yield(output_entry);
    profiler.add_interval("task", work_start, now());

    if (result) {
      auto out_entry = std::get<1>(output_entry);
      out_entry.last = work_entry.last;
      output_work.push(std::make_tuple(std::get<0>(output_entry),
                                       out_entry));

    }

    if (std::getenv("NO_PIPELINING")) {
      no_pipelining_cvar.notify_one();
    }
  }

  VLOG(1) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
          << "): thread finished ";
}

void save_coordinator(OutputEvalQueue& eval_work,
                      std::vector<SaveInputQueue>& save_work) {
  i32 num_save_workers = save_work.size();
  std::map<i32, i32> task_to_worker_mapping;
  i32 last_worker_assigned = 0;
  while (true) {
    auto idle_start = now();

    std::tuple<IOItem, EvalWorkEntry> entry;
    eval_work.pop(entry);
    IOItem& io_item = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    //args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.io_item_index == -1) {
      break;
    }

    i32 task_id = work_entry.io_item_index;
    if (task_to_worker_mapping.count(task_id) == 0) {
      // Assign worker to this task
      task_to_worker_mapping[task_id] =
          last_worker_assigned++ % num_save_workers;
    }

    i32 assigned_worker = task_to_worker_mapping.at(task_id);
    save_work[assigned_worker].push(entry);

    if (work_entry.last) {
      task_to_worker_mapping.erase(task_id);
    }
  }
}

void save_driver(SaveInputQueue& save_work,
                 std::atomic<i64>& retired_items,
                 SaveWorkerArgs args) {
  Profiler& profiler = args.profiler;
  SaveWorker worker(args);

  i32 active_task = -1;
  while (true) {
    auto idle_start = now();

    std::tuple<IOItem, EvalWorkEntry> entry;
    save_work.pop(entry);

    IOItem& io_item = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
            << "): processing item " << work_entry.io_item_index;

    auto work_start = now();

    if (work_entry.io_item_index != active_task) {
      active_task = work_entry.io_item_index;
      worker.new_task(io_item, work_entry.column_types);
    }

    auto input_entry = std::make_tuple(io_item, work_entry);
    worker.feed(input_entry);

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
            << "): finished item " << work_entry.io_item_index;

    args.profiler.add_interval("task", work_start, now());

    retired_items++;
  }

  VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
          << "): thread finished ";
}
}


WorkerImpl::WorkerImpl(DatabaseParameters& db_params,
                       std::string master_address, std::string worker_port)
  : watchdog_awake_(true), db_params_(db_params) {
  set_database_path(db_params.db_path);

  avcodec_register_all();
#ifdef DEBUG
  // Stop SIG36 from grpc when debugging
  grpc_use_signal(-1);
#endif
  // google::protobuf::io::CodedInputStream::SetTotalBytesLimit(67108864 * 4,
  //                                                            67108864 * 2);

  master_ = proto::Master::NewStub(
      grpc::CreateChannel(master_address, grpc::InsecureChannelCredentials()));

  proto::WorkerParams worker_info;
  worker_info.set_port(worker_port);

  proto::MachineParameters* params = worker_info.mutable_params();
  params->set_num_cpus(db_params_.num_cpus);
  params->set_num_load_workers(db_params_.num_cpus);
  params->set_num_save_workers(db_params_.num_cpus);
  for (i32 gpu_id : db_params_.gpu_ids) {
    params->add_gpu_ids(gpu_id);
  }

  grpc::ClientContext context;
  proto::Registration registration;
  grpc::Status status =
      master_->RegisterWorker(&context, worker_info, &registration);
  LOG_IF(FATAL, !status.ok())
      << "Worker could not contact master server at " << master_address << " ("
      << status.error_code() << "): " << status.error_message();

  node_id_ = registration.node_id();

  storage_ =
      storehouse::StorageBackend::make_from_config(db_params_.storage_config);

  // Set up Python runtime if any kernels need it
  Py_Initialize();
  boost::python::numpy::initialize();
}

WorkerImpl::~WorkerImpl() {
  trigger_shutdown_.set();
  if (watchdog_thread_.joinable()) {
    watchdog_thread_.join();
  }
  delete storage_;
  if (memory_pool_initialized_) {
    destroy_memory_allocators();
  }
}

grpc::Status WorkerImpl::NewJob(grpc::ServerContext* context,
                                const proto::JobParameters* job_params,
                                proto::Result* job_result) {
  job_result->set_success(true);
  set_database_path(db_params_.db_path);

  // Load table metadata for use in other operations
  // TODO(apoms): only load needed tables
  DatabaseMetadata meta =
      read_database_metadata(storage_, DatabaseMetadata::descriptor_path());
  std::map<std::string, TableMetadata> table_meta;
  for (const std::string& table_name : meta.table_names()) {
    std::string table_path =
        TableMetadata::descriptor_path(meta.get_table_id(table_name));
    table_meta[table_name] = read_table_metadata(storage_, table_path);
  }

  i32 local_id = job_params->local_id();
  i32 local_total = job_params->local_total();
  i32 node_count = job_params->global_total();

  // Controls if work should be distributed roundrobin or dynamically
  bool distribute_work_dynamically = true;

  timepoint_t base_time = now();
  const i32 work_item_size = job_params->work_item_size();
  const i32 io_item_size = job_params->io_item_size();
  i32 warmup_size = 0;

  OpRegistry* op_registry = get_op_registry();
  auto& ops = job_params->task_set().ops();

  // Analyze op DAG to determine what inputs need to be pipped along
  // and when intermediates can be retired -- essentially liveness analysis
  AnalysisResults analysis_results = analyze_dag(job_params->task_set());
  // The live columns at each op index
  std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns =
      analysis_results.live_columns;
  // The columns to remove for the current kernel
  std::vector<std::vector<i32>> dead_columns =
      analysis_results.dead_columns;
  // Outputs from the current kernel that are not used
  std::vector<std::vector<i32>> unused_outputs =
      analysis_results.unused_outputs;
  // Indices in the live columns list that are the inputs to the current
  // kernel. Starts from the second evalutor (index 1)
  std::vector<std::vector<i32>> column_mapping =
      analysis_results.column_mapping;


  // Read final output columns for use in post-evaluate worker
  // (needed for determining column types)
  std::vector<Column> final_output_columns;
  {
    std::string output_name =
        job_params->task_set().tasks(0).output_table_name();
    TableMetadata& table = table_meta[output_name];
    final_output_columns = table.columns();
  }
  std::vector<ColumnCompressionOptions> final_compression_options;
  for (auto& opts : job_params->task_set().compression()) {
    ColumnCompressionOptions o;
    o.codec = opts.codec();
    for (auto& kv : opts.options()) {
      o.options[kv.first] = kv.second;
    }
    final_compression_options.push_back(o);
  }
  assert(final_output_columns.size() == final_compression_options.size());

  // Setup kernel factories and the kernel configs that will be used
  // to instantiate instances of the op pipeline
  KernelRegistry* kernel_registry = get_kernel_registry();
  std::vector<KernelFactory*> kernel_factories;
  std::vector<KernelConfig> kernel_configs;
  i32 num_cpus = db_params_.num_cpus;
  assert(num_cpus > 0);

  i32 total_gpus = db_params_.gpu_ids.size();
  i32 num_gpus = db_params_.gpu_ids.size() / local_total;
  // Should have at least one gpu if there are gpus
  assert(db_params_.gpu_ids.size() == 0 || num_gpus > 0);
  std::vector<i32> gpu_ids;
  {
    i32 start_idx = num_gpus * local_id;
    for (i32 i = 0; i < num_gpus; ++i) {
      gpu_ids.push_back(db_params_.gpu_ids[(start_idx + i) % total_gpus]);
    }
  }

  for (size_t i = 1; i < ops.size() - 1; ++i) {
    auto& op = ops.Get(i);
    const std::string& name = op.name();
    OpInfo* op_info = op_registry->get_op_info(name);

    DeviceType requested_device_type = op.device_type();
    if (requested_device_type == DeviceType::GPU && num_gpus == 0) {
      RESULT_ERROR(job_result,
                   "Scanner is configured with zero available GPUs but a GPU "
                   "op was requested! Please configure Scanner to have "
                   "at least one GPU using the `gpu_ids` config option.");
      return grpc::Status::OK;
    }

    if (!kernel_registry->has_kernel(name, requested_device_type)) {
      RESULT_ERROR(
          job_result,
          "Requested an instance of op %s with device type %s, but no kernel "
          "exists for that configuration.",
          op.name().c_str(),
          (requested_device_type == DeviceType::CPU ? "CPU" : "GPU"));
      return grpc::Status::OK;
    }

    KernelFactory* kernel_factory =
        kernel_registry->get_kernel(name, requested_device_type);
    kernel_factories.push_back(kernel_factory);

    KernelConfig kernel_config;
    kernel_config.work_item_size = work_item_size;
    kernel_config.node_id = node_id_;
    kernel_config.node_count = node_count;
    kernel_config.args =
        std::vector<u8>(op.kernel_args().begin(), op.kernel_args().end());
    const std::vector<Column>& output_columns = op_info->output_columns();
    for (auto& col : output_columns) {
      kernel_config.output_columns.push_back(col.name());
    }

    for (auto& input : op.inputs()) {
      const proto::Op& input_op = ops.Get(input.op_index());
      if (input_op.name() == "InputTable") {
      } else {
        OpInfo* input_op_info = op_registry->get_op_info(input_op.name());
        // TODO: verify that input.columns() are all in
        // op_info->output_columns()
      }
      kernel_config.input_columns.insert(kernel_config.input_columns.end(),
                                         input.columns().begin(),
                                         input.columns().end());
    }
    kernel_configs.push_back(kernel_config);
  }

  // Break up kernels into groups that run on the same device
  std::vector<std::vector<std::tuple<KernelFactory*, KernelConfig>>>
      kernel_groups;
  std::vector<std::vector<std::vector<std::tuple<i32, std::string>>>>
      kg_live_columns;
  std::vector<std::vector<std::vector<i32>>> kg_dead_columns;
  std::vector<std::vector<std::vector<i32>>> kg_unused_outputs;
  std::vector<std::vector<std::vector<i32>>> kg_column_mapping;
  std::vector<std::vector<std::vector<i32>>> kg_stencils;
  std::vector<std::vector<i32>> kg_batch_sizes;
  if (!kernel_factories.empty()) {
    DeviceType last_device_type = kernel_factories[0]->get_device_type();
    kernel_groups.emplace_back();
    kg_live_columns.emplace_back();
    kg_dead_columns.emplace_back();
    kg_unused_outputs.emplace_back();
    kg_column_mapping.emplace_back();
    kg_stencils.emplace_back();
    kg_batch_sizes.emplace_back();
    for (size_t i = 0; i < kernel_factories.size(); ++i) {
      KernelFactory* factory = kernel_factories[i];
      if (factory->get_device_type() != last_device_type) {
        // Does not use the same device as previous kernel, so push into new
        // group
        last_device_type = factory->get_device_type();
        kernel_groups.emplace_back();
        kg_live_columns.emplace_back();
        kg_dead_columns.emplace_back();
        kg_unused_outputs.emplace_back();
        kg_column_mapping.emplace_back();
        kg_stencils.emplace_back();
        kg_batch_sizes.emplace_back();
      }
      auto& group = kernel_groups.back();
      auto& lc = kg_live_columns.back();
      auto& dc = kg_dead_columns.back();
      auto& uo = kg_unused_outputs.back();
      auto& cm = kg_column_mapping.back();
      auto& st = kg_stencils.back();
      auto& bt = kg_batch_sizes.back();
      group.push_back(std::make_tuple(factory, kernel_configs[i]));
      lc.push_back(live_columns[i]);
      dc.push_back(dead_columns[i]);
      uo.push_back(unused_outputs[i]);
      cm.push_back(column_mapping[i]);
      st.push_back(analysis_results.stencils[i]);
      bt.push_back(analysis_results.batch_sizes[i]);
    }
  }

  i32 num_kernel_groups = static_cast<i32>(kernel_groups.size());
  assert(num_kernel_groups > 0);  // is this actually necessary?

  i32 pipeline_instances_per_node = job_params->pipeline_instances_per_node();
  // If ki per node is -1, we set a smart default. Currently, we calculate the
  // maximum possible kernel instances without oversubscribing any part of the
  // pipeline, either CPU or GPU.
  bool has_gpu_kernel = false;
  if (pipeline_instances_per_node == -1) {
    pipeline_instances_per_node = std::numeric_limits<i32>::max();
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      auto& group = kernel_groups[kg];
      for (i32 k = 0; k < group.size(); ++k) {
        KernelFactory* factory = std::get<0>(group[k]);
        DeviceType device_type = factory->get_device_type();
        i32 max_devices = factory->get_max_devices();
        if (max_devices == Kernel::UnlimitedDevices) {
          pipeline_instances_per_node = 1;
        } else {
          pipeline_instances_per_node =
              std::min(pipeline_instances_per_node,
                       device_type == DeviceType::CPU
                           ? db_params_.num_cpus / local_total / max_devices
                           : (i32)num_gpus / max_devices);
        }
        if (device_type == DeviceType::GPU) {
          has_gpu_kernel = true;
        }
      }
    }
  }

  if (pipeline_instances_per_node <= 0) {
    RESULT_ERROR(job_result,
                 "JobParameters.pipeline_instances_per_node must -1 for "
                 "auto-default or "
                 " greater than 0 for manual configuration.");
    return grpc::Status::OK;
  }

  // Set up memory pool if different than previous memory pool
  if (!memory_pool_initialized_ ||
      job_params->memory_pool_config() != cached_memory_pool_config_) {
    if (db_params_.num_cpus < local_total * pipeline_instances_per_node &&
        job_params->memory_pool_config().cpu().use_pool()) {
      RESULT_ERROR(job_result,
                   "Cannot oversubscribe CPUs and also use CPU memory pool");
      return grpc::Status::OK;
    }
    if (db_params_.gpu_ids.size() < local_total * pipeline_instances_per_node &&
        job_params->memory_pool_config().gpu().use_pool()) {
      RESULT_ERROR(job_result,
                   "Cannot oversubscribe GPUs and also use GPU memory pool");
      return grpc::Status::OK;
    }
    if (memory_pool_initialized_) {
      destroy_memory_allocators();
    }
    init_memory_allocators(job_params->memory_pool_config(), gpu_ids);
    cached_memory_pool_config_ = job_params->memory_pool_config();
    memory_pool_initialized_ = true;
  }

  omp_set_num_threads(std::thread::hardware_concurrency());

  // Setup shared resources for distributing work to processing threads
  i64 accepted_items = 0;
  LoadInputQueue load_work;
  std::vector<EvalQueue> initial_eval_work(pipeline_instances_per_node);
  std::vector<std::vector<EvalQueue>> eval_work(pipeline_instances_per_node);
  OutputEvalQueue output_eval_work(pipeline_instances_per_node);
  std::vector<SaveInputQueue> save_work(db_params_.num_save_workers);
  std::atomic<i64> retired_items{0};

  // Setup load workers
  i32 num_load_workers = db_params_.num_load_workers;
  std::vector<Profiler> load_thread_profilers;
  for (i32 i = 0; i < num_load_workers; ++i) {
    load_thread_profilers.emplace_back(Profiler(base_time));
  }
  std::vector<std::thread> load_threads;
  for (i32 i = 0; i < num_load_workers; ++i) {
    LoadWorkerArgs args{// Uniform arguments
                        node_id_,
                        // Per worker arguments
                        i, db_params_.storage_config, load_thread_profilers[i],
                        job_params->load_sparsity_threshold(), io_item_size,
                        work_item_size};

    load_threads.emplace_back(load_driver, std::ref(load_work),
                              std::ref(initial_eval_work), args);
  }

  // Setup evaluate workers
  std::vector<std::vector<Profiler>> eval_profilers(
      pipeline_instances_per_node);
  std::vector<std::vector<proto::Result>> eval_results(
      pipeline_instances_per_node);

  std::vector<std::tuple<EvalQueue*, EvalQueue*>> pre_eval_queues;
  std::vector<PreEvaluateWorkerArgs> pre_eval_args;
  std::vector<std::vector<std::tuple<EvalQueue*, EvalQueue*>>> eval_queues(
      pipeline_instances_per_node);
  std::vector<std::vector<EvaluateWorkerArgs>> eval_args(
      pipeline_instances_per_node);
  std::vector<std::tuple<EvalQueue*, OutputEvalQueue*>> post_eval_queues;
  std::vector<PostEvaluateWorkerArgs> post_eval_args;

  i32 next_cpu_num = 0;
  i32 next_gpu_idx = 0;
  for (i32 ki = 0; ki < pipeline_instances_per_node; ++ki) {
    auto& work_queues = eval_work[ki];
    std::vector<Profiler>& eval_thread_profilers = eval_profilers[ki];
    std::vector<proto::Result>& results = eval_results[ki];
    work_queues.resize(num_kernel_groups - 1 + 2);  // +2 for pre/post
    results.resize(num_kernel_groups);
    for (auto& result : results) {
      result.set_success(true);
    }
    for (i32 i = 0; i < num_kernel_groups + 2; ++i) {
      eval_thread_profilers.push_back(Profiler(base_time));
    }

    // Evaluate worker
    DeviceHandle first_kernel_type;
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      auto& group = kernel_groups[kg];
      auto& lc = kg_live_columns[kg];
      auto& dc = kg_dead_columns[kg];
      auto& uo = kg_unused_outputs[kg];
      auto& cm = kg_column_mapping[kg];
      auto& st = kg_stencils[kg];
      auto& bt = kg_batch_sizes[kg];
      std::vector<EvaluateWorkerArgs>& thread_args = eval_args[ki];
      std::vector<std::tuple<EvalQueue*, EvalQueue*>>& thread_qs =
          eval_queues[ki];
      // HACK(apoms): we assume all ops in a kernel group use the
      //   same number of devices for now.
      // for (size_t i = 0; i < group.size(); ++i) {
      KernelFactory* factory = std::get<0>(group[0]);
      DeviceType device_type = factory->get_device_type();
      if (device_type == DeviceType::CPU) {
        for (i32 i = 0; i < factory->get_max_devices(); ++i) {
          i32 device_id = 0;
          next_cpu_num++ % num_cpus;
          for (size_t i = 0; i < group.size(); ++i) {
            KernelConfig& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      } else {
        for (i32 i = 0; i < factory->get_max_devices(); ++i) {
          i32 device_id = gpu_ids[next_gpu_idx++ % num_gpus];
          for (size_t i = 0; i < group.size(); ++i) {
            KernelConfig& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      }
      // Get the device handle for the first kernel in the pipeline
      if (kg == 0) {
        first_kernel_type = std::get<1>(group[0]).devices[0];
      }

      // Input work queue
      EvalQueue* input_work_queue = &work_queues[kg];
      // Create new queue for output, reuse previous queue as input
      EvalQueue* output_work_queue = &work_queues[kg + 1];
      // Create eval thread for passing data through neural net
      thread_qs.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      thread_args.emplace_back(EvaluateWorkerArgs{
          // Uniform arguments
          node_id_,

          // Per worker arguments
          ki, kg, group, lc, dc, uo, cm, st, bt, eval_thread_profilers[kg + 1],
          results[kg]});
    }
    // Pre evaluate worker
    {
      EvalQueue* input_work_queue;
      if (distribute_work_dynamically) {
        input_work_queue = &initial_eval_work[ki];
      } else {
        input_work_queue = &initial_eval_work[0];
      }
      EvalQueue* output_work_queue =
          &work_queues[0];
      assert(kernel_groups.size() > 0);
      pre_eval_queues.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      DeviceHandle decoder_type = std::getenv("FORCE_CPU_DECODE")
        ? CPU_DEVICE
        : first_kernel_type;
      pre_eval_args.emplace_back(PreEvaluateWorkerArgs{
          // Uniform arguments
          node_id_, num_cpus,

          // Per worker arguments
          ki, decoder_type, eval_thread_profilers.front(),
      });
    }

    // Post evaluate worker
    {
      auto& output_op = ops.Get(ops.size() - 1);
      std::vector<std::string> column_names;
      for (auto& op_input : output_op.inputs()) {
        for (auto& input : op_input.columns()) {
          column_names.push_back(input);
        }
      }

      EvalQueue* input_work_queue = &work_queues.back();
      OutputEvalQueue* output_work_queue = &output_eval_work;
      post_eval_queues.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      post_eval_args.emplace_back(PostEvaluateWorkerArgs{
          // Uniform arguments
          node_id_,

          // Per worker arguments
          ki, eval_thread_profilers.back(), column_mapping.back(),
          final_output_columns, final_compression_options,
      });
    }
  }

  // Launch eval worker threads
  std::vector<std::thread> pre_eval_threads;
  std::vector<std::vector<std::thread>> eval_threads;
  std::vector<std::thread> post_eval_threads;
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Pre thread
    pre_eval_threads.emplace_back(
        pre_evaluate_driver, std::ref(*std::get<0>(pre_eval_queues[pu])),
        std::ref(*std::get<1>(pre_eval_queues[pu])), pre_eval_args[pu]);
    // Op threads
    eval_threads.emplace_back();
    std::vector<std::thread>& threads = eval_threads.back();
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      threads.emplace_back(
          evaluate_driver, std::ref(*std::get<0>(eval_queues[pu][kg])),
          std::ref(*std::get<1>(eval_queues[pu][kg])), eval_args[pu][kg]);
    }
    // Post threads
    post_eval_threads.emplace_back(
        post_evaluate_driver, std::ref(*std::get<0>(post_eval_queues[pu])),
        std::ref(*std::get<1>(post_eval_queues[pu])), post_eval_args[pu]);
  }

  // Setup save coordinator
  std::thread save_coordinator_thread(
      save_coordinator, std::ref(output_eval_work), std::ref(save_work));

  // Setup save workers
  i32 num_save_workers = db_params_.num_save_workers;
  std::vector<Profiler> save_thread_profilers;
  for (i32 i = 0; i < num_save_workers; ++i) {
    save_thread_profilers.emplace_back(Profiler(base_time));
  }
  std::vector<std::thread> save_threads;
  for (i32 i = 0; i < num_save_workers; ++i) {
    SaveWorkerArgs args{// Uniform arguments
                        node_id_,

                        // Per worker arguments
                        i, db_params_.storage_config, save_thread_profilers[i]};

    save_threads.emplace_back(save_driver, std::ref(save_work[i]),
                              std::ref(retired_items), args);
  }

  if (job_params->profiling()) {
    sleep(10);
  }
  timepoint_t start_time = now();

  // Monitor amount of work left and request more when running low
  // Round robin work
  i32 last_work_queue = 0;
  while (true) {
    i32 local_work = accepted_items - retired_items;
    if (local_work < pipeline_instances_per_node * job_params->tasks_in_queue_per_pu()) {
      grpc::ClientContext context;
      proto::NodeInfo node_info;
      proto::NewWork new_work;

      node_info.set_node_id(node_id_);
      grpc::Status status = master_->NextWork(&context, node_info, &new_work);
      if (!status.ok()) {
        RESULT_ERROR(job_result,
                     "Worker %d could not get next work from master", node_id_);
        break;
      }

      i32 next_item = new_work.io_item().item_id();
      if (next_item == -1) {
        // No more work left
        VLOG(1) << "Node " << node_id_ << " received done signal.";
        break;
      } else {
        // Perform analysis on load work entry to determine upstream
        // requirements and when to discard elements.
        std::deque<TaskStream> task_stream;
        LoadWorkEntry stenciled_entry;
        derive_stencil_requirements(storage_, analysis_results,
                                    new_work.load_work(),
                                    analysis_results.stencils, work_item_size,
                                    stenciled_entry, task_stream);

        i32 target_work_queue =
            distribute_work_dynamically ? last_work_queue++ : 0;
        load_work.push(std::make_tuple(target_work_queue, task_stream,
                                       new_work.io_item(), stenciled_entry));
        last_work_queue %= pipeline_instances_per_node;
        accepted_items++;
      }
    }

    for (size_t i = 0; i < eval_results.size(); ++i) {
      for (size_t j = 0; j < eval_results[i].size(); ++j) {
        auto& result = eval_results[i][j];
        if (!result.success()) {
          LOG(WARNING) << "(N/KI/KG: " << node_id_ << "/" << i << "/" << j
                       << ") returned error result: " << result.msg();
          job_result->set_success(false);
          job_result->set_msg(result.msg());
          goto leave_loop;
        }
      }
    }
    goto remain_loop;
  leave_loop:
    break;
  remain_loop:

    std::this_thread::yield();
  }

  // If the job failed, can't expect queues to have drained, so
  // attempt to flush all all queues here (otherwise we could block
  // on pushing into a queue)
  if (!job_result->success()) {
    load_work.clear();
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      initial_eval_work[pu].clear();
    }
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
        eval_work[pu][kg].clear();
      }
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      eval_work[pu].back().clear();
    }
    output_eval_work.clear();
    for (i32 i = 0; i < num_save_workers; ++i) {
      save_work[i].clear();
    }
  }

  auto push_exit_message = [](EvalQueue& q) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    q.push(std::make_tuple(std::deque<TaskStream>(), IOItem{}, entry));
  };

  auto push_output_eval_exit_message = [](OutputEvalQueue& q) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    q.push(std::make_tuple(IOItem{}, entry));
  };

  auto push_save_exit_message = [](SaveInputQueue& q) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    q.push(std::make_tuple(IOItem{}, entry));
  };

  // Push sentinel work entries into queue to terminate load threads
  for (i32 i = 0; i < num_load_workers; ++i) {
    LoadWorkEntry entry;
    entry.set_io_item_index(-1);
    load_work.push(
        std::make_tuple(0, std::deque<TaskStream>(), IOItem{}, entry));
  }

  for (i32 i = 0; i < num_load_workers; ++i) {
    // Wait until all load threads have finished
    load_threads[i].join();
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    if (distribute_work_dynamically) {
      push_exit_message(initial_eval_work[i]);
    } else {
      push_exit_message(initial_eval_work[0]);
    }
  }

  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    // Wait until pre eval has finished
    pre_eval_threads[i].join();
  }

  for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      push_exit_message(eval_work[pu][kg]);
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      // Wait until eval has finished
      eval_threads[pu][kg].join();
    }
  }

  // Terminate post eval threads
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    push_exit_message(eval_work[pu].back());
  }
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Wait until eval has finished
    post_eval_threads[pu].join();
  }

  // Push sentinel work entries into queue to terminate coordinator thread
  push_output_eval_exit_message(output_eval_work);
  save_coordinator_thread.join();

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < num_save_workers; ++i) {
    push_save_exit_message(save_work[i]);
  }
  for (i32 i = 0; i < num_save_workers; ++i) {
    // Wait until eval has finished
    save_threads[i].join();
  }

  // Ensure all files are flushed
  if (job_params->profiling()) {
    std::fflush(NULL);
    sync();
  }

  if (!job_result->success()) {
    return grpc::Status::OK;
  }

  // Write out total time interval
  timepoint_t end_time = now();

  // Execution done, write out profiler intervals for each worker
  // TODO: job_name -> job_id?
  i32 job_id = meta.get_job_id(job_params->job_name());
  std::string profiler_file_name = job_profiler_path(job_id, node_id_);
  std::unique_ptr<WriteFile> profiler_output;
  BACKOFF_FAIL(
      make_unique_write_file(storage_, profiler_file_name, profiler_output));

  i64 base_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
          .time_since_epoch()
          .count();
  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(start_time)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
          .time_since_epoch()
          .count();
  s_write(profiler_output.get(), start_time_ns);
  s_write(profiler_output.get(), end_time_ns);

  i64 out_rank = node_id_;
  // Load worker profilers
  u8 load_worker_count = num_load_workers;
  s_write(profiler_output.get(), load_worker_count);
  for (i32 i = 0; i < num_load_workers; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "load", "", i,
                           load_thread_profilers[i]);
  }

  // Evaluate worker profilers
  u8 eval_worker_count = pipeline_instances_per_node;
  s_write(profiler_output.get(), eval_worker_count);
  u8 profilers_per_chain = 3;
  s_write(profiler_output.get(), profilers_per_chain);
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    i32 i = pu;
    {
      std::string tag = "pre";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_profilers[pu][0]);
    }
    {
      std::string tag = "eval";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_profilers[pu][1]);
    }
    {
      std::string tag = "post";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_profilers[pu][2]);
    }
  }

  // Save worker profilers
  u8 save_worker_count = num_save_workers;
  s_write(profiler_output.get(), save_worker_count);
  for (i32 i = 0; i < num_save_workers; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "save", "", i,
                           save_thread_profilers[i]);
  }

  BACKOFF_FAIL(profiler_output->save());

  VLOG(1) << "Worker " << node_id_ << " finished NewJob";

  return grpc::Status::OK;
}

grpc::Status WorkerImpl::LoadOp(grpc::ServerContext* context,
                                const proto::OpPath* op_path,
                                proto::Empty* empty) {
  const std::string& so_path = op_path->path();
  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  LOG_IF(FATAL, handle == nullptr)
      << "dlopen of " << so_path << " failed: " << dlerror();
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::Shutdown(grpc::ServerContext* context,
                                  const proto::Empty* empty, Result* result) {
  trigger_shutdown_.set();
  result->set_success(true);
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::PokeWatchdog(grpc::ServerContext* context,
                                      const proto::Empty* empty,
                                      proto::Empty* result) {
  watchdog_awake_ = true;
  return grpc::Status::OK;
}

void WorkerImpl::start_watchdog(grpc::Server* server, i32 timeout_ms) {
  watchdog_thread_ = std::thread([this, server, timeout_ms]() {
    double time_since_check = 0;
    // Wait until shutdown is triggered or watchdog isn't woken up
    while (!trigger_shutdown_.raised()) {
      auto sleep_start = now();
      trigger_shutdown_.wait_for(timeout_ms);
      time_since_check += nano_since(sleep_start) / 1e6;
      if (time_since_check > timeout_ms) {
        if (!watchdog_awake_) {
          // Watchdog not woken, time to bail out
          LOG(ERROR) << "Worker did not receive heartbeat in " << timeout_ms
                     << "ms. Shutting down.";
          trigger_shutdown_.set();
        }
        watchdog_awake_ = false;
        time_since_check = 0;
      }
    }
    // Shutdown self
    server->Shutdown();
  });
}
}
}

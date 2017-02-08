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

#include "scanner/engine/runtime.h"
#include "scanner/engine/db.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/op_registry.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/save_worker.h"

#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <grpc++/security/server_credentials.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc/grpc_posix.h>
#include <grpc/support/log.h>
#include <google/protobuf/io/coded_stream.h>

#include <thread>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {
namespace {
inline bool operator==(const MemoryPoolConfig &lhs,
                       const MemoryPoolConfig &rhs) {
  return (lhs.cpu().use_pool() == rhs.cpu().use_pool()) &&
         (lhs.cpu().free_space() == rhs.cpu().free_space()) &&
         (lhs.gpu().use_pool() == rhs.gpu().use_pool()) &&
         (lhs.gpu().free_space() == rhs.gpu().free_space());
}
inline bool operator!=(const MemoryPoolConfig &lhs,
                       const MemoryPoolConfig &rhs) {
  return !(lhs == rhs);
}
}
void create_io_items(
  const proto::JobParameters* job_params,
  const std::map<std::string, TableMetadata> &table_metas,
  const proto::TaskSet &task_set,
  std::vector<IOItem> &io_items,
  std::vector<LoadWorkEntry> &load_work_entries,
  proto::Result* job_result) {
  const i32 io_item_size = job_params->io_item_size();
  auto &tasks = task_set.tasks();
  i32 warmup_size = 0;
  i32 total_rows = 0;
  for (size_t i = 0; i < tasks.size(); ++i) {
    auto &task = tasks.Get(i);
    if (table_metas.count(task.output_table_name()) == 0) {
      RESULT_ERROR(
        job_result,
        "Output table %s does not exist.", task.output_table_name().c_str());
      return;
    }
    i32 table_id = table_metas.at(task.output_table_name()).id();
    assert(task.samples().size() > 0);
    i64 item_id = 0;
    i64 rows_in_task = static_cast<i64>(task.samples(0).rows().size());
    i64 allocated_rows = 0;
    while (allocated_rows < rows_in_task) {
      i64 rows_to_allocate =
          std::min((i64)io_item_size, rows_in_task - allocated_rows);

      IOItem item;
      item.table_id = table_id;
      item.item_id = item_id++;
      item.start_row = allocated_rows;
      item.end_row = allocated_rows + rows_to_allocate;
      io_items.push_back(item);

      proto::LoadWorkEntry load_item;
      load_item.set_io_item_index(io_items.size() - 1);
      for (auto &sample : task.samples()) {
        if (table_metas.count(sample.table_name()) == 0) {
          RESULT_ERROR(
            job_result,
            "Requested table %s does not exist.", sample.table_name().c_str());
          return;
        }
        const TableMetadata &t_meta = table_metas.at(sample.table_name());
        i32 sample_table_id = t_meta.id();
        proto::LoadSample *load_sample = load_item.add_samples();
        load_sample->set_table_id(sample_table_id);
        for (auto col_name : sample.column_names()) {
          load_sample->add_column_ids(t_meta.column_id(col_name));
        }
        i64 e = allocated_rows + rows_to_allocate;
        // Add extra frames for warmup
        i64 s = std::max(allocated_rows - warmup_size, 0L);
        for (; s < e; ++s) {
          load_sample->add_rows(sample.rows(s));
        }
      }
      load_work_entries.push_back(load_item);

      allocated_rows += rows_to_allocate;
    }
    total_rows += rows_in_task;
  }
}

class WorkerImpl final : public proto::Worker::Service {
public:
  WorkerImpl(DatabaseParameters &db_params, std::string master_address)
      : db_params_(db_params) {
    set_database_path(db_params.db_path);

#ifdef DEBUG
    // Stop SIG36 from grpc when debugging
    grpc_use_signal(-1);
#endif
    // google::protobuf::io::CodedInputStream::SetTotalBytesLimit(67108864 * 4,
    //                                                            67108864 * 2);

    master_ = proto::Master::NewStub(grpc::CreateChannel(
        master_address, grpc::InsecureChannelCredentials()));

    proto::WorkerInfo worker_info;
    char hostname[1024];
    if (gethostname(hostname, 1024)) {
      LOG(FATAL) << "gethostname failed";
    }
    worker_info.set_address(std::string(hostname) + ":5002");

    grpc::ClientContext context;
    proto::Registration registration;
    grpc::Status status =
        master_->RegisterWorker(&context, worker_info, &registration);
    LOG_IF(FATAL, !status.ok()) << "Worker could not contact master server at "
                                << master_address << " (" << status.error_code()
                                << "): " << status.error_message();

    node_id_ = registration.node_id();

    storage_ =
        storehouse::StorageBackend::make_from_config(db_params_.storage_config);

    for (i32 id : db_params_.gpu_ids) {
      gpu_device_ids_.push_back(id);
    }
  }

  ~WorkerImpl() {
    delete storage_;
    if (memory_pool_initialized_) {
      destroy_memory_allocators();
    }
  }

  grpc::Status NewJob(grpc::ServerContext *context,
                      const proto::JobParameters *job_params,
                      proto::Result *job_result) {
    job_result->set_success(true);
    set_database_path(db_params_.db_path);

    // Set up memory pool if different than previous memory pool
    if (!memory_pool_initialized_ ||
        job_params->memory_pool_config() != cached_memory_pool_config_) {
      if (memory_pool_initialized_) {
        destroy_memory_allocators();
      }
      init_memory_allocators(job_params->memory_pool_config(), gpu_device_ids_);
      cached_memory_pool_config_ = job_params->memory_pool_config();
      memory_pool_initialized_ = true;
    }

    i32 kernel_instances_per_node = job_params->kernel_instances_per_node();
    if (kernel_instances_per_node <= 0) {
      RESULT_ERROR(
        job_result, "JobParameters.kernel_instances_per_node must be greater than 0.");
      return grpc::Status::OK;
    }

    timepoint_t base_time = now();
    const i32 work_item_size = job_params->work_item_size();
    i32 warmup_size = 0;

    OpRegistry *op_registry = get_op_registry();
    auto &ops = job_params->task_set().ops();
    assert(ops.Get(0).name() == "InputTable");
    // Analyze op DAG to determine what inputs need to be pipped along
    // and when intermediates can be retired -- essentially liveness analysis
    // Op idx -> column name -> last used index
    std::map<i32, std::map<std::string, i32>> intermediates;
    // Start off with the columns from the gathered tables
    {
      auto &input_op = ops.Get(0);
      for (const std::string &input_col : input_op.inputs(0).columns()) {
        intermediates[0].insert({input_col, 0});
      }
    }
    for (size_t i = 1; i < ops.size(); ++i) {
      auto &op = ops.Get(i);
      // For each input, update the intermediate last used index to the
      // current index
      for (auto &eval_input : op.inputs()) {
        i32 parent_index = eval_input.op_index();
        for (const std::string &parent_col : eval_input.columns()) {
          intermediates.at(parent_index).at(parent_col) = i;
        }
      }
      // Add this op's outputs to the intermediate list
      if (i == ops.size() - 1) {
        continue;
      }
      const auto &op_info =
          op_registry->get_op_info(op.name());
      for (const auto &output_column : op_info->output_columns()) {
        intermediates[i].insert({output_column, i});
      }
    }

    // The live columns at each op index
    std::vector<std::vector<std::tuple<i32, std::string>>> live_columns(
        ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      i32 op_index = i;
      auto &columns = live_columns[i];
      size_t max_i = std::min((size_t)(ops.size() - 2), i);
      for (size_t j = 0; j <= max_i; ++j) {
        for (auto &kv : intermediates.at(j)) {
          i32 last_used_index = kv.second;
          if (last_used_index > op_index) {
            // Last used index is greater than current index, so still live
            columns.push_back(std::make_tuple((i32)j, kv.first));
          }
        }
      }
    }

    // The columns to remove for the current kernel
    std::vector<std::vector<i32>> dead_columns(ops.size() - 1);
    // Outputs from the current kernel that are not used
    std::vector<std::vector<i32>> unused_outputs(ops.size() - 1);
    // Indices in the live columns list that are the inputs to the current
    // kernel. Starts from the second evalutor (index 1)
    std::vector<std::vector<i32>> column_mapping(ops.size() - 1);
    for (size_t i = 1; i < ops.size(); ++i) {
      i32 op_index = i;
      auto &prev_columns = live_columns[i - 1];
      auto &op = ops.Get(op_index);
      // Determine which columns are no longer live
      {
        auto &unused = unused_outputs[i - 1];
        auto &dead = dead_columns[i - 1];
        size_t max_i = std::min((size_t)(ops.size() - 2), (size_t)i);
        for (size_t j = 0; j <= max_i; ++j) {
          i32 parent_index = j;
          for (auto &kv : intermediates.at(j)) {
            i32 last_used_index = kv.second;
            if (last_used_index == op_index) {
              // Column is no longer live, so remove it.
              const std::string &col_name = kv.first;
              if (j == i) {
                // This op has an unused output
                i32 col_index = -1;
                const std::vector<std::string> &op_cols =
                    op_registry->get_op_info(op.name())
                        ->output_columns();
                for (size_t k = 0; k < op_cols.size(); k++) {
                  if (col_name == op_cols[k]) {
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
                  const std::tuple<i32, std::string> &live_input =
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
      auto &mapping = column_mapping[op_index - 1];
      for (const auto &eval_input : op.inputs()) {
        i32 parent_index = eval_input.op_index();
        for (const std::string &col : eval_input.columns()) {
          i32 col_index = -1;
          for (i32 k = 0; k < (i32)prev_columns.size(); ++k) {
            const std::tuple<i32, std::string> &live_input = prev_columns[k];
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

    // Setup kernel factories and the kernel configs that will be used
    // to instantiate instances of the op pipeline
    KernelRegistry *kernel_registry = get_kernel_registry();
    std::vector<KernelFactory *> kernel_factories;
    std::vector<Kernel::Config> kernel_configs;
    i32 num_cpus = db_params_.num_cpus;
    assert(num_cpus > 0);

    i32 num_gpus = static_cast<i32>(gpu_device_ids_.size());
    for (size_t i = 1; i < ops.size() - 1; ++i) {
      auto &op = ops.Get(i);
      const std::string &name = op.name();
      OpInfo *op_info =
          op_registry->get_op_info(name);

      DeviceType requested_device_type = op.device_type();
      if (requested_device_type == DeviceType::GPU && num_gpus == 0) {
        RESULT_ERROR(
          job_result,
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

      KernelFactory *kernel_factory =
          kernel_registry->get_kernel(name, requested_device_type);
      kernel_factories.push_back(kernel_factory);

      Kernel::Config kernel_config;
      kernel_config.work_item_size = work_item_size;
      kernel_config.args = std::vector<u8>(op.kernel_args().begin(),
                                           op.kernel_args().end());
      const std::vector<std::string> &output_columns =
          op_info->output_columns();
      kernel_config.output_columns = std::vector<std::string>(
          output_columns.begin(), output_columns.end());

      for (auto &input : op.inputs()) {
        const proto::Op &input_op =
            ops.Get(input.op_index());
        if (input_op.name() == "InputTable") {
        } else {
          OpInfo *input_op_info =
              op_registry->get_op_info(input_op.name());
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
    std::vector<std::vector<std::tuple<KernelFactory *, Kernel::Config>>>
        kernel_groups;
    std::vector<std::vector<std::vector<std::tuple<i32, std::string>>>>
        kg_live_columns;
    std::vector<std::vector<std::vector<i32>>> kg_dead_columns;
    std::vector<std::vector<std::vector<i32>>> kg_unused_outputs;
    std::vector<std::vector<std::vector<i32>>> kg_column_mapping;
    if (!kernel_factories.empty()) {
      DeviceType last_device_type = kernel_factories[0]->get_device_type();
      kernel_groups.emplace_back();
      kg_live_columns.emplace_back();
      kg_dead_columns.emplace_back();
      kg_unused_outputs.emplace_back();
      kg_column_mapping.emplace_back();
      for (size_t i = 0; i < kernel_factories.size(); ++i) {
        KernelFactory *factory = kernel_factories[i];
        if (factory->get_device_type() != last_device_type) {
          // Does not use the same device as previous kernel, so push into new
          // group
          last_device_type = factory->get_device_type();
          kernel_groups.emplace_back();
          kg_live_columns.emplace_back();
          kg_dead_columns.emplace_back();
          kg_unused_outputs.emplace_back();
          kg_column_mapping.emplace_back();
        }
        auto &group = kernel_groups.back();
        auto &lc = kg_live_columns.back();
        auto &dc = kg_dead_columns.back();
        auto &uo = kg_unused_outputs.back();
        auto &cm = kg_column_mapping.back();
        group.push_back(std::make_tuple(factory, kernel_configs[i]));
        lc.push_back(live_columns[i]);
        dc.push_back(dead_columns[i]);
        uo.push_back(unused_outputs[i]);
        cm.push_back(column_mapping[i]);
      }
    }

    i32 num_kernel_groups = static_cast<i32>(kernel_groups.size());

    // Load table metadata for use in constructing io items
    DatabaseMetadata meta =
        read_database_metadata(storage_, DatabaseMetadata::descriptor_path());
    std::map<std::string, TableMetadata> table_meta;
    for (const std::string &table_name : meta.table_names()) {
      std::string table_path =
          TableMetadata::descriptor_path(meta.get_table_id(table_name));
      table_meta[table_name] = read_table_metadata(storage_, table_path);
    }

    // Setup identical io item list on every node
    std::vector<IOItem> io_items;
    std::vector<LoadWorkEntry> load_work_entries;
    create_io_items(job_params, table_meta, job_params->task_set(), io_items,
                    load_work_entries, job_result);

    // Setup shared resources for distributing work to processing threads
    i64 accepted_items = 0;
    Queue<LoadWorkEntry> load_work;
    Queue<EvalWorkEntry> initial_eval_work;
    std::vector<std::vector<Queue<EvalWorkEntry>>> eval_work(
        kernel_instances_per_node);
    Queue<EvalWorkEntry> save_work;
    std::atomic<i64> retired_items{0};

    // Setup load workers
    i32 num_load_workers = db_params_.num_load_workers;
    std::vector<Profiler> load_thread_profilers(num_load_workers,
                                                Profiler(base_time));
    std::vector<LoadThreadArgs> load_thread_args;
    for (i32 i = 0; i < num_load_workers; ++i) {
      // Create IO thread for reading and decoding data
      load_thread_args.emplace_back(LoadThreadArgs{
          // Uniform arguments
          node_id_, io_items, warmup_size, job_params,

          // Per worker arguments
          i, db_params_.storage_config, load_thread_profilers[i],

          // Queues
          load_work, initial_eval_work,
      });
    }
    std::vector<pthread_t> load_threads(num_load_workers);
    for (i32 i = 0; i < num_load_workers; ++i) {
      pthread_create(&load_threads[i], NULL, load_thread, &load_thread_args[i]);
    }

    // Setup evaluate workers
    std::vector<std::vector<Profiler>> eval_profilers(
        kernel_instances_per_node);
    std::vector<std::vector<proto::Result>> eval_results(
      kernel_instances_per_node);
    std::vector<PreEvaluateThreadArgs> pre_eval_args;
    std::vector<std::vector<EvaluateThreadArgs>> eval_args(
        kernel_instances_per_node);
    std::vector<PostEvaluateThreadArgs> post_eval_args;


    i32 next_cpu_num = 0;
    i32 next_gpu_idx = 0;
    for (i32 ki = 0; ki < kernel_instances_per_node; ++ki) {
      std::vector<Queue<EvalWorkEntry>> &work_queues = eval_work[ki];
      std::vector<Profiler> &eval_thread_profilers = eval_profilers[ki];
      std::vector<proto::Result>& results = eval_results[ki];
      work_queues.resize(num_kernel_groups - 1 + 2); // +2 for pre/post
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
        auto &group = kernel_groups[kg];
        auto &lc = kg_live_columns[kg];
        auto &dc = kg_dead_columns[kg];
        auto &uo = kg_unused_outputs[kg];
        auto &cm = kg_column_mapping[kg];
        std::vector<EvaluateThreadArgs> &thread_args = eval_args[ki];
        // HACK(apoms): we assume all ops in a kernel group use the
        //   same number of devices for now.
        // for (size_t i = 0; i < group.size(); ++i) {
        KernelFactory *factory = std::get<0>(group[0]);
        DeviceType device_type = factory->get_device_type();
        if (device_type == DeviceType::CPU) {
          for (i32 i = 0; i < factory->get_max_devices(); ++i) {
            i32 device_id = next_cpu_num++ % num_cpus;
            for (size_t i = 0; i < group.size(); ++i) {
              Kernel::Config &config = std::get<1>(group[i]);
              config.devices.clear();
              config.devices.push_back({device_type, device_id});
            }
          }
        } else {
          for (i32 i = 0; i < factory->get_max_devices(); ++i) {
            i32 device_id = gpu_device_ids_[next_gpu_idx++ % num_gpus];
            for (size_t i = 0; i < group.size(); ++i) {
              Kernel::Config &config = std::get<1>(group[i]);
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
        Queue<EvalWorkEntry> *input_work_queue = &work_queues[kg];
        // Create new queue for output, reuse previous queue as input
        Queue<EvalWorkEntry> *output_work_queue = &work_queues[kg + 1];
        // Create eval thread for passing data through neural net
        thread_args.emplace_back(EvaluateThreadArgs{
            // Uniform arguments
            node_id_, io_items, warmup_size, job_params,

            // Per worker arguments
            ki, kg, group, lc, dc, uo, cm, eval_thread_profilers[kg+1],
            results[kg],

            // Queues
            *input_work_queue, *output_work_queue});
      }
      // Pre evaluate worker
      {
        Queue<EvalWorkEntry> *input_work_queue = &initial_eval_work;
        Queue<EvalWorkEntry> *output_work_queue = &work_queues[0];
        assert(kernel_groups.size() > 0);
        pre_eval_args.emplace_back(PreEvaluateThreadArgs{
            // Uniform arguments
            node_id_, io_items, warmup_size, num_cpus, job_params,

            // Per worker arguments
              ki, first_kernel_type, eval_thread_profilers.front(),

            // Queues
            *input_work_queue, *output_work_queue});
      }

      // Post evaluate worker
      {
        Queue<EvalWorkEntry> *input_work_queue = &work_queues.back();
        Queue<EvalWorkEntry> *output_work_queue = &save_work;
        post_eval_args.emplace_back(
            PostEvaluateThreadArgs{// Uniform arguments
                                   node_id_, io_items, warmup_size,

                                   // Per worker arguments
                                     ki, eval_thread_profilers.back(),

                                   // Queues
                                   *input_work_queue, *output_work_queue});
      }
    }

    // Launch eval worker threads
    std::vector<pthread_t> pre_eval_threads(kernel_instances_per_node);
    std::vector<std::vector<pthread_t>> eval_threads(kernel_instances_per_node);
    std::vector<pthread_t> post_eval_threads(kernel_instances_per_node);
    for (i32 pu = 0; pu < kernel_instances_per_node; ++pu) {
      // Pre thread
      pthread_create(&pre_eval_threads[pu], NULL, pre_evaluate_thread,
                     &pre_eval_args[pu]);
      // Op threads
      std::vector<pthread_t> &threads = eval_threads[pu];
      threads.resize(num_kernel_groups);
      for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
        pthread_create(&threads[kg], NULL, evaluate_thread, &eval_args[pu][kg]);
      }
      // Post threads
      pthread_create(&post_eval_threads[pu], NULL, post_evaluate_thread,
                     &post_eval_args[pu]);
    }

    // Setup save workers
    i32 num_save_workers = db_params_.num_save_workers;
    std::vector<Profiler> save_thread_profilers(num_save_workers,
                                                Profiler(base_time));
    std::vector<SaveThreadArgs> save_thread_args;
    for (i32 i = 0; i < num_save_workers; ++i) {
      // Create IO thread for reading and decoding data
      save_thread_args.emplace_back(
          SaveThreadArgs{// Uniform arguments
                         node_id_, job_params->job_name(), io_items,

                         // Per worker arguments
                         i, db_params_.storage_config, save_thread_profilers[i],

                         // Queues
                         save_work, retired_items});
    }
    std::vector<pthread_t> save_threads(num_save_workers);
    for (i32 i = 0; i < num_save_workers; ++i) {
      pthread_create(&save_threads[i], NULL, save_thread, &save_thread_args[i]);
    }

    timepoint_t start_time = now();

    // Monitor amount of work left and request more when running low
    while (true) {
      i32 local_work = accepted_items - retired_items;
      if (local_work < kernel_instances_per_node * TASKS_IN_QUEUE_PER_PU) {
        grpc::ClientContext context;
        proto::Empty empty;
        proto::IOItem io_item;

        master_->NextIOItem(&context, empty, &io_item);

        i32 next_item = io_item.item_id();
        if (next_item == -1) {
          // No more work left
          break;
        } else {
          LoadWorkEntry &entry = load_work_entries[next_item];
          load_work.push(entry);
          accepted_items++;
        }
      }

      for (auto& results : eval_results) {
        for (auto& result : results) {
          if (!result.success()) {
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

    // Push sentinel work entries into queue to terminate load threads
    for (i32 i = 0; i < num_load_workers; ++i) {
      LoadWorkEntry entry;
      entry.set_io_item_index(-1);
      load_work.push(entry);
    }

    for (i32 i = 0; i < num_load_workers; ++i) {
      // Wait until load has finished
      void *result;
      i32 err = pthread_join(load_threads[i], &result);
      LOG_IF(FATAL, err != 0) << "error in pthread_join of load thread";
      free(result);
    }

    // Push sentinel work entries into queue to terminate eval threads
    for (i32 i = 0; i < kernel_instances_per_node; ++i) {
      EvalWorkEntry entry;
      entry.io_item_index = -1;
      initial_eval_work.push(entry);
    }

    for (i32 i = 0; i < kernel_instances_per_node; ++i) {
      // Wait until pre eval has finished
      void *result;
      i32 err = pthread_join(pre_eval_threads[i], &result);
      LOG_IF(FATAL, err != 0) << "error in pthread_join of pre eval thread";
      free(result);
    }

    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      for (i32 pu = 0; pu < kernel_instances_per_node; ++pu) {
        EvalWorkEntry entry;
        entry.io_item_index = -1;
        eval_work[pu][kg].push(entry);
      }
      for (i32 pu = 0; pu < kernel_instances_per_node; ++pu) {
        // Wait until eval has finished
        void *result;
        i32 err = pthread_join(eval_threads[pu][kg], &result);
        LOG_IF(FATAL, err != 0) << "error in pthread_join of eval thread";
        free(result);
      }
    }

    // Terminate post eval threads
    for (i32 pu = 0; pu < kernel_instances_per_node; ++pu) {
      EvalWorkEntry entry;
      entry.io_item_index = -1;
      eval_work[pu].back().push(entry);
    }
    for (i32 pu = 0; pu < kernel_instances_per_node; ++pu) {
      // Wait until eval has finished
      void *result;
      i32 err = pthread_join(post_eval_threads[pu], &result);
      LOG_IF(FATAL, err != 0) << "error in pthread_join of post eval thread";
      free(result);
    }

    // Push sentinel work entries into queue to terminate save threads
    for (i32 i = 0; i < num_save_workers; ++i) {
      EvalWorkEntry entry;
      entry.io_item_index = -1;
      save_work.push(entry);
    }

    for (i32 i = 0; i < num_save_workers; ++i) {
      // Wait until eval has finished
      void *result;
      i32 err = pthread_join(save_threads[i], &result);
      LOG_IF(FATAL, err != 0) << "error in pthread_join of save thread";
      free(result);
    }

// Ensure all files are flushed
#ifdef SCANNER_PROFILING
    std::fflush(NULL);
    sync();
#endif

    for (auto& results : eval_results) {
      for (auto& result : results) {
        if (!result.success()) {
          job_result->set_success(false);
          job_result->set_msg(result.msg());
          return grpc::Status::OK;
        }
      }
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
    u8 eval_worker_count = kernel_instances_per_node;
    s_write(profiler_output.get(), eval_worker_count);
    u8 profilers_per_chain = 3;
    s_write(profiler_output.get(), profilers_per_chain);
    for (i32 pu = 0; pu < kernel_instances_per_node; ++pu) {
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

    job_result->set_success(true);
    return grpc::Status::OK;
  }

private:
  std::unique_ptr<proto::Master::Stub> master_;
  storehouse::StorageConfig *storage_config_;
  DatabaseParameters db_params_;
  i32 node_id_;
  storehouse::StorageBackend *storage_;
  std::map<std::string, TableMetadata *> table_metas_;
  std::vector<i32> gpu_device_ids_;
  bool memory_pool_initialized_ = false;
  MemoryPoolConfig cached_memory_pool_config_;
};

class MasterImpl final : public proto::Master::Service {
public:
  MasterImpl(DatabaseParameters &params) : db_params_(params) {
    storage_ =
        storehouse::StorageBackend::make_from_config(db_params_.storage_config);
    set_database_path(params.db_path);
  }

  ~MasterImpl() { delete storage_; }

  grpc::Status RegisterWorker(grpc::ServerContext *context,
                              const proto::WorkerInfo *worker_info,
                              proto::Registration *registration) {
    set_database_path(db_params_.db_path);

    workers_.push_back(proto::Worker::NewStub(grpc::CreateChannel(
        worker_info->address(), grpc::InsecureChannelCredentials())));
    registration->set_node_id(workers_.size() - 1);

    return grpc::Status::OK;
  }

  grpc::Status NextIOItem(grpc::ServerContext *context,
                          const proto::Empty *empty, proto::IOItem *io_item) {
    if (next_io_item_to_allocate_ < num_io_items_) {
      io_item->set_item_id(next_io_item_to_allocate_);
      ++next_io_item_to_allocate_;
      i32 items_left = num_io_items_ - next_io_item_to_allocate_;
      if (items_left % 10 == 0) {
        LOG(INFO) << "IO items remaining: " << items_left;
      }
    } else {
      io_item->set_item_id(-1);
    }
    return grpc::Status::OK;
  }

  grpc::Status NewJob(grpc::ServerContext *context,
                      const proto::JobParameters *job_params,
                      proto::Result *job_result) {
    job_result->set_success(true);
    set_database_path(db_params_.db_path);

    const i32 io_item_size = job_params->io_item_size();
    const i32 work_item_size = job_params->work_item_size();

    i32 warmup_size = 0;
    i32 total_rows = 0;

    proto::JobDescriptor job_descriptor;
    job_descriptor.set_io_item_size(io_item_size);
    job_descriptor.set_work_item_size(work_item_size);
    job_descriptor.set_num_nodes(workers_.size());

    // Get output columns from last output op
    auto &ops = job_params->task_set().ops();
    // OpRegistry* op_registry = get_op_registry();
    // OpInfo* output_op = op_registry->get_op_info(
    //   ops.Get(ops.size()-1).name());
    // const std::vector<std::string>& output_columns =
    //   output_op->output_columns();
    auto &last_op = ops.Get(ops.size() - 1);
    assert(last_op.name() == "OutputTable");
    std::vector<std::string> output_columns;
    for (const auto &eval_input : last_op.inputs()) {
      for (const std::string &name : eval_input.columns()) {
        output_columns.push_back(name);
      }
    }
    for (size_t i = 0; i < output_columns.size(); ++i) {
      auto &col_name = output_columns[i];
      Column *col = job_descriptor.add_columns();
      col->set_id(i);
      col->set_name(col_name);
      col->set_type(ColumnType::Other);
    }

    DatabaseMetadata meta =
        read_database_metadata(storage_, DatabaseMetadata::descriptor_path());

    auto &tasks = job_params->task_set().tasks();
    job_descriptor.mutable_tasks()->CopyFrom(tasks);

    // Add job name into database metadata so we can look up what jobs have
    // been ran
    i32 job_id = meta.add_job(job_params->job_name());
    job_descriptor.set_id(job_id);
    job_descriptor.set_name(job_params->job_name());
    write_job_metadata(storage_, JobMetadata(job_descriptor));

    // Create output tables
    for (auto &task : job_params->task_set().tasks()) {
      i32 table_id = meta.add_table(task.output_table_name());
      proto::TableDescriptor table_desc;
      table_desc.set_id(table_id);
      table_desc.set_name(task.output_table_name());
      // Set columns equal to the last op's output columns
      for (size_t i = 0; i < output_columns.size(); ++i) {
        Column *col = table_desc.add_columns();
        col->set_id(i);
        col->set_name(output_columns[i]);
        col->set_type(ColumnType::Other);
      }
      table_desc.set_num_rows(task.samples(0).rows().size());
      table_desc.set_rows_per_item(io_item_size);
      table_desc.set_job_id(job_id);
      write_table_metadata(storage_, TableMetadata(table_desc));
    }

    // Write out database metadata so that workers can read it
    write_database_metadata(storage_, meta);

    // Read all table metadata
    std::map<std::string, TableMetadata> table_meta;
    for (const std::string &table_name : meta.table_names()) {
      std::string table_path =
          TableMetadata::descriptor_path(meta.get_table_id(table_name));
      table_meta[table_name] = read_table_metadata(storage_, table_path);
    }

    std::vector<IOItem> io_items;
    std::vector<LoadWorkEntry> load_work_entries;
    create_io_items(job_params, table_meta, job_params->task_set(), io_items,
                    load_work_entries, job_result);
    if (!job_result->success()) {
      return grpc::Status::OK;
    }

    next_io_item_to_allocate_ = 0;
    num_io_items_ = io_items.size();

    grpc::CompletionQueue cq;
    std::vector<grpc::ClientContext> client_contexts(workers_.size());
    std::vector<grpc::Status> statuses(workers_.size());
    std::vector<proto::Result> replies(workers_.size());
    std::vector<std::unique_ptr<grpc::ClientAsyncResponseReader<proto::Result>>>
        rpcs;

    proto::JobParameters w_job_params;
    w_job_params.CopyFrom(*job_params);
    for (size_t i = 0; i < workers_.size(); ++i) {
      auto &worker = workers_[i];
      rpcs.emplace_back(
          worker->AsyncNewJob(&client_contexts[i], w_job_params, &cq));
      rpcs[i]->Finish(&replies[i], &statuses[i], (void *)i);
    }

    for (size_t i = 0; i < workers_.size(); ++i) {
      void *got_tag;
      bool ok = false;
      GPR_ASSERT(cq.Next(&got_tag, &ok));
      GPR_ASSERT((i64)got_tag < workers_.size());
      assert(ok);

      if (!replies[i].success()) {
        job_result->set_success(false);
        job_result->set_msg(replies[i].msg());
        return grpc::Status::OK;
      }
    }
    // Write table metadata
    return grpc::Status::OK;
  }

  grpc::Status Ping(grpc::ServerContext *context, const proto::Empty *empty1,
                    proto::Empty *empty2) {
    return grpc::Status::OK;
  }

private:
  i32 next_io_item_to_allocate_;
  i32 num_io_items_;
  std::vector<std::unique_ptr<proto::Worker::Stub>> workers_;
  DatabaseParameters db_params_;
  storehouse::StorageBackend *storage_;
};

proto::Master::Service *get_master_service(DatabaseParameters &param) {
  return new MasterImpl(param);
}

proto::Worker::Service *get_worker_service(DatabaseParameters &params,
                                           const std::string &master_address) {
  return new WorkerImpl(params, master_address);
}
}
}

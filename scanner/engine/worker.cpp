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

#include <arpa/inet.h>
#include <grpc/grpc_posix.h>
#include <grpc/support/log.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>

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

void analyze_dag(
  const proto::TaskSet& task_set,
  std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns,
  std::vector<std::vector<i32>>& dead_columns,
  std::vector<std::vector<i32>>& unused_outputs,
  std::vector<std::vector<i32>>& column_mapping) {
  // Start off with the columns from the gathered tables
  OpRegistry* op_registry = get_op_registry();
  auto& ops = task_set.ops();
  std::map<i32, std::vector<std::tuple<std::string, i32>>> intermediates;
  {
    auto& input_op = ops.Get(0);
    for (const std::string& input_col : input_op.inputs(0).columns()) {
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
    // Add this op's outputs to the intermediate list
    if (i == ops.size() - 1) {
      continue;
    }
    const auto& op_info = op_registry->get_op_info(op.name());
    for (const auto& output_column : op_info->output_columns()) {
      intermediates[i].push_back(std::make_tuple(output_column, i));
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
              const std::vector<std::string>& op_cols =
                op_registry->get_op_info(op.name())->output_columns();
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
}
}

WorkerImpl::WorkerImpl(DatabaseParameters& db_params,
                       std::string master_address, std::string worker_port)
  : watchdog_awake_(true), db_params_(db_params) {
  set_database_path(db_params.db_path);

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
}

WorkerImpl::~WorkerImpl() {
  trigger_shutdown_.set();
  watchdog_thread_.join();
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

  i32 local_id = job_params->local_id();
  i32 local_total = job_params->local_total();
  i32 node_count = job_params->global_total();

  timepoint_t base_time = now();
  const i32 work_item_size = job_params->work_item_size();
  i32 warmup_size = 0;

  OpRegistry* op_registry = get_op_registry();
  auto& ops = job_params->task_set().ops();

  // Analyze op DAG to determine what inputs need to be pipped along
  // and when intermediates can be retired -- essentially liveness analysis
  // The live columns at each op index
  std::vector<std::vector<std::tuple<i32, std::string>>> live_columns;
  // The columns to remove for the current kernel
  std::vector<std::vector<i32>> dead_columns;
  // Outputs from the current kernel that are not used
  std::vector<std::vector<i32>> unused_outputs;
  // Indices in the live columns list that are the inputs to the current
  // kernel. Starts from the second evalutor (index 1)
  std::vector<std::vector<i32>> column_mapping;
  analyze_dag(job_params->task_set(), live_columns, dead_columns,
              unused_outputs, column_mapping);

  // Setup kernel factories and the kernel configs that will be used
  // to instantiate instances of the op pipeline
  KernelRegistry* kernel_registry = get_kernel_registry();
  std::vector<KernelFactory*> kernel_factories;
  std::vector<Kernel::Config> kernel_configs;
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

    Kernel::Config kernel_config;
    kernel_config.work_item_size = work_item_size;
    kernel_config.node_id = node_id_;
    kernel_config.node_count = node_count;
    kernel_config.args =
      std::vector<u8>(op.kernel_args().begin(), op.kernel_args().end());
    const std::vector<std::string>& output_columns = op_info->output_columns();
    kernel_config.output_columns =
      std::vector<std::string>(output_columns.begin(), output_columns.end());

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
  std::vector<std::vector<std::tuple<KernelFactory*, Kernel::Config>>>
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
      }
      auto& group = kernel_groups.back();
      auto& lc = kg_live_columns.back();
      auto& dc = kg_dead_columns.back();
      auto& uo = kg_unused_outputs.back();
      auto& cm = kg_column_mapping.back();
      group.push_back(std::make_tuple(factory, kernel_configs[i]));
      lc.push_back(live_columns[i]);
      dc.push_back(dead_columns[i]);
      uo.push_back(unused_outputs[i]);
      cm.push_back(column_mapping[i]);
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

  // Load table metadata for use in constructing io items
  DatabaseMetadata meta =
    read_database_metadata(storage_, DatabaseMetadata::descriptor_path());
  std::map<std::string, TableMetadata> table_meta;
  for (const std::string& table_name : meta.table_names()) {
    std::string table_path =
      TableMetadata::descriptor_path(meta.get_table_id(table_name));
    table_meta[table_name] = read_table_metadata(storage_, table_path);
  }

  // Setup shared resources for distributing work to processing threads
  i64 accepted_items = 0;
  Queue<std::tuple<IOItem, LoadWorkEntry>> load_work;
  Queue<std::tuple<IOItem, EvalWorkEntry>> initial_eval_work;
  std::vector<std::vector<Queue<std::tuple<IOItem, EvalWorkEntry>>>> eval_work(
    pipeline_instances_per_node);
  Queue<std::tuple<IOItem, EvalWorkEntry>> save_work;
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
      node_id_, job_params,

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
    pipeline_instances_per_node);
  std::vector<std::vector<proto::Result>> eval_results(
    pipeline_instances_per_node);
  std::vector<PreEvaluateThreadArgs> pre_eval_args;
  std::vector<std::vector<EvaluateThreadArgs>> eval_args(
    pipeline_instances_per_node);
  std::vector<PostEvaluateThreadArgs> post_eval_args;

  i32 next_cpu_num = 0;
  i32 next_gpu_idx = 0;
  for (i32 ki = 0; ki < pipeline_instances_per_node; ++ki) {
    std::vector<Queue<std::tuple<IOItem, EvalWorkEntry>>>& work_queues =
      eval_work[ki];
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
      std::vector<EvaluateThreadArgs>& thread_args = eval_args[ki];
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
            Kernel::Config& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      } else {
        for (i32 i = 0; i < factory->get_max_devices(); ++i) {
          i32 device_id = gpu_ids[next_gpu_idx++ % num_gpus];
          for (size_t i = 0; i < group.size(); ++i) {
            Kernel::Config& config = std::get<1>(group[i]);
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
      Queue<std::tuple<IOItem, EvalWorkEntry>>* input_work_queue =
        &work_queues[kg];
      // Create new queue for output, reuse previous queue as input
      Queue<std::tuple<IOItem, EvalWorkEntry>>* output_work_queue =
        &work_queues[kg + 1];
      // Create eval thread for passing data through neural net
      thread_args.emplace_back(EvaluateThreadArgs{
        // Uniform arguments
        node_id_, job_params,

        // Per worker arguments
        ki, kg, group, lc, dc, uo, cm, eval_thread_profilers[kg + 1],
        results[kg],

        // Queues
        *input_work_queue, *output_work_queue});
    }
    // Pre evaluate worker
    {
      Queue<std::tuple<IOItem, EvalWorkEntry>>* input_work_queue =
        &initial_eval_work;
      Queue<std::tuple<IOItem, EvalWorkEntry>>* output_work_queue =
        &work_queues[0];
      assert(kernel_groups.size() > 0);
      pre_eval_args.emplace_back(PreEvaluateThreadArgs{
        // Uniform arguments
        node_id_, num_cpus, job_params,

        // Per worker arguments
        ki, first_kernel_type, eval_thread_profilers.front(),

        // Queues
        *input_work_queue, *output_work_queue});
    }

    // Post evaluate worker
    {
      auto& output_op = ops.Get(0);
      std::vector<std::string> column_names;
      for (auto& op_input : output_op.inputs()) {
        for (auto& input : op_input.columns()) {
          column_names.push_back(input);
        }
      }

      Queue<std::tuple<IOItem, EvalWorkEntry>>* input_work_queue =
        &work_queues.back();
      Queue<std::tuple<IOItem, EvalWorkEntry>>* output_work_queue = &save_work;
      post_eval_args.emplace_back(PostEvaluateThreadArgs{
        // Uniform arguments
        node_id_,

        // Per worker arguments
        ki, eval_thread_profilers.back(), column_mapping.back(),

            column_names,

        // Queues
        *input_work_queue, *output_work_queue});
    }
  }

  // Launch eval worker threads
  std::vector<pthread_t> pre_eval_threads(pipeline_instances_per_node);
  std::vector<std::vector<pthread_t>> eval_threads(pipeline_instances_per_node);
  std::vector<pthread_t> post_eval_threads(pipeline_instances_per_node);
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Pre thread
    pthread_create(&pre_eval_threads[pu], NULL, pre_evaluate_thread,
                   &pre_eval_args[pu]);
    // Op threads
    std::vector<pthread_t>& threads = eval_threads[pu];
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
    save_thread_args.emplace_back(SaveThreadArgs{
      // Uniform arguments
      node_id_, job_params->job_name(),

      // Per worker arguments
      i, db_params_.storage_config, save_thread_profilers[i],

      // Queues
      save_work, retired_items});
  }
  std::vector<pthread_t> save_threads(num_save_workers);
  for (i32 i = 0; i < num_save_workers; ++i) {
    pthread_create(&save_threads[i], NULL, save_thread, &save_thread_args[i]);
  }

#ifdef SCANNER_PROFILING
  sleep(10);
#endif
  timepoint_t start_time = now();

  // Monitor amount of work left and request more when running low
  while (true) {
    i32 local_work = accepted_items - retired_items;
    if (local_work < pipeline_instances_per_node * TASKS_IN_QUEUE_PER_PU) {
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
        load_work.push(
          std::make_tuple(new_work.io_item(), new_work.load_work()));
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
    initial_eval_work.clear();
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
        eval_work[pu][kg].clear();
      }
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      eval_work[pu].back().clear();
    }
    save_work.clear();
  }

  // Push sentinel work entries into queue to terminate load threads
  for (i32 i = 0; i < num_load_workers; ++i) {
    LoadWorkEntry entry;
    entry.set_io_item_index(-1);
    load_work.push(std::make_tuple(IOItem{}, entry));
  }

  for (i32 i = 0; i < num_load_workers; ++i) {
    // Wait until load has finished
    void* result;
    i32 err = pthread_join(load_threads[i], &result);
    LOG_IF(FATAL, err != 0) << "error in pthread_join of load thread";
    free(result);
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    initial_eval_work.push(std::make_tuple(IOItem{}, entry));
  }

  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    // Wait until pre eval has finished
    void* result;
    i32 err = pthread_join(pre_eval_threads[i], &result);
    LOG_IF(FATAL, err != 0) << "error in pthread_join of pre eval thread";
    free(result);
  }

  for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      EvalWorkEntry entry;
      entry.io_item_index = -1;
      eval_work[pu][kg].push(std::make_tuple(IOItem{}, entry));
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      // Wait until eval has finished
      void* result;
      i32 err = pthread_join(eval_threads[pu][kg], &result);
      LOG_IF(FATAL, err != 0) << "error in pthread_join of eval thread";
      free(result);
    }
  }

  // Terminate post eval threads
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    eval_work[pu].back().push(std::make_tuple(IOItem{}, entry));
  }
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Wait until eval has finished
    void* result;
    i32 err = pthread_join(post_eval_threads[pu], &result);
    LOG_IF(FATAL, err != 0) << "error in pthread_join of post eval thread";
    free(result);
  }

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < num_save_workers; ++i) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    save_work.push(std::make_tuple(IOItem{}, entry));
  }
  for (i32 i = 0; i < num_save_workers; ++i) {
    // Wait until eval has finished
    void* result;
    i32 err = pthread_join(save_threads[i], &result);
    LOG_IF(FATAL, err != 0) << "error in pthread_join of save thread";
    free(result);
  }

// Ensure all files are flushed
#ifdef SCANNER_PROFILING
  std::fflush(NULL);
  sync();
#endif

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

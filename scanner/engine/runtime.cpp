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
#include "scanner/engine/save_worker.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/db.h"

#include "scanner/evaluators/serialize.h"

#include "scanner/util/common.h"
#include "scanner/util/memory.h"
#include "scanner/util/profiler.h"
#include "scanner/util/queue.h"
#include "scanner/util/storehouse.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

#include <libgen.h>
#include <mpi.h>
#include <pthread.h>
#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "scanner/util/cuda.h"
#endif

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// run_job
void run_job(JobParameters& params) {
  storehouse::StorageBackend* storage =
    storehouse::StorageBackend::make_from_config(params.storage_config);

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  i32 num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  // Get node-local info
  // http://stackoverflow.com/questions/9022496/how-to-determine-mpi-rank-process-number-local-to-a-socket-node
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &shmcomm);
  i32 local_rank;
  MPI_Comm_rank(shmcomm, &local_rank);

  i32 local_num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &local_num_nodes);

  // Load database metadata
  DatabaseMetadata db_meta{};
  {
    std::string db_meta_path = database_metadata_path();
    std::unique_ptr<RandomReadFile> meta_in_file;
    BACKOFF_FAIL(
        make_unique_random_read_file(storage, db_meta_path, meta_in_file));
    u64 pos = 0;
    db_meta = deserialize_database_metadata(meta_in_file.get(), pos);
  }

  // Load the dataset descriptor to find all data files
  i32 dataset_id = db_meta.get_dataset_id(params.dataset_name);
  DatasetDescriptor descriptor;
  {
    std::unique_ptr<RandomReadFile> file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage, dataset_descriptor_path(params.dataset_name), file));
    u64 pos = 0;
    descriptor = deserialize_dataset_descriptor(file.get(), pos);
  }
  DatasetMetadata dataset_meta(descriptor);

  // Establish base time to use for profilers
  timepoint_t base_time = now();

  // Get metadata for all dataset items for distributing to evaluators
  std::vector<std::string> paths(dataset_meta.item_names());

  std::vector<VideoMetadata> video_metadata;
  std::vector<ImageFormatGroupMetadata> image_metadata;
  std::vector<InputFormat> input_formats;
  for (size_t i = 0; i < paths.size(); ++i) {
    const std::string& path = paths.at(i);
    std::unique_ptr<RandomReadFile> metadata_file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage, dataset_item_metadata_path(params.dataset_name, path),
        metadata_file));
    if (dataset_meta.type() == DatasetType_Video) {
      u64 pos = 0;
      video_metadata.push_back(
          deserialize_video_metadata(metadata_file.get(), pos));
      VideoMetadata& meta = video_metadata.back();
      input_formats.emplace_back(meta.width(), meta.height());
    } else if (dataset_meta.type() == DatasetType_Image) {
      u64 pos = 0;
      image_metadata.push_back(
          deserialize_image_format_group_metadata(metadata_file.get(), pos));
      ImageFormatGroupMetadata& meta = image_metadata.back();
      input_formats.emplace_back(meta.width(), meta.height());
    }
  }

  // Generate the pipeline description by feeding in the dataset information
  // into the user supplied pipeline generator function
  std::vector<std::string> dataset_job_names;
  for (i32 job_id : db_meta.dataset_job_ids.at(dataset_id)) {
    dataset_job_names.push_back(db_meta.job_names.at(job_id));
  }
  PipelineDescription pipeline_description;
  {
    DatasetInformation info(params.dataset_name, dataset_job_names, storage);
    pipeline_description = params.pipeline_gen_fn(info);
  }

  // Validate pipeline description and load job metadata for jobs listed in
  // pipeline description tasks
  LOG_IF(FATAL, pipeline_description.tasks.empty())
      << "No tasks specified for pipeline description!";
  std::map<i32, JobMetadata> job_meta;
  for (Task& task : pipeline_description.tasks) {
    LOG_IF(FATAL, task.samples.empty())
        << "No samples specified for task with table name " << task.table_name
        << "!";
    for (TableSample& sample : task.samples) {
      LOG_IF(FATAL, !db_meta.has_job(dataset_id, sample.job_name))
          << "Requested job " << sample.job_name
          << " does not exist in dataset " << params.dataset_name << "!";
      i32 job_id = db_meta.get_job_id(dataset_id, sample.job_name);
      if (job_meta.count(job_id) == 0) {
        JobDescriptor descriptor;
        std::unique_ptr<RandomReadFile> file;
        BACKOFF_FAIL(make_unique_random_read_file(
            storage, job_descriptor_path(params.dataset_name, sample.job_name),
            file));
        u64 pos = 0;
        JobDescriptor desc = deserialize_job_descriptor(file.get(), pos);
        job_meta.insert({job_id, JobMetadata(desc)});
      }
      JobMetadata& meta = job_meta.at(job_id);
      LOG_IF(FATAL, !meta.has_table(sample.table_name))
          << "Requested table " << sample.table_name << " does not exist in "
          << "job " << sample.job_name << " in dataset " << params.dataset_name
          << "!";
      LOG_IF(FATAL, sample.columns.empty())
          << "No columns specified for sampling from table "
          << sample.table_name << " in job " << sample.job_name
          << " in dataset " << params.dataset_name << "!";
      std::set<std::string> job_columns(meta.columns().begin(),
                                        meta.columns().end());
      assert(!job_columns.empty());
      std::string available_columns = *job_columns.begin();
      for (auto it = ++job_columns.begin(); it != job_columns.end(); ++it) {
        available_columns += ", " + *it;
      }
      for (const std::string &column : sample.columns) {
        LOG_IF(FATAL, job_columns.count(column) == 0)
            << "Requested column " << column << " does not exist in table "
            << sample.table_name << " in job " << sample.job_name
            << " in dataset " << params.dataset_name << "! Available columns "
            << "are: " << available_columns;
      }
    }
  }

  // Unwrap factories into raw pointers and get capabilities
  std::vector<EvaluatorFactory*> evaluator_factories;
  for (auto& f : pipeline_description.evaluator_factories) {
    evaluator_factories.push_back(f.get());
  }
  std::vector<EvaluatorCapabilities> evaluator_caps;
  for (EvaluatorFactory* factory : evaluator_factories) {
    evaluator_caps.push_back(factory->get_capabilities());
  }

  // Setup format metadata for each task
  std::map<i32, BatchConfig> format_metadata;
  for (size_t i = 0; i < pipeline_description.tasks.size(); ++i) {
    const auto& task = pipeline_description.tasks[i];
    BatchConfig batch_config;
    for (const auto& sample : task.samples) {
      if (sample.job_name == base_job_name()) {
        i32 table_id = std::atoi(sample.table_name.c_str());
        batch_config.formats.push_back(input_formats[table_id]);
      }
    }
    format_metadata.insert({(i32)i, batch_config});
  }

  // We break up work into IO items which are then broken up into work items
  // to be processed by evaluators
  const i32 io_item_size = rows_per_io_item();
  const i32 work_item_size = rows_per_work_item();

  // It is necessary to track how work was broken up for each video so that the
  // system can later figure out where each output row is located

  // We need to know the maximum warmup size across all evaluators to pass
  // enough work items through the pipeline after a reset, even if it is more
  // than a specific evaluator needed for warmup
  i32 warmup_size = 0;

  // Only calculate the warmup for video datasets
  if (dataset_meta.type() == DatasetType_Video) {
    for (EvaluatorCapabilities& caps : evaluator_caps) {
      warmup_size = std::max(warmup_size, caps.warmup_size);
    }
  }

  // Create job descriptor and list of work
  u32 total_rows = 0;
  std::vector<IOItem> io_items;
  std::vector<LoadWorkEntry> load_work_entries;
  std::vector<size_t> item_task_delimeters;
  std::vector<std::string> final_column_names;
  {
    std::vector<std::string> input_columns;
    auto& task = pipeline_description.tasks[0];
    for (auto& sample : task.samples) {
      input_columns.insert(input_columns.end(),
                           sample.columns.begin(),
                           sample.columns.end());
    }
    for (auto factory : evaluator_factories) {
      input_columns = factory->get_output_columns(input_columns);
    }
    final_column_names = input_columns;
  }
  JobDescriptor job_descriptor;
  job_descriptor.set_io_item_size(io_item_size);
  job_descriptor.set_work_item_size(work_item_size);
  job_descriptor.set_num_nodes(num_nodes);
  for (i32 i = 0; i < (i32)(pipeline_description.tasks.size()); ++i) {
    // Keep track of where tasks start and end so we can try and partition
    // all items associated with a single task to the same evaluator
    item_task_delimeters.push_back(io_items.size());

    Task& task = pipeline_description.tasks.at(i);
    JobDescriptor::Task* jd_task = job_descriptor.add_tasks();
    jd_task->set_table_id(i);
    jd_task->set_table_name(task.table_name);
    for (TableSample &sample : task.samples) {
      i32 sample_job_id = db_meta.get_job_id(dataset_id, sample.job_name);
      JobMetadata& meta = job_meta.at(sample_job_id);

      JobDescriptor::Task::TableSample* jd_sample = jd_task->add_samples();
      jd_sample->set_job_id(sample_job_id);
      i32 sample_table_id =
          meta.table_id(sample.table_name);
      jd_sample->set_table_id(sample_table_id);
      for (const std::string& col : sample.columns) {
        JobDescriptor::Column* jd_col = jd_sample->add_columns();
        jd_col->set_id(meta.column_id(col));
        jd_col->set_name(col);
      }
      for (i64 r : sample.rows) {
        jd_sample->add_rows(r);
      }
    }

    // Split up task into IOItems
    assert(task.samples.size() > 0);
    i64 item_id = 0;
    i64 rows_in_task = static_cast<i64>(task.samples[0].rows.size());
    i64 allocated_rows = 0;
    while (allocated_rows < rows_in_task) {
      i64 rows_to_allocate =
          std::min((i64)io_item_size, rows_in_task - allocated_rows);

      IOItem item;
      item.table_id = i;
      item.item_id = item_id++;
      item.start_row = allocated_rows;
      item.end_row = allocated_rows + rows_to_allocate;
      io_items.push_back(item);

      LoadWorkEntry load_item;
      load_item.io_item_index = io_items.size() - 1;
      for (TableSample &sample : task.samples) {
        i32 sample_job_id = db_meta.get_job_id(dataset_id, sample.job_name);
        JobMetadata& meta = job_meta.at(sample_job_id);
        i32 sample_table_id = meta.table_id(sample.table_name);

        load_item.samples.emplace_back();
        LoadWorkEntry::Sample& load_sample = load_item.samples.back();
        load_sample.job_id = sample_job_id;
        load_sample.table_id = sample_table_id;
        load_sample.columns = sample.columns;
        i64 e = allocated_rows + rows_to_allocate;
        // Add extra frames for warmup
        i64 s = std::max(allocated_rows - warmup_size, 0L);
        for (; s < e; ++s) {
          load_sample.rows.push_back(sample.rows[s]);
        }
      }
      load_work_entries.push_back(load_item);

      allocated_rows += rows_to_allocate;
    }
    total_rows += rows_in_task;
  }
  for (size_t j = 0; j < final_column_names.size(); ++j) {
    JobDescriptor_Column* column = job_descriptor.add_columns();
    column->set_id(j);
    column->set_name(final_column_names[j]);
  }

  if (is_master(rank)) {
    printf("Total IO items: %lu, Total rows: %u\n", io_items.size(),
           total_rows);
  }

  // Setup shared resources for distributing work to processing threads
  i64 accepted_items = 0;
  Queue<LoadWorkEntry> load_work;
  Queue<EvalWorkEntry> initial_eval_work;
  std::vector<std::vector<Queue<EvalWorkEntry>>> eval_work(PUS_PER_NODE);
  Queue<EvalWorkEntry> save_work;
  std::atomic<i64> retired_items{0};

  // Setup load workers
  std::vector<Profiler> load_thread_profilers(LOAD_WORKERS_PER_NODE,
                                              Profiler(base_time));
  std::vector<LoadThreadArgs> load_thread_args;
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    load_thread_args.emplace_back(LoadThreadArgs{
        // Uniform arguments
        dataset_meta, job_meta, video_metadata, image_metadata,
        io_items, warmup_size,

        // Per worker arguments
        i, params.storage_config, load_thread_profilers[i],

        // Queues
        load_work, initial_eval_work,
    });
  }
  std::vector<pthread_t> load_threads(LOAD_WORKERS_PER_NODE);
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    pthread_create(&load_threads[i], NULL, load_thread, &load_thread_args[i]);
  }

  // Setup evaluate workers

  // Initialize factory groups which determine which evaluators run in the
  // same thread. Evaluators running in different threads should be using
  // different physical resources
  std::vector<std::vector<EvaluatorFactory*>> factory_groups;
  if (evaluator_caps.size() == 1) {
    factory_groups.push_back({evaluator_factories.front()});
  } else if (evaluator_caps.size() == 2 &&
             (evaluator_caps.front().can_overlap ||
              evaluator_caps.back().can_overlap)) {
    factory_groups.push_back({evaluator_factories.front()});
    factory_groups.push_back({evaluator_factories.back()});
  } else {
    i32 evaluator_offset_start = 0;
    i32 evaluator_offset_end = static_cast<i32>(evaluator_factories.size() - 1);
    std::vector<EvaluatorFactory*> main_factories;
    if (evaluator_caps.front().can_overlap) {
      factory_groups.push_back({evaluator_factories.front()});
      evaluator_offset_start++;
    }
    main_factories.insert(main_factories.end(),
                          evaluator_factories.begin() + evaluator_offset_start,
                          evaluator_factories.begin() + evaluator_offset_end);
    if (evaluator_caps.back().can_overlap) {
      factory_groups.push_back(main_factories);
      factory_groups.push_back({evaluator_factories.back()});
    } else {
      main_factories.push_back(evaluator_factories.back());
      factory_groups.push_back(main_factories);
    }
  }

  i32 factory_groups_per_chain = static_cast<i32>(factory_groups.size());
  assert(factory_groups_per_chain > 0);

  std::vector<std::vector<Profiler>> eval_chain_profilers(PUS_PER_NODE);
  std::vector<PreEvaluateThreadArgs> pre_eval_args;
  std::vector<std::vector<EvaluateThreadArgs>> eval_chain_args(PUS_PER_NODE);
  std::vector<PostEvaluateThreadArgs> post_eval_args;

  i32 num_gpus = static_cast<i32>(GPU_DEVICE_IDS.size());
  std::set<i32> gpu_device_ids;
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    std::vector<Queue<EvalWorkEntry>>& work_queues = eval_work[pu];
    std::vector<Profiler>& eval_thread_profilers = eval_chain_profilers[pu];
    std::vector<EvaluateThreadArgs>& eval_thread_args = eval_chain_args[pu];
    work_queues.resize(factory_groups_per_chain - 1 + 2 /* for pre/post */);
    // Setup profilers and thread args
    for (i32 fg = 0; fg < factory_groups_per_chain + 2 /* for pre/post */;
         ++fg) {
      eval_thread_profilers.push_back(Profiler(base_time));
    }
    // Pre evaluate worker
    {
      Queue<EvalWorkEntry>* input_work_queue = &initial_eval_work;
      Queue<EvalWorkEntry>* output_work_queue = &work_queues[0];
      pre_eval_args.emplace_back(PreEvaluateThreadArgs{
          // Uniform arguments
          format_metadata, io_items, warmup_size,

          // Per worker arguments
          pu, eval_thread_profilers.front(),

          // Queues
          *input_work_queue, *output_work_queue});
    }

    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      std::vector<EvaluatorConfig> eval_configs;
      for (size_t i = 0; i < factory_groups[fg].size(); ++i) {
        DeviceType evaluator_device_type =
            factory_groups[fg][i]->get_capabilities().device_type;

        EvaluatorConfig eval_config;
        eval_config.max_input_count =
            std::max(rows_per_work_item(), warmup_size);
        eval_config.max_frame_width = dataset_meta.max_width();
        eval_config.max_frame_height = dataset_meta.max_height();

        i32 device_id;
        if (evaluator_device_type == DeviceType::GPU) {
          LOG_IF(FATAL, num_gpus == 0)
              << "Scanner is configured with zero available GPUs but a GPU "
              << "evaluator was reque(PUS_PER_NODE)sted! Please configure Scanner to have "
              << "at least one GPU using the `gpu_device_ids` config option.";

          // If we have more than one MPI process on a single machine, then
          // we should round robin the GPUs between the nodes if possible.
          // This case occurs if having multiple PUs per process would conflict,
          // e.g. Caffe with Python layers.
          i32 base_index = num_gpus / local_num_nodes * local_rank;
          device_id = GPU_DEVICE_IDS[(base_index + pu) % num_gpus];
          gpu_device_ids.insert(device_id);
        } else {
          device_id = pu;
        }

        eval_config.device_ids = {device_id};
        eval_configs.push_back(eval_config);
      }
      // Input work queue
      Queue<EvalWorkEntry>* input_work_queue = &work_queues[fg];
      // Create new queue for output, reuse previous queue as input
      Queue<EvalWorkEntry>* output_work_queue = &work_queues[fg + 1];
      // Create eval thread for passing data through neural net
      eval_thread_args.emplace_back(EvaluateThreadArgs{
          // Uniform arguments
          format_metadata, io_items, warmup_size,

          // Per worker arguments
          pu, fg, factory_groups[fg], eval_configs,
          eval_thread_profilers[fg + 1],

          // Queues
          *input_work_queue, *output_work_queue});
    }
    // Post evaluate worker
    {
      Queue<EvalWorkEntry>* input_work_queue = &work_queues.back();
      Queue<EvalWorkEntry>* output_work_queue = &save_work;
      post_eval_args.emplace_back(PostEvaluateThreadArgs{
          // Uniform arguments
          format_metadata, io_items, warmup_size,

          // Per worker arguments
          pu, eval_thread_profilers.back(),

          // Queues
          *input_work_queue, *output_work_queue});
    }
  }

  std::vector<i32> gpu_device_ids_vec;
  std::copy(gpu_device_ids.begin(), gpu_device_ids.end(),
            std::back_inserter(gpu_device_ids_vec));
  init_memory_allocators(gpu_device_ids_vec, params.memory_pool_config);

  // Launch eval worker threads
  std::vector<pthread_t> pre_eval_threads(PUS_PER_NODE);
  std::vector<std::vector<pthread_t>> eval_chain_threads(PUS_PER_NODE);
  std::vector<pthread_t> post_eval_threads(PUS_PER_NODE);
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    // Pre thread
    pthread_create(&pre_eval_threads[pu], NULL, pre_evaluate_thread,
                   &pre_eval_args[pu]);
    // Evaluator threads
    std::vector<pthread_t>& eval_threads = eval_chain_threads[pu];
    eval_threads.resize(factory_groups_per_chain);
    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      pthread_create(&eval_threads[fg], NULL, evaluate_thread,
                     &eval_chain_args[pu][fg]);
    }
    // Post threads
    pthread_create(&post_eval_threads[pu], NULL, post_evaluate_thread,
                   &post_eval_args[pu]);
  }

  // Setup save workers
  std::vector<Profiler> save_thread_profilers(SAVE_WORKERS_PER_NODE,
                                              Profiler(base_time));
  std::vector<SaveThreadArgs> save_thread_args;
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    save_thread_args.emplace_back(SaveThreadArgs{
        // Uniform arguments
        params.dataset_name, params.out_job_name, io_items,

        // Per worker arguments
        i, params.storage_config, save_thread_profilers[i],

        // Queues
        save_work, retired_items});
  }
  std::vector<pthread_t> save_threads(SAVE_WORKERS_PER_NODE);
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    pthread_create(&save_threads[i], NULL, save_thread, &save_thread_args[i]);
  }

  // Push work into load queues
  if (is_master(rank)) {
    // Begin distributing work on master node
    i32 next_io_item_to_allocate = 0;
    // Wait for clients to ask for work
    while (next_io_item_to_allocate < static_cast<i32>(io_items.size())) {
      // Check if we need to allocate work to our own processing thread
      i32 local_work = accepted_items - retired_items;
      if (local_work < PUS_PER_NODE * TASKS_IN_QUEUE_PER_PU) {
        LoadWorkEntry& entry = load_work_entries[next_io_item_to_allocate++];
        load_work.push(entry);

        accepted_items++;
        if ((static_cast<i32>(io_items.size()) - next_io_item_to_allocate) %
                10 ==
            0) {
          printf("Work items left: %d\n", static_cast<i32>(io_items.size()) -
                                              next_io_item_to_allocate);
          fflush(stdout);
        }
        continue;
      }

      if (num_nodes > 1) {
        i32 more_work;
        MPI_Status status;
        MPI_Recv(&more_work, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        i32 next_item = next_io_item_to_allocate++;
        MPI_Send(&next_item, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        std::this_thread::yield();
      }
    }
    i32 workers_done = 1;
    while (workers_done < num_nodes) {
      i32 more_work;
      MPI_Status status;
      MPI_Recv(&more_work, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
               MPI_COMM_WORLD, &status);
      i32 next_item = -1;
      MPI_Send(&next_item, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      workers_done += 1;
      std::this_thread::yield();
    }
  } else {
    // Monitor amount of work left and request more when running low
    while (true) {
      i32 local_work = accepted_items - retired_items;
      if (local_work < PUS_PER_NODE * TASKS_IN_QUEUE_PER_PU) {
        // Request work when there is only a few unprocessed items left
        i32 more_work = true;
        MPI_Send(&more_work, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        i32 next_item;
        MPI_Recv(&next_item, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        if (next_item == -1) {
          // No more work left
          break;
        } else {
          LoadWorkEntry& entry = load_work_entries[next_item];
          load_work.push(entry);
          accepted_items++;
        }
      }
      std::this_thread::yield();
    }
  }

  // Push sentinel work entries into queue to terminate load threads
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    LoadWorkEntry entry;
    entry.io_item_index = -1;
    load_work.push(entry);
  }

  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    // Wait until load has finished
    void* result;
    i32 err = pthread_join(load_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of load thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    initial_eval_work.push(entry);
  }

  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    // Wait until pre eval has finished
    void* result;
    i32 err = pthread_join(pre_eval_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of pre eval thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
    for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
      EvalWorkEntry entry;
      entry.io_item_index = -1;
      eval_work[pu][fg].push(entry);
    }
    for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
      // Wait until eval has finished
      void* result;
      i32 err = pthread_join(eval_chain_threads[pu][fg], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join of eval thread\n");
        exit(EXIT_FAILURE);
      }
      free(result);
    }
  }
  // Terminate post eval threads
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    eval_work[pu][factory_groups_per_chain].push(entry);
  }
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    // Wait until eval has finished
    void* result;
    i32 err = pthread_join(post_eval_threads[pu], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of post eval thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    EvalWorkEntry entry;
    entry.io_item_index = -1;
    save_work.push(entry);
  }

  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    // Wait until eval has finished
    void* result;
    i32 err = pthread_join(save_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of save thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  if (is_master(rank)) {
    // Add job name into database metadata so we can look up what jobs have
    // been ran
    i32 job_id;
    {
      const std::string db_meta_path = database_metadata_path();

      std::unique_ptr<RandomReadFile> meta_in_file;
      BACKOFF_FAIL(
          make_unique_random_read_file(storage, db_meta_path, meta_in_file));
      u64 pos = 0;
      DatabaseMetadata meta =
          deserialize_database_metadata(meta_in_file.get(), pos);

      job_id = meta.add_job(dataset_id, params.out_job_name);

      std::unique_ptr<WriteFile> meta_out_file;
      BACKOFF_FAIL(
          make_unique_write_file(storage, db_meta_path, meta_out_file));
      serialize_database_metadata(meta_out_file.get(), meta);
    }

    job_descriptor.set_id(job_id);
    job_descriptor.set_name(params.out_job_name);

    // Write out metadata to describe where the output results are for each
    // video
    {
      const std::string job_file_path =
          job_descriptor_path(params.dataset_name, params.out_job_name);
      std::unique_ptr<WriteFile> output_file;
      BACKOFF_FAIL(make_unique_write_file(storage, job_file_path, output_file));

      serialize_job_descriptor(output_file.get(), job_descriptor);

      BACKOFF_FAIL(output_file->save());
    }
  }

  // Write out total time interval
  timepoint_t end_time = now();

  // Execution done, write out profiler intervals for each worker
  std::string profiler_file_name =
      job_profiler_path(params.dataset_name, params.out_job_name, rank);
  std::unique_ptr<WriteFile> profiler_output;
  BACKOFF_FAIL(
      make_unique_write_file(storage, profiler_file_name, profiler_output));

  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
          .time_since_epoch()
          .count();
  write(profiler_output.get(), start_time_ns);
  write(profiler_output.get(), end_time_ns);

  i64 out_rank = rank;
  // Load worker profilers
  u8 load_worker_count = LOAD_WORKERS_PER_NODE;
  write(profiler_output.get(), load_worker_count);
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "load", "", i,
                           load_thread_profilers[i]);
  }

  // Evaluate worker profilers
  u8 eval_worker_count = PUS_PER_NODE;
  write(profiler_output.get(), eval_worker_count);
  u8 groups_per_chain = factory_groups_per_chain;
  write(profiler_output.get(), groups_per_chain);
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      i32 i = pu;
      std::string tag = "fg" + std::to_string(fg);
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_chain_profilers[pu][fg]);
    }
  }

  // Save worker profilers
  u8 save_worker_count = SAVE_WORKERS_PER_NODE;
  write(profiler_output.get(), save_worker_count);
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "save", "", i,
                           save_thread_profilers[i]);
  }

  BACKOFF_FAIL(profiler_output->save());

  delete storage;
}
}

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
#include "scanner/engine/ingest.h"
#include "scanner/eval/pipeline_description.h"

#include "scanner/util/common.h"
#include "scanner/util/profiler.h"
#include "scanner/util/queue.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"
#include "storehouse/storage_config.h"

#include "toml/toml.h"

#include "scanner/video/video_decoder.h"

#ifdef HAVE_SERVER
#include "scanner/server/video_handler_factory.h"
#endif

// For parsing command line args
#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#include "scanner/util/cuda.h"
#endif

#include <libgen.h>
#include <mpi.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

// For setting up libav*
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// For serve command
#ifdef HAVE_SERVER
#include <folly/Memory.h>
#include <folly/Portability.h>
#include <folly/io/async/EventBaseManager.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <unistd.h>
#endif

using namespace scanner;
namespace po = boost::program_options;

#ifdef HAVE_SERVER
namespace pg = proxygen;
#endif

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

const std::string CONFIG_DEFAULT_PATH = "%s/.scanner.toml";

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
  FLAGS_minloglevel = 0;
#ifdef HAVE_CUDA
  CUD_CHECK(cuInit(0));
#endif
}

void shutdown() { MPI_Finalize(); }

class Config {
 public:
  Config(po::variables_map vm, toml::ParseResult pr, bool has_toml)
      : vm(vm), pr(pr), has_toml(has_toml) {}

  bool has(std::string prefix, std::string key) {
    return vm.count(key) ||
           (has_toml && pr.value.find(prefix + "." + key) != nullptr);
  }

  template <typename T>
  T get(std::string prefix, std::string key) {
    if (vm.count(key)) {
      return vm[key].as<T>();
    } else if (has_toml) {
      return pr.value.find(prefix + "." + key)->as<T>();
    } else {
      LOG(FATAL) << "Config key `" << key << "` not found";
    }
  }

 private:
  po::variables_map vm;
  toml::ParseResult pr;
  bool has_toml;
};

int main(int argc, char** argv) {
  // Variables for holding parsed command line arguments

  std::string cmd;  // sub-command to execute
  // Common among sub-commands
  std::string dataset_name;  // name of dataset to create/operate on
  bool force;
  std::string in_job_name;    
  // For ingest sub-command
  std::string dataset_type;           // type of datset to ingest
  std::string paths_file;             // paths of files to turn into dataset
  bool compute_web_metadata = false;  // whether to compute metadata on ingest
  // For run sub-command
  std::string pipeline_name;  // name of pipeline to run
  std::string out_job_name;   // name of job to refer to after run
  // For rm sub-command
  std::string resource_type;  // dataset or job
  std::string resource_name;  // name of resource to rm
  storehouse::StorageConfig* storage_config;
  MemoryPoolConfig memory_pool_config;
  {
    po::variables_map vm;

    po::options_description main_desc("Allowed options");
    main_desc.add_options()("help", "Produce help message")(
        "command", po::value<std::string>()->required(), "Command to execute")(

        "subargs", po::value<std::vector<std::string>>(),
        "Arguments for command")(

        "config_file", po::value<std::string>(),
        "System configuration (# pus, batch, etc) in toml format. "
        "Explicit command line options will override file settings.")(

        "db_path", po::value<std::string>(),
        "Path to the persistent database.")(

        "pus_per_node", po::value<int>(), "Number of PUs per node")(

        "io_item_size", po::value<int>(),
        "Number of rows to load and save together.")(

        "work_item_size", po::value<int>(),
        "Maximum number of rows that will be fed to an evaluator. "
        "Will always smaller than the io item size.")(

        "tasks_in_queue_per_pu", po::value<int>(),
        "Number of tasks a node will try to maintain in the work queue per PU")(

        "load_workers_per_node", po::value<int>(),
        "Number of worker threads processing load jobs per node")(

        "save_workers_per_node", po::value<int>(),
        "Number of worker threads processing save jobs per node")(

        "gpu_device_ids", po::value<std::vector<int>>(),
        "The set of GPUs that scanner should use.");

    po::positional_options_description main_pos;
    main_pos.add("command", 1);
    main_pos.add("subargs", -1);

    std::vector<std::string> opts;
    try {
      auto parsed = po::command_line_parser(argc, argv)
                        .options(main_desc)
                        .positional(main_pos)
                        .allow_unregistered()
                        .run();
      po::store(parsed, vm);
      po::notify(vm);

      // Collect all the unrecognized options from the first pass.
      // This will include the (positional) command name, so we need to erase
      // that.
      opts = po::collect_unrecognized(parsed.options, po::include_positional);
      opts.erase(opts.begin());

    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << main_desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }

    Config* config;
    char path[256];
    snprintf(path, 256, CONFIG_DEFAULT_PATH.c_str(), getenv("HOME"));
    std::string config_path = vm.count("config_file")
                                  ? vm["config_file"].as<std::string>()
                                  : std::string(path);
    std::ifstream config_ifs(config_path);
    if (config_ifs.good()) {
      toml::ParseResult pr = toml::parse(config_ifs);
      config = new Config(vm, pr, true);
    } else {
      toml::Value val;
      toml::ParseResult pr(val, "");
      config = new Config(vm, pr, false);
    }

    if (config->has("job", "pus_per_node")) {
      PUS_PER_NODE = config->get<int>("job", "pus_per_node");
    }
    if (config->has("job", "io_item_size")) {
      IO_ITEM_SIZE = config->get<int>("job", "io_item_size");
    }
    if (config->has("job", "work_item_size")) {
      WORK_ITEM_SIZE = config->get<int>("job", "work_item_size");
    }
    LOG_IF(FATAL, WORK_ITEM_SIZE > IO_ITEM_SIZE)
        << "Work item size (" << WORK_ITEM_SIZE
        << ") must be <= to IO item size (" << IO_ITEM_SIZE << ")!";

    if (config->has("job", "tasks_in_queue_per_pu")) {
      TASKS_IN_QUEUE_PER_PU = config->get<int>("job", "tasks_in_queue_per_pu");
    }
    if (config->has("job", "load_workers_per_node")) {
      LOAD_WORKERS_PER_NODE = config->get<int>("job", "load_workers_per_node");
    }
    if (config->has("job", "save_workers_per_node")) {
      SAVE_WORKERS_PER_NODE = config->get<int>("job", "save_workers_per_node");
    }
    if (config->has("job", "gpu_device_ids")) {
      GPU_DEVICE_IDS = config->get<std::vector<int>>("job", "gpu_device_ids");
    } else {
#ifdef HAVE_CUDA
      i32 num_gpus;
      cudaGetDeviceCount(&num_gpus);
      for (i32 i = 0; i < num_gpus; ++i) {
        GPU_DEVICE_IDS.push_back(i);
      }
#endif
    }

    LOG_IF(FATAL, !config->has("storage", "db_path"))
        << "Scanner config must contain storage.db_path";
    std::string db_path = config->get<std::string>("storage", "db_path");
    set_database_path(db_path);

    if (config->has("memory_pool", "use_pool")) {
      memory_pool_config.use_pool = config->get<bool>("memory_pool", "use_pool");
      if (config->has("memory_pool", "pool_size")) {
        memory_pool_config.pool_size = config->get<i64>("memory_pool", "pool_size");
      } else {
        memory_pool_config.pool_size = DEFAULT_POOL_SIZE;
      }
    } else {
      memory_pool_config.use_pool = false;
    }

    std::string storage_type = config->get<std::string>("storage", "type");
    if (storage_type == "posix") {
      storage_config = storehouse::StorageConfig::make_posix_config();
    } else if (storage_type == "gcs") {
      LOG_IF(FATAL, !config->has("storage", "cert_path"))
          << "Scanner config must contain storage.cert_path";
      std::string cert_path = config->get<std::string>("storage", "cert_path");
      LOG_IF(FATAL, !config->has("storage", "key_path"))
          << "Scanner config must contain storage.key_path";
      std::string key_path = config->get<std::string>("storage", "key_path");
      LOG_IF(FATAL, !config->has("storage", "bucket"))
          << "Scanner config must contain storage.bucket";
      std::string bucket = config->get<std::string>("storage", "bucket");
      std::ifstream ifs(key_path);
      LOG_IF(FATAL, !ifs.is_open()) << "GCS key " << key_path
                                    << " does not exist.";
      std::string key_content((std::istreambuf_iterator<char>(ifs)),
                              (std::istreambuf_iterator<char>()));
      storage_config = storehouse::StorageConfig::make_gcs_config(
          cert_path, key_content, bucket);
    } else {
      LOG(FATAL) << "Unsupported storage type " << storage_type;
    }

    cmd = config->get<std::string>("", "command");

    if (cmd == "ingest") {
      po::options_description ingest_desc("ingest options");
      ingest_desc.add_options()("help", "Produce help message")(
          "dataset_type", po::value<std::string>()->required(),
          "Type of dataset to ingest. One of 'video' or 'image'.")(

          "dataset_name", po::value<std::string>()->required(),
          "Unique name of the dataset to store persistently")(

          "paths_file", po::value<std::string>()->required(),
          "File which contains paths to files to ingest")(

          "compute_web_metadata", po::bool_switch()->default_value(false),
          "If true, generate metadata for Scanner server")(

          "force,f", po::bool_switch()->default_value(false),
          "Overwrite the job if it already exists.");

      po::positional_options_description ingest_pos;
      ingest_pos.add("dataset_type", 1);
      ingest_pos.add("dataset_name", 1);
      ingest_pos.add("paths_file", 1);

      try {
        vm.clear();
        po::store(po::command_line_parser(opts)
                      .options(ingest_desc)
                      .positional(ingest_pos)
                      .run(),
                  vm);
        po::notify(vm);
      } catch (const po::required_option& e) {
        if (vm.count("help")) {
          std::cout << ingest_desc << std::endl;
          return 1;
        } else {
          throw e;
        }
      }

      dataset_type = vm["dataset_type"].as<std::string>();
      dataset_name = vm["dataset_name"].as<std::string>();
      paths_file = vm["paths_file"].as<std::string>();
      compute_web_metadata = vm["compute_web_metadata"].as<bool>();
      force = vm["force"].as<bool>();

    } else if (cmd == "run") {
      po::options_description run_desc("run options");
      run_desc.add_options()("help", "Produce help message")(
          "dataset_name", po::value<std::string>()->required(),
          "Unique name of the dataset to store persistently")(

          "pipeline_name", po::value<std::string>()->required(),
          "Name of the pipeline to run on the given dataset")(

          "out_job_name", po::value<std::string>()->required(),
          "Unique name to refer to the pipeline outputs after completion")(

          "force,f", po::bool_switch()->default_value(false),
          "Overwrite the job if it already exists");

      po::positional_options_description run_pos;
      run_pos.add("dataset_name", 1);
      run_pos.add("pipeline_name", 1);
      run_pos.add("out_job_name", 1);

      try {
        po::store(po::command_line_parser(opts)
                      .options(run_desc)
                      .positional(run_pos)
                      .run(),
                  vm);
        po::notify(vm);
      } catch (const po::required_option& e) {
        if (vm.count("help")) {
          std::cout << run_desc << std::endl;
          return 1;
        } else {
          throw e;
        }
      }

      dataset_name = vm["dataset_name"].as<std::string>();
      pipeline_name = vm["pipeline_name"].as<std::string>();
      out_job_name = vm["out_job_name"].as<std::string>();
      force = vm["force"].as<bool>();

    } else if (cmd == "rm") {
      po::options_description rm_desc("rm options");
      rm_desc.add_options()("help", "Produce help message")(
          "resource_type", po::value<std::string>()->required(),
          "Type of resource to remove: dataset or job")(
          "dataset_name", po::value<std::string>()->required(),
          "Name of dataset.")("job_name", po::value<std::string>(),
                              "Name of job.");

      po::positional_options_description rm_pos;
      rm_pos.add("resource_type", 1);
      rm_pos.add("dataset_name", 1);
      rm_pos.add("job_name", 1);

      try {
        po::store(po::command_line_parser(opts)
                      .options(rm_desc)
                      .positional(rm_pos)
                      .run(),
                  vm);
        po::notify(vm);
      } catch (const po::required_option& e) {
        if (vm.count("help")) {
          std::cout << rm_desc << std::endl;
          return 1;
        } else {
          throw e;
        }
      }

      resource_type = vm["resource_type"].as<std::string>();
      dataset_name = vm["dataset_name"].as<std::string>();
      if (vm.count("job_name")) {
        in_job_name = vm["resource_name"].as<std::string>();
      }

    } else if (cmd == "serve") {
#ifdef HAVE_SERVER
      po::options_description serve_desc("serve options");
      serve_desc.add_options()("help", "Produce help message");

      po::positional_options_description serve_pos;

      try {
        po::store(po::command_line_parser(opts)
                      .options(serve_desc)
                      .positional(serve_pos)
                      .run(),
                  vm);
        po::notify(vm);
      } catch (const po::required_option& e) {
        if (vm.count("help")) {
          std::cout << serve_desc << std::endl;
          return 1;
        } else {
          throw e;
        }
      }

#else
      std::cout << "Scanner not built with results serving support."
                << std::endl;
      return 1;
#endif
    } else {
      std::cout << "Command must be one of "
                << "'ingest', 'run', 'rm', or 'serve'." << std::endl;
      return 1;
    }
  }

  startup(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(storage_config);

  // Setup db metadata if it does not exist yet
  DatabaseMetadata meta{};
  {
    std::string db_meta_path = database_metadata_path();
    storehouse::FileInfo info;
    storehouse::StoreResult result = storage->get_file_info(db_meta_path, info);

    if (result == storehouse::StoreResult::FileDoesNotExist) {
      // Need to initialize db metadata
      std::unique_ptr<storehouse::WriteFile> meta_out_file;
      BACKOFF_FAIL(
          make_unique_write_file(storage, db_meta_path, meta_out_file));
      serialize_database_metadata(meta_out_file.get(), meta);
      BACKOFF_FAIL(meta_out_file->save());
    } else {
      std::unique_ptr<RandomReadFile> meta_in_file;
      BACKOFF_FAIL(
          make_unique_random_read_file(storage, db_meta_path, meta_in_file));
      u64 pos = 0;
      meta = deserialize_database_metadata(meta_in_file.get(), pos);
    }
  }

  if (cmd == "ingest") {
    // The ingest command takes 1) a new dataset name, 2) a file with paths to
    // videos  on the local filesystem and preprocesses the videos into a
    // persistently stored dataset which can then be operated on by the run
    // command.

    if (meta.has_dataset(dataset_name)) {
      if (force) {
        meta.remove_dataset(meta.get_dataset_id(dataset_name));

        std::string db_meta_path = database_metadata_path();
        std::unique_ptr<storehouse::WriteFile> meta_out_file;
        make_unique_write_file(storage, db_meta_path, meta_out_file);
        serialize_database_metadata(meta_out_file.get(), meta);
        BACKOFF_FAIL(meta_out_file->save());
      } else {
        LOG(FATAL) << "Dataset with that name already exists.";
      }
    }

    DatasetType type;
    if (!string_to_dataset_type(dataset_type, type)) {
      LOG(FATAL) << dataset_type << " is not a valid type of dataset. "
                 << "Available dataset types are: video, image";
    }

    ingest(storage_config, type, dataset_name, paths_file,
           compute_web_metadata);
  } else if (cmd == "run") {
    // The run command takes 1) a name for the job, 2) an existing dataset name,
    // 3) a toml file describing the target network to evaluate and evaluates
    // the network on every frame of the given dataset, saving the results and
    // the metadata for the job persistently. The metadata file for the job can
    // be used to find the results for any given video frame.

    if (!meta.has_dataset(dataset_name)) {
      LOG(FATAL) << "Dataset with that name does not exist.";
    }
    i32 dataset_id = meta.get_dataset_id(dataset_name);

    if (meta.has_job(dataset_id, out_job_name)) {
      if (force) {
        meta.remove_job(meta.get_job_id(dataset_id, out_job_name));

        std::string db_meta_path = database_metadata_path();
        std::unique_ptr<storehouse::WriteFile> meta_out_file;
        make_unique_write_file(storage, db_meta_path, meta_out_file);
        serialize_database_metadata(meta_out_file.get(), meta);
        BACKOFF_FAIL(meta_out_file->save());

      } else {
        LOG(FATAL) << "Out job with name " << out_job_name << " already exists "
                   << "for dataset " << dataset_name;
      }
    }

    PipelineGeneratorFn pipe_gen = get_pipeline(pipeline_name);
    JobParameters params;
    params.storage_config = storage_config;
    params.memory_pool_config = memory_pool_config;
    params.dataset_name = dataset_name;
    params.pipeline_gen_fn = pipe_gen;
    params.out_job_name = out_job_name;
    run_job(params);
  } else if (cmd == "rm") {
    // TODO(apoms): properly delete the excess files for the resource we are
    // removing instead of just clearing the metadata
    if (!meta.has_dataset(dataset_name)) {
      LOG(FATAL) << "Cannot remove: dataset with that name does not exist";
    }
    i32 dataset_id = meta.get_dataset_id(dataset_name);
    if (resource_type == "dataset") {
      meta.remove_dataset(dataset_id);
    } else if (resource_type == "job") {
      if (!meta.has_job(dataset_id, in_job_name)) {
        LOG(FATAL) << "Cannot remove: job with that name does not exist";
      }
      meta.remove_job(meta.get_job_id(dataset_id, in_job_name));
    } else {
      LOG(FATAL) << "No resource type named: " << resource_type;
    }

    std::string db_meta_path = database_metadata_path();
    std::unique_ptr<storehouse::WriteFile> meta_out_file;
    make_unique_write_file(storage, db_meta_path, meta_out_file);
    serialize_database_metadata(meta_out_file.get(), meta);
    BACKOFF_FAIL(meta_out_file->save());
  } else if (cmd == "serve") {
#ifdef HAVE_SERVER
    std::string ip = "0.0.0.0";
    i32 http_port = 12000;
    i32 spdy_port = 12001;
    i32 http2_port = 12002;
    i32 threads = 8;
    std::vector<pg::HTTPServer::IPConfig> IPs = {
        {folly::SocketAddress(ip, http_port, true),
         pg::HTTPServer::Protocol::HTTP},

        {folly::SocketAddress(ip, spdy_port, true),
         pg::HTTPServer::Protocol::SPDY},

        {folly::SocketAddress(ip, http2_port, true),
         pg::HTTPServer::Protocol::HTTP2},
    };

    pg::HTTPServerOptions options;
    options.threads = static_cast<size_t>(threads);
    options.idleTimeout = std::chrono::milliseconds(60000);
    options.shutdownOn = {SIGINT, SIGTERM};
    options.enableContentCompression = false;
    options.handlerFactories = pg::RequestHandlerChain()
                                   .addThen<VideoHandlerFactory>(storage_config)
                                   .build();

    pg::HTTPServer server(std::move(options));
    server.bind(IPs);

    // Start HTTPServer mainloop in a separate thread
    std::thread t([&]() { server.start(); });

    t.join();
#endif
  }

  // Cleanup
  delete storage;
  delete storage_config;
  shutdown();

  return EXIT_SUCCESS;
}

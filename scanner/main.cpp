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

#include "scanner/engine.h"
#include "scanner/ingest.h"

#include "scanner/util/common.h"
#include "scanner/util/profiler.h"
#include "scanner/util/queue.h"

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

  Config(po::variables_map vm, toml::ParseResult pr, bool has_toml) : vm(vm), pr(pr), has_toml(has_toml) {}

  bool has(std::string key) {
    return vm.count(key) || (has_toml && pr.value.find(key) != nullptr);
  }

  template<typename T>
  T get(std::string key) {
    if (vm.count(key)) {
      return vm[key].as<T>();
    } else if (has_toml) {
      return pr.value.find(key)->as<T>();
    } else {
      LOG(FATAL) << "Config key `" << key << "` not found";
    }
  }

private:
  po::variables_map vm;
  toml::ParseResult pr;
  bool has_toml;
};

extern std::vector<std::unique_ptr<EvaluatorFactory>>
setup_evaluator_pipeline();

int main(int argc, char** argv) {
  // Variables for holding parsed command line arguments

  std::string db_path;
  std::string cmd;  // sub-command to execute
  // Common among sub-commands
  std::string dataset_name;  // name of dataset to create/operate on
  // For ingest sub-command
  std::string video_paths_file;  // paths of video files to turn into dataset
  // For run sub-command
  std::string job_name;  // name of job to refer to after run
  // For rm sub-command
  std::string resource_type;  // dataset or job
  std::string resource_name;  // name of resource to rm
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

        "pus_per_node", po::value<int>(), "Number of PUs per node")(

        "work_item_size", po::value<int>(), "Size of a work item")(

        "tasks_in_queue_per_pu", po::value<int>(),
        "Number of tasks a node will try to maintain in the work queue per PU")(

        "load_workers_per_node", po::value<int>(),
        "Number of worker threads processing load jobs per node")(

        "save_workers_per_node", po::value<int>(),
        "Number of worker threads processing save jobs per node")(

        "db_path", po::value<std::string>(),
        "Absolute path to scanner database directory");

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

    if (vm.count("help")) {
      std::cout << main_desc << std::endl;
      return 1;
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

    if (config->has("pus_per_node")) {
      PUS_PER_NODE = config->get<int>("pus_per_node");
    }
    if (config->has("work_item_size")) {
      WORK_ITEM_SIZE = config->get<int>("work_item_size");
    }
    if (config->has("tasks_in_queue_per_pu")) {
      TASKS_IN_QUEUE_PER_PU = config->get<int>("tasks_in_queue_per_pu");
    }
    if (config->has("load_workers_per_node")) {
      LOAD_WORKERS_PER_NODE = config->get<int>("load_workers_per_node");
    }
    if (config->has("save_workers_per_node")) {
      SAVE_WORKERS_PER_NODE = config->get<int>("save_workers_per_node");
    }

    db_path = config->get<std::string>("db_path");
    cmd = config->get<std::string>("command");

    if (cmd == "ingest") {
      po::options_description ingest_desc("ingest options");
      ingest_desc.add_options()("help", "Produce help message")(
          "dataset_name", po::value<std::string>()->required(),
          "Unique name of the dataset to store persistently")(
          "video_paths_file", po::value<std::string>()->required(),
          "File which contains paths to video files to process");

      po::positional_options_description ingest_pos;
      ingest_pos.add("dataset_name", 1);
      ingest_pos.add("video_paths_file", 1);

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

      dataset_name = vm["dataset_name"].as<std::string>();
      video_paths_file = vm["video_paths_file"].as<std::string>();

    } else if (cmd == "run") {
      po::options_description run_desc("run options");
      run_desc.add_options()("help", "Produce help message")(
          "job_name", po::value<std::string>()->required(),
          "Unique name to refer to the output of the job after completion")(
          "dataset_name", po::value<std::string>()->required(),
          "Unique name of the dataset to store persistently");

      po::positional_options_description run_pos;
      run_pos.add("job_name", 1);
      run_pos.add("dataset_name", 1);

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

      job_name = vm["job_name"].as<std::string>();
      dataset_name = vm["dataset_name"].as<std::string>();

    } else if (cmd == "rm") {
      po::options_description rm_desc("rm options");
      rm_desc.add_options()("help", "Produce help message")(
          "resource_type", po::value<std::string>()->required(),
          "Type of resource to remove: dataset or job")(
          "resource_name", po::value<std::string>()->required(),
          "Unique name of the resource to remove");

      po::positional_options_description rm_pos;
      rm_pos.add("resource_type", 1);
      rm_pos.add("resource_name", 1);

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
      resource_name = vm["resource_name"].as<std::string>();

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

  // For now, we use a disk based persistent storage with a hardcoded
  // path for storing video and output data persistently
  storehouse::StorageConfig* config =
      storehouse::StorageConfig::make_posix_config(db_path);

  storehouse::StorageBackend *storage =
      storehouse::StorageBackend::make_from_config(config);

  // Setup db metadata if it does not exist yet
  DatabaseMetadata meta{};
  {
    std::string db_meta_path = database_metadata_path();
    storehouse::FileInfo info;
    storehouse::StoreResult result = storage->get_file_info(db_meta_path, info);

    if (result == storehouse::StoreResult::FileDoesNotExist) {
      // Need to initialize db metadata
      std::unique_ptr<storehouse::WriteFile> meta_out_file;
      make_unique_write_file(storage, db_meta_path, meta_out_file);
      serialize_database_metadata(meta_out_file.get(), meta);
    } else {
      std::unique_ptr<RandomReadFile> meta_in_file;
      make_unique_random_read_file(storage, db_meta_path, meta_in_file);
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
      LOG(FATAL) << "Dataset with that name already exists.";
    }

    ingest(config, dataset_name, video_paths_file);
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
    if (meta.has_job(job_name)) {
      LOG(FATAL) << "Job with that name already exists for that dataset";
    }

    VideoDecoderType decoder_type = VideoDecoderType::SOFTWARE;
    std::vector<std::unique_ptr<EvaluatorFactory>> factories =
        setup_evaluator_pipeline();
    std::vector<EvaluatorFactory*> pfactories;
    for (auto& fact : factories) {
      pfactories.push_back(fact.get());
    }
    run_job(config, decoder_type, pfactories, job_name, dataset_name);
  } else if (cmd == "rm") {
    // TODO(apoms): properly delete the excess files for the resource we are
    // removing instead of just clearing the metadata
    if (resource_type == "dataset") {
      if (!meta.has_dataset(resource_name)) {
        LOG(FATAL) << "Cannot remove: dataset with that name does not exist";
      }
      meta.remove_dataset(meta.get_dataset_id(resource_name));
    } else if (resource_type == "job") {
      if (!meta.has_job(resource_name)) {
        LOG(FATAL) << "Cannot remove: job with that name does not exist";
      }
      meta.remove_job(meta.get_job_id(resource_name));
    } else {
      LOG(FATAL) << "No resource type named: " << resource_type;
    }

    std::string db_meta_path = database_metadata_path();
    std::unique_ptr<storehouse::WriteFile> meta_out_file;
    make_unique_write_file(storage, db_meta_path, meta_out_file);
    serialize_database_metadata(meta_out_file.get(), meta);
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
    options.handlerFactories =
        pg::RequestHandlerChain()
            .addThen<VideoHandlerFactory>(config)
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
  delete config;
  shutdown();

  return EXIT_SUCCESS;
}

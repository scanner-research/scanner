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

#include "scanner/video/video_decoder.h"

#ifdef HAVE_CAFFE
#include "scanner/evaluators/caffe/caffe_cpu_evaluator.h"
#include "scanner/evaluators/caffe/facenet/facenet_cpu_input_transformer.h"
#include "scanner/evaluators/caffe/net_descriptor.h"
#endif
#include "scanner/evaluators/image_processing/blur_evaluator.h"

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

namespace {

const std::string DB_PATH = "/Users/apoms/scanner_db";
}

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
  FLAGS_minloglevel = 3;
#ifdef HAVE_CUDA
  CUD_CHECK(cuInit(0));
#endif
}

void shutdown() { MPI_Finalize(); }

int main(int argc, char** argv) {
  // Variables for holding parsed command line arguments

  std::string cmd;  // sub-command to execute
  // Common among sub-commands
  std::string dataset_name;  // name of dataset to create/operate on
  // For ingest sub-command
  std::string video_paths_file;  // paths of video files to turn into dataset
  // For run sub-command
  std::string job_name;  // name of job to refer to after run
  {
    po::variables_map vm;

    po::options_description main_desc("Allowed options");
    main_desc.add_options()("help", "Produce help message")(
        "command", po::value<std::string>()->required(), "Command to execute")(
        "subargs", po::value<std::vector<std::string> >(),
        "Arguments for command")(
        "config_file", po::value<std::string>(),
        "System configuration (# pus, batch, etc) in toml format. "
        "Explicit command line options will overide file settings.")(
        "pus_per_node", po::value<int>(), "Number of PUs per node")(
        "batch_size", po::value<int>(), "Neural Net input batch size")(
        "batches_per_work_item", po::value<int>(),
        "Number of batches in each work item")(
        "tasks_in_queue_per_pu", po::value<int>(),
        "Number of tasks a node will try to maintain in the work queue per PU")(
        "load_workers_per_node", po::value<int>(),
        "Number of worker threads processing load jobs per node")(
        "save_workers_per_node", po::value<int>(),
        "Number of worker threads processing save jobs per node");

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

    if (vm.count("config_file")) {
      std::string config_file_path = vm["config_file"].as<std::string>();
    }

    if (vm.count("pus_per_node")) {
      PUS_PER_NODE = vm["pus_per_node"].as<int>();
    }
    if (vm.count("batch_size")) {
      GLOBAL_BATCH_SIZE = vm["batch_size"].as<int>();
    }
    if (vm.count("batches_per_work_item")) {
      BATCHES_PER_WORK_ITEM = vm["batches_per_work_item"].as<int>();
    }
    if (vm.count("tasks_in_queue_per_pu")) {
      TASKS_IN_QUEUE_PER_PU = vm["tasks_in_queue_per_pu"].as<int>();
    }
    if (vm.count("load_workers_per_node")) {
      LOAD_WORKERS_PER_NODE = vm["load_workers_per_node"].as<int>();
    }
    if (vm.count("save_workers_per_node")) {
      SAVE_WORKERS_PER_NODE = vm["save_workers_per_node"].as<int>();
    }

    cmd = vm["command"].as<std::string>();

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

    } else if (cmd == "serve") {
#ifdef HAVE_SERVER
      po::options_description serve_desc("serve options");
      serve_desc.add_options()("help", "Produce help message")(
          "job_name", po::value<std::string>()->required(),
          "Name of job to serve results for");

      po::positional_options_description serve_pos;
      serve_pos.add("job_name", 1);

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

      job_name = vm["job_name"].as<std::string>();

#else
      std::cout << "Scanner not built with results serving support."
                << std::endl;
      return 1;
#endif
    } else {
      std::cout << "Command must be one of "
                << "'ingest', 'run', or 'serve'." << std::endl;
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
      storehouse::StorageConfig::make_posix_config(DB_PATH);

  // Setup db metadata if it does not exist yet
  {
    storehouse::StorageBackend* storage =
        storehouse::StorageBackend::make_from_config(config);

    std::string db_meta_path = database_metadata_path();
    storehouse::FileInfo info;
    storehouse::StoreResult result = storage->get_file_info(db_meta_path, info);
    if (result == storehouse::StoreResult::FileDoesNotExist) {
      // Need to initialize db metadata
      DatabaseMetadata meta;
      std::unique_ptr<storehouse::WriteFile> meta_out_file;
      make_unique_write_file(storage, db_meta_path, meta_out_file);
      serialize_database_metadata(meta_out_file.get(), meta);
    }

    delete storage;
  }

  if (cmd == "ingest") {
    // The ingest command takes 1) a new dataset name, 2) a file with paths to
    // videos  on the local filesystem and preprocesses the videos into a
    // persistently stored dataset which can then be operated on by the run
    // command.

    ingest(config, dataset_name, video_paths_file);
  } else if (cmd == "run") {
    // The run command takes 1) a name for the job, 2) an existing dataset name,
    // 3) a toml file describing the target network to evaluate and evaluates
    // the network on every frame of the given dataset, saving the results and
    // the metadata for the job persistently. The metadata file for the job can
    // be used to find the results for any given video frame.

    VideoDecoderType decoder_type = VideoDecoderType::SOFTWARE;

#ifdef HAVE_CAFFE
    std::string net_descriptor_file = "features/caffe_facenet.toml";
    NetDescriptor descriptor;
    {
      std::ifstream net_file{net_descriptor_file};
      descriptor = descriptor_from_net_file(net_file);
    }
    FacenetCPUInputTransformerFactory* factory =
        new FacenetCPUInputTransformerFactory();
    CaffeCPUEvaluatorFactory evaluator_factory(descriptor, factory);
#else
    // HACK(apoms): hardcoding the blur evaluator for now. Will allow user code
    //   to specify their own evaluator soon.

    FaceEvaluatorFactory evaluator_factory;
#endif

    run_job(config, decoder_type, &evaluator_factory, job_name, dataset_name);
  } else if (cmd == "serve") {
#ifdef HAVE_SERVER
    std::string ip = "0.0.0.0";
    i32 http_port = 11000;
    i32 spdy_port = 11001;
    i32 http2_port = 11002;
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
            .addThen<VideoHandlerFactory>(config, job_name)
            .build();

    pg::HTTPServer server(std::move(options));
    server.bind(IPs);

    // Start HTTPServer mainloop in a separate thread
    std::thread t([&]() { server.start(); });

    t.join();
#endif
  }

  // Cleanup
  delete config;
  shutdown();

  return EXIT_SUCCESS;
}

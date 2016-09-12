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

#include "scanner/ingest.h"
#include "scanner/engine.h"

#include "scanner/video/video_decoder.h"

#ifdef HAVE_CAFFE
#include "scanner/eval/caffe/net_descriptor.h"
#include "scanner/eval/caffe/caffe_cpu_evaluator.h"
#else
#include "scanner/eval/image_processing/blur_evaluator.h"
#endif

#include "scanner/util/common.h"
#include "scanner/util/queue.h"
#include "scanner/util/profiler.h"

#include "storehouse/storage_config.h"
#include "storehouse/storage_backend.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#ifdef HAVE_CUDA
#include <cuda.h>
#include "scanner/util/cuda.h"
#endif

#include <mpi.h>
#include <cstdlib>
#include <string>
#include <libgen.h>
#include <atomic>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace scanner;
namespace po = boost::program_options;

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace {

const std::string DB_PATH = "/Users/abpoms/kcam";

/* read_last_processed_video - read from persistent storage the index 
 *   of the last succesfully processed video for the given dataset. 
 *   Used to recover from failures midway through the ingest process.
 *
 *   @return: index of the last successfully processed video
 */
int read_last_processed_video(
  storehouse::StorageBackend* storage,
  const std::string& dataset_name)
{
  StoreResult result;

  const std::string last_written_path =
    dataset_name + "_dataset/last_written.bin";

  // File will not exist when first running ingest so check first
  // and return default value if not there
  storehouse::FileInfo info;
  result = storage->get_file_info(last_written_path, info);
  (void) info;
  if (result == StoreResult::FileDoesNotExist) {
    return -1;
  }

  std::unique_ptr<RandomReadFile> file;
  result = make_unique_random_read_file(storage, last_written_path, file);

  uint64_t pos = 0;
  size_t size_read;

  int32_t last_processed_video;
  EXP_BACKOFF(
    file->read(pos,
               sizeof(int32_t),
               reinterpret_cast<char*>(&last_processed_video),
               size_read),
    result);
  assert(result == StoreResult::Success ||
         result == StoreResult::EndOfFile);
  assert(size_read == sizeof(int32_t));

  return last_processed_video;
}

/* write_last_processed_video - write to persistent storage the index 
 *   of the last succesfully processed video for the given dataset. 
 *   Used to recover from failures midway through the ingest process.
 *
 */
void write_last_processed_video(
  storehouse::StorageBackend* storage,
  const std::string& dataset_name,
  int file_index)
{
  const std::string last_written_path =
    dataset_name + "_dataset/last_written.bin";
  std::unique_ptr<WriteFile> file;
  make_unique_write_file(storage, last_written_path, file);

  StoreResult result;
  EXP_BACKOFF(
    file->append(sizeof(int32_t),
                 reinterpret_cast<const char*>(&file_index)),
    result);
  exit_on_error(result);
}

}

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
  FLAGS_minloglevel = 2;
#ifdef HAVE_CUDA
  CUD_CHECK(cuInit(0));
#endif
}

void shutdown() {
  MPI_Finalize();
}

int main(int argc, char** argv) {
  // Variables for holding parsed command line arguments

  std::string cmd; // sub-command to execute
  // Common among sub-commands
  std::string dataset_name; // name of dataset to create/operate on
  // For ingest sub-command
  std::string video_paths_file; // paths of video files to turn into dataset
  // For run sub-command
  std::string job_name; // name of job to refer to after run
  std::string net_descriptor_file; // path to file describing network to use
  {
    po::variables_map vm;

    po::options_description main_desc("Allowed options");
    main_desc.add_options()
      ("help", "Produce help message")
      ("command", po::value<std::string>()->required(),
       "Command to execute")
      ("subargs", po::value<std::vector<std::string> >(),
       "Arguments for command")
      ("config_file", po::value<std::string>(),
       "System configuration (# pus, batch, etc) in toml format. "
       "Explicit command line options will overide file settings.")
      ("pus_per_node", po::value<int>(), "Number of PUs per node")
      ("batch_size", po::value<int>(), "Neural Net input batch size")
      ("batches_per_work_item", po::value<int>(),
       "Number of batches in each work item")
      ("tasks_in_queue_per_pu", po::value<int>(),
       "Number of tasks a node will try to maintain in the work queue per PU")
      ("load_workers_per_node", po::value<int>(),
       "Number of worker threads processing load jobs per node")
      ("save_workers_per_node", po::value<int>(),
       "Number of worker threads processing save jobs per node");

      po::positional_options_description main_pos;
      main_pos.add("command", 1);
      main_pos.add("subargs", -1);

      std::vector<std::string> opts;
      try {
        auto parsed = po::command_line_parser(argc, argv).
          options(main_desc).
          positional(main_pos).
          allow_unregistered().
          run();
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
        ingest_desc.add_options()
          ("help", "Produce help message")
          ("dataset_name", po::value<std::string>()->required(),
           "Unique name of the dataset to store persistently")
          ("video_paths_file", po::value<std::string>()->required(),
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
        run_desc.add_options()
          ("help", "Produce help message")
          ("job_name", po::value<std::string>()->required(),
           "Unique name to refer to the output of the job after completion")
          ("dataset_name", po::value<std::string>()->required(),
           "Unique name of the dataset to store persistently")
          ("net_descriptor_file", po::value<std::string>()->required(),
           "File which contains a description of the net to use");

        po::positional_options_description run_pos;
        run_pos.add("job_name", 1);
        run_pos.add("dataset_name", 1);
        run_pos.add("net_descriptor_file", 1);

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
        net_descriptor_file = vm["net_descriptor_file"].as<std::string>();

      } else {
        std::cout << "Command must be one of 'run' or 'ingest'." << std::endl;
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

  if (cmd == "ingest") {
    // The ingest command takes 1) a new dataset name, 2) a file with paths to videos
    // on the local filesystem and preprocesses the videos into a persistently
    // stored dataset which can then be operated on by the run command.

    log_ls.print("Creating dataset %s...\n", dataset_name.c_str());
    // Read in list of video paths and assign unique name to each
    DatasetDescriptor descriptor;
    std::vector<std::string>& video_paths = descriptor.original_video_paths;
    std::vector<std::string>& item_names = descriptor.item_names;
    {
      int video_count = 0;
      std::fstream fs(video_paths_file, std::fstream::in);
      while (fs) {
        std::string path;
        fs >> path;
        if (path.empty()) continue;
        video_paths.push_back(path);
        item_names.push_back(std::to_string(video_count++));
      }
    }

    storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(config);

    // Start from the file after the one we last processed succesfully before
    // crashing/exiting
    int last_processed_index = read_last_processed_video(storage, dataset_name);

    // Keep track of videos which we can't parse
    std::vector<std::string> bad_paths;
    for (size_t i = last_processed_index + 1; i < video_paths.size(); ++i) {
      const std::string& path = video_paths[i];
      const std::string& item_name = item_names[i];

      if (is_master(rank)) {
        log_ls.print("Ingesting video %s...\n", path.c_str());
        bool valid_video =
          preprocess_video(storage, dataset_name, path, item_name);
        if (!valid_video) {
          bad_paths.push_back(path);
        }

        // Track the last succesfully processed dataset so we know where
        // to resume if we crash or exit early
        write_last_processed_video(storage, dataset_name, static_cast<int>(i));
      }
    }
    if (!bad_paths.empty()) {
      std::fstream bad_paths_file("bad_videos.txt", std::fstream::out);
      for (const std::string& bad_path : bad_paths) {
        bad_paths_file << bad_path << std::endl;
      }
      bad_paths_file.close();
    }

    // Write out dataset descriptor
    {
      const std::string dataset_file_path =
        dataset_descriptor_path(dataset_name);
      std::unique_ptr<WriteFile> output_file;
      make_unique_write_file(storage, dataset_file_path, output_file);

      serialize_dataset_descriptor(output_file.get(), descriptor);
    }
    // Reset last processed so that we start from scratch next time
    // TODO(apoms): alternatively we could delete the file but apparently
    // that was never designed into the storage interface!
    write_last_processed_video(storage, dataset_name, -1);

    delete storage;

  } else if (cmd == "run") {
    // The run command takes 1) a name for the job, 2) an existing dataset name,
    // 3) a toml file describing the target network to evaluate and evaluates 
    // the network on every frame of the given dataset, saving the results and
    // the metadata for the job persistently. The metadata file for the job can 
    // be used to find the results for any given video frame.

    // HACK(apoms): hardcoding the caffe evaluator for now. Will allow user code
    //   to specify their own evaluator soon.

    #ifdef HAVE_CAFFE
    NetDescriptor descriptor;
    {
      std::ifstream net_file{net_descriptor_file};
      descriptor = descriptor_from_net_file(net_file);
    }
    CaffeCPUEvaluatorConstructor evaluator_constructor(descriptor);
    #else
    BlurEvaluatorConstructor evaluator_constructor(3.0);
    #endif

    VideoDecoderType decoder_type = VideoDecoderType::SOFTWARE;

    run_job(
      config,
      decoder_type,
      &evaluator_constructor,
      job_name,
      dataset_name);
  }

  // Cleanup
  delete config;
  shutdown();

  return EXIT_SUCCESS;
}

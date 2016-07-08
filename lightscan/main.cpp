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

#include "lightscan/storage/storage_config.h"
#include "lightscan/storage/storage_backend.h"
#include "lightscan/util/common.h"
#include "lightscan/util/video.h"
#include "lightscan/util/caffe.h"

#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <string>
#include <libgen.h>

extern "C" {
#include "libavformat/avformat.h"
}

using namespace lightscan;

const std::string DB_PATH = "/Users/abpoms/kcam";
const std::string IFRAME_PATH_POSTFIX = "_iframes";
const std::string PROCESSED_VIDEO_POSTFIX = "_processed";
const int NUM_GPUS = 1;

#define THREAD_RETURN_SUCCESS() \
  do {                                           \
    void* val = malloc(sizeof(int));             \
    *((int*)val) = EXIT_SUCCESS;                 \
    pthread_exit(val);                           \
  } while (0);

///////////////////////////////////////////////////////////////////////////////
/// Path utils

std::string dirname_s(const std::string& path) {
  char* path_copy = strdup(path.c_str());
  char* dir = dirname(path_copy);
  return std::string(dir);
}

std::string basename_s(const std::string& path) {
  char* path_copy = strdup(path.c_str());
  char* base = basename(path_copy);
  return std::string(base);
}

std::string processed_video_path(const std::string& video_path) {
  return dirname_s(video_path) + "/" +
    basename_s(video_path) + PROCESSED_VIDEO_POSTFIX + ".mp4";
}

std::string iframe_path(const std::string& video_path) {
  return dirname_s(video_path) + "/" +
    basename_s(video_path) + IFRAME_PATH_POSTFIX + ".bin";
}

///////////////////////////////////////////////////////////////////////////////
/// MPI utils
inline bool is_master(int rank) {
  return rank == 0;
}

///////////////////////////////////////////////////////////////////////////////
///

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
struct LoadVideoArgs {
  // Input arguments
  StorageConfig* storage_config;
  std::string video_path;
  int frame_offset;
  // Output arguments
  int x;
};

void* load_video_thread(void* arg) {
  // Setup connection to load video
  LoadVideoArgs& args = *reinterpret_cast<LoadVideoArgs*>(arg);

  // Setup a distinct storage backend for each IO thread
  StorageBackend* storage =
    StorageBackend::make_from_config(args.storage_config);

  // Open the video file for reading
  RandomReadFile* file;
  storage->make_random_read_file(args.video_path, file);

  // Cleanup
  delete file;
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously save out results
struct SaveVideoArgs {
};

void* save_video_thread(void* arg) {
  // Setup connection to save video
  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Main processing thread that runs the read, evaluate net, write loop
struct ProcessArgs {
  int gpu_device_id;
  StorageConfig* storage_config;
  std::string video_path;
  int frame_offset;
};

void* process_thread(void* arg) {
  ProcessArgs& args = *reinterpret_cast<ProcessArgs*>(arg);

  // Create IO threads for reading and writing
  LoadVideoArgs load_args;
  load_args.storage_config = args.storage_config;
  load_args.video_path = args.video_path;
  load_args.frame_offset = args.frame_offset;
  pthread_t load_thread;
  pthread_create(&load_thread, NULL, load_video_thread, &load_args);

  // pthread_t* save_thread;
  // pthread_create(save_thread, NULL, save_video_thread, NULL);

  // Setup caffe net
  NetInfo net_info = load_neural_net(NetType::VGG, args.gpu_device_id);
  caffe::Net<float>* net = net_info.net;

  // Load
  while (true) {
    // Read batch of frames

    // Decompress batch of frame

    // Process batch of frames

    // Save batch of frames
    break;
  }

  // Cleanup
  THREAD_RETURN_SUCCESS();
}

void shutdown() {
  MPI_Finalize();
}

int main(int argc, char **argv) {
  startup(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string video_path =
    "kcam-videos-20140910_195012_247.mp4";

  // Setup storage config
  StorageConfig* config =
    StorageConfig::make_disk_config(DB_PATH);
  StorageBackend* storage = StorageBackend::make_from_config(config);

  // Check if we have already preprocessed the video
  FileInfo video_info;
  StoreResult result =
    storage->get_file_info(processed_video_path(video_path), video_info);
  if (result == StoreResult::FileDoesNotExist) {
    // Preprocess video and then exit
    if (is_master(rank)) {
      log_ls.print("Video not processed yet. Processing now...\n");
      preprocess_video(storage,
                       video_path,
                       processed_video_path(video_path),
                       iframe_path(video_path));
    }
  } else {
    // Determine video size

    // Parse args to determine video offset

    // Create processing threads for each gpu
    ProcessArgs processing_thread_args[NUM_GPUS];
    pthread_t processing_threads[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; ++i) {
      ProcessArgs& args = processing_thread_args[i];
      args.gpu_device_id = i;
      args.storage_config = config;
      args.video_path = video_path;
      args.frame_offset = 0;
      pthread_create(&processing_threads[i],
                     NULL,
                     process_thread,
                     &processing_thread_args[i]);
    }

    // Wait till done
    for (int i = 0; i < NUM_GPUS; ++i) {
      void* result;

      int err = pthread_join(processing_threads[i], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join\n");
        exit(EXIT_FAILURE);
      }

      printf("Joined with thread %d; returned value was %d\n",
             i, *((int *)result));
      free(result);      /* Free memory allocated by thread */
    }
  }

 // Cleanup
 delete storage;
 delete config;

 shutdown();

 return EXIT_SUCCESS;
}

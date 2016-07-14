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

#include <opencv2/opencv.hpp>

#ifdef HARDWARE_DECODE
#include <cuda.h>
#endif

#include <thread>
#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <string>
#include <libgen.h>
#include <atomic>

extern "C" {
#include "libavformat/avformat.h"
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace lightscan;

const int WORK_ITEM_AMPLIFICATION = 4;
const int LOAD_EVAL_BUFFERS = 4;
const int WORK_SURPLUS_FACTOR = 2;

const bool NO_PCIE_TRANSFER = false;
const std::string DB_PATH = "/Users/abpoms/kcam";
const std::string IFRAME_PATH_POSTFIX = "_iframes";
const std::string METADATA_PATH_POSTFIX = "_metadata";
const std::string PROCESSED_VIDEO_POSTFIX = "_processed";
int global_batch_size;

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

std::string metadata_path(const std::string& video_path) {
  return dirname_s(video_path) + "/" +
    basename_s(video_path) + METADATA_PATH_POSTFIX + ".bin";
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
/// Misc

struct VideoWorkItem {
  int video_index;
  int start_frame;
  int end_frame;
};

inline int max_work_item_size() {
  return global_batch_size * WORK_ITEM_AMPLIFICATION;
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
void convert_av_frame_to_rgb(
  SwsContext*& sws_context,
  AVFrame* frame,
  char* buffer)
{
  size_t buffer_size =
    av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame->width, frame->height, 1);

  // Convert image to RGB
  sws_context = sws_getCachedContext(
    sws_context,

    frame->width, frame->height,
    static_cast<AVPixelFormat>(frame->format),

    frame->width, frame->height, AV_PIX_FMT_RGB24,
    SWS_BICUBIC, 0, 0, 0);

  if (sws_context == nullptr) {
    fprintf(stderr, "Error trying to get sws context\n");
    assert(false);
  }

  AVFrame rgb_format;
  int alloc_fail = av_image_alloc(rgb_format.data,
                                  rgb_format.linesize,
                                  frame->width,
                                  frame->height,
                                  AV_PIX_FMT_RGB24,
                                  1);

  if (alloc_fail < 0) {
    fprintf(stderr, "Error while allocating avpicture for conversion\n");
    assert(false);
  }

  sws_scale(sws_context,
            frame->data /* input data */,
            frame->linesize /* input layout */,
            0 /* x start location */,
            frame->height /* height of input image */,
            rgb_format.data /* output data */,
            rgb_format.linesize /* output layout */);

  av_image_copy_to_buffer(reinterpret_cast<uint8_t*>(buffer),
                          buffer_size,
                          rgb_format.data,
                          rgb_format.linesize,
                          AV_PIX_FMT_RGB24,
                          frame->width,
                          frame->height,
                          1);

  av_freep(&rgb_format.data[0]);
}

struct LoadVideoArgs {
  // Input arguments
  int gpu_device_id; // for hardware decode, need to know gpu
  StorageConfig* storage_config;
  const std::vector<std::string>& video_paths;
  const std::vector<VideoMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  SpinLock& item_lock;
  std::vector<int>& allocated_work_items;
  std::atomic<int>& next_unprocessed_item;
  std::atomic<bool>& finished;

#ifdef HARDWARE_DECODE
  CUcontext cuda_ctx; // context to use to decode frames
#endif

  // Output arguments
  size_t buffer_size;
  size_t frame_buffers_count;
  char** frame_buffers;
  std::atomic<bool>* buffer_ready_for_eval;
  int* work_item_index;
};

void* load_video_thread(void* arg) {
  LoadVideoArgs& args = *reinterpret_cast<LoadVideoArgs*>(arg);

  // Setup a distinct storage backend for each IO thread
  StorageBackend* storage =
    StorageBackend::make_from_config(args.storage_config);

  int next_buffer = 0;
  while (true) {
    args.item_lock.lock();
    int work_items_left =
      args.allocated_work_items.size() - args.next_unprocessed_item;
    // Check if we are done
    if (args.finished && work_items_left == 0) {
      // Need to check if finished first because there is a race condition if
      // work_items_left == 0 is checked before finished.
      args.item_lock.unlock();
      break;
    } else if (work_items_left == 0) {
      args.item_lock.unlock();
      continue;
    }
    // Process next item
    int next_item = args.next_unprocessed_item++;
    int work_item_index = args.allocated_work_items[next_item];
    VideoWorkItem work_item = args.work_items[work_item_index];
    // Keep locked until this point so we do not have multiple threads try to
    // process another item simultaneously and go over the allocated work
    args.item_lock.unlock();

    const std::string& video_path = args.video_paths[work_item.video_index];
    const VideoMetadata& metadata = args.metadata[work_item.video_index];

    // Open the iframe file to setup keyframe data
    std::string iframe_file_path = iframe_path(video_path);
    std::vector<int> keyframe_positions;
    std::vector<int64_t> keyframe_timestamps;
    {
      RandomReadFile* iframe_file;
      storage->make_random_read_file(iframe_file_path, iframe_file);

      (void)read_keyframe_info(
        iframe_file, 0, keyframe_positions, keyframe_timestamps);

      delete iframe_file;
    }

    // Open the video file for reading
    RandomReadFile* file;
    storage->make_random_read_file(video_path, file);

    VideoDecoder decoder(file, keyframe_positions, keyframe_timestamps);
#ifdef HARDWARE_DECODE
    decoder.set_gpu_context(args.cuda_ctx);
#endif
    decoder.seek(work_item.start_frame);

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                               metadata.width,
                               metadata.height,
                               1);

    // Wait for next buffer to have been consumed by eval thread before
    // overwritting
    while (args.buffer_ready_for_eval[next_buffer]) {
      std::this_thread::yield();
    }

    char* frame_buffer = args.frame_buffers[next_buffer];
    args.work_item_index[next_buffer] = work_item_index;

    SwsContext* sws_context;
    int current_frame = work_item.start_frame;
    while (current_frame < work_item.end_frame) {
      AVFrame* frame = decoder.decode();
      assert(frame != nullptr);

      size_t frames_buffer_offset =
        frame_size * (current_frame - work_item.start_frame);
      assert(frames_buffer_offset < args.buffer_size);
      char* current_frame_buffer_pos =
        frame_buffer + frames_buffer_offset;

      convert_av_frame_to_rgb(sws_context, frame, current_frame_buffer_pos);
    }

    args.buffer_ready_for_eval[next_buffer] = true;
    next_buffer += 1;
    if (next_buffer >= args.frame_buffers_count) {
      next_buffer = 0;
    }

    delete file;
  }

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to run net evaluation
struct EvaluateArgs {
  int gpu_device_id; // for hardware decode, need to know gpu
  const std::vector<VideoMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  std::atomic<int>& num_processed_items;
  std::atomic<bool>& load_finished;
  size_t buffer_size;
  size_t frame_buffers_count;
  char** frame_buffers;
  std::atomic<bool>* buffer_ready_for_eval;
  int* work_item_index;
};

void* evaluate_thread(void* arg) {
  EvaluateArgs& args = *reinterpret_cast<EvaluateArgs*>(arg);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup caffe net
  NetInfo net_info = load_neural_net(NetType::ALEX_NET, args.gpu_device_id);
  caffe::Net<float>* net = net_info.net;

  // Resize net input blob for batch size
  const boost::shared_ptr<caffe::Blob<float>> data_blob{
    net->blob_by_name("data")};
  if (data_blob->shape(0) != global_batch_size) {
    data_blob->Reshape({
        global_batch_size, 3, net_info.input_size, net_info.input_size});
  }

  int dim = net_info.input_size;

  cv::Mat unsized_mean_mat(
    net_info.mean_width, net_info.mean_height, CV_32FC3, net_info.mean_image);
  cv::Mat mean_mat;
  cv::resize(unsized_mean_mat, mean_mat, cv::Size(dim, dim));


  caffe::Blob<float> net_input{global_batch_size, 3, dim, dim};
  // For avoiding transferring over PCIE
  caffe::Blob<float> no_pcie_dummy{global_batch_size, 3, dim, dim};

  int next_buffer = 0;
  while (true) {
    // Wait for buffer to process
    while (!args.buffer_ready_for_eval[next_buffer]) {
      if (args.load_finished) break;
      std::this_thread::yield();
    }
    if (args.load_finished && !args.buffer_ready_for_eval[next_buffer]) break;

    VideoWorkItem work_item =
      args.work_items[args.work_item_index[next_buffer]];
    const VideoMetadata& metadata = args.metadata[work_item.video_index];

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                               metadata.width,
                               metadata.height,
                               1);

    int current_frame = work_item.start_frame;
    while (current_frame + global_batch_size < work_item.end_frame) {
      int frame_offset = current_frame - work_item.start_frame;
      if (frame_offset % 128 == 0) {
        printf("Node %d, GPU %d, frame %d\n",
               rank, args.gpu_device_id, current_frame);
      }
      // Decompress batch of frame

      float* net_input_buffer;
      if (NO_PCIE_TRANSFER) {
        net_input_buffer = no_pcie_dummy.mutable_cpu_data();
        net_input.mutable_gpu_data();
      } else {
        net_input_buffer = net_input.mutable_cpu_data();
      }

      // Process batch of frames
      char* frame_buffer = args.frame_buffers[next_buffer];
      for (int i = 0; i < global_batch_size; ++i) {
        char* buffer = frame_buffer + frame_size * (i + frame_offset);
        cv::Mat input_mat(metadata.height, metadata.width, CV_8UC3, buffer);
        cv::cvtColor(input_mat, input_mat, CV_RGB2BGR);
        cv::Mat conv_input;
        cv::resize(input_mat, conv_input, cv::Size(dim, dim));
        cv::Mat float_conv_input;
        conv_input.convertTo(float_conv_input, CV_32FC3);
        cv::Mat normed_input = float_conv_input - mean_mat;
        //to_conv_input(&std::get<0>(in_vec[i]), &conv_input, &mean);
        memcpy(net_input_buffer + i * (dim * dim * 3),
               normed_input.data,
               dim * dim * 3 * sizeof(float));
      }

      net->Forward({&net_input});

      // Save batch of frames
      current_frame += global_batch_size;
    }

    // Epilogue for processing less than a batch of frames
    if (current_frame < work_item.end_frame) {
      int batch_size = work_item.end_frame - current_frame;

      // Resize for our smaller batch size
      if (data_blob->shape(0) != batch_size) {
        data_blob->Reshape({
            batch_size, 3, net_info.input_size, net_info.input_size});
      }

      int frame_offset = current_frame - work_item.start_frame;

      // Process batch of frames
      caffe::Blob<float> net_input{batch_size, 3, dim, dim};
      caffe::Blob<float> no_pcie_dummy{batch_size, 3, dim, dim};

      float* net_input_buffer;
      if (NO_PCIE_TRANSFER) {
        net_input_buffer = no_pcie_dummy.mutable_cpu_data();
        net_input.mutable_gpu_data();
      } else {
        net_input_buffer = net_input.mutable_cpu_data();
      }

      char* frame_buffer = args.frame_buffers[next_buffer];
      for (int i = 0; i < batch_size; ++i) {
        char* buffer = frame_buffer + frame_size * (i + frame_offset);
        cv::Mat input_mat(metadata.height, metadata.width, CV_8UC3, buffer);
        cv::cvtColor(input_mat, input_mat, CV_RGB2BGR);
        cv::Mat conv_input;
        cv::resize(input_mat, conv_input, cv::Size(dim, dim));
        cv::Mat float_conv_input;
        conv_input.convertTo(float_conv_input, CV_32FC3);
        cv::Mat normed_input = float_conv_input - mean_mat;
        //to_conv_input(&std::get<0>(in_vec[i]), &conv_input, &mean);
        memcpy(net_input_buffer + i * (dim * dim * 3),
               normed_input.data,
               dim * dim * 3 * sizeof(float));
      }

      net->Forward({&net_input});

      // Save batch of frames
      current_frame += batch_size;
    }
    args.buffer_ready_for_eval[next_buffer] = false;
    args.num_processed_items += 1;
    next_buffer++;
    if (next_buffer >= args.frame_buffers_count) {
      next_buffer = 0;
    }
  }
  delete net;

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
  const std::vector<std::string>& video_paths;
  const std::vector<VideoMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  SpinLock& item_lock;
  std::vector<int>& allocated_work_items;
  std::atomic<int>& next_unprocessed_item;
  std::atomic<int>& num_processed_items;
  std::atomic<bool>& finished;
};

void* process_thread(void* arg) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ProcessArgs& args = *reinterpret_cast<ProcessArgs*>(arg);

  // Allocate several buffers to hold the intermediate of an entire work item
  // to allow pipelining of load/eval

  // HACK(apoms): we are assuming that all videos have the same frame size here
  // alternatively, we should allocate the buffer in the load thread instead
  size_t frame_size =
    av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                             args.metadata[0].width,
                             args.metadata[0].height,
                             1);
  size_t frame_buffer_size = frame_size * max_work_item_size();
  const int frame_buffers_count = LOAD_EVAL_BUFFERS;
  char* frame_buffers[frame_buffers_count];
  std::atomic<bool> buffer_ready_for_eval[frame_buffers_count];
  int buffer_work_item_index[frame_buffers_count];
  for (int i = 0; i < frame_buffers_count; ++i) {
    frame_buffers[i] = new char[frame_buffer_size];
    buffer_ready_for_eval[i] = false;
    buffer_work_item_index[i] = -1;
  }

  std::atomic<bool> load_finished{false};

  // Retain primary context to use for decoder
#ifdef HARDWARE_DECODE
  CUcontext cuda_ctx;
  cuDevicePrimaryCtxRetain(&cuda_ctx, args.gpu_device_id);
#endif

  // Create IO thread for reading and decoding data
  LoadVideoArgs load_args = {
    args.gpu_device_id,
    args.storage_config,
    args.video_paths,
    args.metadata,
    args.work_items,

    args.item_lock,
    args.allocated_work_items,
    args.next_unprocessed_item,
    args.finished,

#ifdef HARDWARE_DECODE
    cuda_ctx,
#endif
    frame_buffer_size,
    frame_buffers_count,
    frame_buffers,
    buffer_ready_for_eval,
    buffer_work_item_index
  };
  pthread_t load_thread;
  pthread_create(&load_thread, NULL, load_video_thread, &load_args);

  // Create eval thread for passing data through neural net
  EvaluateArgs eval_args = {
    args.gpu_device_id,
    args.metadata,
    args.work_items,

    args.num_processed_items,
    load_finished,
    frame_buffer_size,
    frame_buffers_count,
    frame_buffers,
    buffer_ready_for_eval,
    buffer_work_item_index
  };
  pthread_t eval_thread;
  pthread_create(&eval_thread, NULL, evaluate_thread, &eval_args);

  // Wait until load has finished
  void* result;
  int err = pthread_join(load_thread, &result);
  if (err != 0) {
    fprintf(stderr, "error in pthread_join of load thread\n");
    exit(EXIT_FAILURE);
  }
  free(result);

  load_finished = true;

  // Wait until eval has finished
  err = pthread_join(eval_thread, &result);
  if (err != 0) {
    fprintf(stderr, "error in pthread_join of eval thread\n");
    exit(EXIT_FAILURE);
  }
  free(result);

  // Cleanup
#ifdef HARDWARE_DECODE
  cuDevicePrimaryCtxRelease(args.gpu_device_id);
#endif

  THREAD_RETURN_SUCCESS();
}

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
  FLAGS_minloglevel = 2;
  av_mutex = PTHREAD_MUTEX_INITIALIZER;
}

void shutdown() {
  MPI_Finalize();
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <gpus_per_node> <batch_size> <video_paths_file>\n",
           argv[0]);
    exit(EXIT_FAILURE);
  }

  int gpus_per_node = atoi(argv[1]);
  global_batch_size = atoi(argv[2]);
  std::string video_paths_file{argv[3]};

  startup(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  // Setup storage config
  StorageConfig* config =
    StorageConfig::make_disk_config(DB_PATH);
  StorageBackend* storage = StorageBackend::make_from_config(config);

  // Read in list of video paths
  std::vector<std::string> video_paths;
  {
    std::fstream fs(video_paths_file, std::fstream::in);
    while (fs) {
      std::string path;
      fs >> path;
      if (path.empty()) continue;
      video_paths.push_back(path);
    }
  }

  // Check if we have already preprocessed the videos
  bool all_preprocessed = true;
  for (const std::string& path : video_paths) {
    FileInfo video_info;
    StoreResult result =
      storage->get_file_info(processed_video_path(path), video_info);
    if (result == StoreResult::FileDoesNotExist) {
      all_preprocessed = false;
      // Preprocess video and then exit
      if (is_master(rank)) {
        log_ls.print("Video %s not processed yet. Processing now...\n",
                     path.c_str());
        preprocess_video(storage,
                         path,
                         processed_video_path(path),
                         metadata_path(path),
                         iframe_path(path));
      }
    }
  }
  if (all_preprocessed) {
    // Get video metadata for all videos for distributing with work items
    std::vector<VideoMetadata> video_metadata;
    for (const std::string& path : video_paths) {
      std::unique_ptr<RandomReadFile> metadata_file;
      exit_on_error(
        make_unique_random_read_file(storage,
                                     metadata_path(path),
                                     metadata_file));
      VideoMetadata metadata;
      (void) read_video_metadata(metadata_file.get(), 0, metadata);
      video_metadata.push_back(metadata);
    }

    // Break up videos and their frames into equal sized work items
    const int WORK_ITEM_SIZE = max_work_item_size();
    std::vector<VideoWorkItem> work_items;
    for (size_t i = 0; i < video_paths.size(); ++i) {
      const VideoMetadata& meta = video_metadata[i];

      int32_t allocated_frames = 0;
      while (allocated_frames < meta.frames) {
        int32_t frames_to_allocate =
          std::min(WORK_ITEM_SIZE, meta.frames - allocated_frames);

        VideoWorkItem item;
        item.video_index = i;
        item.start_frame = allocated_frames;
        item.start_frame = allocated_frames + frames_to_allocate;
        work_items.push_back(item);

        allocated_frames += frames_to_allocate;
      }
    }
    if (is_master(rank)) {
      printf("Total work items: %lu\n", work_items.size());
    }

    // Setup shared resources for distributing work to processing threads
    SpinLock item_lock;
    std::vector<int> allocated_work_items;
    std::atomic<int> next_unprocessed_item{0};
    std::atomic<int> num_processed_items{0};
    std::atomic<bool> finished{false};

    // Create processing threads for each gpu
    std::vector<ProcessArgs> processing_thread_args;
    std::vector<pthread_t> processing_threads(gpus_per_node);

    processing_thread_args.reserve(gpus_per_node);
    for (int i = 0; i < gpus_per_node; ++i) {
      processing_thread_args.emplace_back(ProcessArgs{
        i, // gpu device id
        config, // storage config
        video_paths,
        video_metadata,
        work_items,
        item_lock,
        allocated_work_items,
        next_unprocessed_item,
        num_processed_items,
        finished
      });

      pthread_create(&processing_threads[i],
                     NULL,
                     process_thread,
                     &processing_thread_args[i]);

    }

    // Begin distributing work
    if (is_master(rank)) {
      int next_work_item_to_allocate = 0;
      // Wait for clients to ask for work
      while (next_work_item_to_allocate < static_cast<int>(work_items.size())) {
        // Check if we need to allocate work to our own processing thread
        int items_left = allocated_work_items.size() - num_processed_items;
        if (items_left < gpus_per_node * WORK_SURPLUS_FACTOR) {
          item_lock.lock();
          allocated_work_items.push_back(next_work_item_to_allocate++);
          item_lock.unlock();
          std::this_thread::yield();
          continue;
        }

        if (num_nodes > 1) {
          int more_work;
          MPI_Status status;
          MPI_Recv(&more_work, 1, MPI_INT,
                   MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          int next_item = next_work_item_to_allocate++;
          MPI_Send(&next_item, 1, MPI_INT,
                   status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }
        std::this_thread::yield();
      }
      finished = true;
      int workers_done = 1;
      while (workers_done < num_nodes) {
        int more_work;
        MPI_Status status;
        MPI_Recv(&more_work, 1, MPI_INT,
                 MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int next_item = -1;
        MPI_Send(&next_item, 1, MPI_INT,
                 status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        std::this_thread::yield();
      }
    } else {
      // Monitor amount of work left and request more when running low
      while (true) {
        int items_left = allocated_work_items.size() - num_processed_items;
        if (items_left < gpus_per_node * WORK_SURPLUS_FACTOR) {
          // Request work when there is only a few unprocessed items left
          int more_work = true;
          MPI_Send(&more_work, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
          int next_item;
          MPI_Recv(&next_item, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          if (next_item == -1) {
            // No more work left
            break;
          } else {
            item_lock.lock();
            allocated_work_items.push_back(next_item);
            item_lock.unlock();
          }
        }
        std::this_thread::yield();
      }
      finished = true;
    }

    // Wait till done
    for (int i = 0; i < gpus_per_node; ++i) {
      void* result;

      int err = pthread_join(processing_threads[i], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join\n");
        exit(EXIT_FAILURE);
      }

      printf("Node %d, GPU %d finished; ret=%d\n",
             rank, i, *((int *)result));
      free(result);      /* Free memory allocated by thread */
    }
  }

 // Cleanup
 delete storage;
 delete config;

 shutdown();

 return EXIT_SUCCESS;
}

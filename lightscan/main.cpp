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
#include "lightscan/util/queue.h"
#include "lightscan/util/jpeg/JPEGWriter.h"
#include "lightscan/util/profiler.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "lightscan/util/opencv.h"

#include <cuda.h>
#include "lightscan/util/cuda.h"

#include <thread>
#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <string>
#include <libgen.h>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace lightscan;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
/// Global constants
int GPUS_PER_NODE = 1;           // Number of available GPUs per node
int GLOBAL_BATCH_SIZE = 64;      // Batch size for network
int BATCHES_PER_WORK_ITEM = 4;   // How many batches per work item
int TASKS_IN_QUEUE_PER_GPU = 4;  // How many tasks per GPU to allocate to a node
int LOAD_WORKERS_PER_NODE = 2;   // Number of worker threads loading data
int NUM_CUDA_STREAMS = 32;       // Number of cuda streams for image processing

const std::string DB_PATH = "/Users/abpoms/kcam";
const std::string IFRAME_PATH_POSTFIX = "_iframes";
const std::string METADATA_PATH_POSTFIX = "_metadata";
const std::string PROCESSED_VIDEO_POSTFIX = "_processed";

///////////////////////////////////////////////////////////////////////////////
/// Helper functions

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

inline int frames_per_work_item() {
  return GLOBAL_BATCH_SIZE * BATCHES_PER_WORK_ITEM;
}

template <typename T>
T sum(const std::vector<T>& vec) {
  T result{};
  for (const T& v: vec) {
    result += v;
  }
  return result;
}

template <typename T>
T nano_to_ms(T ns) {
  return ns / 1000000;
}

///////////////////////////////////////////////////////////////////////////////
/// Work structs
struct VideoWorkItem {
  int video_index;
  int start_frame;
  int end_frame;
};

struct LoadWorkEntry {
  int work_item_index;
};

struct DecodeWorkEntry {
  int work_item_index;
  int start_keyframe;
  int end_keyframe;
  size_t encoded_data_size;
  char* buffer;
};

struct DecodeBufferEntry {
  size_t buffer_size;
  char* buffer;
};

struct EvalWorkEntry {
  int work_item_index;
  size_t decoded_frames_size;
  char* buffer;
};

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct LoadThreadArgs {
  // Uniform arguments
  const std::vector<std::string>& video_paths;
  const std::vector<VideoMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<LoadWorkEntry>& load_work;
  Queue<DecodeWorkEntry>& decode_work;
};

struct DecodeThreadArgs {
  // Uniform arguments
  const std::vector<VideoMetadata>& metadata;
  const std::vector<std::vector<char>>& metadata_packets;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  int gpu_device_id;
  CUcontext cuda_context; // context to use to decode frames
  Profiler& profiler;

  // Queues for communicating work
  Queue<DecodeWorkEntry>& decode_work;
  Queue<DecodeBufferEntry>& empty_decode_buffers;
  Queue<EvalWorkEntry>& eval_work;
};

struct EvaluateThreadArgs {
  // Uniform arguments
  const std::vector<VideoMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  int gpu_device_id; // for hardware decode, need to know gpu
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& eval_work;
  Queue<DecodeBufferEntry>& empty_decode_buffers;
};

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
void* load_video_thread(void* arg) {
  LoadThreadArgs& args = *reinterpret_cast<LoadThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a distinct storage backend for each IO thread
  StorageBackend* storage =
    StorageBackend::make_from_config(args.storage_config);

  std::vector<double> task_times;
  std::vector<double> idle_times;

  std::vector<double> io_times;

  std::string last_video_path;
  RandomReadFile* video_file = nullptr;
  uint64_t file_size;
  std::vector<int> keyframe_positions;
  std::vector<int64_t> keyframe_byte_offsets;

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    LoadWorkEntry load_work_entry;
    args.load_work.pop(load_work_entry);

    if (load_work_entry.work_item_index == -1) {
      break;
    }

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
      args.work_items[load_work_entry.work_item_index];

    const std::string& video_path = args.video_paths[work_item.video_index];
    const VideoMetadata& metadata = args.metadata[work_item.video_index];

    if (video_path != last_video_path) {
      if (video_file != nullptr) {
        delete video_file;
        video_file = nullptr;
      }

      keyframe_positions.clear();
      keyframe_byte_offsets.clear();

      // Open the iframe file to setup keyframe data
      std::string iframe_file_path = iframe_path(video_path);
      {
        RandomReadFile* iframe_file;
        storage->make_random_read_file(iframe_file_path, iframe_file);

        (void)read_keyframe_info(
          iframe_file, 0, keyframe_positions, keyframe_byte_offsets);

        delete iframe_file;
      }

      // Open the video file for reading
      storage->make_random_read_file(processed_video_path(video_path),
                                     video_file);

      video_file->get_size(file_size);
    }
    last_video_path = video_path;

    // Read the bytes from the file that correspond to the sequences
    // of frames we are interested in decoding. This sequence will contain
    // the bytes starting at the iframe at or preceding the first frame we are
    // interested and will continue up to the bytes before the iframe at or
    // after the last frame we are interested in.

    size_t start_keyframe_index = 0;
    for (size_t i = 1; i < keyframe_positions.size(); ++i) {
      if (keyframe_positions[i] > work_item.start_frame) {
        start_keyframe_index = i - 1;
        break;
      }
    }

    size_t end_keyframe_index = 0;
    for (size_t i = start_keyframe_index; i < keyframe_positions.size(); ++i) {
      if (keyframe_positions[i] > work_item.end_frame) {
        end_keyframe_index = i;
        break;
      }
    }
    uint64_t start_keyframe_byte_offset =
      static_cast<uint64_t>(keyframe_byte_offsets[start_keyframe_index]);
    uint64_t end_keyframe_byte_offset;
    if (end_keyframe_index == 0) {
      end_keyframe_byte_offset = file_size;
    } else {
      end_keyframe_byte_offset =
        static_cast<uint64_t>(keyframe_byte_offsets[end_keyframe_index]);
    }
    size_t data_size = end_keyframe_byte_offset - start_keyframe_byte_offset;

    char* buffer = new char[data_size];

    auto io_start = now();

    size_t size_read;
    StoreResult result;
    EXP_BACKOFF(
      video_file->read(
        start_keyframe_byte_offset, data_size, buffer, size_read),
      result);
    assert(size_read == data_size);
    assert(result == StoreResult::Success ||
           result == StoreResult::EndOfFile);

    args.profiler.add_interval("io", io_start, now());

    args.profiler.add_interval("task", work_start, now());

    DecodeWorkEntry decode_work_entry;
    decode_work_entry.work_item_index = load_work_entry.work_item_index;
    decode_work_entry.start_keyframe = keyframe_positions[start_keyframe_index];
    decode_work_entry.end_keyframe = keyframe_positions[end_keyframe_index];
    decode_work_entry.encoded_data_size = data_size;
    decode_work_entry.buffer = buffer;
    args.decode_work.push(decode_work_entry);
  }

  printf("(N: %d) Load thread finished.\n",
         rank);

  // Cleanup
  if (video_file != nullptr) {
    delete video_file;
  }
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to decode video
void* decode_thread(void* arg) {
  DecodeThreadArgs& args = *reinterpret_cast<DecodeThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // HACK(apoms): For the metadata that the VideoDecoder cares about (chroma and
  //              codec type) all videos should be the same for now so just use
  //              the first.
  VideoDecoder decoder{
    args.cuda_context,
    args.metadata[0],
    args.metadata_packets[0]};

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    DecodeWorkEntry decode_work_entry;
    args.decode_work.pop(decode_work_entry);

    if (decode_work_entry.work_item_index == -1) {
      break;
    }

    DecodeBufferEntry decode_buffer_entry;
    args.empty_decode_buffers.pop(decode_buffer_entry);

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
      args.work_items[decode_work_entry.work_item_index];
    const VideoMetadata& metadata = args.metadata[work_item.video_index];

    size_t encoded_buffer_size = decode_work_entry.encoded_data_size;
    char* encoded_buffer = decode_work_entry.buffer;

    size_t decoded_buffer_size = decode_buffer_entry.buffer_size;
    char* decoded_buffer = decode_buffer_entry.buffer;

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_NV12,
                               metadata.width,
                               metadata.height,
                               1);

    size_t encoded_buffer_offset = 0;

    int current_frame = work_item.start_frame;
    while (current_frame < work_item.end_frame) {
      auto video_start = now();

      int encoded_packet_size =
        *reinterpret_cast<int*>(encoded_buffer + encoded_buffer_offset);
      encoded_buffer_offset += sizeof(int);
      char* encoded_packet = encoded_buffer + encoded_buffer_offset;
      encoded_buffer_offset += encoded_packet_size;

      size_t frames_buffer_offset =
        frame_size * (current_frame - work_item.start_frame);
      assert(frames_buffer_offset < decoded_buffer_size);
      char* current_frame_buffer_pos =
        decoded_buffer + frames_buffer_offset;

      bool new_frame = decoder.decode(
        encoded_packet,
        encoded_packet_size,
        current_frame_buffer_pos,
        frame_size);

      if (!new_frame) {
        continue;
      }

      // HACK(apoms): NVIDIA GPU decoder only outputs NV12 format so we rely
      //              on that here to copy the data properly
      // auto memcpy_start = now();
      // for (int i = 0; i < 2; i++) {
      //   CU_CHECK(cudaMemcpy2D(
      //              current_frame_buffer_pos + i * metadata.width * metadata.height,
      //              metadata.width, // dst pitch
      //              frame->data[i], // src
      //              frame->linesize[i], // src pitch
      //              frame->width, // width
      //              i == 0 ? frame->height : frame->height / 2, // height
      //              cudaMemcpyDeviceToDevice));
      // }
      // memcpy_time += nano_since(memcpy_start);
      current_frame++;
    }

    // Must clean up buffer allocated by load thread
    delete[] encoded_buffer;

    //decode_times.push_back(decoder.time_spent_on_decode());
    //memcpy_times.push_back(memcpy_time);

    args.profiler.add_interval("task", work_start, now());

    EvalWorkEntry eval_work_entry;
    eval_work_entry.work_item_index = decode_work_entry.work_item_index;
    eval_work_entry.decoded_frames_size = decoded_buffer_size;
    eval_work_entry.buffer = decoded_buffer;
    args.eval_work.push(eval_work_entry);
  }

  printf("(N/GPU: %d/%d) Decode thread finished.\n",
         rank, args.gpu_device_id);
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to run net evaluation
void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CU_CHECK(cudaSetDevice(args.gpu_device_id));
  // Setup caffe net
  NetInfo net_info = load_neural_net(NetType::ALEX_NET, args.gpu_device_id);
  caffe::Net<float>* net = net_info.net;

  int dim = net_info.input_size;

  cv::cuda::setDevice(args.gpu_device_id);

  cv::Mat cpu_mean_mat(
    net_info.mean_width, net_info.mean_height, CV_32FC3, net_info.mean_image);
  cv::cuda::GpuMat unsized_mean_mat(cpu_mean_mat);
  cv::cuda::GpuMat mean_mat;
  cv::cuda::resize(unsized_mean_mat, mean_mat, cv::Size(dim, dim));


  caffe::Blob<float> net_input{GLOBAL_BATCH_SIZE, 3, dim, dim};

  // OpenCV matrices
  std::vector<cv::cuda::Stream> cv_streams(NUM_CUDA_STREAMS);

  std::vector<cv::cuda::GpuMat> input_mats(
    NUM_CUDA_STREAMS,
    cv::cuda::GpuMat(args.metadata[0].height + args.metadata[0].height / 2,
                     args.metadata[0].width,
                     CV_8UC1));

  std::vector<cv::cuda::GpuMat> rgba_mat(
    NUM_CUDA_STREAMS,
    cv::cuda::GpuMat(args.metadata[0].height, args.metadata[0].width, CV_8UC4));

  std::vector<cv::cuda::GpuMat> rgb_mat(
    NUM_CUDA_STREAMS,
    cv::cuda::GpuMat(args.metadata[0].height, args.metadata[0].width, CV_8UC3));

  std::vector<cv::cuda::GpuMat> conv_input(
    NUM_CUDA_STREAMS,
    cv::cuda::GpuMat(dim, dim, CV_8UC4));

  std::vector<cv::cuda::GpuMat> float_conv_input(
    NUM_CUDA_STREAMS,
    cv::cuda::GpuMat(dim, dim, CV_32FC3));

  std::vector<cv::cuda::GpuMat> normed_input(
    NUM_CUDA_STREAMS,
    cv::cuda::GpuMat(dim, dim, CV_32FC3));

  const boost::shared_ptr<caffe::Blob<float>> data_blob{
    net->blob_by_name("data")};

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();
    // Wait for buffer to process
    EvalWorkEntry work_entry;
    args.eval_work.pop(work_entry);

    if (work_entry.work_item_index == -1) {
      break;
    }

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    char* frame_buffer = work_entry.buffer;

    const VideoWorkItem& work_item =
      args.work_items[work_entry.work_item_index];
    const VideoMetadata& metadata = args.metadata[work_item.video_index];

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_NV12,
                               metadata.width,
                               metadata.height,
                               1);

    int current_frame = work_item.start_frame;
    while (current_frame < work_item.end_frame) {
      int frame_offset = current_frame - work_item.start_frame;
      int batch_size =
        std::min(GLOBAL_BATCH_SIZE, work_item.end_frame - current_frame);

      if (data_blob->shape(0) != batch_size) {
        data_blob->Reshape({
            batch_size, 3, dim ,dim});
        net_input.Reshape({
            batch_size, 3, dim, dim});
      }

      float* net_input_buffer = net_input.mutable_gpu_data();

      // Process batch of frames
      auto cv_start = now();
      for (int i = 0; i < batch_size; ++i) {
        int sid = i % NUM_CUDA_STREAMS;
        cv::cuda::Stream& cv_stream = cv_streams[sid];
        char* buffer = frame_buffer + frame_size * (i + frame_offset);
        cv::cuda::GpuMat input_mat(
          metadata.height + metadata.height / 2,
          metadata.width,
          CV_8UC1,
          buffer);

        convertNV12toRGBA(input_mat, rgba_mat[sid],
                          metadata.width, metadata.height,
                          cv_stream);
        cv::cuda::cvtColor(rgba_mat[sid], rgb_mat[sid], CV_RGBA2BGR, 0,
                           cv_stream);
        cv::cuda::resize(rgb_mat[sid], conv_input[sid], cv::Size(dim, dim),
                         0, 0, cv::INTER_LINEAR, cv_stream);
        conv_input[sid].convertTo(float_conv_input[sid], CV_32FC3, cv_stream);
        cv::cuda::subtract(float_conv_input[sid], mean_mat, normed_input[sid],
                           cv::noArray(), -1, cv_stream);
        cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
        CU_CHECK(cudaMemcpyAsync(
                   net_input_buffer + i * (dim * dim * 3),
                   normed_input[sid].data,
                   dim * dim * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice,
                   s));

        if (sid == 0 && i != 0) {
          CU_CHECK(cudaDeviceSynchronize());
        }

        // For checking for proper encoding
        if (false && ((current_frame + i) % 512) == 0) {
          size_t image_size = metadata.width * metadata.height * 3;
          uint8_t* image_buff = new uint8_t[image_size];
          CU_CHECK(cudaMemcpy(image_buff, rgb_mat[sid].data, image_size,
                              cudaMemcpyDeviceToHost));
          JPEGWriter writer;
          writer.header(metadata.width, metadata.height, 3, JPEG::COLOR_RGB);
          std::vector<uint8_t*> rows(metadata.height);
          for (int i = 0; i < metadata.height; ++i) {
            rows[i] = image_buff + metadata.width * 3 * i;
          }
          std::string image_path =
            "frame" + std::to_string(current_frame + i) + ".jpg";
          writer.write(image_path, rows.begin());
          delete[] image_buff;
        }
      }
      CU_CHECK(cudaDeviceSynchronize());
      args.profiler.add_interval("cv", cv_start, now());

      auto net_start = now();
      net->Forward({&net_input});
      args.profiler.add_interval("net", idle_start, now());

      // Save batch of frames
      current_frame += batch_size;
    }
    args.profiler.add_interval("task", work_start, now());

    DecodeBufferEntry empty_buffer_entry;
    empty_buffer_entry.buffer_size = work_entry.decoded_frames_size;
    empty_buffer_entry.buffer = frame_buffer;
    args.empty_decode_buffers.push(empty_buffer_entry);
  }

  delete net;

  printf("(N/GPU: %d/%d) Evaluate thread finished.\n",
         rank, args.gpu_device_id);

  THREAD_RETURN_SUCCESS();
}

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
  FLAGS_minloglevel = 2;
  CUD_CHECK(cuInit(0));
}

void shutdown() {
  MPI_Finalize();
}

int main(int argc, char** argv) {
  std::string video_paths_file;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Produce help message")
      ("video_paths_file", po::value<std::string>()->required(),
       "File which contains paths to video files to process")
      ("gpus_per_node", po::value<int>(), "Number of GPUs per node")
      ("batch_size", po::value<int>(), "Neural Net input batch size")
      ("batches_per_work_item", po::value<int>(),
       "Number of batches in each work item")
      ("tasks_in_queue_per_gpu", po::value<int>(),
       "Number of tasks a node will try to maintain in the work queue per GPU")
      ("load_workers_per_node", po::value<int>(),
       "Number of worker threads processing load jobs per node");
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      if (vm.count("gpus_per_node")) {
        GPUS_PER_NODE = vm["gpus_per_node"].as<int>();
      }
      if (vm.count("batch_size")) {
        GLOBAL_BATCH_SIZE = vm["batch_size"].as<int>();
      }
      if (vm.count("batches_per_work_item")) {
        BATCHES_PER_WORK_ITEM = vm["batches_per_work_item"].as<int>();
      }
      if (vm.count("tasks_in_queue_per_gpu")) {
        TASKS_IN_QUEUE_PER_GPU = vm["tasks_in_queue_per_gpu"].as<int>();
      }
      if (vm.count("load_workers_per_node")) {
        LOAD_WORKERS_PER_NODE = vm["load_workers_per_node"].as<int>();
      }

      video_paths_file = vm["video_paths_file"].as<std::string>();

    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }

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
  std::vector<std::string> bad_paths;
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
        bool valid_video = preprocess_video(storage,
                                            path,
                                            processed_video_path(path),
                                            metadata_path(path),
                                            iframe_path(path));
        if (!valid_video) {
          bad_paths.push_back(path);
        }
      }
    }
  }
  if (!all_preprocessed) {
    if (!bad_paths.empty()) {
      std::fstream bad_paths_file("bad_videos.txt", std::fstream::out);
      for (const std::string& bad_path : bad_paths) {
        bad_paths_file << bad_path << std::endl;
      }
      bad_paths_file.close();
    }
  } else {
    // Get video metadata for all videos for distributing with work items
    std::vector<VideoMetadata> video_metadata(video_paths.size());
    std::vector<std::vector<char>> metadata_packets(video_paths.size());
    for (size_t i = 0; i < video_paths.size(); ++i) {
      const std::string& path = video_paths[i];
      std::unique_ptr<RandomReadFile> metadata_file;
      exit_on_error(
        make_unique_random_read_file(storage,
                                     metadata_path(path),
                                     metadata_file));
      (void) read_video_metadata(metadata_file.get(), 0,
                                 video_metadata[i],
                                 metadata_packets[i]);
    }

    // Break up videos and their frames into equal sized work items
    const int WORK_ITEM_SIZE = frames_per_work_item();
    std::vector<VideoWorkItem> work_items;
    uint32_t total_frames = 0;
    for (size_t i = 0; i < video_paths.size(); ++i) {
      const VideoMetadata& meta = video_metadata[i];

      int32_t allocated_frames = 0;
      while (allocated_frames < meta.frames) {
        int32_t frames_to_allocate =
          std::min(WORK_ITEM_SIZE, meta.frames - allocated_frames);

        VideoWorkItem item;
        item.video_index = i;
        item.start_frame = allocated_frames;
        item.end_frame = allocated_frames + frames_to_allocate;
        work_items.push_back(item);

        allocated_frames += frames_to_allocate;
      }
      total_frames += meta.frames;
    }
    if (is_master(rank)) {
      printf("Total work items: %lu, Total frames: %u\n",
             work_items.size(),
             total_frames);
    }

    // Setup shared resources for distributing work to processing threads
    Queue<LoadWorkEntry> load_work;
    Queue<DecodeWorkEntry> decode_work;
    std::vector<Queue<DecodeBufferEntry>> empty_decode_buffers(GPUS_PER_NODE);
    std::vector<Queue<EvalWorkEntry>> eval_work(GPUS_PER_NODE);

    // Allocate several buffers to hold the intermediate of an entire work item
    // to allow pipelining of load/eval
    // HACK(apoms): we are assuming that all videos have the same frame size
    // We should allocate the buffer in the load thread if we need to support
    // multiple sizes or analyze all the videos an allocate buffers for the
    // largest possible size
    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_NV12,
                               video_metadata[0].width,
                               video_metadata[0].height,
                               1);
    size_t frame_buffer_size = frame_size * frames_per_work_item();
    const int LOAD_BUFFERS = TASKS_IN_QUEUE_PER_GPU;
    char*** gpu_frame_buffers = new char**[GPUS_PER_NODE];
    for (int gpu = 0; gpu < GPUS_PER_NODE; ++gpu) {
      CU_CHECK(cudaSetDevice(gpu));
      gpu_frame_buffers[gpu] = new char*[LOAD_BUFFERS];
      char** frame_buffers = gpu_frame_buffers[gpu];
      for (int i = 0; i < LOAD_BUFFERS; ++i) {
        CU_CHECK(cudaMalloc(&frame_buffers[i], frame_buffer_size));
        // Add the buffer index into the empty buffer queue so workers can
        // fill it to pass to the eval worker
        empty_decode_buffers[gpu].emplace(
          DecodeBufferEntry{frame_buffer_size, frame_buffers[i]});
      }
    }

    // Establish base time to use for profilers
    timepoint_t base_time = now();

    // Setup load workers
    std::vector<Profiler> load_thread_profilers(
      LOAD_WORKERS_PER_NODE,
      Profiler(base_time));
    std::vector<LoadThreadArgs> load_thread_args;
    for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
      // Create IO thread for reading and decoding data
      load_thread_args.emplace_back(LoadThreadArgs{
        // Uniform arguments
        video_paths,
        video_metadata,
        work_items,

        // Per worker arguments
        config,
        load_thread_profilers[i],

        // Queues
        load_work,
        decode_work,
      });
    }
    std::vector<pthread_t> load_threads(LOAD_WORKERS_PER_NODE);
    for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
      pthread_create(&load_threads[i], NULL, load_video_thread,
                     &load_thread_args[i]);
    }

    // Setup load workers
    std::vector<Profiler> decode_thread_profilers(
      GPUS_PER_NODE,
      Profiler(base_time));
    std::vector<DecodeThreadArgs> decode_thread_args;
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      // Retain primary context to use for decoder
      CUcontext cuda_context;
      CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, i));
      // Create IO thread for reading and decoding data
      decode_thread_args.emplace_back(DecodeThreadArgs{
        // Uniform arguments
        video_metadata,
        metadata_packets,
        work_items,

        // Per worker arguments
        i % GPUS_PER_NODE,
        cuda_context,
        decode_thread_profilers[i],

        // Queues
        decode_work,
        empty_decode_buffers[i],
        eval_work[i],
      });
    }
    std::vector<pthread_t> decode_threads(GPUS_PER_NODE);
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      pthread_create(&decode_threads[i], NULL, decode_thread,
                     &decode_thread_args[i]);
    }

    // Setup evaluate workers
    std::vector<Profiler> eval_thread_profilers(
      GPUS_PER_NODE,
      Profiler(base_time));
    std::vector<EvaluateThreadArgs> eval_thread_args;
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      int gpu_device_id = i;

      // Create eval thread for passing data through neural net
      eval_thread_args.emplace_back(EvaluateThreadArgs{
        // Uniform arguments
        video_metadata,
        work_items,

        // Per worker arguments
        gpu_device_id,
        eval_thread_profilers[i],

        // Queues
        eval_work[i],
        empty_decode_buffers[i],
      });
    }
    std::vector<pthread_t> eval_threads(GPUS_PER_NODE);
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      pthread_create(&eval_threads[i], NULL, evaluate_thread,
                     &eval_thread_args[i]);
    }

    // Push work into load queues
    if (is_master(rank)) {
      // Begin distributing work on master node
      int next_work_item_to_allocate = 0;
      // Wait for clients to ask for work
      while (next_work_item_to_allocate < static_cast<int>(work_items.size())) {
        // Check if we need to allocate work to our own processing thread
        int local_work = load_work.size() + decode_work.size();
        for (size_t i = 0; i < eval_work.size(); ++i) {
          local_work += eval_work[i].size();
        }
        if (local_work < GPUS_PER_NODE * TASKS_IN_QUEUE_PER_GPU) {
          LoadWorkEntry entry;
          entry.work_item_index = next_work_item_to_allocate++;
          load_work.push(entry);

          if (next_work_item_to_allocate % 10 == 0) {
            printf("Work items left: %d\n",
                   static_cast<int>(work_items.size()) -
                   next_work_item_to_allocate);
          }
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

          if (next_work_item_to_allocate % 10 == 0) {
            printf("Work items left: %d\n",
                   static_cast<int>(work_items.size()) -
                   next_work_item_to_allocate);
          }
        }
        std::this_thread::yield();
      }
      int workers_done = 1;
      while (workers_done < num_nodes) {
        int more_work;
        MPI_Status status;
        MPI_Recv(&more_work, 1, MPI_INT,
                 MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int next_item = -1;
        MPI_Send(&next_item, 1, MPI_INT,
                 status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        workers_done += 1;
        std::this_thread::yield();
      }
    } else {
      // Monitor amount of work left and request more when running low
      while (true) {
        int local_work = load_work.size() + decode_work.size();
        for (size_t i = 0; i < eval_work.size(); ++i) {
          local_work += eval_work[i].size();
        }
        if (local_work < GPUS_PER_NODE * TASKS_IN_QUEUE_PER_GPU) {
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
            LoadWorkEntry entry;
            entry.work_item_index = next_item;
            load_work.push(entry);
          }
        }
        std::this_thread::yield();
      }
    }

    // Push sentinel work entries into queue to terminate load threads
    for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
      LoadWorkEntry entry;
      entry.work_item_index = -1;
      load_work.push(entry);
    }

    for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
      // Wait until load has finished
      void* result;
      int err = pthread_join(load_threads[i], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join of load thread\n");
        exit(EXIT_FAILURE);
      }
      free(result);

    }
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      DecodeWorkEntry entry;
      entry.work_item_index = -1;
      decode_work.push(entry);
    }

    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      // Wait until eval has finished
      void* result;
      int err = pthread_join(decode_threads[i], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join of decode thread\n");
        exit(EXIT_FAILURE);
      }
      free(result);
    }

    // Cleanup
    for (int gpu = 0; gpu < GPUS_PER_NODE; ++gpu) {
      CUD_CHECK(cuDevicePrimaryCtxRelease(gpu));
    }

    // Push sentinel work entries into queue to terminate eval threads
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      EvalWorkEntry entry;
      entry.work_item_index = -1;
      eval_work[i].push(entry);
    }

    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      // Wait until eval has finished
      void* result;
      int err = pthread_join(eval_threads[i], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join of eval thread\n");
        exit(EXIT_FAILURE);
      }
      free(result);
    }

    // Execution done, write out profiler intervals for each worker
    std::string profiler_file_name =
      "profiler_" + std::to_string(rank) + ".bin";
    std::ofstream profiler_output(profiler_file_name, std::fstream::binary);

    auto write_profiler_to_file = [&profiler_output]
      (int64_t node,
       std::string type_name,
       int64_t worker_num,
       const Profiler& profiler)
    {
      // Write worker header information
      // Node
      profiler_output.write((char*)&node, sizeof(node));
      // Worker type
      profiler_output.write(type_name.c_str(), type_name.size() + 1);
      // Worker number
      profiler_output.write((char*)&worker_num, sizeof(worker_num));
      // Intervals
      const std::vector<lightscan::Profiler::TaskRecord>& records =
        load_thread_profilers[i].get_records();
      // Perform dictionary compression on interval key names
      int64_t record_key_id = 0;
      std::map<std::string, int64_t> key_names;
      for (size_t j = 0; j < records.size(); j++) {
        const std::string& key = records[i].key;
        if (key_names.count(key) == 0) {
          key_names.insert({key, record_key_id++});
        }
      }
      // Write out key name dictionary
      int64_t num_keys = static_cast<int64_t>(key_names.size());
      profiler_output.write(&num_keys, sizeof(num_keys));
      for (auto& kv : key_names) {
        std::string key = kv.first;
        int64_t key_index = kv.second;
        profiler_output.write(key.c_str(), key.size() + 1);
        profiler_output.write((char*)&key_index, sizeof(key_index));
      }
      // Number of intervals
      int64_t num_records = static_cast<int64_t>(records.size());
      profiler_output.write(&num_records, sizeof(num_records));
      for (size_t j = 0; j < records.size(); j++) {
        const lightscan::Profiler::TaskRecord& record = records[j];
        int64_t key_index = key_names[record.key];
        int64_t start = record.start;
        int64_t end = record.end;
        profiler_output.write((char*)&key_index, sizeof(key_index));
        profiler_output.write((char*)&start, sizeof(start));
        profiler_output.write((char*)&end, sizeof(end));
      }
    };

    // Load worker profilers
    int64_t out_rank = rank;
    for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
      write_profiler_to_file(out_rank, "load", i, load_thread_profilers[i]);
    }

    // Decode worker profilers
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      write_profiler_to_file(out_rank, "decode", i, decode_thread_profilers[i]);
    }

    // Evaluate worker profilers
    for (int i = 0; i < GPUS_PER_NODE; ++i) {
      write_profiler_to_file(out_rank, "eval", i, eval_thread_profilers[i]);
    }

    profiler_output.close();

    // Cleanup
    for (int gpu = 0; gpu < GPUS_PER_NODE; ++gpu) {
      char** frame_buffers = gpu_frame_buffers[gpu];
      for (int i = 0; i < LOAD_BUFFERS; ++i) {
        CU_CHECK(cudaSetDevice(gpu));
        CU_CHECK(cudaFree(frame_buffers[i]));
      }
      delete[] frame_buffers;
    }
    delete[] gpu_frame_buffers;
  }

  // Cleanup
  delete storage;
  delete config;

  shutdown();

  return EXIT_SUCCESS;
}

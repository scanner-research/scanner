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

#include "scanner/util/cuda.h"
#include "scanner/util/video.h"
#include "scanner/util/common.h"
#include "scanner/util/caffe.h"
#include "scanner/util/profiler.h"
#include "scanner/util/queue.h"
#include "scanner/util/util.h"
#include "scanner/util/opencv.h"
#include "scanner/util/jpeg/JPEGWriter.h"

#include "storehouse/storage_backend.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

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

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;
using storehouse::exit_on_error;

namespace scanner {

///////////////////////////////////////////////////////////////////////////////
/// Functionality object
class PipelineFunctions {
  virtual void new_buffer(char** buffer, size_t buffer_size) = 0;

  virtual void delete_buffer(char* buffer) = 0;
};

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct LoadThreadArgs {
  // Uniform arguments
  std::string dataset_name;
  const std::vector<std::string>& video_paths;
  const std::vector<DatasetItemMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<LoadWorkEntry>& load_work;
  Queue<DecodeWorkEntry>& decode_work;
};

struct DecodeThreadArgs {
  // Uniform arguments
  const std::vector<DatasetItemMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  int device_id;
  Profiler& profiler;

  // Queues for communicating work
  Queue<DecodeWorkEntry>& decode_work;
  Queue<DecodeBufferEntry>& empty_decode_buffers;
  Queue<EvalWorkEntry>& eval_work;
};

struct EvaluateThreadArgs {
  // Uniform arguments
  const std::vector<DatasetItemMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;
  const NetDescriptor& net_descriptor;

  // Per worker arguments
  int device_id; // for hardware decode, need to know which device
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& eval_work;
  Queue<DecodeBufferEntry>& empty_decode_buffers;
  Queue<SaveWorkEntry>& save_work;
};

struct SaveThreadArgs {
  // Uniform arguments
  std::string job_name;
  const std::vector<std::string>& video_paths;
  const std::vector<DatasetItemMetadata>& metadata;
  const std::vector<std::string>& output_layer_names;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<SaveWorkEntry>& save_work;
};

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
void* load_video_thread(void* arg) {
  LoadThreadArgs& args = *reinterpret_cast<LoadThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
    storehouse::StorageBackend::make_from_config(args.storage_config);

  std::string last_video_path;
  RandomReadFile* video_file = nullptr;
  uint64_t file_size;

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
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    if (video_path != last_video_path) {
      if (video_file != nullptr) {
        delete video_file;
        video_file = nullptr;
      }

      // Open the video file for reading
      storage->make_random_read_file(
        dataset_item_data_path(args.dataset_name, video_path),
        video_file);

      video_file->get_size(file_size);
    }
    last_video_path = video_path;

    // Place end of file and num frame at end of iframe to handle edge case
    std::vector<int64_t> keyframe_positions = metadata.keyframe_positions;
    std::vector<int64_t> keyframe_byte_offsets = metadata.keyframe_byte_offsets;
    keyframe_positions.push_back(metadata.frames);
    keyframe_byte_offsets.push_back(file_size);

    // Read the bytes from the file that correspond to the sequences
    // of frames we are interested in decoding. This sequence will contain
    // the bytes starting at the iframe at or preceding the first frame we are
    // interested and will continue up to the bytes before the iframe at or
    // after the last frame we are interested in.

    size_t start_keyframe_index = std::numeric_limits<size_t>::max();
    for (size_t i = 1; i < keyframe_positions.size(); ++i) {
      if (keyframe_positions[i] > work_item.start_frame) {
        start_keyframe_index = i - 1;
        break;
      }
    }
    assert(start_keyframe_index != std::numeric_limits<size_t>::max());
    uint64_t start_keyframe_byte_offset =
      static_cast<uint64_t>(keyframe_byte_offsets[start_keyframe_index]);

    size_t end_keyframe_index = 0;
    for (size_t i = start_keyframe_index; i < keyframe_positions.size(); ++i) {
      if (keyframe_positions[i] >= work_item.end_frame) {
        end_keyframe_index = i;
        break;
      }
    }
    assert(end_keyframe_index != 0);
    uint64_t end_keyframe_byte_offset =
      static_cast<uint64_t>(keyframe_byte_offsets[end_keyframe_index]);

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

  CU_CHECK(cudaSetDevice(args.device_id));

  // HACK(apoms): For the metadata that the VideoDecoder cares about (chroma and
  //              codec type) all videos should be the same for now so just use
  //              the first.
  VideoDecoder decoder{
    args.cuda_context,
    args.metadata[0]};
  decoder.set_profiler(&args.profiler);

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
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

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

    bool discontinuity = true;
    int current_frame = decode_work_entry.start_keyframe;
    while (current_frame < work_item.end_frame) {
      auto video_start = now();

      int encoded_packet_size = 0;
      char* encoded_packet = nullptr;
      if (encoded_buffer_offset < encoded_buffer_size) {
        encoded_packet_size =
          *reinterpret_cast<int*>(encoded_buffer + encoded_buffer_offset);
        encoded_buffer_offset += sizeof(int);
        encoded_packet = encoded_buffer + encoded_buffer_offset;
        encoded_buffer_offset += encoded_packet_size;
      }

      if (decoder.feed(encoded_packet, encoded_packet_size, discontinuity)) {
        // New frames
        bool more_frames = true;
        while (more_frames && current_frame < work_item.end_frame) {
          if (current_frame >= work_item.start_frame) {
            size_t frames_buffer_offset =
              frame_size * (current_frame - work_item.start_frame);
            assert(frames_buffer_offset < decoded_buffer_size);
            char* current_frame_buffer_pos =
              decoded_buffer + frames_buffer_offset;

            more_frames =
              decoder.get_frame(current_frame_buffer_pos, frame_size);
          } else {
            more_frames = decoder.discard_frame();
          }
          current_frame++;
        }
      }
      discontinuity = false;
    }
    // Wait on all memcpys from frames to be done
    decoder.wait_until_frames_copied();

    if (decoder.decoded_frames_buffered() > 0) {
      while (decoder.discard_frame()) {};
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

  printf("(N/PU: %d/%d) Decode thread finished.\n",
         rank, args.gpu_device_id);

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to run evaluation
void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


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
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_NV12,
                               metadata.width,
                               metadata.height,
                               1);

    // Create output buffer to hold results from net evaluation for all frames
    // in the current work item
    std::vector<size_t> output_buffer_sizes;
    std::vector<char*> output_buffers;
    for (size_t output_size_per_frame : output_sizes) {
      size_t output_buffer_size =
        (work_item.end_frame - work_item.start_frame) * output_size_per_frame;
      output_buffer_sizes.push_back(output_buffer_size);
      output_buffers.push_back(new char[output_buffer_size]);
    }

    int current_frame = work_item.start_frame;
    while (current_frame < work_item.end_frame) {
      int frame_offset = current_frame - work_item.start_frame;
      int batch_size =
        std::min(GLOBAL_BATCH_SIZE, work_item.end_frame - current_frame);

      if (input_blob->shape(0) != batch_size) {
        input_blob->Reshape({batch_size, 3, inputHeight, inputWidth});
      }

      float* net_input_buffer = input_blob->mutable_gpu_data();

      // Process batch of frames
      auto cv_start = now();
      for (int i = 0; i < batch_size; ++i) {
        int sid = i % NUM_CUDA_STREAMS;
        cv::cuda::Stream& cv_stream = cv_streams[sid];

        char* buffer = frame_buffer + frame_size * (i + frame_offset);

        input_mats[sid] =
          cv::cuda::GpuMat(
            metadata.height + metadata.height / 2,
            metadata.width,
            CV_8UC1,
            buffer);

        convertNV12toRGBA(input_mats[sid], rgba_mat[sid],
                          metadata.width, metadata.height,
                          cv_stream);
        // BGR -> RGB for helnet
        cv::cuda::cvtColor(rgba_mat[sid], rgb_mat[sid], CV_BGRA2RGB, 0,
                           cv_stream);
        cv::cuda::resize(rgb_mat[sid], conv_input[sid],
                         cv::Size(inputWidth, inputHeight),
                         0, 0, cv::INTER_LINEAR, cv_stream);
        // Changed from interleaved BGR to planar BGR
        convertRGBInterleavedToPlanar(conv_input[sid], conv_planar_input[sid],
                                      inputWidth, inputHeight,
                                      cv_stream);
        conv_planar_input[sid].convertTo(
          float_conv_input[sid], CV_32FC1, cv_stream);
        cv::cuda::subtract(float_conv_input[sid], mean_mat, normed_input[sid],
                           cv::noArray(), -1, cv_stream);
        // For helnet, we need to transpose so width is fasting moving dim
        // and normalize to 0 - 1
        cv::cuda::divide(normed_input[sid], 255.0f, scaled_input[sid],
                         1, -1, cv_stream);
        cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
        CU_CHECK(cudaMemcpy2DAsync(
                   net_input_buffer + i * (inputWidth * inputHeight * 3),
                   inputWidth * sizeof(float),
                   scaled_input[sid].data,
                   scaled_input[sid].step,
                   inputWidth * sizeof(float),
                   inputHeight * 3,
                   cudaMemcpyDeviceToDevice,
                   s));

        // For checking for proper encoding
        if (false && ((current_frame + i) % 512) == 0) {
          CU_CHECK(cudaDeviceSynchronize());
          size_t image_size = metadata.width * metadata.height * 3;
          uint8_t* image_buff = new uint8_t[image_size];

          for (int i = 0; i < rgb_mat[sid].rows; ++i) {
            CU_CHECK(cudaMemcpy(image_buff + metadata.width * 3 * i,
                                rgb_mat[sid].ptr<uint8_t>(i),
                                metadata.width * 3,
                                cudaMemcpyDeviceToHost));
          }
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

      // Compute features
      auto net_start = now();
      net.Forward();
      args.profiler.add_interval("net", net_start, now());

      // Save batch of frames
      for (size_t i = 0; i < output_buffer_sizes.size(); ++i) {
        const std::string& output_layer_name =
          args.net_descriptor.output_layer_names[i];
        const boost::shared_ptr<caffe::Blob<float>> output_blob{
          net.blob_by_name(output_layer_name)};
        CU_CHECK(cudaMemcpy(
                   output_buffers[i] + frame_offset * output_sizes[i],
                   output_blob->gpu_data(),
                   batch_size * output_sizes[i],
                   cudaMemcpyDeviceToHost));
      }

      current_frame += batch_size;
    }
    args.profiler.add_interval("task", work_start, now());

    DecodeBufferEntry empty_buffer_entry;
    empty_buffer_entry.buffer_size = work_entry.decoded_frames_size;
    empty_buffer_entry.buffer = frame_buffer;
    args.empty_decode_buffers.push(empty_buffer_entry);

    SaveWorkEntry save_work_entry;
    save_work_entry.work_item_index = work_entry.work_item_index;
    save_work_entry.output_buffer_sizes = output_buffer_sizes;
    save_work_entry.output_buffers = output_buffers;
    args.save_work.push(save_work_entry);
  }

  printf("(N/PU: %d/%d) Evaluate thread finished.\n",
         rank, args.gpu_device_id);

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously save result buffers
void* save_thread(void* arg) {
  SaveThreadArgs& args = *reinterpret_cast<SaveThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
    storehouse::StorageBackend::make_from_config(args.storage_config);

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    SaveWorkEntry save_work_entry;
    args.save_work.pop(save_work_entry);

    if (save_work_entry.work_item_index == -1) {
      break;
    }

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
      args.work_items[save_work_entry.work_item_index];

    const std::string& video_path = args.video_paths[work_item.video_index];
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    // Write out each output layer to an individual data file
    for (size_t i = 0; i < args.output_layer_names.size(); ++i) {
      const std::string output_path =
        job_item_output_path(args.job_name,
                             video_path,
                             args.output_layer_names[i],
                             work_item.start_frame,
                             work_item.end_frame);

      size_t buffer_size = save_work_entry.output_buffer_sizes[i];
      char* buffer = save_work_entry.output_buffers[i];

      auto io_start = now();

      WriteFile* output_file = nullptr;
      storage->make_write_file(output_path, output_file);

      StoreResult result;
      EXP_BACKOFF(
        output_file->append(buffer_size, buffer),
        result);
      assert(result == StoreResult::Success ||
             result == StoreResult::EndOfFile);

      output_file->save();

      args.profiler.add_interval("io", io_start, now());

      delete output_file;
      delete[] buffer;
    }

    args.profiler.add_interval("task", work_start, now());
  }

  printf("(N: %d) Save thread finished.\n", rank);

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}

std::vector<NetFrameworkType> get_supported_net_framework_types() {
  std::vector<NetFrameworkType> net_types;
#ifdef HAVE_GPU
  net_types.push_back(NetFrameworkType::GPU);
#endif
  net_types.push_back(NetFrameworkType::CPU);

  return net_types;
}

bool has_net_framework_type(NetFrameworkType type) {
  std::vector<NetFrameworkType> types =
    get_supported_net_framework_types();

  for (const NetFrameworkType& supported_type : types) {
    if (type == supported_type) return true;
  }

  return false;
}

///////////////////////////////////////////////////////////////////////////////
/// run_job

void run_job(
  storehouse::StorageConfig* config,
  VideoDecoderType decoder_type,
  NetFrameworkType net_framework_type,
  const std::string& job_name,
  const std::string& dataset_name,
  const std::string& net_descriptor_file)
{
  storehouse::StorageBackend* storage =
    storehouse::StorageBackend::make_from_config(config);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  // Load the dataset descriptor to find all data files
  DatasetDescriptor descriptor;
  {
    std::unique_ptr<RandomReadFile> file;
    exit_on_error(
      make_unique_random_read_file(storage,
                                   dataset_descriptor_path(dataset_name),
                                   file));
    uint64_t pos = 0;
    descriptor = deserialize_dataset_descriptor(file.get(), pos);
  }

  // Load net descriptor for specifying target network
  NetDescriptor net_descriptor;
  {
    std::ifstream s{net_descriptor_file};
    net_descriptor = descriptor_from_net_file(s);
  }

  // Establish base time to use for profilers
  timepoint_t base_time = now();

  // Get video metadata for all videos for distributing with work items
  std::vector<std::string>& video_paths = descriptor.item_names;

  std::vector<DatasetItemMetadata> video_metadata;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    const std::string& path = video_paths[i];
    std::unique_ptr<RandomReadFile> metadata_file;
    exit_on_error(
      make_unique_random_read_file(
        storage,
        dataset_item_metadata_path(dataset_name, path),
        metadata_file));
    uint64_t pos = 0;
    video_metadata.push_back(
      deserialize_dataset_item_metadata(metadata_file.get(), pos));
  }

  // Break up videos and their frames into equal sized work items
  const int WORK_ITEM_SIZE = frames_per_work_item();
  std::vector<VideoWorkItem> work_items;

  // Track how work was broken up for each video so we can know how the
  // output will be chunked up when saved out
  JobDescriptor job_descriptor;
  job_descriptor.dataset_name = dataset_name;
  uint32_t total_frames = 0;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    const DatasetItemMetadata& meta = video_metadata[i];

    std::vector<std::tuple<int, int>>& work_intervals =
      job_descriptor.intervals[video_paths[i]];
    int32_t allocated_frames = 0;
    while (allocated_frames < meta.frames) {
      int32_t frames_to_allocate =
        std::min(WORK_ITEM_SIZE, meta.frames - allocated_frames);

      VideoWorkItem item;
      item.video_index = i;
      item.start_frame = allocated_frames;
      item.end_frame = allocated_frames + frames_to_allocate;
      work_items.push_back(item);
      work_intervals.emplace_back(item.start_frame, item.end_frame);

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
  std::vector<Queue<DecodeBufferEntry>> empty_decode_buffers(PUS_PER_NODE);
  std::vector<Queue<EvalWorkEntry>> eval_work(PUS_PER_NODE);
  Queue<SaveWorkEntry> save_work;

  // Allocate several buffers to hold the intermediate of an entire work item
  // to allow pipelining of load/eval
  // HACK(apoms): we are assuming that all videos have the same frame size.
  //   We should allocate the buffer in the load thread if we need to support
  //   multiple sizes or analyze all the videos an allocate buffers for the
  //   largest possible size
  size_t frame_size =
    av_image_get_buffer_size(AV_PIX_FMT_NV12,
                             video_metadata[0].width,
                             video_metadata[0].height,
                             1);
  size_t frame_buffer_size = frame_size * frames_per_work_item();
  const int LOAD_BUFFERS = TASKS_IN_QUEUE_PER_PU;
  char*** pu_frame_buffers = new char**[PUS_PER_NODE];
  std::vector<Evaluator*> evaluators;
  for (int pu = 0; pu < PUS_PER_NODE; ++pu) {
    evaluators.push_back(
      evaluator_constructor->new_evaluator(
        pu,
        GLOBAL_BATCH_SIZE,
        LOAD_BUFFERS,
        frame_buffer_size));

    pu_frame_buffers[gpu] = new char*[LOAD_BUFFERS];
    char** frame_buffers = pu_frame_buffers[gpu];
    CU_CHECK(cudaSetDevice(gpu));
    for (int i = 0; i < LOAD_BUFFERS; ++i) {
      CU_CHECK(cudaMalloc(&frame_buffers[i], frame_buffer_size));
      // Add the buffer index into the empty buffer queue so workers can
      // fill it to pass to the eval worker
      empty_decode_buffers[gpu].emplace(
        DecodeBufferEntry{frame_buffer_size, frame_buffers[i]});
    }
  }

  // Setup load workers
  std::vector<Profiler> load_thread_profilers(
    LOAD_WORKERS_PER_NODE,
    Profiler(base_time));
  std::vector<LoadThreadArgs> load_thread_args;
  for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    load_thread_args.emplace_back(LoadThreadArgs{
        // Uniform arguments
          dataset_name,
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

  // Setup decode workers
  std::vector<Profiler> decode_thread_profilers(
    PUS_PER_NODE,
    Profiler(base_time));
  std::vector<DecodeThreadArgs> decode_thread_args;
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    decode_thread_args.emplace_back(DecodeThreadArgs{
        // Uniform arguments
        video_metadata,
          work_items,

          // Per worker arguments
          i % PUS_PER_NODE,
          decode_thread_profilers[i],

          // Queues
          decode_work,
          empty_decode_buffers[i],
          eval_work[i],
          });
  }
  std::vector<pthread_t> decode_threads(PUS_PER_NODE);
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    pthread_create(&decode_threads[i], NULL, decode_thread,
                   &decode_thread_args[i]);
  }

  // Setup evaluate workers
  std::vector<Profiler> eval_thread_profilers(
    PUS_PER_NODE,
    Profiler(base_time));
  std::vector<EvaluateThreadArgs> eval_thread_args;
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    int pu_device_id = i;

    // Create eval thread for passing data through neural net
    eval_thread_args.emplace_back(EvaluateThreadArgs{
        // Uniform arguments
        video_metadata,
          work_items,
          net_descriptor,

          // Per worker arguments
          pu_device_id,
          eval_thread_profilers[i],

          // Queues
          eval_work[i],
          empty_decode_buffers[i],
          save_work
          });
  }
  std::vector<pthread_t> eval_threads(PUS_PER_NODE);
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    pthread_create(&eval_threads[i], NULL, evaluate_thread,
                   &eval_thread_args[i]);
  }

  // Setup save workers
  std::vector<Profiler> save_thread_profilers(
    SAVE_WORKERS_PER_NODE,
    Profiler(base_time));
  std::vector<SaveThreadArgs> save_thread_args;
  for (int i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    save_thread_args.emplace_back(SaveThreadArgs{
        // Uniform arguments
          job_name,
          video_paths,
          video_metadata,
          net_descriptor.output_layer_names,
          work_items,

          // Per worker arguments
          config,
          save_thread_profilers[i],

          // Queues
          save_work,
          });
  }
  std::vector<pthread_t> save_threads(SAVE_WORKERS_PER_NODE);
  for (int i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    pthread_create(&save_threads[i], NULL, save_thread,
                   &save_thread_args[i]);
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
      if (local_work < PUS_PER_NODE * TASKS_IN_QUEUE_PER_PU) {
        LoadWorkEntry entry;
        entry.work_item_index = next_work_item_to_allocate++;
        load_work.push(entry);

        if ((static_cast<int>(work_items.size()) - next_work_item_to_allocate)
            % 10 == 0)
        {
          printf("Work items left: %d\n",
                 static_cast<int>(work_items.size()) -
                 next_work_item_to_allocate);
          fflush(stdout);
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
      if (local_work < PUS_PER_NODE * TASKS_IN_QUEUE_PER_PU) {
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

  // Push sentinel work entries into queue to terminate decode threads
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    DecodeWorkEntry entry;
    entry.work_item_index = -1;
    decode_work.push(entry);
  }

  for (int i = 0; i < PUS_PER_NODE; ++i) {
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
  for (int gpu = 0; gpu < PUS_PER_NODE; ++gpu) {
    CUD_CHECK(cuDevicePrimaryCtxRelease(gpu));
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    EvalWorkEntry entry;
    entry.work_item_index = -1;
    eval_work[i].push(entry);
  }

  for (int i = 0; i < PUS_PER_NODE; ++i) {
    // Wait until eval has finished
    void* result;
    int err = pthread_join(eval_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of eval thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  // Push sentinel work entries into queue to terminate save threads
  for (int i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    SaveWorkEntry entry;
    entry.work_item_index = -1;
    save_work.push(entry);
  }

  for (int i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    // Wait until eval has finished
    void* result;
    int err = pthread_join(save_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of save thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  // Write out metadata to describe where the output results are for each
  // video
  {
    const std::string job_file_path = job_descriptor_path(job_name);
    std::unique_ptr<WriteFile> output_file;
    make_unique_write_file(storage, job_file_path, output_file);

    serialize_job_descriptor(output_file.get(), job_descriptor);

    output_file->save();
  }

  // Execution done, write out profiler intervals for each worker
  std::string profiler_file_name = job_profiler_path(job_name, rank);
  std::ofstream profiler_output(profiler_file_name, std::fstream::binary);

  // Write out total time interval
  timepoint_t end_time = now();

  int64_t start_time_ns =
    std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
    .time_since_epoch()
    .count();
  int64_t end_time_ns =
    std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
    .time_since_epoch()
    .count();
  profiler_output.write((char*)&start_time_ns, sizeof(start_time_ns));
  profiler_output.write((char*)&end_time_ns, sizeof(end_time_ns));

  int64_t out_rank = rank;
  // Load worker profilers
  uint8_t load_worker_count = LOAD_WORKERS_PER_NODE;
  profiler_output.write((char*)&load_worker_count, sizeof(load_worker_count));
  for (int i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(
      profiler_output, out_rank, "load", i, load_thread_profilers[i]);
  }

  // Decode worker profilers
  uint8_t decode_worker_count = PUS_PER_NODE;
  profiler_output.write((char*)&decode_worker_count,
                        sizeof(decode_worker_count));
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    write_profiler_to_file(
      profiler_output, out_rank, "decode", i, decode_thread_profilers[i]);
  }

  // Evaluate worker profilers
  uint8_t eval_worker_count = PUS_PER_NODE;
  profiler_output.write((char*)&eval_worker_count,
                        sizeof(eval_worker_count));
  for (int i = 0; i < PUS_PER_NODE; ++i) {
    write_profiler_to_file(
      profiler_output, out_rank, "eval", i, eval_thread_profilers[i]);
  }

  // Save worker profilers
  uint8_t save_worker_count = SAVE_WORKERS_PER_NODE;
  profiler_output.write((char*)&save_worker_count, sizeof(save_worker_count));
  for (int i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(
      profiler_output, out_rank, "save", i, save_thread_profilers[i]);
  }

  profiler_output.close();

  // Cleanup
  for (int gpu = 0; gpu < PUS_PER_NODE; ++gpu) {
    char** frame_buffers = gpu_frame_buffers[gpu];
    for (int i = 0; i < LOAD_BUFFERS; ++i) {
      CU_CHECK(cudaSetDevice(gpu));
      CU_CHECK(cudaFree(frame_buffers[i]));
    }
    delete[] frame_buffers;
  }
  delete[] gpu_frame_buffers;

  delete storage;
}

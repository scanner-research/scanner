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

#include "scanner/util/common.h"
#include "scanner/util/jpeg/JPEGWriter.h"
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

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "scanner/util/cuda.h"
#endif

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;
using storehouse::exit_on_error;

namespace scanner {
namespace {

u8* new_buffer(DeviceType type, int device_id, size_t size) {
  u8* buffer = nullptr;
  if (type == DeviceType::CPU) {
    buffer = new u8[size];
  }
#ifdef HAVE_CUDA
  else if (type == DeviceType::GPU) {
    CU_CHECK(cudaSetDevice(device_id));
    CU_CHECK(cudaMalloc((void**)&buffer, size));
  }
#endif
  else {
    LOG(FATAL) << "Tried to allocate buffer of unsupported device type";
  }
  return buffer;
}

void delete_buffer(DeviceType type, int device_id, u8* buffer) {
  assert(buffer != nullptr);
  if (type == DeviceType::CPU) {
    delete[] buffer;
  }
#ifdef HAVE_CUDA
  else if (type == DeviceType::GPU) {
    CU_CHECK(cudaSetDevice(device_id));
    CU_CHECK(cudaFree(buffer));
  }
#endif
  else {
    LOG(FATAL) << "Tried to delete buffer of unsupported device type";
  }
  buffer = nullptr;
}

} // end anonymous namespace

///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct LoadThreadArgs {
  // Uniform arguments
  std::string dataset_name;
  const std::vector<std::string>& video_paths;
  const std::vector<DatasetItemMetadata>& metadata;
  const std::vector<VideoWorkItem>& work_items;

  // Per worker arguments
  int id;
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
  int id;
  DeviceType device_type;
  i32 device_id;
  VideoDecoderType decoder_type;
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

  // Per worker arguments
  int id;
  DeviceType device_type;
  i32 device_id;
  EvaluatorFactory& evaluator_factory;
  EvaluatorConfig evaluator_config;
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
  const std::vector<VideoWorkItem>& work_items;
  std::vector<std::string> output_names;

  // Per worker arguments
  int id;
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

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(args.storage_config);

  std::string last_video_path;
  RandomReadFile* video_file = nullptr;
  u64 file_size;

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    LoadWorkEntry load_work_entry;
    args.load_work.pop(load_work_entry);

    if (load_work_entry.work_item_index == -1) {
      break;
    }

    LOG(INFO) << "Load (N/PU: " << rank << "/" << args.id
               << "): processing item " << load_work_entry.work_item_index;

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
          dataset_item_data_path(args.dataset_name, video_path), video_file);

      video_file->get_size(file_size);
    }
    last_video_path = video_path;

    // Place end of file and num frame at end of iframe to handle edge case
    std::vector<i64> keyframe_positions = metadata.keyframe_positions;
    std::vector<i64> keyframe_byte_offsets = metadata.keyframe_byte_offsets;
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
    u64 start_keyframe_byte_offset =
        static_cast<u64>(keyframe_byte_offsets[start_keyframe_index]);

    size_t end_keyframe_index = 0;
    for (size_t i = start_keyframe_index; i < keyframe_positions.size(); ++i) {
      if (keyframe_positions[i] >= work_item.end_frame) {
        end_keyframe_index = i;
        break;
      }
    }
    assert(end_keyframe_index != 0);
    u64 end_keyframe_byte_offset =
        static_cast<u64>(keyframe_byte_offsets[end_keyframe_index]);

    size_t data_size = end_keyframe_byte_offset - start_keyframe_byte_offset;

    u8* buffer = new u8[data_size];

    auto io_start = now();

    u64 pos = start_keyframe_byte_offset;
    read(video_file, buffer, data_size, pos);

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

  LOG(INFO) << "Load (N/PU: " << rank << "/" << args.id
             << "): thread finished";

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

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // HACK(apoms): For the metadata that the VideoDecoder cares about (chroma and
  //              codec type) all videos should be the same for now so just use
  //              the first.
  std::unique_ptr<VideoDecoder> decoder{VideoDecoder::make_from_config(
      args.device_type, args.device_id, args.decoder_type)};
  assert(decoder.get());

  decoder->set_profiler(&args.profiler);

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    DecodeWorkEntry decode_work_entry;
    args.decode_work.pop(decode_work_entry);

    if (decode_work_entry.work_item_index == -1) {
      break;
    }

    LOG(INFO) << "Decode (N/PU: " << rank << "/" << args.id
               << "): processing item " << decode_work_entry.work_item_index;

    DecodeBufferEntry decode_buffer_entry;
    args.empty_decode_buffers.pop(decode_buffer_entry);

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
        args.work_items[decode_work_entry.work_item_index];
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    decoder->configure(metadata);

    size_t encoded_buffer_size = decode_work_entry.encoded_data_size;
    u8* encoded_buffer = decode_work_entry.buffer;

    size_t decoded_buffer_size = decode_buffer_entry.buffer_size;
    u8* decoded_buffer = decode_buffer_entry.buffer;

    size_t frame_size = av_image_get_buffer_size(
        AV_PIX_FMT_RGB24, metadata.width, metadata.height, 1);

    size_t encoded_buffer_offset = 0;

    bool discontinuity = true;
    i32 current_frame = decode_work_entry.start_keyframe;
    while (current_frame < work_item.end_frame) {
      auto video_start = now();

      i32 encoded_packet_size = 0;
      u8* encoded_packet = NULL;
      if (encoded_buffer_offset < encoded_buffer_size) {
        encoded_packet_size =
            *reinterpret_cast<i32*>(encoded_buffer + encoded_buffer_offset);
        encoded_buffer_offset += sizeof(i32);
        encoded_packet = encoded_buffer + encoded_buffer_offset;
        encoded_buffer_offset += encoded_packet_size;
      }

      if (decoder->feed(encoded_packet, encoded_packet_size, discontinuity)) {
        // New frames
        bool more_frames = true;
        while (more_frames && current_frame < work_item.end_frame) {
          if (current_frame >= work_item.start_frame) {
            size_t frames_buffer_offset =
                frame_size * (current_frame - work_item.start_frame);
            assert(frames_buffer_offset < decoded_buffer_size);
            u8* current_frame_buffer_pos =
                decoded_buffer + frames_buffer_offset;

            more_frames =
                decoder->get_frame(current_frame_buffer_pos, frame_size);
          } else {
            more_frames = decoder->discard_frame();
          }
          current_frame++;
        }
      }
      discontinuity = false;
    }
    // Wait on all memcpys from frames to be done
    decoder->wait_until_frames_copied();

    if (decoder->decoded_frames_buffered() > 0) {
      while (decoder->discard_frame()) {
      };
    }

    // Must clean up buffer allocated by load thread
    delete[] encoded_buffer;

    args.profiler.add_interval("task", work_start, now());

    LOG(INFO) << "Decode (N/PU: " << rank << "/" << args.id
               << "): finished item " << decode_work_entry.work_item_index;

    EvalWorkEntry eval_work_entry;
    eval_work_entry.work_item_index = decode_work_entry.work_item_index;
    eval_work_entry.decoded_frames_size = decoded_buffer_size;
    eval_work_entry.buffer = decoded_buffer;
    args.eval_work.push(eval_work_entry);
  }

  LOG(INFO) << "Decode (N/PU: " << rank << "/" << args.id
             << "): thread finished";

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to run evaluation
void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  EvaluatorCapabilities caps = args.evaluator_factory.get_capabilities();

  std::unique_ptr<Evaluator> evaluator{
      args.evaluator_factory.new_evaluator(args.evaluator_config)};
  i32 num_outputs = args.evaluator_factory.get_number_of_outputs();

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();
    // Wait for buffer to process
    EvalWorkEntry work_entry;
    args.eval_work.pop(work_entry);

    if (work_entry.work_item_index == -1) {
      break;
    }

    LOG(INFO) << "Evaluate (N/PU: " << rank << "/" << args.id
               << "): processing item " << work_entry.work_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
        args.work_items[work_entry.work_item_index];
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    // Make the evaluator aware of the format of the data we are about to
    // feed it
    evaluator->configure(metadata);

    size_t frame_size = metadata.width * metadata.height * 3 * sizeof(u8);

    // Create output buffer to hold results from net evaluation for all frames
    // in the current work item
    i32 num_inputs = work_item.end_frame - work_item.start_frame;

    SaveWorkEntry save_work_entry;
    save_work_entry.buffer_type = args.device_type;
    save_work_entry.buffer_device_id = args.device_id;
    save_work_entry.work_item_index = work_entry.work_item_index;

    std::vector<std::vector<size_t>>& work_item_output_sizes =
        save_work_entry.output_buffer_sizes;
    std::vector<std::vector<u8*>>& work_item_output_buffers =
        save_work_entry.output_buffers;
    work_item_output_sizes.resize(num_outputs);
    work_item_output_buffers.resize(num_outputs);

    i32 current_frame = work_item.start_frame;
    while (current_frame < work_item.end_frame) {
      i32 frame_offset = current_frame - work_item.start_frame;
      i32 batch_size =
          std::min(GLOBAL_BATCH_SIZE, work_item.end_frame - current_frame);

      u8* frame_buffer = work_entry.buffer + frame_size * frame_offset;

      std::vector<std::vector<u8*>> output_buffers(num_outputs);
      std::vector<std::vector<size_t>> output_sizes(num_outputs);
      evaluator->evaluate(batch_size, frame_buffer, output_buffers,
                          output_sizes);
      for (i32 i = 0; i < num_outputs; ++i) {
        assert(output_sizes[i].size() == output_buffers[i].size());
        work_item_output_sizes[i].insert(work_item_output_sizes[i].end(),
                                         output_sizes[i].begin(),
                                         output_sizes[i].end());
        work_item_output_buffers[i].insert(work_item_output_buffers[i].end(),
                                           output_buffers[i].begin(),
                                           output_buffers[i].end());
      }

      current_frame += batch_size;
    }

    args.profiler.add_interval("task", work_start, now());

    LOG(INFO) << "Evaluate (N/PU: " << rank << "/" << args.id
               << "): finished item " << work_entry.work_item_index;

    DecodeBufferEntry empty_buffer_entry;
    empty_buffer_entry.buffer_size = work_entry.decoded_frames_size;
    empty_buffer_entry.buffer = work_entry.buffer;
    args.empty_decode_buffers.push(empty_buffer_entry);

    args.save_work.push(save_work_entry);
  }

  LOG(INFO) << "Evaluate (N/PU: " << rank << "/" << args.id
             << "): thread finished";

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously save result buffers
void* save_thread(void* arg) {
  SaveThreadArgs& args = *reinterpret_cast<SaveThreadArgs*>(arg);

  auto setup_start = now();

  i32 rank;
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

    LOG(INFO) << "Save (N/PU: " << rank << "/" << args.id
               << "): processing item " << save_work_entry.work_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
        args.work_items[save_work_entry.work_item_index];

    const std::string& video_path = args.video_paths[work_item.video_index];
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    // HACK(apoms): debugging
    if (false) {
      u8* frame_buffer =
          reinterpret_cast<u8*>(save_work_entry.output_buffers[0][0]);

      JPEGWriter writer;
      writer.header(metadata.width, metadata.height, 3, JPEG::COLOR_RGB);
      std::vector<u8*> rows(metadata.height);
      for (i32 i = 0; i < metadata.height; ++i) {
        rows[i] = frame_buffer + metadata.width * 3 * i;
      }
      std::string image_path =
          "frame" + std::to_string(work_item.start_frame) + ".jpg";
      writer.write(image_path, rows.begin());
    }

    // Write out each output layer to an individual data file
    size_t num_frames = work_item.end_frame - work_item.start_frame;
    for (size_t out_idx = 0; out_idx < args.output_names.size(); ++out_idx) {
      const std::string output_path = job_item_output_path(
          args.job_name, video_path, args.output_names[out_idx],
          work_item.start_frame, work_item.end_frame);

      auto io_start = now();

      WriteFile* output_file = nullptr;
      {
        StoreResult result;
        EXP_BACKOFF(storage->make_write_file(output_path, output_file), result);
        exit_on_error(result);
      }

      assert(save_work_entry.output_buffers[out_idx].size() == num_frames);
      // Write out all output sizes first so we can easily index into the file
      for (size_t i = 0; i < num_frames; ++i) {
        i64 buffer_size = save_work_entry.output_buffer_sizes[out_idx][i];
        write(output_file, buffer_size);
      }
      // Write actual output data
      for (size_t i = 0; i < num_frames; ++i) {
        i64 buffer_size = save_work_entry.output_buffer_sizes[out_idx][i];
        u8* buffer = save_work_entry.output_buffers[out_idx][i];
        write(output_file, buffer, buffer_size);
      }

      output_file->save();

      // TODO(apoms): For now, all evaluators are expected to return CPU
      //   buffers as output so just assume CPU
      for (size_t i = 0; i < num_frames; ++i) {
        delete_buffer(DeviceType::CPU, // save_work_entry.buffer_type,
                      save_work_entry.buffer_device_id,
                      save_work_entry.output_buffers[out_idx][i]);
      }

      delete output_file;

      args.profiler.add_interval("io", io_start, now());
    }

    LOG(INFO) << "Save (N/PU: " << rank << "/" << args.id
               << "): finished item " << save_work_entry.work_item_index;

    args.profiler.add_interval("task", work_start, now());
  }

  LOG(INFO) << "Save (N/PU: " << rank << "/" << args.id
             << "): thread finished ";

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// run_job
void run_job(storehouse::StorageConfig* config, VideoDecoderType decoder_type,
             EvaluatorFactory* evaluator_factory, const std::string& job_name,
             const std::string& dataset_name) {
  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(config);

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  i32 num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  // Load the dataset descriptor to find all data files
  DatasetDescriptor descriptor;
  {
    std::unique_ptr<RandomReadFile> file;
    exit_on_error(make_unique_random_read_file(
        storage, dataset_descriptor_path(dataset_name), file));
    u64 pos = 0;
    descriptor = deserialize_dataset_descriptor(file.get(), pos);
  }

  // Establish base time to use for profilers
  timepoint_t base_time = now();

  // Get video metadata for all videos for distributing with work items
  std::vector<std::string>& video_paths = descriptor.item_names;

  std::vector<DatasetItemMetadata> video_metadata;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    const std::string& path = video_paths[i];
    std::unique_ptr<RandomReadFile> metadata_file;
    exit_on_error(make_unique_random_read_file(
        storage, dataset_item_metadata_path(dataset_name, path),
        metadata_file));
    u64 pos = 0;
    video_metadata.push_back(
        deserialize_dataset_item_metadata(metadata_file.get(), pos));
  }

  // Break up videos and their frames into equal sized work items
  const i32 WORK_ITEM_SIZE = frames_per_work_item();
  std::vector<VideoWorkItem> work_items;

  // Track how work was broken up for each video so we can know how the
  // output will be chunked up when saved out
  JobDescriptor job_descriptor;
  job_descriptor.dataset_name = dataset_name;
  u32 total_frames = 0;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    const DatasetItemMetadata& meta = video_metadata[i];

    std::vector<std::tuple<i32, i32>>& work_intervals =
        job_descriptor.intervals[video_paths[i]];
    i32 allocated_frames = 0;
    while (allocated_frames < meta.frames) {
      i32 frames_to_allocate =
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
    printf("Total work items: %lu, Total frames: %u\n", work_items.size(),
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
  EvaluatorCapabilities caps = evaluator_factory->get_capabilities();

  size_t frame_size =
      descriptor.max_width * descriptor.max_height * 3 * sizeof(u8);
  const int LOAD_BUFFERS = TASKS_IN_QUEUE_PER_PU;
  std::vector<std::vector<u8*>> staging_buffers(PUS_PER_NODE);
  size_t buffer_size = frame_size * frames_per_work_item();
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    int device_id = pu;
    for (i32 i = 0; i < LOAD_BUFFERS; ++i) {
      u8* staging_buffer = new_buffer(caps.device_type, device_id, buffer_size);
      staging_buffers[pu].push_back(staging_buffer);
      empty_decode_buffers[pu].emplace(
          DecodeBufferEntry{buffer_size, staging_buffer});
    }
  }

  // Setup load workers
  std::vector<Profiler> load_thread_profilers(LOAD_WORKERS_PER_NODE,
                                              Profiler(base_time));
  std::vector<LoadThreadArgs> load_thread_args;
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    load_thread_args.emplace_back(LoadThreadArgs{
        // Uniform arguments
        dataset_name, video_paths, video_metadata, work_items,

        // Per worker arguments
        i, config, load_thread_profilers[i],

        // Queues
        load_work, decode_work,
    });
  }
  std::vector<pthread_t> load_threads(LOAD_WORKERS_PER_NODE);
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    pthread_create(&load_threads[i], NULL, load_video_thread,
                   &load_thread_args[i]);
  }

  // Setup decode workers
  std::vector<Profiler> decode_thread_profilers(PUS_PER_NODE,
                                                Profiler(base_time));
  std::vector<DecodeThreadArgs> decode_thread_args;
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    decode_thread_args.emplace_back(DecodeThreadArgs{
        // Uniform arguments
        video_metadata, work_items,

        // Per worker arguments
        i, caps.device_type, i % PUS_PER_NODE, decoder_type,
        decode_thread_profilers[i],

        // Queues
        decode_work, empty_decode_buffers[i], eval_work[i],
    });
  }
  std::vector<pthread_t> decode_threads(PUS_PER_NODE);
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    pthread_create(&decode_threads[i], NULL, decode_thread,
                   &decode_thread_args[i]);
  }

  // Setup evaluate workers
  std::vector<Profiler> eval_thread_profilers(PUS_PER_NODE,
                                              Profiler(base_time));
  std::vector<EvaluateThreadArgs> eval_thread_args;

  EvaluatorConfig eval_config;
  eval_config.max_input_count = frames_per_work_item();
  eval_config.max_frame_width = descriptor.max_width;
  eval_config.max_frame_height = descriptor.max_height;
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    eval_config.device_ids = {i};
    // Create eval thread for passing data through neural net
    eval_thread_args.emplace_back(EvaluateThreadArgs{
        // Uniform arguments
        video_metadata, work_items,

        // Per worker arguments
        i, caps.device_type, i, *evaluator_factory, eval_config,
        eval_thread_profilers[i],

        // Queues
        eval_work[i], empty_decode_buffers[i], save_work});
  }
  std::vector<pthread_t> eval_threads(PUS_PER_NODE);
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    pthread_create(&eval_threads[i], NULL, evaluate_thread,
                   &eval_thread_args[i]);
  }

  // Setup save workers
  std::vector<Profiler> save_thread_profilers(SAVE_WORKERS_PER_NODE,
                                              Profiler(base_time));
  std::vector<SaveThreadArgs> save_thread_args;
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    save_thread_args.emplace_back(SaveThreadArgs{
        // Uniform arguments
        job_name, video_paths, video_metadata, work_items,
        evaluator_factory->get_output_names(),

        // Per worker arguments
        i, config, save_thread_profilers[i],

        // Queues
        save_work,
    });
  }
  std::vector<pthread_t> save_threads(SAVE_WORKERS_PER_NODE);
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    pthread_create(&save_threads[i], NULL, save_thread, &save_thread_args[i]);
  }

  // Push work into load queues
  if (is_master(rank)) {
    // Begin distributing work on master node
    i32 next_work_item_to_allocate = 0;
    // Wait for clients to ask for work
    while (next_work_item_to_allocate < static_cast<i32>(work_items.size())) {
      // Check if we need to allocate work to our own processing thread
      i32 local_work = load_work.size() + decode_work.size();
      for (size_t i = 0; i < eval_work.size(); ++i) {
        local_work += eval_work[i].size();
      }
      if (local_work < PUS_PER_NODE * TASKS_IN_QUEUE_PER_PU) {
        LoadWorkEntry entry;
        entry.work_item_index = next_work_item_to_allocate++;
        load_work.push(entry);

        if ((static_cast<i32>(work_items.size()) - next_work_item_to_allocate) %
            10 == 0) {
          printf("Work items left: %d\n", static_cast<i32>(work_items.size()) -
                                              next_work_item_to_allocate);
          fflush(stdout);
        }
        continue;
      }

      if (num_nodes > 1) {
        i32 more_work;
        MPI_Status status;
        MPI_Recv(&more_work, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        i32 next_item = next_work_item_to_allocate++;
        MPI_Send(&next_item, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);

        if (next_work_item_to_allocate % 10 == 0) {
          printf("Work items left: %d\n", static_cast<i32>(work_items.size()) -
                                              next_work_item_to_allocate);
        }
      }
      std::this_thread::yield();
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
      i32 local_work = load_work.size() + decode_work.size();
      for (size_t i = 0; i < eval_work.size(); ++i) {
        local_work += eval_work[i].size();
      }
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
          LoadWorkEntry entry;
          entry.work_item_index = next_item;
          load_work.push(entry);
        }
      }
      std::this_thread::yield();
    }
  }

  // Push sentinel work entries into queue to terminate load threads
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    LoadWorkEntry entry;
    entry.work_item_index = -1;
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

  // Push sentinel work entries into queue to terminate decode threads
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    DecodeWorkEntry entry;
    entry.work_item_index = -1;
    decode_work.push(entry);
  }

  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    // Wait until eval has finished
    void* result;
    i32 err = pthread_join(decode_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of decode thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    EvalWorkEntry entry;
    entry.work_item_index = -1;
    eval_work[i].push(entry);
  }

  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    // Wait until eval has finished
    void* result;
    i32 err = pthread_join(eval_threads[i], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of eval thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    SaveWorkEntry entry;
    entry.work_item_index = -1;
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
    // Write out metadata to describe where the output results are for each
    // video
    {
      const std::string job_file_path = job_descriptor_path(job_name);
      std::unique_ptr<WriteFile> output_file;
      make_unique_write_file(storage, job_file_path, output_file);

      serialize_job_descriptor(output_file.get(), job_descriptor);

      output_file->save();
    }

    // Add job name into database metadata so we can look up what jobs have been
    // ran
    {
      const std::string db_meta_path = database_metadata_path();

      std::unique_ptr<RandomReadFile> meta_in_file;
      make_unique_random_read_file(storage, db_meta_path, meta_in_file);
      u64 pos = 0;
      DatabaseMetadata meta =
          deserialize_database_metadata(meta_in_file.get(), pos);

      size_t dataset_id = 0;
      for (size_t i = 0; i < meta.dataset_names.size(); ++i) {
        if (meta.dataset_names[i] == dataset_name) {
          dataset_id = i;
          break;
        }
      }
      meta.dataset_job_names[dataset_id].push_back(dataset_name);

      std::unique_ptr<WriteFile> meta_out_file;
      make_unique_write_file(storage, db_meta_path, meta_out_file);
      serialize_database_metadata(meta_out_file.get(), meta);
    }
  }

  // Execution done, write out profiler intervals for each worker
  std::string profiler_file_name = job_profiler_path(job_name, rank);
  std::ofstream profiler_output(profiler_file_name, std::fstream::binary);

  // Write out total time interval
  timepoint_t end_time = now();

  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
          .time_since_epoch()
          .count();
  profiler_output.write((char*)&start_time_ns, sizeof(start_time_ns));
  profiler_output.write((char*)&end_time_ns, sizeof(end_time_ns));

  i64 out_rank = rank;
  // Load worker profilers
  u8 load_worker_count = LOAD_WORKERS_PER_NODE;
  profiler_output.write((char*)&load_worker_count, sizeof(load_worker_count));
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output, out_rank, "load", i,
                           load_thread_profilers[i]);
  }

  // Decode worker profilers
  u8 decode_worker_count = PUS_PER_NODE;
  profiler_output.write((char*)&decode_worker_count,
                        sizeof(decode_worker_count));
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output, out_rank, "decode", i,
                           decode_thread_profilers[i]);
  }

  // Evaluate worker profilers
  u8 eval_worker_count = PUS_PER_NODE;
  profiler_output.write((char*)&eval_worker_count, sizeof(eval_worker_count));
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output, out_rank, "eval", i,
                           eval_thread_profilers[i]);
  }

  // Save worker profilers
  u8 save_worker_count = SAVE_WORKERS_PER_NODE;
  profiler_output.write((char*)&save_worker_count, sizeof(save_worker_count));
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output, out_rank, "save", i,
                           save_thread_profilers[i]);
  }

  profiler_output.close();

  // Cleanup the input buffers for the evaluators
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    std::vector<u8*> frame_buffers = staging_buffers[pu];
    for (u8* buffer : frame_buffers) {
      delete_buffer(caps.device_type, pu, buffer);
    }
  }

  delete storage;
}
}

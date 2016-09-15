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
#include "scanner/util/profiler.h"
#include "scanner/util/queue.h"
#include "scanner/util/storehouse.h"
#include "scanner/util/jpeg/JPEGWriter.h"
#include "scanner/util/util.h"

#include "storehouse/storage_backend.h"

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
  DeviceType device_type;
  int device_id;
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
  DeviceType device_type;
  int device_id; 
  EvaluatorConstructor& evaluator_constructor;
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

    uint64_t pos = start_keyframe_byte_offset;
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
  std::unique_ptr<VideoDecoder> decoder{
    VideoDecoder::make_from_config(
      args.device_type,
      args.device_id,
      args.decoder_type)};
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

    DecodeBufferEntry decode_buffer_entry;
    args.empty_decode_buffers.pop(decode_buffer_entry);

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const VideoWorkItem& work_item =
      args.work_items[decode_work_entry.work_item_index];
    const DatasetItemMetadata& metadata = args.metadata[work_item.video_index];

    decoder->configure(metadata);

    size_t encoded_buffer_size = decode_work_entry.encoded_data_size;
    char* encoded_buffer = decode_work_entry.buffer;

    size_t decoded_buffer_size = decode_buffer_entry.buffer_size;
    char* decoded_buffer = decode_buffer_entry.buffer;

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_RGB24,
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

      if (decoder->feed(encoded_packet, encoded_packet_size, discontinuity)) {
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
      while (decoder->discard_frame()) {};
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
         rank, args.device_id);

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to run evaluation
void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::unique_ptr<Evaluator> evaluator{
    args.evaluator_constructor.new_evaluator(args.evaluator_config)};
  int num_outputs = args.evaluator_constructor.get_number_of_outputs();

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

    // Make the evaluator aware of the format of the data we are about to
    // feed it
    evaluator->configure(metadata);

    size_t frame_size =
      av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                               metadata.width,
                               metadata.height,
                               1);

    // Create output buffer to hold results from net evaluation for all frames
    // in the current work item
    int num_inputs = work_item.end_frame - work_item.start_frame;

    SaveWorkEntry save_work_entry;
    save_work_entry.work_item_index = work_entry.work_item_index;

    std::vector<std::vector<size_t>>& work_item_output_sizes =
      save_work_entry.output_buffer_sizes;
    std::vector<std::vector<char*>>& work_item_output_buffers =
      save_work_entry.output_buffers;
    work_item_output_sizes.resize(num_outputs);
    work_item_output_buffers.resize(num_outputs);

    int current_frame = work_item.start_frame;
    while (current_frame < work_item.end_frame) {
      int frame_offset = current_frame - work_item.start_frame;
      int batch_size =
        std::min(GLOBAL_BATCH_SIZE, work_item.end_frame - current_frame);

      char* frame_buffer = work_entry.buffer + frame_size * frame_offset;

      std::vector<std::vector<char*>> output_buffers(num_outputs);
      std::vector<std::vector<size_t>> output_sizes(num_outputs);

      evaluator->evaluate(
        frame_buffer,
        output_buffers,
        output_sizes,
        batch_size);
      for (int i = 0; i < num_outputs; ++i) {
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

    DecodeBufferEntry empty_buffer_entry;
    empty_buffer_entry.buffer_size = work_entry.decoded_frames_size;
    empty_buffer_entry.buffer = frame_buffer;
    args.empty_decode_buffers.push(empty_buffer_entry);

    args.save_work.push(save_work_entry);
  }

  printf("(N/PU: %d/%d) Evaluate thread finished.\n",
         rank, args.device_id);

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

    // HACK(apoms): debugging
    if (true) {
      uint8_t* frame_buffer =
        reinterpret_cast<uint8_t*>(save_work_entry.output_buffers[0][0]);

      JPEGWriter writer;
      writer.header(metadata.width, metadata.height, 3, JPEG::COLOR_RGB);
      std::vector<uint8_t*> rows(metadata.height);
      for (int i = 0; i < metadata.height; ++i) {
        rows[i] = frame_buffer + metadata.width * 3 * i;
      }
      std::string image_path =
        "frame" + std::to_string(work_item.start_frame) + ".jpg";
      writer.write(image_path, rows.begin());
    }

    // Write out each output layer to an individual data file
    size_t num_outputs = work_item.end_frame - work_item.start_frame;
    for (size_t i = 0; i < args.output_names.size(); ++i) {
      const std::string output_path =
        job_item_output_path(args.job_name,
                             video_path,
                             args.output_names[i],
                             work_item.start_frame,
                             work_item.end_frame);

        auto io_start = now();

        WriteFile* output_file = nullptr;
        {
          StoreResult result;
          EXP_BACKOFF(storage->make_write_file(output_path, output_file),
                      result);
          exit_on_error(result);
        }

        assert(save_work_entry.output_buffers[i].size() == num_outputs);
        for (size_t output_idx = 0; output_idx < num_outputs; ++output_idx) {
          int64_t buffer_size =
            save_work_entry.output_buffer_sizes[i][output_idx];
          char* buffer =
            save_work_entry.output_buffers[i][output_idx];

          write(output_file, buffer_size);
          write(output_file, buffer, buffer_size);
        }

        output_file->save();

        delete output_file;

        args.profiler.add_interval("io", io_start, now());
    }
    // TODO(apoms): Use evaluator constructor to delete buffers

    args.profiler.add_interval("task", work_start, now());
  }

  printf("(N: %d) Save thread finished.\n", rank);

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// run_job
void run_job(
  storehouse::StorageConfig* config,
  VideoDecoderType decoder_type,
  EvaluatorConstructor* evaluator_constructor,
  const std::string& job_name,
  const std::string& dataset_name)
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
  
  size_t frame_size = av_image_get_buffer_size(
    AV_PIX_FMT_RGB24,
    descriptor.max_width,
    descriptor.max_height,
    1);
  size_t frame_buffer_size = frame_size * frames_per_work_item();
  const int LOAD_BUFFERS = TASKS_IN_QUEUE_PER_PU;
  std::vector<std::vector<char*>> staging_buffers(PUS_PER_NODE);

  EvaluatorConfig eval_config;
  eval_config.max_batch_size = frames_per_work_item();
  eval_config.staging_buffer_size = frame_buffer_size;
  eval_config.max_frame_width = descriptor.max_width;
  eval_config.max_frame_height = descriptor.max_height;
  for (int pu = 0; pu < PUS_PER_NODE; ++pu) {
    eval_config.device_id = pu;

    for (int i = 0; i < LOAD_BUFFERS; ++i) {
      char* staging_buffer =
        evaluator_constructor->new_input_buffer(eval_config);
      staging_buffers[pu].push_back(staging_buffer);
      empty_decode_buffers[pu].emplace(
        DecodeBufferEntry{frame_buffer_size, staging_buffer});
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
          evaluator_constructor->get_input_buffer_type(),
          i % PUS_PER_NODE,
          decoder_type,
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

    eval_config.device_id = pu_device_id;
    // Create eval thread for passing data through neural net
    eval_thread_args.emplace_back(EvaluateThreadArgs{
        // Uniform arguments
        video_metadata,
          work_items,

          // Per worker arguments
          evaluator_constructor->get_input_buffer_type(),
          pu_device_id,
          *evaluator_constructor,
          eval_config,
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
          work_items,
          evaluator_constructor->get_output_names(),

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

  // Cleanup the input buffers for the evaluators
  for (int pu = 0; pu < PUS_PER_NODE; ++pu) {
    eval_config.device_id = pu;

    std::vector<char*> frame_buffers = staging_buffers[pu];
    for (char* buffer : frame_buffers) {
      evaluator_constructor->delete_input_buffer(eval_config, buffer);
    }
  }

  delete storage;
}

}

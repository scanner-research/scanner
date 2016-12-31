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
#include "scanner/engine/db.h"

#include "scanner/evaluators/serialize.h"

#include "scanner/util/common.h"
#include "scanner/util/memory.h"
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

namespace scanner {
///////////////////////////////////////////////////////////////////////////////
/// Worker thread arguments
struct LoadThreadArgs {
  // Uniform arguments
  const DatasetMetadata& dataset;
  const std::map<i32, JobMetadata&> job_meta;
  const std::vector<VideoMetadata>& video_meta;
  const std::vector<ImageFormatGroupMetadata>& image_meta;
  const std::vector<IOItem>& io_items;
  i32 warmup_count;

  // Per worker arguments
  int id;
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<LoadWorkEntry>& load_work;  // in
  Queue<EvalWorkEntry>& eval_work;  // out
};

struct EvaluateThreadArgs {
  // Uniform arguments
  const std::vector<InputFormat>& metadata;
  const std::vector<IOItem>& io_items;
  i32 warmup_count;

  // Per worker arguments
  int id;
  int evaluator_group;
  bool last_evaluator_group;
  std::vector<EvaluatorFactory*> evaluator_factories;
  std::vector<EvaluatorConfig> evaluator_configs;
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& input_work;
  Queue<EvalWorkEntry>& output_work;
};

struct SaveThreadArgs {
  // Uniform arguments
  std::string dataset_name;
  std::string job_name;
  const std::vector<std::string>& video_paths;
  const std::vector<InputFormat>& metadata;
  const std::vector<WorkItem>& work_items;
  std::vector<std::string> output_names;

  // Per worker arguments
  int id;
  storehouse::StorageConfig* storage_config;
  Profiler& profiler;

  // Queues for communicating work
  Queue<EvalWorkEntry>& input_work;
  std::atomic<i64>& retired_items;
};

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
std::tuple<size_t, size_t> find_keyframe_indices(
    i32 start_frame, i32 end_frame,
    const std::vector<i64>& keyframe_positions) {
  size_t start_keyframe_index = std::numeric_limits<size_t>::max();
  for (size_t i = 1; i < keyframe_positions.size(); ++i) {
    if (keyframe_positions[i] > start_frame) {
      start_keyframe_index = i - 1;
      break;
    }
  }
  assert(start_keyframe_index != std::numeric_limits<size_t>::max());

  size_t end_keyframe_index = 0;
  for (size_t i = start_keyframe_index; i < keyframe_positions.size(); ++i) {
    if (keyframe_positions[i] >= end_frame) {
      end_keyframe_index = i;
      break;
    }
  }
  assert(end_keyframe_index != 0);
  return std::make_tuple(start_keyframe_index, end_keyframe_index);
}

void* load_thread(void* arg) {
  LoadThreadArgs& args = *reinterpret_cast<LoadThreadArgs*>(arg);

  auto setup_start = now();

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(args.storage_config);

  // To ammortize opening files
  i32 last_table_id = -1;
  std::vector<RandomReadFile*> files;
  std::vector<u64> file_sizes;
  std::vector<std::vector<i64>> keyframe_positions;
  std::vector<std::vector<i64>> keyframe_byte_offsets;
  std::vector<i64> image_compressed_sizes;
  std::vector<i64> image_compressed_offsets;

  args.profiler.add_interval("setup", setup_start, now());
  while (true) {
    auto idle_start = now();

    LoadWorkEntry load_work_entry;
    args.load_work.pop(load_work_entry);

    if (load_work_entry.io_item_index == -1) {
      break;
    }

    LOG(INFO) << "Load (N/PU: " << rank << "/" << args.id
              << "): processing item " << load_work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const IOItem& io_item = args.io_items[load_work_entry.io_item_index];
    const std::vector<LoadWorkEntry::Sample>& samples = load_work_entry.samples;

    if (io_item.table_id != last_table_id) {
      // Not from the same task so clear cached data
      last_table_id = io_item.table_id;
      for (auto file : files) {
        delete file;
      }
      files.clear();
      file_sizes.clear();
      keyframe_positions.clear();
      keyframe_byte_offsets.clear();
      image_compressed_sizes.clear();
      image_compressed_offsets.clear();
    }

    EvalWorkEntry eval_work_entry;
    eval_work_entry.io_item_index = load_work_entry.io_item_index;
    eval_work_entry.video_decode_item = false;

    // Aggregate all sample columns so we know the tuple size
    for (size_t i = 0; i < samples.size(); ++i) {
      for (const std::string& c : samples[i].columns) {
        eval_work_entry.column_names.push_back(c);
        if (c == base_column_name()) {
          eval_work_entry.video_decode_item = true;
        }
      }
    }
    i32 num_columns = static_cast<i32>(eval_work_entry.column_names.size());
    eval_work_entry.columns.resize(num_columns);
    eval_work_entry.buffer_handle = CPU_DEVICE;

    // Find the work item files that we will need to load the requested rows
    JobMetadata::RowLocations row_locations;
    if (!(args.in_columns.size() == 1 &&
          args.in_columns[0] == base_column_name())) {
      row_locations = args.in_job.row_work_item_locations(
          args.sampling, work_item.video_index, load_work_entry);
    }

    i32 special_col_idx = 0;
    i32 col_idx = 0;
    for (LoadWorkEntry::Sample& sample : samples) {
      for (const std::string& column_name : samples.columns) {
        // If reading from base job and special visual data column...
        if (sample.job_id == base_job_id() &&
            column_name == base_column_name()) {
          if (args.dataset.type() == DatasetType_Video) {
            // Special video column
            const VideoMetadata& metadata = args.video_meta[sample.table_id];
            if (files.size() <= col_idx) {
              // Open the video file for reading
              RandomReadFile* video_file;
              BACKOFF_FAIL(storage->make_random_read_file(
                  dataset_item_data_path(
                      args.dataset.name(),
                      args.dataset.item_names[sample.table_id]),
                  video_file));
              u64 file_size;
              BACKOFF_FAIL(video_file->get_size(file_size));

              keyframe_positions.push_back(metadata.keyframe_positions());
              keyframe_byte_offsets.push_back(metadata.keyframe_byte_offsets());
              // Place end of file and num frame at end of iframe to handle edge
              // case
              keyframe_positions.back().push_back(metadata.frames());
              keyframe_byte_offsets.back().push_back(file_size);
            }

            // Read the bytes from the file that correspond to the sequences of
            // frames we are interested in decoding. This sequence will contain
            // the bytes starting at the iframe at or preceding the first frame
            // we are interested and will continue up to the bytes before the
            // iframe at or after the last frame we are interested in.
            JobMetadata::FrameLocations locations = args.in_job.frame_locations(
                args.sampling, work_item.video_index, load_work_entry);
            std::vector<Interval>& intervals = locations.intervals;
            std::vector<DecodeArgs>& dargs = locations.video_args;
            assert(intervals.size() == dargs.size());
            size_t num_intervals = intervals.size();
            for (size_t i = 0; i < num_intervals; ++i) {
              i32 start_frame = intervals[i].start;
              i32 end_frame = intervals[i].end;
              size_t start_keyframe_index;
              size_t end_keyframe_index;
              std::tie(start_keyframe_index, end_keyframe_index) =
                  find_keyframe_indices(start_frame, end_frame,
                                        keyframe_positions);

              u64 start_keyframe_byte_offset =
                  static_cast<u64>(keyframe_byte_offsets[start_keyframe_index]);

              u64 end_keyframe_byte_offset =
                  static_cast<u64>(keyframe_byte_offsets[end_keyframe_index]);

              size_t buffer_size =
                  end_keyframe_byte_offset - start_keyframe_byte_offset;

              u8* buffer = new_buffer(CPU_DEVICE, buffer_size);

              auto io_start = now();

              u64 pos = start_keyframe_byte_offset;
              read(video_file, buffer, buffer_size, pos);

              args.profiler.add_interval("io", io_start, now());
              args.profiler.increment("io_read", static_cast<i64>(buffer_size));

              // Encoded buffer
              INSERT_ROW(eval_work_entry.columns[out_col], buffer, buffer_size);

              // Decode args
              DecodeArgs& decode_args = dargs[i];

              decode_args.set_warmup_count(args.warmup_count);
              decode_args.set_rows_from_start(work_item.rows_from_start);
              decode_args.set_start_keyframe(
                  keyframe_positions[start_keyframe_index]);
              decode_args.set_end_keyframe(
                  keyframe_positions[end_keyframe_index]);

              u8* decode_args_buffer = nullptr;
              size_t size;
              serialize_decode_args(decode_args, decode_args_buffer, size);

              INSERT_ROW(eval_work_entry.columns[out_col + 1],
                         decode_args_buffer, size);
            }
            // Jump over the next output column because we wrote two columns for
            // this iteration (frame and frame_args)
            out_col++;
          } else if (args.dataset.type() == DatasetType_Image) {
            // Special image column
            const ImageFormatGroupMetadata& metadata =
                args.image_meta[work_item.video_index];
            if (video_path != last_video_path) {
              if (image_file != nullptr) {
                delete image_file;
                image_file = nullptr;
              }
              image_compressed_sizes.clear();
              image_compressed_offsets.clear();

              // Open the video file for reading
              BACKOFF_FAIL(storage->make_random_read_file(
                  dataset_item_data_path(args.dataset.name(), video_path),
                  image_file));

              BACKOFF_FAIL(image_file->get_size(file_size));

              i64 s = 0;
              for (i64 size : metadata.compressed_sizes()) {
                image_compressed_sizes.push_back(size);
                image_compressed_offsets.push_back(s);
                s += size;
              }
              image_compressed_offsets.push_back(s);
            }
            last_video_path = video_path;

            // Read the bytes from the file that correspond to the sequences
            // of images we are interested in decoding.
            JobMetadata::FrameLocations locations = args.in_job.frame_locations(
                args.sampling, work_item.video_index, load_work_entry);
            std::vector<Interval>& intervals = locations.intervals;
            std::vector<ImageDecodeArgs>& dargs = locations.image_args;
            assert(intervals.size() == dargs.size());
            size_t num_intervals = intervals.size();
            for (size_t i = 0; i < num_intervals; ++i) {
              i32 start_frame = intervals[i].start;
              i32 end_frame = intervals[i].end;

              u64 start_byte_offset =
                  static_cast<u64>(image_compressed_offsets[start_frame]);

              u64 end_byte_offset =
                  static_cast<u64>(image_compressed_offsets[end_frame]);

              size_t buffer_size = end_byte_offset - start_byte_offset;

              u8* buffer = new_buffer(CPU_DEVICE, buffer_size);

              auto io_start = now();

              u64 pos = start_byte_offset;
              read(image_file, buffer, buffer_size, pos);

              args.profiler.add_interval("io", io_start, now());
              args.profiler.increment("io_read", static_cast<i64>(buffer_size));

              // Encoded buffer
              INSERT_ROW(eval_work_entry.columns[out_col], buffer, buffer_size);

              // Decode args
              ImageDecodeArgs& decode_args = dargs[i];

              decode_args.set_warmup_count(args.warmup_count);
              decode_args.set_rows_from_start(work_item.rows_from_start);
              decode_args.set_encoding_type(metadata.encoding_type());
              decode_args.set_color_space(metadata.color_space());
              for (i32 f = start_frame; f < end_frame; ++f) {
                decode_args.add_compressed_sizes(image_compressed_sizes[f]);
              }

              u8* decode_args_buffer = nullptr;
              size_t size;
              serialize_image_decode_args(decode_args, decode_args_buffer,
                                          size);

              INSERT_ROW(eval_work_entry.columns[out_col + 1],
                         decode_args_buffer, size);
            }
            // Jump over the next output column because we wrote two columns for
            // this iteration (frame and frame_args)
          } else {
            assert(false);
          }
        } else {
          // Regular column load
          i32 in_job_work_item_size = args.in_job.work_item_size();
          // Read each work item
          for (size_t i = 0; i < row_locations.work_items.size(); ++i) {
            i32 wi = row_locations.work_items[i];
            Interval interval = row_locations.work_item_intervals[i];

            std::unique_ptr<RandomReadFile> row_file;
            StoreResult result;
            BACKOFF_FAIL(make_unique_random_read_file(
                storage,
                job_item_output_path(args.dataset.name(), args.in_job.name(),
                                     video_path, column_name, wi),
                row_file));

            u64 row_file_size = 0;
            BACKOFF_FAIL(row_file->get_size(row_file_size));

            // Read number of rows in file
            u64 pos = 0;
            u64 num_rows = read<u64>(row_file.get(), pos);

            // Read row sizes from work item file header
            std::vector<i64> row_sizes(num_rows);
            read(row_file.get(), reinterpret_cast<u8*>(row_sizes.data()),
                 row_sizes.size() * sizeof(i64), pos);

            // Determine start and end position of rows to read in file
            u64 start_offset;
            for (i32 i = 0; i < interval.start; ++i) {
              start_offset += row_sizes[i];
            }
            u64 end_offset = start_offset;
            for (i32 i = interval.start; i < interval.end; ++i) {
              end_offset += row_sizes[i];
            }
            u64 row_data_size = end_offset - start_offset;
            std::vector<u8> row_data(row_data_size);

            // Read chunk of file corresponding to requested rows
            pos += start_offset;
            read(row_file.get(), row_data.data(), row_data.size(), pos);

            // Extract individual rows and insert into output work entry
            u64 offset = 0;
            for (i32 i = interval.start; i < interval.end; ++i) {
              size_t buffer_size = static_cast<size_t>(row_sizes[i]);
              u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
              memcpy(buffer, row_data.data() + offset, buffer_size);
              offset += buffer_size;
              INSERT_ROW(eval_work_entry.columns[out_col], buffer, buffer_size);
            }
          }
        }
        col_idx++;
      }
    }

    args.profiler.add_interval("task", work_start, now());

    args.eval_work.push(eval_work_entry);
  }

  LOG(INFO) << "Load (N/PU: " << rank << "/" << args.id << "): thread finished";

  // Cleanup
  for (auto file : files) {
    delete file;
  }
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to run evaluation
void* pre_evaluate_thread(void* arg) {
  // Split up a work entry into work item size chunks
}

void* evaluate_thread(void* arg) {
  EvaluateThreadArgs& args = *reinterpret_cast<EvaluateThreadArgs*>(arg);

  auto setup_start = now();

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  assert(args.evaluator_factories.size() == args.evaluator_configs.size());
  std::vector<EvaluatorCapabilities> evaluator_caps;
  std::vector<std::unique_ptr<Evaluator>> evaluators;
  std::vector<i32> num_evaluator_outputs;
  for (size_t i = 0; i < args.evaluator_factories.size(); ++i) {
    EvaluatorFactory* factory = args.evaluator_factories[i];
    const EvaluatorConfig& config = args.evaluator_configs[i];
    evaluator_caps.push_back(factory->get_capabilities());
    evaluators.emplace_back(factory->new_evaluator(config));
    num_evaluator_outputs.push_back(factory->get_output_names().size());
  }
  assert(evaluators.size() > 0);

  for (auto& evaluator : evaluators) {
    evaluator->set_profiler(&args.profiler);
  }

  i32 last_evaluator_num_columns =
      args.evaluator_factories.back()->get_output_names().size();
  i32 last_evaluator_device_id = args.evaluator_configs.back().device_ids[0];
  DeviceType last_evaluator_device_type = evaluator_caps.back().device_type;

  args.profiler.add_interval("setup", setup_start, now());

  int last_video_index = -1;
  int last_next_item_id = -1;
  while (true) {
    auto idle_start = now();
    // Wait for next work item to process
    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.work_item_index == -1) {
      break;
    }

    LOG(INFO) << "Evaluate (N/PU/G: " << rank << "/" << args.id << "/"
              << args.evaluator_group << "): processing item "
              << work_entry.work_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const WorkItem& work_item = args.work_items[work_entry.work_item_index];
    const InputFormat& metadata = args.metadata[work_item.video_index];

    bool needs_configure = !(work_item.video_index == last_video_index);
    bool needs_reset = (!(work_item.video_index == last_video_index &&
                          work_item.item_id == last_next_item_id));
    for (auto& evaluator : evaluators) {
      // Make the evaluator aware of the format of the data we are about to
      // feed it
      if (needs_configure) {
        evaluator->configure(metadata);
      }
      if (needs_reset) {
        evaluator->reset();
      }
    }
    last_video_index = work_item.video_index;
    last_next_item_id = work_item.next_item_id;

    size_t frame_size = metadata.width() * metadata.height() * 3 * sizeof(u8);

    EvalWorkEntry output_work_entry;
    output_work_entry.work_item_index = work_entry.work_item_index;
    output_work_entry.buffer_type = evaluator_caps.back().device_type;
    output_work_entry.buffer_device_id =
        args.evaluator_configs.back().device_ids[0];
    output_work_entry.video_decode_item = false;

    BatchedColumns& work_item_output_columns = output_work_entry.columns;
    work_item_output_columns.resize(last_evaluator_num_columns);

    i32 current_input = 0;
    i32 total_inputs = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      total_inputs =
          std::max(total_inputs, (i32)work_entry.columns[i].rows.size());
    }
    while (current_input < total_inputs) {
      i32 batch_size = std::min(total_inputs - current_input, WORK_ITEM_SIZE);

      std::vector<std::string> input_names;
      BatchedColumns input_columns;
      DeviceType input_buffer_type;
      i32 input_device_id;
      // Initialize the output buffers with the frame input because we
      // perform a swap from output to input on each iterator to pass outputs
      // from the previous evaluator into the input of the next one
      std::vector<std::string> output_names = work_entry.column_names;
      BatchedColumns output_columns;
      output_columns.resize(work_entry.columns.size());
      for (size_t i = 0; i < work_entry.columns.size(); ++i) {
        i32 batch =
            std::min(batch_size, (i32)work_entry.columns[i].rows.size());
        assert(batch > 0);
        output_columns[i].rows.insert(
            output_columns[i].rows.end(),
            work_entry.columns[i].rows.begin() + current_input,
            work_entry.columns[i].rows.begin() + current_input + batch);
      }
      DeviceType output_buffer_type = work_entry.buffer_type;
      i32 output_device_id = work_entry.buffer_device_id;

      for (size_t e = 0; e < evaluators.size(); ++e) {
        i32 device_id = args.evaluator_configs[e].device_ids[0];
        EvaluatorCapabilities& caps = evaluator_caps[e];
        std::unique_ptr<Evaluator>& evaluator = evaluators[e];
        i32 num_outputs = num_evaluator_outputs[e];

        input_names.swap(output_names);
        input_columns.swap(output_columns);
        input_buffer_type = output_buffer_type;
        input_device_id = output_device_id;

        i32 num_inputs = input_columns.size();
        // If current evaluator type and input buffer type differ, then move
        // the data in the input buffer into a new buffer which has the same
        // type as the evaluator input
        auto copy_start = now();
        if (input_buffer_type != caps.device_type ||
            input_device_id != device_id) {
          for (i32 i = 0; i < num_inputs; ++i) {
            std::vector<u8*> dest_buffers, src_buffers;
            std::vector<size_t> sizes;

            Column& column = input_columns[i];
            size_t total_size = 0;
            for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
              total_size += column.rows[b].size;
            }

            if (column.rows.size() > 0) {
              u8* block = new_block_buffer({caps.device_type, device_id},
                                           total_size,
                                           column.rows.size());
              for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
                size_t size = column.rows[b].size;
                dest_buffers.push_back(block);
                block += size;
                src_buffers.push_back(column.rows[b].buffer);
                sizes.push_back(size);
              }

              auto memcpy_start = now();
              memcpy_vec(dest_buffers, {caps.device_type, device_id},
                         src_buffers, {input_buffer_type, input_device_id},
                         sizes);
              args.profiler.add_interval("memcpy", memcpy_start, now());

              auto delete_start = now();
              for (i32 b = 0; b < (i32)column.rows.size(); ++b) {
                delete_buffer({input_buffer_type, input_device_id},
                              column.rows[b].buffer);
                column.rows[b].buffer = dest_buffers[b];
              }
            }
          }

          input_buffer_type = caps.device_type;
          input_device_id = device_id;
        }
        args.profiler.add_interval("evaluator_marshal", copy_start, now());

        // Setup output buffers to receive evaluator output
        output_columns.clear();
        output_buffer_type = caps.device_type;
        output_device_id = device_id;
        output_columns.resize(num_outputs);
        output_names = args.evaluator_factories[e]->get_output_names();

        auto eval_start = now();
        evaluator->evaluate(input_columns, output_columns);
        args.profiler.add_interval("evaluate", eval_start, now());
        // Do not verify outputs == inputs if we are decoding encoded video as
        // there is an increase of 1 encoded chunk to multiple frames
        if (false && !(e == 0 && work_entry.video_decode_item)) {
          for (size_t i = 0; i < output_columns.size(); ++i) {
            LOG_IF(FATAL, output_columns[i].rows.size() != batch_size)
                << "Evaluator " << e << " produced "
                << output_columns[i].rows.size() << " output rows for column "
                << output_names[i] << ". Expected " << batch_size
                << " outputs.";
          }
        }
        // HACK(apoms): Handle the case where the video decode evaluator gets a
        //   single input but produces multiple outputs. Should be removed if we
        //   add flatmap esque increases in output element count
        if (e == 0 && work_entry.video_decode_item) {
          batch_size = output_columns[0].rows.size();
        }

        // Allow passing input buffers through to an evaluator output
        // by tracking the pointers and comparing the output pointers
        // for equality
        std::set<u8*> all_output_buffers_set;
        for (Column& column : output_columns) {
          for (Row& row : column.rows) {
            all_output_buffers_set.insert(row.buffer);
          }
        }

        // Delete input buffers after they are used
        for (size_t i = 0; i < num_inputs; ++i) {
          Column& column = input_columns[i];
          for (Row& row : column.rows) {
            u8* buff = row.buffer;
            if (all_output_buffers_set.count(buff) == 0) {
              delete_buffer({input_buffer_type, input_device_id}, buff);
            }
          }
        }
      }
      // Only discard warmup frames for last evaluator group because otherwise
      // they need to be forwarded to warm up later evaluator groups
      i32 warmup_frames;
      if (args.last_evaluator_group && needs_reset) {
        i32 total_warmup_frames =
            std::min(args.warmup_count, work_item.rows_from_start);
        warmup_frames = std::min(
            batch_size, std::max(0, total_warmup_frames - current_input));
      } else {
        warmup_frames = 0;
      }
      for (i32 i = 0; i < last_evaluator_num_columns; ++i) {
        // Delete warmup frame outputs
        for (i32 w = 0; w < warmup_frames; ++w) {
          delete_buffer({last_evaluator_device_type, last_evaluator_device_id},
                        output_columns[i].rows[w].buffer);
        }

        // Make sure all outputs are in CPU memory so downstream code does not
        // need to condition on buffer type
        i32 num_output_rows = static_cast<i32>(output_columns[i].rows.size());
        // Keep non-warmup frame outputs
        work_item_output_columns[i].rows.insert(
            work_item_output_columns[i].rows.end(),
            output_columns[i].rows.begin() + warmup_frames,
            output_columns[i].rows.end());
      }
      current_input += batch_size;
    }

    args.profiler.add_interval("task", work_start, now());

    LOG(INFO) << "Evaluate (N/PU/G: " << rank << "/" << args.id << "/"
              << args.evaluator_group << "): finished item "
              << work_entry.work_item_index;

    args.output_work.push(output_work_entry);
  }

  LOG(INFO) << "Evaluate (N/PU: " << rank << "/" << args.id
            << "): thread finished";

  THREAD_RETURN_SUCCESS();
}

void* post_evaluate_thread(void* arg) {
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

    EvalWorkEntry work_entry;
    args.input_work.pop(work_entry);

    if (work_entry.work_item_index == -1) {
      break;
    }

    LOG(INFO) << "Save (N/PU: " << rank << "/" << args.id
              << "): processing item " << work_entry.work_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const WorkItem& work_item = args.work_items[work_entry.work_item_index];

    const std::string& video_path = args.video_paths[work_item.video_index];
    const InputFormat& metadata = args.metadata[work_item.video_index];

    // Write out each output layer to an individual data file
    for (size_t out_idx = 0; out_idx < args.output_names.size(); ++out_idx) {
      u64 num_rows = static_cast<u64>(work_entry.columns[out_idx].rows.size());

      const std::string output_path = job_item_output_path(
          args.dataset_name, args.job_name, video_path,
          args.output_names[out_idx], work_item.item_index);

      auto io_start = now();

      WriteFile* output_file = nullptr;
      {
        StoreResult result;
        BACKOFF_FAIL(storage->make_write_file(output_path, output_file));
      }

      if (work_entry.columns[out_idx].rows.size() != num_rows) {
        LOG(FATAL) << "Output layer's row vector has wrong length";
      }

      if (work_entry.buffer_type != DeviceType::CPU) {
        std::vector<u8*> dest_buffers, src_buffers;
        std::vector<size_t> sizes;
        size_t total_size = 0;
        for (i32 f = 0; f < num_rows; ++f) {
          Row& row = work_entry.columns[out_idx].rows[f];
          total_size += row.size;
        }

        if (num_rows > 0) {
          u8* output_block = new_block_buffer(CPU_DEVICE, total_size, num_rows);
          for (i32 f = 0; f < num_rows; ++f) {
            Row& row = work_entry.columns[out_idx].rows[f];
            size_t size = row.size;
            u8* src_buffer = row.buffer;
            u8* dest_buffer = output_block;

            dest_buffers.push_back(dest_buffer);
            src_buffers.push_back(src_buffer);
            sizes.push_back(size);

            output_block += size;
          }

          memcpy_vec(dest_buffers, CPU_DEVICE, src_buffers,
                     {work_entry.buffer_type, work_entry.buffer_device_id},
                     sizes);

          for (i32 f = 0; f < num_rows; ++f) {
            delete_buffer({work_entry.buffer_type, work_entry.buffer_device_id},
                          src_buffers[f]);
            work_entry.columns[out_idx].rows[f].buffer = dest_buffers[f];
          }
        }
      }

      // Write number of rows in the file
      write(output_file, num_rows);
      // Write out all output sizes first so we can easily index into the file
      i64 size_written = 0;
      for (size_t i = 0; i < num_rows; ++i) {
        i64 buffer_size = work_entry.columns[out_idx].rows[i].size;
        write(output_file, buffer_size);
        size_written += sizeof(i64);
      }
      // Write actual output data
      for (size_t i = 0; i < num_rows; ++i) {
        i64 buffer_size = work_entry.columns[out_idx].rows[i].size;
        u8* buffer = work_entry.columns[out_idx].rows[i].buffer;
        write(output_file, buffer, buffer_size);
        size_written += buffer_size;
      }

      BACKOFF_FAIL(output_file->save());

      // TODO(apoms): For now, all evaluators are expected to return CPU
      //   buffers as output so just assume CPU
      for (size_t i = 0; i < num_rows; ++i) {
        delete_buffer(
          {DeviceType::CPU,  // work_entry.buffer_type,
              work_entry.buffer_device_id},
          work_entry.columns[out_idx].rows[i].buffer);
      }

      delete output_file;

      args.profiler.add_interval("io", io_start, now());
      args.profiler.increment("io_write", size_written);
    }

    LOG(INFO) << "Save (N/PU: " << rank << "/" << args.id << "): finished item "
              << work_entry.work_item_index;

    args.profiler.add_interval("task", work_start, now());

    args.retired_items++;
  }

  LOG(INFO) << "Save (N/PU: " << rank << "/" << args.id
            << "): thread finished ";

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// run_job
void run_job(JobParameters& params) {
  storehouse::StorageBackend* storage =
    storehouse::StorageBackend::make_from_config(params.storage_config);

  i32 rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  i32 num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  // Get node-local info
  // http://stackoverflow.com/questions/9022496/how-to-determine-mpi-rank-process-number-local-to-a-socket-node
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &shmcomm);
  i32 local_rank;
  MPI_Comm_rank(shmcomm, &local_rank);

  i32 local_num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &local_num_nodes);

  // Load the dataset descriptor to find all data files
  i32 dataset_id = meta.get_dataset_id(params.dataset_name);
  DatasetDescriptor descriptor;
  {
    std::unique_ptr<RandomReadFile> file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage, dataset_descriptor_path(params.dataset_name), file));
    u64 pos = 0;
    descriptor = deserialize_dataset_descriptor(file.get(), pos);
  }
  DatasetMetadata dataset_meta(descriptor);

  // Establish base time to use for profilers
  timepoint_t base_time = now();

  // Get metadata for all dataset items for distributing to evaluators
  std::vector<std::string> paths{dataset_meta.item_names()};

  std::vector<VideoMetadata> video_metadata;
  std::vector<ImageFormatGroupMetadata> image_metadata;
  std::vector<InputFormat> input_formats;
  std::vector<DatasetItemMetadata> item_descriptors;
  for (size_t i = 0; i < paths.size(); ++i) {
    const std::string& path = paths.at(i);
    std::unique_ptr<RandomReadFile> metadata_file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage, dataset_item_metadata_path(params.dataset_name, path),
        metadata_file));
    if (dataset_meta.type() == DatasetType_Video) {
      u64 pos = 0;
      video_metadata.push_back(
          deserialize_video_metadata(metadata_file.get(), pos));
      VideoMetadata& meta = video_metadata.back();
      input_formats.emplace_back(meta.width(), meta.height());
      item_descriptors.emplace_back(meta.frames(), meta.width(), meta.height());
    } else if (dataset_meta.type() == DatasetType_Image) {
      u64 pos = 0;
      image_metadata.push_back(
          deserialize_image_format_group_metadata(metadata_file.get(), pos));
      ImageFormatGroupMetadata& meta = image_metadata.back();
      input_formats.emplace_back(meta.width(), meta.height());
      item_descriptors.emplace_back(meta.num_images(), meta.width(),
                                    meta.height());
    }
  }

  // Generate the pipeline description by feeding in the dataset information
  // into the user supplied pipeline generator function
  std::vector<std::string> dataset_job_names;
  for (i32 job_id : db_meta.dataset_job_ids.at(dataset_id)) {
    dataset_job_names.push_back(db_meta.job_names.at(job_id));
  }
  PipelineDescription pipeline_description;
  {
    DatasetInformation info(params.dataset_name, dataset_job_names);
    pipeline_description = params.pipeline_gen_fn(info);
  }

  // Validate pipeline description and load job metadata for jobs listed in
  // pipeline description tasks
  LOG_IF(FATAL, pipeline_description.tasks.empty())
      << "No tasks specified for pipeline description!";
  std::map<i32, JobMetadata> job_meta;
  for (Task& task : pipeline_description.tasks) {
    LOG_IF(FATAL, task.samples.empty())
        << "No samples specified for task with table name " << task.table_name
        << "!";
    for (TableSample &sample : task.samples) {
      i32 job_id = db.get_job_id(dataset_id, sample.job_name);
      if (job_meta.count(job_id) == 0) {
        LOG_IF(FATAL, !db_meta.has_job(job_name))
            << "Requested job " << job_name << " does not exist in dataset "
            << params.dataset_name << "!";
        JobDescriptor descriptor;
        std::unique_ptr<RandomReadFile> file;
        BACKOFF_FAIL(make_unique_random_read_file(
            storage, job_descriptor_path(params.dataset_name, job_name), file));
        u64 pos = 0;
        JobDescriptor descriptor = deserialize_job_descriptor(file.get(), pos);
        job_meta.insert({job_id, JobMetadata(descriptor)});
      }
      JobMetadata& meta = job_meta.at(job_id);
      LOG_IF(FATAL, !meta.has_table(sample.table_name))
          << "Requested table " << sample.table_name << " does not exist in "
          << "job " << sample.job_name << " in dataset " << params.dataset_name
          << "!";
      LOG_IF(FATAL, sample.columns.empty())
          << "No columns specified for sampling from table "
          << sample.table_name << " in job " << sample.job_name
          << " in dataset " << params.dataset_name << "!";
      std::set<std::string> job_columns(meta.columns().begin(),
                                        meta.columns().end());
      assert(!job_columns.empty());
      std::string available_columns = *job_columns.begin();
      for (auto it = job_columns.begin() + 1; it != job_columns.end(); ++it) {
        available_columns += ", " + *it;
      }
      for (const std::string &column : sample.columns) {
        LOG_IF(FATAL, job_columns.count(column) == 0)
            << "Requested column " << column << " does not exist in table "
            << sample.table_name << " in job " << sample.job_name
            << " in dataset " << params.dataset_name << "! Available columns "
            << "are: " << available_columns;
      }
    }
  }

  // Unwrap factories into raw pointers and get capabilities
  std::vector<EvaluatorFactory*> evaluator_factories;
  for (auto& f : pipeline_description.evaluator_factories) {
    evaluator_factories.push_back(f.get());
  }
  std::vector<EvaluatorCapabilities> evaluator_caps;
  for (EvaluatorFactory* factory : evaluator_factories) {
    evaluator_caps.push_back(factory->get_capabilities());
  }

  // We break up work into IO items which are then broken up into work items
  // to be processed by evaluators
  const i32 io_item_size = rows_per_io_item();
  const i32 work_item_size = rows_per_work_item();

  // It is necessary to track how work was broken up for each video so that the
  // system can later figure out where each output row is located

  // We need to know the maximum warmup size across all evaluators to pass
  // enough work items through the pipeline after a reset, even if it is more
  // than a specific evaluator needed for warmup
  i32 warmup_size = 0;

  // Only calculate the warmup for video datasets
  if (dataset_meta.type() == DatasetType_Video) {
    for (EvaluatorCapabilities& caps : evaluator_caps) {
      warmup_size = std::max(warmup_size, caps.warmup_size);
    }
  }

  // Create job descriptor and list of work
  u32 total_rows = 0;
  std::vector<IOItem> io_items;
  std::vector<LoadWorkEntry> load_work_entries;
  std::vector<size_t> item_task_delimeters;
  std::vector<std::string> final_column_names =
      evaluator_factories.back()->get_output_names();
  JobDescriptor job_descriptor;
  job_descriptor.set_io_item_size(io_item_size);
  job_descriptor.set_work_item_size(work_item_size);
  job_descriptor.set_num_nodes(num_nodes);
  for (i32 i = 0; i < (i32)(pipeline_description.tasks.size()); ++i) {
    // Keep track of where tasks start and end so we can try and partition
    // all items associated with a single task to the same evaluator
    item_task_delimeters.push_back(io_items.size());

    Task& task = pipeline_description.tasks.at(i);
    JobDescriptor::Task* jd_task = job_descriptor.add_tasks();
    jd_task->set_table_id(i);
    jd_task->set_table_name(task.table_name);
    for (TableSample &sample : task.samples) {
      i32 sample_job_id = db_meta.get_job_id(dataset_id, sample.job_name);
      JobMeta& meta = job_meta.at(sample_job_id);

      JobDescriptor::Task::TableSample* jd_sample = jd_task.add_samples();
      jd_sample->set_job_id(sample_job_id);
      i32 sample_table_id =
          meta.table_id(sample.table_name);
      jd_sample->set_table_id(sample_table_id);
      for (const std::string& col : sample.columns) {
        JobDescriptor::Column* jd_col = jd_sample->add_columns();
        jd_col->set_id(meta.column_id(col));
        jd_col->set_name(col);
      }
      for (i64 r : sample.rows) {
        jd_sample->add_rows(r);
      }
    }

    // Split up task into IOItems
    assert(task.samples.size() > 0);
    i64 rows_in_task = static_cast<i64>(task.samples[0].rows.size());
    i64 allocated_rows = 0;
    while (allocated_rows < rows_in_task) {
      i64 rows_to_allocate =
          std::min(io_item_size, rows_in_sample - allocated_rows);

      IOItem item;
      item.table_id = i;
      item.start_row = allocated_rows;
      item.end_row = allocated_rows + rows_to_allocate;
      io_items.push_back(item);

      LoadWorkEntry load_item;
      load_item.io_item_index = io_items.size() - 1;
      for (TableSample &sample : task.samples) {
        i32 sample_job_id = db_meta.get_job_id(dataset_id, sample.job_name);
        JobMeta& meta = job_meta.at(sample_job_id);
        i32 sample_table_id = meta.table_id(sample.table_name);

        LoadWorkEntry::Sample load_sample;
        load_sample.job_id = sample_job_id;
        load_sample.table_id = sample_table_id;
        load_sample.columns = sample.columns;
        i64 e = allocated_rows + rows_to_allocate;
        for (i64 s = allocated_rows; s < e; ++s) {
          load_sample.rows.push_back(sample.rows[s]);
        }
      }
      load_work_entries.push_back(load_item);

      allocated_rows += rows_to_allocate;
    }
    total_rows += rows_in_task;
  }
  for (size_t j = 0; j < final_column_names.size(); ++j) {
    JobDescriptor_Column* column = job_descriptor.add_columns();
    column->set_id(j);
    column->set_name(final_column_names[j]);
  }

  if (is_master(rank)) {
    printf("Total IO items: %lu, Total rows: %u\n", io_items.size(),
           total_rows);
  }

  // Setup shared resources for distributing work to processing threads
  i64 accepted_items = 0;
  Queue<LoadWorkEntry> load_work;
  Queue<EvalWorkEntry> initial_eval_work;
  std::vector<std::vector<Queue<EvalWorkEntry>>> eval_work(PUS_PER_NODE);
  Queue<EvalWorkEntry> save_work;
  std::atomic<i64> retired_items{0};

  // Setup load workers
  std::vector<Profiler> load_thread_profilers(LOAD_WORKERS_PER_NODE,
                                              Profiler(base_time));
  std::vector<LoadThreadArgs> load_thread_args;
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    load_thread_args.emplace_back(LoadThreadArgs{
        // Uniform arguments
        dataset_meta, job_meta, video_metadata, image_metadata, input_formats,
        io_items, warmup_size,

        // Per worker arguments
        i, params.storage_config, load_thread_profilers[i],

        // Queues
        load_work, initial_eval_work,
    });
  }
  std::vector<pthread_t> load_threads(LOAD_WORKERS_PER_NODE);
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    pthread_create(&load_threads[i], NULL, load_thread, &load_thread_args[i]);
  }

  // Setup evaluate workers

  // Initialize factory groups which determine which evaluators run in the
  // same thread. Evaluators running in different threads should be using
  // different physical resources
  std::vector<std::vector<EvaluatorFactory*>> factory_groups;
  if (evaluator_caps.size() == 1) {
    factory_groups.push_back({evaluator_factories.front()});
  } else if (evaluator_caps.size() == 2 &&
             (evaluator_caps.front().can_overlap ||
              evaluator_caps.back().can_overlap)) {
    factory_groups.push_back({evaluator_factories.front()});
    factory_groups.push_back({evaluator_factories.back()});
  } else {
    i32 evaluator_offset_start = 0;
    i32 evaluator_offset_end = static_cast<i32>(evaluator_factories.size() - 1);
    std::vector<EvaluatorFactory*> main_factories;
    if (evaluator_caps.front().can_overlap) {
      factory_groups.push_back({evaluator_factories.front()});
      evaluator_offset_start++;
    }
    main_factories.insert(main_factories.end(),
                          evaluator_factories.begin() + evaluator_offset_start,
                          evaluator_factories.begin() + evaluator_offset_end);
    if (evaluator_caps.back().can_overlap) {
      factory_groups.push_back(main_factories);
      factory_groups.push_back({evaluator_factories.back()});
    } else {
      main_factories.push_back(evaluator_factories.back());
      factory_groups.push_back(main_factories);
    }
  }

  i32 factory_groups_per_chain = static_cast<i32>(factory_groups.size());
  assert(factory_groups_per_chain > 0);

  std::vector<std::vector<Profiler>> eval_chain_profilers(PUS_PER_NODE);
  std::vector<std::vector<EvaluateThreadArgs>> eval_chain_args(PUS_PER_NODE);

  i32 num_gpus = static_cast<i32>(GPU_DEVICE_IDS.size());
  std::set<i32> gpu_device_ids;
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    std::vector<Queue<EvalWorkEntry>>& work_queues = eval_work[pu];
    std::vector<Profiler>& eval_thread_profilers = eval_chain_profilers[pu];
    std::vector<EvaluateThreadArgs>& eval_thread_args = eval_chain_args[pu];
    work_queues.resize(factory_groups_per_chain - 1);
    // Setup profilers and thread args
    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      eval_thread_profilers.push_back(Profiler(base_time));
    }
    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      std::vector<EvaluatorConfig> eval_configs;
      for (size_t i = 0; i < factory_groups[fg].size(); ++i) {
        DeviceType evaluator_device_type =
            factory_groups[fg][i]->get_capabilities().device_type;

        EvaluatorConfig eval_config;
        eval_config.max_input_count =
            std::max(frames_per_work_item(), warmup_size);
        eval_config.max_frame_width = dataset_meta.max_width();
        eval_config.max_frame_height = dataset_meta.max_height();

        i32 device_id;
        if (evaluator_device_type == DeviceType::GPU) {
          LOG_IF(FATAL, num_gpus == 0)
              << "Scanner is configured with zero available GPUs but a GPU "
              << "evaluator was requested! Please configure Scanner to have "
              << "at least one GPU using the `gpu_device_ids` config option.";

          // If we have more than one MPI process on a single machine, then
          // we should round robin the GPUs between the nodes if possible.
          // This case occurs if having multiple PUs per process would conflict,
          // e.g. Caffe with Python layers.
          i32 base_index = num_gpus / local_num_nodes * local_rank;
          device_id = GPU_DEVICE_IDS[(base_index + pu) % num_gpus];
          gpu_device_ids.insert(device_id);
        } else {
          device_id = pu;
        }

        eval_config.device_ids = {device_id};
        eval_configs.push_back(eval_config);
      }
      // Input work queue
      Queue<EvalWorkEntry>* input_work_queue;
      bool first_evaluator_group = (fg == 0);
      if (first_evaluator_group) {
        input_work_queue = &initial_eval_work;
      } else {
        input_work_queue = &work_queues[fg - 1];
      }
      // Create new queue for output, reuse previous queue as input
      bool last_evaluator_group = (fg == factory_groups_per_chain - 1);
      Queue<EvalWorkEntry>* output_work_queue;
      if (last_evaluator_group) {
        output_work_queue = &save_work;
      } else {
        output_work_queue = &work_queues[fg];
      }
      // Create eval thread for passing data through neural net
      eval_thread_args.emplace_back(EvaluateThreadArgs{
          // Uniform arguments
          warmup_size, input_formats, work_items,

          // Per worker arguments
          pu, fg, last_evaluator_group, factory_groups[fg], eval_configs,
          eval_thread_profilers[fg],

          // Queues
          *input_work_queue, *output_work_queue});
    }
  }

  std::vector<i32> gpu_device_ids_vec;
  std::copy(gpu_device_ids.begin(), gpu_device_ids.end(),
            std::back_inserter(gpu_device_ids_vec));
  init_memory_allocators(gpu_device_ids_vec, params.memory_pool_config);

  std::vector<std::vector<pthread_t>> eval_chain_threads(PUS_PER_NODE);
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    std::vector<pthread_t>& eval_threads = eval_chain_threads[pu];
    eval_threads.resize(factory_groups_per_chain);
    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      pthread_create(&eval_threads[fg], NULL, evaluate_thread,
                     &eval_chain_args[pu][fg]);
    }
  }

  // Setup save workers
  std::vector<Profiler> save_thread_profilers(SAVE_WORKERS_PER_NODE,
                                              Profiler(base_time));
  std::vector<SaveThreadArgs> save_thread_args;
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    // Create IO thread for reading and decoding data
    save_thread_args.emplace_back(SaveThreadArgs{
        // Uniform arguments
        params.dataset_name, params.out_job_name, paths, input_formats, work_items,
        evaluator_factories.back()->get_output_names(),

        // Per worker arguments
        i, params.storage_config, save_thread_profilers[i],

        // Queues
        save_work, retired_items});
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
      i32 local_work = accepted_items - retired_items;
      if (local_work < PUS_PER_NODE * TASKS_IN_QUEUE_PER_PU) {
        LoadWorkEntry& entry = load_work_items[next_work_item_to_allocate++];
        load_work.push(entry);

        accepted_items++;
        if ((static_cast<i32>(work_items.size()) - next_work_item_to_allocate) %
                10 ==
            0) {
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
        std::this_thread::yield();
      }
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
      i32 local_work = accepted_items - retired_items;
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
          LoadWorkEntry& entry = load_work_items[next_item];
          load_work.push(entry);
          accepted_items++;
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

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    EvalWorkEntry entry;
    entry.work_item_index = -1;
    initial_eval_work.push(entry);
  }

  for (i32 i = 0; i < PUS_PER_NODE; ++i) {
    // Wait until first eval has finished
    void* result;
    i32 err = pthread_join(eval_chain_threads[i][0], &result);
    if (err != 0) {
      fprintf(stderr, "error in pthread_join of eval thread\n");
      exit(EXIT_FAILURE);
    }
    free(result);
  }

  for (i32 fg = 1; fg < factory_groups_per_chain; ++fg) {
    for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
      EvalWorkEntry entry;
      entry.work_item_index = -1;
      eval_work[pu][fg - 1].push(entry);
    }
    for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
      // Wait until eval has finished
      void* result;
      i32 err = pthread_join(eval_chain_threads[pu][fg], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join of eval thread\n");
        exit(EXIT_FAILURE);
      }
      free(result);
    }
  }

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    EvalWorkEntry entry;
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
    // Add job name into database metadata so we can look up what jobs have
    // been ran
    i32 job_id;
    {
      const std::string db_meta_path = database_metadata_path();

      std::unique_ptr<RandomReadFile> meta_in_file;
      BACKOFF_FAIL(
          make_unique_random_read_file(storage, db_meta_path, meta_in_file));
      u64 pos = 0;
      DatabaseMetadata meta =
          deserialize_database_metadata(meta_in_file.get(), pos);

      job_id = meta.add_job(dataset_id, params.out_job_name);

      std::unique_ptr<WriteFile> meta_out_file;
      BACKOFF_FAIL(
          make_unique_write_file(storage, db_meta_path, meta_out_file));
      serialize_database_metadata(meta_out_file.get(), meta);
    }

    job_descriptor.set_id(job_id);
    job_descriptor.set_name(params.out_job_name);

    // Write out metadata to describe where the output results are for each
    // video
    {
      const std::string job_file_path =
          job_descriptor_path(params.dataset_name, params.out_job_name);
      std::unique_ptr<WriteFile> output_file;
      BACKOFF_FAIL(make_unique_write_file(storage, job_file_path, output_file));

      serialize_job_descriptor(output_file.get(), job_descriptor);

      BACKOFF_FAIL(output_file->save());
    }
  }

  // Write out total time interval
  timepoint_t end_time = now();

  // Execution done, write out profiler intervals for each worker
  std::string profiler_file_name =
      job_profiler_path(params.dataset_name, params.out_job_name, rank);
  std::unique_ptr<WriteFile> profiler_output;
  BACKOFF_FAIL(
      make_unique_write_file(storage, profiler_file_name, profiler_output));

  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
          .time_since_epoch()
          .count();
  write(profiler_output.get(), start_time_ns);
  write(profiler_output.get(), end_time_ns);

  i64 out_rank = rank;
  // Load worker profilers
  u8 load_worker_count = LOAD_WORKERS_PER_NODE;
  write(profiler_output.get(), load_worker_count);
  for (i32 i = 0; i < LOAD_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "load", "", i,
                           load_thread_profilers[i]);
  }

  // Evaluate worker profilers
  u8 eval_worker_count = PUS_PER_NODE;
  write(profiler_output.get(), eval_worker_count);
  u8 groups_per_chain = factory_groups_per_chain;
  write(profiler_output.get(), groups_per_chain);
  for (i32 pu = 0; pu < PUS_PER_NODE; ++pu) {
    for (i32 fg = 0; fg < factory_groups_per_chain; ++fg) {
      i32 i = pu;
      std::string tag = "fg" + std::to_string(fg);
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_chain_profilers[pu][fg]);
    }
  }

  // Save worker profilers
  u8 save_worker_count = SAVE_WORKERS_PER_NODE;
  write(profiler_output.get(), save_worker_count);
  for (i32 i = 0; i < SAVE_WORKERS_PER_NODE; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "save", "", i,
                           save_thread_profilers[i]);
  }

  BACKOFF_FAIL(profiler_output->save());

  delete storage;
}
}

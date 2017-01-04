/* Copyright 2016 Carnegie Mellon University
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

#include "scanner/engine/load_worker.h"
#include "scanner/engine/sampling.h"
#include "scanner/util/storehouse.h"

#include "scanner/evaluators/serialize.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
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

  const i32 io_item_size = rows_per_io_item();
  const i32 work_item_size = rows_per_work_item();

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(args.storage_config);

  // To ammortize opening files
  i32 last_table_id = -1;
  std::vector<RandomReadFile*> files;
  std::vector<u64> file_sizes;
  std::vector<std::vector<i64>> all_keyframe_positions;
  std::vector<std::vector<i64>> all_keyframe_byte_offsets;
  std::vector<std::vector<i64>> all_image_compressed_sizes;
  std::vector<std::vector<i64>> all_image_compressed_offsets;

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
      all_keyframe_positions.clear();
      all_keyframe_byte_offsets.clear();
      all_image_compressed_sizes.clear();
      all_image_compressed_offsets.clear();
    }

    EvalWorkEntry eval_work_entry;
    eval_work_entry.io_item_index = load_work_entry.io_item_index;

    // Aggregate all sample columns so we know the tuple size
    assert(!samples.empty());
    for (size_t i = 0; i < samples.size(); ++i) {
      for (const std::string& c : samples[i].columns) {
        eval_work_entry.column_names.push_back(c);
        if (c == base_column_name()) {
          eval_work_entry.column_names.push_back(base_column_args_name());
        }
      }
    }
    i32 num_columns = static_cast<i32>(eval_work_entry.column_names.size());
    eval_work_entry.columns.resize(num_columns);
    eval_work_entry.buffer_handle = CPU_DEVICE;

    i32 media_col_idx = 0;
    i32 col_idx = 0;
    for (const LoadWorkEntry::Sample& sample : samples) {
      const std::vector<i64>& rows = sample.rows;
      const JobMetadata& job = args.job_meta.at(sample.job_id);
      if (job.name() == base_job_name() &&
          sample.columns[0] == base_column_name()) {
        // If reading from base job and special visual data column...
        if (args.dataset.type() == DatasetType_Video) {
          // Special video column
          const VideoMetadata& metadata = args.video_meta[sample.table_id];
          if (files.size() <= media_col_idx) {
            // Open the video file for reading
            RandomReadFile* video_file;
            BACKOFF_FAIL(storage->make_random_read_file(
                dataset_item_data_path(
                    args.dataset.name(),
                    args.dataset.item_names()[sample.table_id]),
                video_file));
            files.push_back(video_file);
            u64 file_size;
            BACKOFF_FAIL(video_file->get_size(file_size));
            file_sizes.push_back(file_size);

            all_keyframe_positions.push_back(metadata.keyframe_positions());
            all_keyframe_byte_offsets.push_back(
                metadata.keyframe_byte_offsets());
            // Place end of file and num frame at end of iframe to handle edge
            // case
            all_keyframe_positions.back().push_back(metadata.frames());
            all_keyframe_byte_offsets.back().push_back(file_size);
          }
          RandomReadFile* video_file = files.at(media_col_idx);
          u64 file_size = file_sizes.at(media_col_idx);
          std::vector<i64>& keyframe_positions =
              all_keyframe_positions[media_col_idx];
          std::vector<i64>& keyframe_byte_offsets =
              all_keyframe_byte_offsets[media_col_idx];

          // Read the bytes from the file that correspond to the sequences of
          // frames we are interested in decoding. This sequence will contain
          // the bytes starting at the iframe at or preceding the first frame
          // we are interested and will continue up to the bytes before the
          // iframe at or after the last frame we are interested in.
          VideoIntervals intervals =
              slice_into_video_intervals(keyframe_positions, rows);
          size_t num_intervals = intervals.keyframe_index_intervals.size();
          for (size_t i = 0; i < num_intervals; ++i) {
            size_t start_keyframe_index;
            size_t end_keyframe_index;
            std::tie(start_keyframe_index, end_keyframe_index) =
                intervals.keyframe_index_intervals[i];

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

            i64 num_non_warmup_frames = io_item.end_row - io_item.start_row;
            i64 first_non_warmup_index =
                static_cast<i64>(rows.size()) - num_non_warmup_frames;
            for (size_t j = 0; j < intervals.valid_frames[i].size(); ++j) {
              u8* b = nullptr;
              size_t size = 0;
              if (j == 0) {
                // Encoded buffer
                b = buffer;
                size = buffer_size;
              } else if (j == first_non_warmup_index) {
                u8* non_warmup_buffer = new_buffer(CPU_DEVICE, buffer_size);
                memcpy(non_warmup_buffer, buffer, buffer_size);
                b = non_warmup_buffer;
                size = buffer_size;
              } else {
                size = 1;
                b = new_buffer(CPU_DEVICE, size);
              }
              INSERT_ROW(eval_work_entry.columns[col_idx], b, size);

              // Decode args
              DecodeArgs decode_args;
              decode_args.set_start_keyframe(
                  keyframe_positions[start_keyframe_index]);
              decode_args.set_end_keyframe(
                  keyframe_positions[end_keyframe_index]);
              decode_args.set_valid_frame(intervals.valid_frames[i][j]);

              u8* decode_args_buffer = nullptr;
              serialize_decode_args(decode_args, decode_args_buffer, size);

              INSERT_ROW(eval_work_entry.columns[col_idx + 1],
                         decode_args_buffer, size);
            }
          }
          // Jump over the next output column because we wrote two columns for
          // this iteration (frame and frame_args)
          col_idx++;
        } else if (args.dataset.type() == DatasetType_Image) {
          assert(false);
          // Special image column
          // const ImageFormatGroupMetadata& metadata =
          //     args.image_meta[work_item.video_index];
          // if (video_path != last_video_path) {
          //   if (image_file != nullptr) {
          //     delete image_file;
          //     image_file = nullptr;
          //   }
          //   image_compressed_sizes.clear();
          //   image_compressed_offsets.clear();

          //   // Open the video file for reading
          //   BACKOFF_FAIL(storage->make_random_read_file(
          //       dataset_item_data_path(args.dataset.name(), video_path),
          //       image_file));

          //   BACKOFF_FAIL(image_file->get_size(file_size));

          //   i64 s = 0;
          //   for (i64 size : metadata.compressed_sizes()) {
          //     image_compressed_sizes.push_back(size);
          //     image_compressed_offsets.push_back(s);
          //     s += size;
          //   }
          //   image_compressed_offsets.push_back(s);
          // }
          // last_video_path = video_path;

          // // Read the bytes from the file that correspond to the sequences
          // // of images we are interested in decoding.
          // JobMetadata::FrameLocations locations = args.in_job.frame_locations(
          //     args.sampling, work_item.video_index, load_work_entry);
          // std::vector<Interval>& intervals = locations.intervals;
          // std::vector<ImageDecodeArgs>& dargs = locations.image_args;
          // assert(intervals.size() == dargs.size());
          // size_t num_intervals = intervals.size();
          // for (size_t i = 0; i < num_intervals; ++i) {
          //   i32 start_frame = intervals[i].start;
          //   i32 end_frame = intervals[i].end;

          //   u64 start_byte_offset =
          //       static_cast<u64>(image_compressed_offsets[start_frame]);

          //   u64 end_byte_offset =
          //       static_cast<u64>(image_compressed_offsets[end_frame]);

          //   size_t buffer_size = end_byte_offset - start_byte_offset;

          //   u8* buffer = new_buffer(CPU_DEVICE, buffer_size);

          //   auto io_start = now();

          //   u64 pos = start_byte_offset;
          //   read(image_file, buffer, buffer_size, pos);

          //   args.profiler.add_interval("io", io_start, now());
          //   args.profiler.increment("io_read", static_cast<i64>(buffer_size));

          //   // Encoded buffer
          //   INSERT_ROW(eval_work_entry.columns[out_col], buffer, buffer_size);

          //   // Decode args
          //   ImageDecodeArgs& decode_args = dargs[i];

          //   decode_args.set_warmup_count(args.warmup_count);
          //   decode_args.set_rows_from_start(work_item.rows_from_start);
          //   decode_args.set_encoding_type(metadata.encoding_type());
          //   decode_args.set_color_space(metadata.color_space());
          //   for (i32 f = start_frame; f < end_frame; ++f) {
          //     decode_args.add_compressed_sizes(image_compressed_sizes[f]);
          //   }

          //   u8* decode_args_buffer = nullptr;
          //   size_t size;
          //   serialize_image_decode_args(decode_args, decode_args_buffer, size);

          //   INSERT_ROW(eval_work_entry.columns[out_col + 1], decode_args_buffer,
          //              size);
          // }
          // Jump over the next output column because we wrote two columns for
          // this iteration (frame and frame_args)
        } else {
          assert(false);
        }
        media_col_idx++;
        col_idx++;
      } else {
        // Regular column load
        RowIntervals intervals = slice_into_row_intervals(job, rows);
        size_t num_items = intervals.item_ids.size();
        for (const std::string& column_name : sample.columns) {
          for (size_t i = 0; i < num_items; ++i) {
            i32 item_id = intervals.item_ids[i];
            i64 item_start;
            i64 item_end;
            std::tie(item_start, item_end) = intervals.item_intervals[i];
            const std::vector<i64>& valid_offsets = intervals.valid_offsets[i];

            std::unique_ptr<RandomReadFile> file;
            StoreResult result;
            BACKOFF_FAIL(make_unique_random_read_file(
                storage,
                job_item_output_path(args.dataset.name(), job.name(),
                                     sample.table_id, column_name, item_id),
                file));

            u64 file_size = 0;
            BACKOFF_FAIL(file->get_size(file_size));

            // Read number of rows in file
            u64 pos = 0;
            u64 num_rows = read<u64>(file.get(), pos);

            // Read row sizes from work item file header
            std::vector<i64> row_sizes(num_rows);
            read(file.get(), reinterpret_cast<u8*>(row_sizes.data()),
                 row_sizes.size() * sizeof(i64), pos);

            // Determine start and end position of rows to read in file
            u64 start_offset;
            for (i64 i = 0; i < item_start; ++i) {
              start_offset += row_sizes[i];
            }
            u64 end_offset = start_offset;
            for (i64 i = item_start; i < item_end; ++i) {
              end_offset += row_sizes[i];
            }
            u64 row_data_size = end_offset - start_offset;
            std::vector<u8> row_data(row_data_size);

            // Read chunk of file corresponding to requested rows
            pos += start_offset;
            read(file.get(), row_data.data(), row_data.size(), pos);

            // Extract individual rows and insert into output work entry
            u64 offset = 0;
            size_t valid_idx = 0;
            for (i32 i = item_start; i < item_end; ++i) {
              size_t buffer_size = static_cast<size_t>(row_sizes[i]);
              if (i == valid_offsets[valid_idx]) {
                u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
                memcpy(buffer, row_data.data() + offset, buffer_size);
                INSERT_ROW(eval_work_entry.columns[col_idx], buffer,
                           buffer_size);
                valid_idx++;
              }
              offset += buffer_size;
            }
            assert(valid_idx == valid_offsets.size());
          }
          col_idx++;
        }
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
}

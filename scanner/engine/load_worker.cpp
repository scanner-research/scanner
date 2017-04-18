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

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {
namespace {

struct RowIntervals {
  std::vector<i32> item_ids;
  std::vector<std::tuple<i64, i64>> item_intervals;
  std::vector<std::vector<i64>> valid_offsets;
};

struct VideoIntervals {
  std::vector<std::tuple<size_t, size_t>> keyframe_index_intervals;
  std::vector<std::vector<i64>> valid_frames;
};

// Gets the list of work items for a sequence of rows in the job
RowIntervals slice_into_row_intervals(const TableMetadata& table,
                                      const std::vector<i64>& rows) {
  RowIntervals info;
  // Analyze rows and table to determine what item ids and offsets in them to
  // sample from
  std::vector<i64> end_rows = table.end_rows();
  auto item_from_row = [&end_rows](i64 r) -> i32 {
    i64 i = 0;
    for (; i < end_rows.size(); ++i) {
      if (r < end_rows[i]) {
        break;
      }
    }
    assert(i != end_rows.size());
    return i;
  };

  auto offset_from_row = [&end_rows](i64 r) -> i64 {
    i64 i = 0;
    i64 last_end_row = 0;
    for (; i < end_rows.size(); ++i) {
      if (r < end_rows[i]) {
        break;
      }
      last_end_row = end_rows[i];
    }
    assert(i != end_rows.size());
    return r - last_end_row;
  };

  assert(!rows.empty());
  i32 current_item = item_from_row(rows[0]);
  i64 item_start = offset_from_row(rows[0]);
  i64 item_end = item_start + 1;
  i64 prev_row = -1;
  std::vector<i64> valid_offsets;
  for (i64 row : rows) {
    i32 item = item_from_row(row);
    i64 item_offset = offset_from_row(row);
    // We check two cases:
    //   1. if the row is in a new item, then we have found all the consecutive
    //      increasing rows that will be in this item and we should move on
    //      to the next one.
    //   2. if the row we are asking for is the same as the existing row or
    //      before it, we end the current item and start back with the item
    //      for this new row, even if the item is the same as the current item.
    //      NOTE(apoms): We could fuse these together and only load the item
    //      once, but to do so requires reordering the data after it is read
    //      from disk to match the ordering requested.
    if (item != current_item || row <= prev_row) {
      // Start a new item and push the current one into the list
      info.item_ids.push_back(current_item);
      info.item_intervals.push_back(std::make_tuple(item_start, item_end));
      info.valid_offsets.push_back(valid_offsets);

      current_item = item;
      item_start = item_offset;
      item_end = item_offset + 1;
      valid_offsets.clear();
    }

    valid_offsets.push_back(item_offset);
    item_end = item_offset + 1;
    prev_row = row;
  }
  info.item_ids.push_back(current_item);
  info.item_intervals.push_back(std::make_tuple(item_start, item_end));
  info.valid_offsets.push_back(valid_offsets);

  return info;
}

VideoIntervals slice_into_video_intervals(
    const std::vector<i64>& keyframe_positions, const std::vector<i64>& rows) {
  VideoIntervals info;
  assert(keyframe_positions.size() >= 2);
  size_t start_keyframe_index = 0;
  size_t end_keyframe_index = 1;
  i64 next_keyframe = keyframe_positions[end_keyframe_index];
  std::vector<i64> valid_frames;
  for (i64 row : rows) {
    if (row >= next_keyframe) {
      assert(end_keyframe_index < keyframe_positions.size() - 1);
      next_keyframe = keyframe_positions[++end_keyframe_index];
      if (row >= next_keyframe) {
        // Skipped a keyframe, so make a new interval
        if (!valid_frames.empty()) {
          info.keyframe_index_intervals.push_back(
              std::make_tuple(start_keyframe_index, end_keyframe_index - 1));
          info.valid_frames.push_back(valid_frames);
        }

        while (row >= keyframe_positions[end_keyframe_index]) {
          end_keyframe_index++;
          assert(end_keyframe_index < keyframe_positions.size());
        }
        valid_frames.clear();
        start_keyframe_index = end_keyframe_index - 1;
        next_keyframe = keyframe_positions[end_keyframe_index];
      }
    }
    valid_frames.push_back(row);
  }
  info.keyframe_index_intervals.push_back(
      std::make_tuple(start_keyframe_index, end_keyframe_index));
  info.valid_frames.push_back(valid_frames);
  return info;
}

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

struct VideoIndexEntry {
  i32 width;
  i32 height;
  i32 channels;
  FrameType frame_type;
  proto::VideoDescriptor::VideoCodecType codec_type;
  std::unique_ptr<RandomReadFile> file;
  u64 file_size;
  std::vector<i64> keyframe_positions;
  std::vector<i64> keyframe_byte_offsets;
};

VideoIndexEntry read_video_index(storehouse::StorageBackend* storage,
                                 i32 table_id, i32 column_id, i32 item_id) {
  VideoIndexEntry index_entry;
  VideoMetadata video_meta = read_video_metadata(
      storage, VideoMetadata::descriptor_path(table_id, column_id, item_id));

  // Open the video file for reading
  index_entry.width = video_meta.width();
  index_entry.height = video_meta.height();
  index_entry.channels = video_meta.channels();
  index_entry.frame_type = video_meta.frame_type();
  index_entry.codec_type = video_meta.codec_type();

  BACKOFF_FAIL(storehouse::make_unique_random_read_file(
      storage, table_item_output_path(table_id, column_id, item_id),
      index_entry.file));
  BACKOFF_FAIL(index_entry.file->get_size(index_entry.file_size));
  index_entry.keyframe_positions = video_meta.keyframe_positions();
  index_entry.keyframe_byte_offsets = video_meta.keyframe_byte_offsets();
  // Place total frames at the end of keyframe positions and total file size
  // at the end of byte offsets to make interval calculation not need to
  // deal with edge cases surrounding those
  index_entry.keyframe_positions.push_back(video_meta.frames());
  index_entry.keyframe_byte_offsets.push_back(index_entry.file_size);

  return index_entry;
}

void read_video_column(Profiler& profiler, const VideoIndexEntry& index_entry,
                       const std::vector<i64>& rows,
                       ElementList& element_list) {
  RandomReadFile* video_file = index_entry.file.get();
  u64 file_size = index_entry.file_size;
  const std::vector<i64>& keyframe_positions = index_entry.keyframe_positions;
  const std::vector<i64>& keyframe_byte_offsets =
      index_entry.keyframe_byte_offsets;

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

    i64 start_keyframe = keyframe_positions[start_keyframe_index];
    i64 end_keyframe = keyframe_positions[end_keyframe_index];
    std::vector<i64> all_keyframes;
    for (size_t i = start_keyframe_index; i < end_keyframe_index + 1; ++i) {
      all_keyframes.push_back(keyframe_positions[i]);
    }

    std::vector<i64> all_keyframes_byte_offsets;
    for (size_t i = start_keyframe_index; i < end_keyframe_index + 1; ++i) {
      all_keyframes_byte_offsets.push_back(keyframe_byte_offsets[i] -
                                           start_keyframe_byte_offset);
    }

    size_t buffer_size = end_keyframe_byte_offset - start_keyframe_byte_offset;
    u8* buffer = new_buffer(CPU_DEVICE, buffer_size);

    auto io_start = now();

    u64 pos = start_keyframe_byte_offset;
    s_read(video_file, buffer, buffer_size, pos);

    profiler.add_interval("io", io_start, now());
    profiler.increment("io_read", static_cast<i64>(buffer_size));

    proto::DecodeArgs decode_args;
    decode_args.set_width(index_entry.width);
    decode_args.set_height(index_entry.height);
    decode_args.set_start_keyframe(keyframe_positions[start_keyframe_index]);
    decode_args.set_end_keyframe(keyframe_positions[end_keyframe_index]);
    for (i64 k : all_keyframes) {
      decode_args.add_keyframes(k);
    }
    for (i64 k : all_keyframes_byte_offsets) {
      decode_args.add_keyframe_byte_offsets(k);
    }
    for (size_t j = 0; j < intervals.valid_frames[i].size(); ++j) {
      decode_args.add_valid_frames(intervals.valid_frames[i][j]);
    }
    decode_args.set_encoded_video(buffer, buffer_size);

    size_t size = decode_args.ByteSize();
    u8* decode_args_buffer = new_buffer(CPU_DEVICE, size);
    bool result = decode_args.SerializeToArray(decode_args_buffer, size);
    assert(result);
    insert_element(element_list, decode_args_buffer, size);

    delete_buffer(CPU_DEVICE, buffer);
  }
}

void read_other_column(storehouse::StorageBackend* storage, i32 table_id,
                       i32 column_id, i32 item_id, i32 item_start, i32 item_end,
                       const std::vector<i64>& rows,
                       ElementList& element_list) {
  const std::vector<i64>& valid_offsets = rows;

  std::unique_ptr<RandomReadFile> file;
  StoreResult result;
  BACKOFF_FAIL(make_unique_random_read_file(
      storage, table_item_output_path(table_id, column_id, item_id), file));

  u64 file_size = 0;
  BACKOFF_FAIL(file->get_size(file_size));

  // Read number of elements in file
  u64 pos = 0;
  u64 num_elements = s_read<u64>(file.get(), pos);

  // Read element sizes from work item file header
  std::vector<i64> element_sizes(num_elements);
  s_read(file.get(), reinterpret_cast<u8*>(element_sizes.data()),
         element_sizes.size() * sizeof(i64), pos);

  // Determine start and end position of elements to read in file
  u64 start_offset = 0;
  for (i64 i = 0; i < item_start; ++i) {
    start_offset += element_sizes[i];
  }
  u64 end_offset = start_offset;
  for (i64 i = item_start; i < item_end; ++i) {
    end_offset += element_sizes[i];
  }
  u64 element_data_size = end_offset - start_offset;
  std::vector<u8> element_data(element_data_size);

  // Read chunk of file corresponding to requested elements
  pos += start_offset;
  s_read(file.get(), element_data.data(), element_data.size(), pos);

  // Extract individual elements and insert into output work entry
  u64 offset = 0;
  size_t valid_idx = 0;
  for (i32 i = item_start; i < item_end; ++i) {
    size_t buffer_size = static_cast<size_t>(element_sizes[i]);
    if (i == valid_offsets[valid_idx]) {
      u8* buffer = new_buffer(CPU_DEVICE, buffer_size);
      memcpy(buffer, element_data.data() + offset, buffer_size);
      insert_element(element_list, buffer, buffer_size);
      valid_idx++;
    }
    offset += buffer_size;
  }
  assert(valid_idx == valid_offsets.size());
}
}

void* load_thread(void* arg) {
  LoadThreadArgs& args = *reinterpret_cast<LoadThreadArgs*>(arg);

  auto setup_start = now();

  const i32 work_item_size = args.job_params->work_item_size();

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(args.storage_config);

  // Caching table metadata
  std::map<i32, TableMetadata> table_metadata;

  // To ammortize opening files
  i32 last_table_id = -1;
  std::map<std::tuple<i32, i32, i32>, VideoIndexEntry> index;

  args.profiler.add_interval("setup", setup_start, now());
  while (true) {
    auto idle_start = now();

    std::tuple<IOItem, LoadWorkEntry> entry;
    args.load_work.pop(entry);
    IOItem& io_item = std::get<0>(entry);
    LoadWorkEntry& load_work_entry = std::get<1>(entry);

    if (load_work_entry.io_item_index() == -1) {
      break;
    }

    VLOG(2) << "Load (N/PU: " << args.node_id << "/" << args.id
            << "): processing item " << load_work_entry.io_item_index();

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();

    const auto& samples = load_work_entry.samples();

    if (io_item.table_id() != last_table_id) {
      // Not from the same task so clear cached data
      last_table_id = io_item.table_id();
      index.clear();
    }

    EvalWorkEntry eval_work_entry;
    eval_work_entry.io_item_index = load_work_entry.io_item_index();

    // Aggregate all sample columns so we know the tuple size
    assert(!samples.empty());
    eval_work_entry.warmup_rows = samples.Get(0).warmup_rows_size();

    i32 num_columns = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
      num_columns += samples.Get(i).column_ids_size();
    }
    eval_work_entry.columns.resize(num_columns);

    i32 media_col_idx = 0;
    i32 out_col_idx = 0;
    for (const proto::LoadSample& sample : samples) {
      i32 table_id = sample.table_id();
      auto it = table_metadata.find(table_id);
      if (it == table_metadata.end()) {
        table_metadata[table_id] = read_table_metadata(
            storage, TableMetadata::descriptor_path(table_id));
        it = table_metadata.find(table_id);
      }
      const TableMetadata& table_meta = it->second;

      const google::protobuf::RepeatedField<i64>& sample_warmup_rows =
          sample.warmup_rows();
      const google::protobuf::RepeatedField<i64>& sample_rows = sample.rows();
      std::vector<i64> rows(sample_warmup_rows.begin(),
                            sample_warmup_rows.end());
      rows.insert(rows.end(), sample_rows.begin(), sample_rows.end());
      RowIntervals intervals = slice_into_row_intervals(table_meta, rows);
      size_t num_items = intervals.item_ids.size();
      for (i32 col_id : sample.column_ids()) {
        ColumnType column_type = ColumnType::Other;
        if (table_meta.column_type(col_id) == ColumnType::Video) {
          column_type = ColumnType::Video;
          // video frame column
          FrameInfo info;
          proto::VideoDescriptor::VideoCodecType encoding_type;
          for (size_t i = 0; i < num_items; ++i) {
            i32 item_id = intervals.item_ids[i];
            const std::vector<i64>& valid_offsets = intervals.valid_offsets[i];

            auto key = std::make_tuple(table_id, col_id, item_id);
            if (index.count(key) == 0) {
              index[key] = read_video_index(storage, table_id, col_id, item_id);
            }
            const VideoIndexEntry& entry = index.at(key);
            info = FrameInfo(entry.height, entry.width, entry.channels,
                             entry.frame_type);
            encoding_type = entry.codec_type;
            if (entry.codec_type == proto::VideoDescriptor::H264) {
              // Video was encoded using h264
              read_video_column(args.profiler, entry, valid_offsets,
                                eval_work_entry.columns[out_col_idx]);
            } else {
              // Video was encoded as individual images
              i32 item_id = intervals.item_ids[i];
              i64 item_start;
              i64 item_end;
              std::tie(item_start, item_end) = intervals.item_intervals[i];

              read_other_column(storage, table_id, col_id, item_id, item_start,
                                item_end, valid_offsets,
                                eval_work_entry.columns[out_col_idx]);
            }
          }
          assert(num_items > 0);
          eval_work_entry.frame_sizes.push_back(info);
          eval_work_entry.video_encoding_type.push_back(encoding_type);
          media_col_idx++;
        } else {
          // regular column
          for (size_t i = 0; i < num_items; ++i) {
            i32 item_id = intervals.item_ids[i];
            i64 item_start;
            i64 item_end;
            std::tie(item_start, item_end) = intervals.item_intervals[i];
            const std::vector<i64>& valid_offsets = intervals.valid_offsets[i];

            read_other_column(storage, table_id, col_id, item_id, item_start,
                              item_end, valid_offsets,
                              eval_work_entry.columns[out_col_idx]);
          }
        }
        eval_work_entry.column_types.push_back(column_type);
        eval_work_entry.column_handles.push_back(CPU_DEVICE);
        out_col_idx++;
      }
    }

    args.profiler.add_interval("task", work_start, now());

    args.eval_work.push(std::make_tuple(io_item, eval_work_entry));
  }

  VLOG(1) << "Load (N/PU: " << args.node_id << "/" << args.id
          << "): thread finished";

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}
}
}

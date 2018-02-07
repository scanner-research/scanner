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
  std::vector<i64> item_start_offsets;
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
      info.item_start_offsets.push_back(current_item == 0 ? 0 :
                                        end_rows[current_item - 1]);
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
  info.item_start_offsets.push_back(
      current_item == 0 ? 0 : end_rows[current_item - 1]);
  info.item_intervals.push_back(std::make_tuple(item_start, item_end));
  info.valid_offsets.push_back(valid_offsets);

  return info;
}

VideoIntervals slice_into_video_intervals(
    const std::vector<u64>& keyframe_positions,
    const std::vector<u64>& sample_offsets,
    const std::vector<u64>& sample_sizes,
    const std::vector<i64>& rows) {
  VideoIntervals info;
  assert(keyframe_positions.size() >= 2);
  size_t start_keyframe_index = 0;
  size_t end_keyframe_index = 1;
  i64 next_keyframe = keyframe_positions[end_keyframe_index];
  std::vector<i64> valid_frames;

  i64 prev_row = 0;
  if (rows.size() > 0) {
    prev_row = rows[0];
  }
  for (i64 row : rows) {
    if (row >= next_keyframe) {
      // Check if this keyframe is adjacent
      uint64_t last_endpoint = sample_offsets.at(next_keyframe - 1) +
                               sample_sizes.at(next_keyframe - 1);
      bool is_adjacent =
          (last_endpoint == sample_offsets.at(next_keyframe));

      assert(end_keyframe_index < keyframe_positions.size() - 1);
      i64 prev_prev_keyframe = keyframe_positions[end_keyframe_index - 1];
      i64 prev_keyframe = next_keyframe;
      next_keyframe = keyframe_positions[++end_keyframe_index];
      uint64_t keyframe_interval = next_keyframe - prev_keyframe;
      // If the interval between the previous row and the current row is larger
      // than half the keyframe interval or if they are exactly equal to the
      // keyframes, then we should make a new interval so that we can avoid
      // decoding everything in between
      // bool large_distance = (row - prev_row) > (keyframe_interval / 2);
      // bool exactly_keyframes =
      //     (row == prev_keyframe) && (prev_row == prev_prev_keyframe);
      // if (exactly_keyframes || large_distance || row >= next_keyframe ||
      //     !is_adjacent) {
      if (row >= next_keyframe || !is_adjacent) {
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
    prev_row = row;
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
}

LoadWorker::LoadWorker(const LoadWorkerArgs& args)
  : node_id_(args.node_id),
    worker_id_(args.worker_id),
    profiler_(args.profiler),
    table_metadata_(args.table_metadata),
    load_sparsity_threshold_(args.load_sparsity_threshold),
    io_packet_size_(args.io_packet_size),
    work_packet_size_(args.work_packet_size) {
  storage_.reset(
      storehouse::StorageBackend::make_from_config(args.storage_config));
  meta_ = read_database_metadata(storage_.get(),
                                 DatabaseMetadata::descriptor_path());
}

void LoadWorker::feed(LoadWorkEntry& input_entry) {
  LoadWorkEntry& load_work_entry = input_entry;

  if (load_work_entry.table_id() != last_table_id_) {
    // Not from the same task so clear cached data
    last_table_id_ = load_work_entry.table_id();
    index_.clear();
  }

  entry_ = input_entry;
  current_row_ = 0;
  total_rows_ = 0;
  for (auto& sample : load_work_entry.samples()) {
    total_rows_ = std::max((i64)sample.input_row_ids_size(), total_rows_);
  }
}

bool LoadWorker::yield(i32 item_size,
                       EvalWorkEntry& output_entry) {
  LoadWorkEntry& load_work_entry = entry_;

  // Ignoring item size for now and just yielding one IO item at a time
  if (current_row_ >= total_rows_) {
    return false;
  }

  EvalWorkEntry eval_work_entry;
  eval_work_entry.table_id = load_work_entry.table_id();
  eval_work_entry.job_index = load_work_entry.job_index();
  eval_work_entry.task_index = load_work_entry.task_index();

  const auto& samples = load_work_entry.samples();
  assert(!samples.empty());

  // Aggregate all sample columns so we know the tuple size
  i32 num_columns = samples.size();
  eval_work_entry.columns.resize(num_columns);

  // For each sample, insert the row ids and read the rows from disk
  // NOTE(apoms): if the requested rows are different for each column,
  // some of the output work entries will have an uneven number of rows
  i32 media_col_idx = 0;
  i32 out_col_idx = 0;
  for (const proto::LoadSample& sample : samples) {
    i32 table_id = sample.table_id();
    const TableMetadata& table_meta = table_metadata_.at(table_id);

    i64 total_rows = sample.input_row_ids_size();
    i64 row_start = current_row_;
    i64 row_end = std::min(current_row_ + item_size, total_rows);

    const auto& sample_rows = sample.input_row_ids();
    const auto& output_row_ids = sample.output_row_ids();
    std::vector<i64> rows(sample_rows.begin() + row_start,
                          sample_rows.begin() + row_end);
    std::vector<i64> output_rows(output_row_ids.begin() + row_start,
                                 output_row_ids.begin() + row_end);
    eval_work_entry.row_ids.push_back(output_rows);

    RowIntervals intervals = slice_into_row_intervals(table_meta, rows);
    size_t num_items = intervals.item_ids.size();
    i32 col_id = sample.column_id();

    ColumnType column_type = ColumnType::Other;
    if (table_meta.column_type(col_id) == ColumnType::Video) {
      column_type = ColumnType::Video;
      // video frame column
      FrameInfo info;
      proto::VideoDescriptor::VideoCodecType encoding_type;
      bool inplace = false;
      for (size_t i = 0; i < num_items; ++i) {
        i32 item_id = intervals.item_ids[i];
        i64 item_start_row = intervals.item_start_offsets[i];
        const std::vector<i64>& valid_offsets = intervals.valid_offsets[i];

        auto key = std::make_tuple(table_id, col_id, item_id);
        if (index_.count(key) == 0) {
          index_[key] =
              read_video_index(storage_.get(), table_id, col_id, item_id);
        }
        const VideoIndexEntry& entry = index_.at(key);
        inplace = entry.inplace;
        info = FrameInfo(entry.height, entry.width, entry.channels,
                         entry.frame_type);
        encoding_type = entry.codec_type;
        if (entry.codec_type == proto::VideoDescriptor::H264) {
          // Video was encoded using h264
          read_video_column(profiler_, entry, valid_offsets, item_start_row,
                            eval_work_entry.columns[out_col_idx]);
        } else {
          // Video was encoded as individual images
          i32 item_id = intervals.item_ids[i];
          i64 item_start;
          i64 item_end;
          std::tie(item_start, item_end) = intervals.item_intervals[i];

          read_other_column(table_id, col_id, item_id, item_start, item_end,
                            valid_offsets,
                            eval_work_entry.columns[out_col_idx]);
        }
      }
      if (num_items == 0) {
        eval_work_entry.frame_sizes.emplace_back();
        eval_work_entry.video_encoding_type.emplace_back();
        eval_work_entry.inplace_video.push_back(false);
      } else {
        eval_work_entry.frame_sizes.push_back(info);
        eval_work_entry.video_encoding_type.push_back(encoding_type);
        eval_work_entry.inplace_video.push_back(inplace);
      }
      media_col_idx++;
    } else {
      // regular column
      for (size_t i = 0; i < num_items; ++i) {
        i32 item_id = intervals.item_ids[i];
        i64 item_start;
        i64 item_end;
        std::tie(item_start, item_end) = intervals.item_intervals[i];
        const std::vector<i64>& valid_offsets = intervals.valid_offsets[i];

        read_other_column(table_id, col_id, item_id, item_start, item_end,
                          valid_offsets, eval_work_entry.columns[out_col_idx]);
      }
      eval_work_entry.inplace_video.push_back(false);
    }
    eval_work_entry.column_types.push_back(column_type);
    eval_work_entry.column_handles.push_back(CPU_DEVICE);
    out_col_idx++;
  }

  output_entry = eval_work_entry;

  current_row_ += item_size;

  return true;
}

bool LoadWorker::done() { return current_row_ >= total_rows_; }

void read_video_column(Profiler& profiler, const VideoIndexEntry& index_entry,
                       const std::vector<i64>& rows, i64 start_frame,
                       Elements& element_list) {
  std::unique_ptr<RandomReadFile> video_file = index_entry.open_file();
  u64 file_size = index_entry.file_size;
  const std::vector<u64>& keyframe_indices = index_entry.keyframe_indices;
  const std::vector<u64>& sample_offsets = index_entry.sample_offsets;
  const std::vector<u64>& sample_sizes = index_entry.sample_sizes;

  // Read the bytes from the file that correspond to the sequences of
  // frames we are interested in decoding. This sequence will contain
  // the bytes starting at the first iframe at or preceding the first frame
  // we are interested and will continue up to the bytes before the
  // first iframe at or after the last frame we are interested in.
  VideoIntervals intervals = slice_into_video_intervals(
      keyframe_indices, sample_offsets, sample_sizes, rows);
  size_t num_intervals = intervals.keyframe_index_intervals.size();
  for (size_t i = 0; i < num_intervals; ++i) {
    size_t start_keyframe_index;
    size_t end_keyframe_index;
    std::tie(start_keyframe_index, end_keyframe_index) =
        intervals.keyframe_index_intervals[i];

    i64 start_keyframe = keyframe_indices[start_keyframe_index];
    i64 end_keyframe = keyframe_indices[end_keyframe_index];

    u64 start_keyframe_byte_offset =
        static_cast<u64>(sample_offsets[start_keyframe]);
    u64 end_keyframe_byte_offset =
        static_cast<u64>(sample_offsets[end_keyframe]);

    std::vector<i64> all_keyframes;
    std::vector<i64> all_keyframe_indices;
    for (size_t i = start_keyframe_index; i <= end_keyframe_index; ++i) {
      all_keyframes.push_back(keyframe_indices[i]);
      all_keyframe_indices.push_back(keyframe_indices[i] - keyframe_indices[0]);
    }

    std::vector<u64> all_offsets;
    std::vector<u64> all_sizes;
    for (size_t i = start_keyframe; i <= end_keyframe; ++i) {
      all_offsets.push_back(sample_offsets[i] - start_keyframe_byte_offset);
      all_sizes.push_back(sample_sizes[i]);
    }

    size_t buffer_size = end_keyframe_byte_offset - start_keyframe_byte_offset;
    u8* buffer = new_buffer(CPU_DEVICE, buffer_size);

    auto io_start = now();

    u64 pos = start_keyframe_byte_offset;
    size_t size_read;
    storehouse::StoreResult r =
        video_file->read(pos, buffer_size, buffer, size_read);
    //s_read(video_file.get(), buffer, buffer_size, pos);

    profiler.add_interval("io", io_start, now());
    profiler.increment("io_read", static_cast<i64>(buffer_size));

    proto::DecodeArgs decode_args;
    decode_args.set_width(index_entry.width);
    decode_args.set_height(index_entry.height);
    // We add the start frame of this item to all frames since the decoder
    // works in terms of absolute frame numbers, instead of item relative
    // frame numbers
    decode_args.set_start_keyframe(keyframe_indices[start_keyframe_index] +
                                   start_frame);
    decode_args.set_end_keyframe(keyframe_indices[end_keyframe_index] +
                                 start_frame);
    for (i64 k : all_keyframes) {
      decode_args.add_keyframes(k + start_frame);
    }
    for (i64 k : all_keyframe_indices) {
      decode_args.add_keyframe_indices(k);
    }
    for (u64 k : all_offsets) {
      decode_args.add_sample_offsets(k);
    }
    for (u64 k : all_sizes) {
      decode_args.add_sample_sizes(k);
    }
    for (size_t j = 0; j < intervals.valid_frames[i].size(); ++j) {
      decode_args.add_valid_frames(intervals.valid_frames[i][j] + start_frame);
    }
    decode_args.set_encoded_video((i64)buffer);
    decode_args.set_encoded_video_size(buffer_size);
    decode_args.set_metadata(index_entry.metadata.data(),
                             index_entry.metadata.size());

    size_t size = decode_args.ByteSizeLong();
    u8* decode_args_buffer = new_buffer(CPU_DEVICE, size);
    bool result = decode_args.SerializeToArray(decode_args_buffer, size);
    assert(result);
    insert_element(element_list, decode_args_buffer, size);
  }
}

void LoadWorker::read_other_column(i32 table_id, i32 column_id, i32 item_id,
                                   i32 item_start, i32 item_end,
                                   const std::vector<i64>& rows,
                                   Elements& element_list) {
  const std::vector<i64>& valid_offsets = rows;

  // Read metadata file to determine num rows and sizes
  u64 num_elements = 0;
  std::vector<i64> element_sizes;
  {
    std::unique_ptr<RandomReadFile> file;
    BACKOFF_FAIL(make_unique_random_read_file(
        storage_.get(), table_item_metadata_path(table_id, column_id, item_id),
        file));

    u64 file_size = 0;
    BACKOFF_FAIL(file->get_size(file_size));

    // Read number of elements in file
    u64 pos = 0;
    while (pos < file_size) {
      u64 elements = s_read<u64>(file.get(), pos);

      // Read element sizes from work item file header
      size_t prev_size = element_sizes.size();
      element_sizes.resize(prev_size + elements);
      s_read(file.get(),
             reinterpret_cast<u8*>(element_sizes.data() + prev_size),
             elements * sizeof(i64), pos);

      num_elements += elements;
    }
    assert(pos == file_size);
  }

  std::unique_ptr<RandomReadFile> file;
  BACKOFF_FAIL(make_unique_random_read_file(
      storage_.get(), table_item_output_path(table_id, column_id, item_id),
      file));

  u64 file_size = 0;
  BACKOFF_FAIL(file->get_size(file_size));

  u64 pos = 0;
  // Determine start and end position of elements to read in file
  u64 start_offset = 0;
  assert(item_start <= element_sizes.size());
  for (i64 i = 0; i < item_start; ++i) {
    start_offset += element_sizes[i];
  }
  u64 end_offset = start_offset;
  assert(item_end <= element_sizes.size());
  for (i64 i = item_start; i < item_end; ++i) {
    end_offset += element_sizes[i];
  }

  size_t total_size = 0;
  for (i32 row : rows) {
    total_size += static_cast<size_t>(element_sizes[row]);
  }
  u8* block_buffer = new_block_buffer(CPU_DEVICE, total_size, rows.size());

  // If the requested elements are sufficiently sparse by some threshold, we
  // read each element individually. Otherwise, we read the entire block and
  // copy out only the necessary elements.
  if ((item_end - item_start) / rows.size() >= load_sparsity_threshold_) {

    for (i32 row : rows) {
      size_t buffer_size = static_cast<size_t>(element_sizes[row]);
      u8* buffer = block_buffer;
      u64 row_offset = pos + start_offset;
      for (i32 i = item_start; i < row; ++i) {
        row_offset += element_sizes[i];
      }
      s_read(file.get(), buffer, buffer_size, row_offset);
      insert_element(element_list, buffer, buffer_size);
      block_buffer += buffer_size;
    }
  } else {
    pos += start_offset;

    u64 element_data_size = end_offset - start_offset;
    std::vector<u8> element_data(element_data_size);

    // Read chunk of file corresponding to requested elements
    s_read(file.get(), element_data.data(), element_data.size(), pos);

    // Extract individual elements and insert into output work entry
    u64 offset = 0;
    size_t valid_idx = 0;
    for (i32 i = item_start; i < item_end; ++i) {
      size_t buffer_size = static_cast<size_t>(element_sizes[i]);
      if (i == valid_offsets[valid_idx]) {
        u8* buffer = block_buffer;
        memcpy(buffer, element_data.data() + offset, buffer_size);
        insert_element(element_list, buffer, buffer_size);
        valid_idx++;
        block_buffer += buffer_size;
      }
      offset += buffer_size;
    }
    assert(valid_idx == valid_offsets.size());
  }
}
}
}

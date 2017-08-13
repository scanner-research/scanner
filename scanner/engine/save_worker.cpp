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

#include "scanner/engine/save_worker.h"

#include "scanner/engine/metadata.h"
#include "scanner/util/common.h"
#include "scanner/util/storehouse.h"
#include "scanner/video/h264_byte_stream_index_creator.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {

SaveWorker::SaveWorker(const SaveWorkerArgs& args)
    : node_id_(args.node_id), worker_id_(args.worker_id), profiler_(args.profiler) {
  auto setup_start = now();
  // Setup a distinct storage backend for each IO thread
  storage_.reset(
      storehouse::StorageBackend::make_from_config(args.storage_config));

  args.profiler.add_interval("setup", setup_start, now());

}

SaveWorker::~SaveWorker() {
  for (auto& file : output_) {
    file->save();
  }
  for (auto& file : output_metadata_) {
    file->save();
  }
  for (auto& meta : video_metadata_) {
    write_video_metadata(storage_.get(), meta);
  }
  output_.clear();
  output_metadata_.clear();
  video_metadata_.clear();
}

void SaveWorker::feed(std::tuple<IOItem, EvalWorkEntry>& input_entry) {
  IOItem& io_item = std::get<0>(input_entry);
  EvalWorkEntry& work_entry = std::get<1>(input_entry);

  // Write out each output column to an individual data file
  i32 video_col_idx = 0;
  for (size_t out_idx = 0; out_idx < work_entry.columns.size(); ++out_idx) {
    u64 num_elements = static_cast<u64>(work_entry.columns[out_idx].size());

    auto io_start = now();

    WriteFile* output_file = output_.at(out_idx).get();
    WriteFile* output_metadata_file = output_metadata_.at(out_idx).get();

    if (work_entry.columns[out_idx].size() != num_elements) {
      LOG(FATAL) << "Output layer's element vector has wrong length";
    }

    // Ensure the data is on the CPU
    move_if_different_address_space(profiler_,
                                    work_entry.column_handles[out_idx],
                                    CPU_DEVICE, work_entry.columns[out_idx]);

    bool compressed = work_entry.compressed[out_idx];
    // If this is a video...
    i64 size_written = 0;
    if (work_entry.column_types[out_idx] == ColumnType::Video) {
      // Read frame info column
      assert(work_entry.columns[out_idx].size() > 0);
      FrameInfo frame_info = work_entry.frame_sizes[video_col_idx];

      // Create index column
      VideoMetadata& video_meta = video_metadata_[video_col_idx];
      proto::VideoDescriptor& video_descriptor = video_meta.get_descriptor();

      video_descriptor.set_width(frame_info.width());
      video_descriptor.set_height(frame_info.height());
      video_descriptor.set_channels(frame_info.channels());
      video_descriptor.set_frame_type(frame_info.type);

      video_descriptor.set_time_base_num(1);
      video_descriptor.set_time_base_denom(25);

      video_descriptor.set_num_encoded_videos(
          video_descriptor.num_encoded_videos() + 1);

      if (compressed && frame_info.type == FrameType::U8 &&
          frame_info.channels() == 3) {
        H264ByteStreamIndexCreator index_creator(output_file);
        for (size_t i = 0; i < num_elements; ++i) {
          Element& element = work_entry.columns[out_idx][i];
          if (!index_creator.feed_packet(element.buffer, element.size)) {
            LOG(FATAL) << "Error in save worker h264 index creator: "
                       << index_creator.error_message();
          }
          size_written += element.size;
        }

        i64 frame = index_creator.frames();
        i32 num_non_ref_frames = index_creator.num_non_ref_frames();
        const std::vector<u8>& metadata_bytes = index_creator.metadata_bytes();
        const std::vector<i64>& keyframe_positions =
            index_creator.keyframe_positions();
        const std::vector<i64>& keyframe_timestamps =
            index_creator.keyframe_timestamps();
        const std::vector<i64>& keyframe_byte_offsets =
            index_creator.keyframe_byte_offsets();

        video_descriptor.set_chroma_format(proto::VideoDescriptor::YUV_420);
        video_descriptor.set_codec_type(proto::VideoDescriptor::H264);

        video_descriptor.set_frames(video_descriptor.frames() + frame);
        video_descriptor.add_frames_per_video(frame);
        video_descriptor.add_keyframes_per_video(keyframe_positions.size());
        video_descriptor.add_size_per_video(index_creator.bytestream_pos());
        video_descriptor.set_metadata_packets(metadata_bytes.data(),
                                              metadata_bytes.size());

        for (i64 v : keyframe_positions) {
          video_descriptor.add_keyframe_positions(v);
        }
        for (i64 v : keyframe_timestamps) {
          video_descriptor.add_keyframe_timestamps(v);
        }
        for (i64 v : keyframe_byte_offsets) {
          video_descriptor.add_keyframe_byte_offsets(v);
        }
      } else {
        // Non h264 compressible video column
        video_descriptor.set_codec_type(proto::VideoDescriptor::RAW);
        // Need to specify but not used for this type
        video_descriptor.set_chroma_format(proto::VideoDescriptor::YUV_420);
        video_descriptor.set_frames(video_descriptor.frames() + num_elements);

        // Write number of elements in the file
        s_write(output_metadata_file, num_elements);
        // Write out all output sizes first so we can easily index into the
        // file
        for (size_t i = 0; i < num_elements; ++i) {
          Frame* frame = work_entry.columns[out_idx][i].as_frame();
          u64 buffer_size = frame->size();
          s_write(output_metadata_file, buffer_size);
          size_written += sizeof(u64);
        }
        // Write actual output data
        for (size_t i = 0; i < num_elements; ++i) {
          Frame* frame = work_entry.columns[out_idx][i].as_frame();
          i64 buffer_size = frame->size();
          u8* buffer = frame->data;
          s_write(output_file, buffer, buffer_size);
          size_written += buffer_size;
        }
      }

      video_col_idx++;
    } else {
      // Write number of elements in the file
      s_write(output_metadata_file, num_elements);
      // Write out all output sizes to metadata file  so we can easily index into the data file
      for (size_t i = 0; i < num_elements; ++i) {
        u64 buffer_size = work_entry.columns[out_idx][i].size;
        s_write(output_metadata_file, buffer_size);
        size_written += sizeof(u64);
      }
      // Write actual output data
      for (size_t i = 0; i < num_elements; ++i) {
        i64 buffer_size = work_entry.columns[out_idx][i].size;
        u8* buffer = work_entry.columns[out_idx][i].buffer;
        s_write(output_file, buffer, buffer_size);
        size_written += buffer_size;
      }
    }

    // TODO(apoms): For now, all evaluators are expected to return CPU
    //   buffers as output so just assume CPU
    for (size_t i = 0; i < num_elements; ++i) {
      delete_element(CPU_DEVICE, work_entry.columns[out_idx][i]);
    }

    profiler_.add_interval("io", io_start, now());
    profiler_.increment("io_write", size_written);
  }
}

void SaveWorker::new_task(IOItem item, std::vector<ColumnType> column_types) {
  auto io_start = now();
  for (auto& file : output_) {
    file->save();
  }
  for (auto& file : output_metadata_) {
    file->save();
  }
  for (auto& meta : video_metadata_) {
    write_video_metadata(storage_.get(), meta);
  }
  output_.clear();
  output_metadata_.clear();
  video_metadata_.clear();

  profiler_.add_interval("io", io_start, now());

  for (size_t out_idx = 0; out_idx < column_types.size(); ++out_idx) {
    const std::string output_path =
        table_item_output_path(item.table_id(), out_idx, item.item_id());
    const std::string output_metdata_path = table_item_metadata_path(
        item.table_id(), out_idx, item.item_id());

    WriteFile* output_file = nullptr;
    BACKOFF_FAIL(storage_->make_write_file(output_path, output_file));
    output_.emplace_back(output_file);

    WriteFile* output_metadata_file = nullptr;
    BACKOFF_FAIL(
        storage_->make_write_file(output_metdata_path, output_metadata_file));
    output_metadata_.emplace_back(output_metadata_file);

    if (column_types[out_idx] == ColumnType::Video) {
      video_metadata_.emplace_back();

      VideoMetadata& video_meta = video_metadata_.back();
      proto::VideoDescriptor& video_descriptor = video_meta.get_descriptor();
      video_descriptor.set_table_id(item.table_id());
      video_descriptor.set_column_id(out_idx);
      video_descriptor.set_item_id(item.item_id());
    }
  }
}
}
}

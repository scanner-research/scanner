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
#include "scanner/video/h264_byte_stream_index_creator.h"
#include "scanner/util/common.h"
#include "scanner/util/storehouse.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;

namespace scanner {
namespace internal {

void* save_thread(void* arg) {
  SaveThreadArgs& args = *reinterpret_cast<SaveThreadArgs*>(arg);

  auto setup_start = now();

  // Setup a distinct storage backend for each IO thread
  storehouse::StorageBackend* storage =
      storehouse::StorageBackend::make_from_config(args.storage_config);

  args.profiler.add_interval("setup", setup_start, now());

  while (true) {
    auto idle_start = now();

    std::tuple<IOItem, EvalWorkEntry> entry;
    args.input_work.pop(entry);
    IOItem& io_item = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    if (work_entry.io_item_index == -1) {
      break;
    }

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.id
            << "): processing item " << work_entry.io_item_index;

    args.profiler.add_interval("idle", idle_start, now());

    auto work_start = now();


    // Write out each output column to an individual data file
    for (size_t out_idx = 0; out_idx < work_entry.columns.size(); ++out_idx) {
      u64 num_elements = static_cast<u64>(work_entry.columns[out_idx].size());

      const std::string output_path = table_item_output_path(
          io_item.table_id(), out_idx, io_item.item_id());

      auto io_start = now();

      WriteFile* output_file = nullptr;
      BACKOFF_FAIL(storage->make_write_file(output_path, output_file));

      if (work_entry.columns[out_idx].size() != num_elements) {
        LOG(FATAL) << "Output layer's element vector has wrong length";
      }

      if (!work_entry.column_handles[out_idx].is_same_address_space(
              CPU_DEVICE)) {
        std::vector<u8*> dest_buffers, src_buffers;
        std::vector<size_t> sizes;
        size_t total_size = 0;
        for (i32 f = 0; f < num_elements; ++f) {
          Element& element = work_entry.columns[out_idx][f];
          total_size += element.size;
        }

        if (num_elements > 0) {
          u8* output_block = new_block_buffer(CPU_DEVICE, total_size, num_elements);
          for (i32 f = 0; f < num_elements; ++f) {
            Element& element = work_entry.columns[out_idx][f];
            size_t size = element.size;
            u8* src_buffer = element.buffer;
            u8* dest_buffer = output_block;

            dest_buffers.push_back(dest_buffer);
            src_buffers.push_back(src_buffer);
            sizes.push_back(size);

            output_block += size;
          }

          memcpy_vec(dest_buffers, CPU_DEVICE, src_buffers,
                     work_entry.column_handles[out_idx], sizes);

          for (i32 f = 0; f < num_elements; ++f) {
            delete_buffer(work_entry.column_handles[out_idx], src_buffers[f]);
            work_entry.columns[out_idx][f].buffer = dest_buffers[f];
          }
        }
      }

      // If this is a video...
      i64 size_written = 0;
      if (work_entry.column_types[out_idx] == ColumnType::Video) {
        // Read frame info column
        FrameInfo frame_info =
          work_entry.columns[out_idx][0].as_frame()->as_frame_info();

        H264ByteStreamIndexCreator index_creator(output_file);
        for (size_t i = 0; i < num_elements; ++i) {
          i64 buffer_size = work_entry.columns[out_idx][i].size;
          u8* buffer = work_entry.columns[out_idx][i].buffer;
          if (!index_creator.feed_packet(buffer, buffer_size)) {
            LOG(FATAL) << "Error in save worker h264 index creator: "
                       << index_creator.error_message();
          }
          size_written += buffer_size;
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

        // Create index column
        VideoMetadata video_meta;
        proto::VideoDescriptor& video_descriptor = video_meta.get_descriptor();
        video_descriptor.set_table_id(io_item.table_id());
        video_descriptor.set_column_id(out_idx);
        video_descriptor.set_item_id(io_item.item_id());

        assert(frame_info.channels() == 3);
        assert(frame_info.type == FrameType::U8);
        video_descriptor.set_width(frame_info.width());
        video_descriptor.set_height(frame_info.height());
        video_descriptor.set_chroma_format(proto::VideoDescriptor::YUV_420);
        video_descriptor.set_codec_type(proto::VideoDescriptor::H264);

        video_descriptor.set_frames(frame);
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

        // Save our metadata for the frame column
        write_video_metadata(storage, video_meta);
      } else {
        // Write number of elements in the file
        s_write(output_file, num_elements);
        // Write out all output sizes first so we can easily index into the file
        for (size_t i = 0; i < num_elements; ++i) {
          i64 buffer_size = work_entry.columns[out_idx][i].size;
          s_write(output_file, buffer_size);
          size_written += sizeof(i64);
        }
        // Write actual output data
        for (size_t i = 0; i < num_elements; ++i) {
          i64 buffer_size = work_entry.columns[out_idx][i].size;
          u8* buffer = work_entry.columns[out_idx][i].buffer;
          s_write(output_file, buffer, buffer_size);
          size_written += buffer_size;
        }
      }

      BACKOFF_FAIL(output_file->save());

      // TODO(apoms): For now, all evaluators are expected to return CPU
      //   buffers as output so just assume CPU
      for (size_t i = 0; i < num_elements; ++i) {
        delete_buffer(CPU_DEVICE, work_entry.columns[out_idx][i].buffer);
      }

      delete output_file;

      args.profiler.add_interval("io", io_start, now());
      args.profiler.increment("io_write", size_written);
    }

    VLOG(2) << "Save (N/KI: " << args.node_id << "/" << args.id
            << "): finished item " << work_entry.io_item_index;

    args.profiler.add_interval("task", work_start, now());

    args.retired_items++;
  }

  VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.id
          << "): thread finished ";

  // Cleanup
  delete storage;

  THREAD_RETURN_SUCCESS();
}
}
}

/* Copyright 2018 Carnegie Mellon University
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

#include "scanner/engine/column_sink.h"
#include "scanner/engine/metadata.h"
#include "scanner/sink_args.pb.h"
#include "scanner/engine/video_index_entry.h"
#include "scanner/video/h264_byte_stream_index_creator.h"

#include "storehouse/storage_backend.h"

#include <glog/logging.h>
#include <vector>

using storehouse::StorageBackend;
using storehouse::StorageConfig;
using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;


namespace scanner {
namespace internal {

// FIXME(apoms): This should be a configuration option
const i32 MAX_SINK_THREADS = 16;

ColumnSink::ColumnSink(const SinkConfig& config)
  : Sink(config),
    thread_pool_(MAX_SINK_THREADS) {
  // Deserialize ColumnSinkConfig
  scanner::proto::ColumnSinkArgs args;
  bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
  if (!parsed || config.args.size() == 0) {
    RESULT_ERROR(&valid_, "Could not parse ColumnSinkArgs");
    return;
  }

  // Setup storagebackend using config arguments
  std::map<std::string, std::string> storage_args;
  storage_args["bucket"] = args.bucket();
  storage_args["region"] = args.region();
  storage_args["endpoint"] = args.endpoint();
  StorageConfig* sc_config = StorageConfig::make_config(args.storage_type(), storage_args);
  if (sc_config == nullptr) {
    LOG(FATAL) << "Invalid storage config";
  }
  storage_.reset(storehouse::StorageBackend::make_from_config(sc_config));
  assert(storage_.get());
}

ColumnSink::~ColumnSink() {
}

void ColumnSink::new_stream(const std::vector<u8>& args) {
  // This space intentionally left blank
}

void ColumnSink::write(const BatchedElements& input_columns) {
  // Write out each output column to an individual data file
  auto write_start = now();
  i32 video_col_idx = 0;
  for (size_t out_idx = 0; out_idx < input_columns.size(); ++out_idx) {
    u64 num_elements = static_cast<u64>(input_columns[out_idx].size());

    auto io_start = now();

    WriteFile* output_file = output_.at(out_idx).get();
    WriteFile* output_metadata_file = output_metadata_.at(out_idx).get();

    // If this is a video...
    i64 size_written = 0;
    if (column_types_[out_idx] == ColumnType::Video) {
      // Read frame info column
      assert(input_columns[out_idx].size() > 0);
      FrameInfo& frame_info = frame_info_[out_idx];

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

      if (compressed_[out_idx] && frame_info.type == FrameType::U8 &&
          frame_info.channels() == 3) {
        H264ByteStreamIndexCreator index_creator(output_file);
        for (size_t i = 0; i < num_elements; ++i) {
          const Element& element = input_columns[out_idx][i];
          if (!index_creator.feed_packet(element.buffer, element.size)) {
            LOG(FATAL) << "Error in save worker h264 index creator: "
                       << index_creator.error_message();
          }
          size_written += element.size;
        }

        i64 frame = index_creator.frames();
        i32 num_non_ref_frames = index_creator.num_non_ref_frames();
        const std::vector<u8>& metadata_bytes = index_creator.metadata_bytes();
        const std::vector<u64>& keyframe_indices =
            index_creator.keyframe_indices();
        const std::vector<u64>& sample_offsets =
            index_creator.sample_offsets();
        const std::vector<u64>& sample_sizes =
            index_creator.sample_sizes();

        video_descriptor.set_chroma_format(proto::VideoDescriptor::YUV_420);
        video_descriptor.set_codec_type(proto::VideoDescriptor::H264);

        video_descriptor.set_frames(video_descriptor.frames() + frame);
        video_descriptor.add_frames_per_video(frame);
        video_descriptor.add_keyframes_per_video(keyframe_indices.size());
        video_descriptor.add_size_per_video(index_creator.bytestream_pos());
        video_descriptor.set_metadata_packets(metadata_bytes.data(),
                                              metadata_bytes.size());

        const std::string output_path =
            table_item_output_path(video_descriptor.table_id(), out_idx,
                                   video_descriptor.item_id());
        video_descriptor.set_data_path(output_path);
        video_descriptor.set_inplace(false);

        for (u64 v : keyframe_indices) {
          video_descriptor.add_keyframe_indices(v);
        }
        for (u64 v : sample_offsets) {
          video_descriptor.add_sample_offsets(v);
        }
        for (u64 v : sample_sizes) {
          video_descriptor.add_sample_sizes(v);
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
          const Frame* frame = input_columns[out_idx][i].as_const_frame();
          u64 buffer_size = frame->size();
          s_write(output_metadata_file, buffer_size);
          size_written += sizeof(u64);
        }
        // Write actual output data
        for (size_t i = 0; i < num_elements; ++i) {
          const Frame* frame = input_columns[out_idx][i].as_const_frame();
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
        u64 buffer_size = input_columns[out_idx][i].size;
        s_write(output_metadata_file, buffer_size);
        size_written += sizeof(u64);
      }
      // Write actual output data
      for (size_t i = 0; i < num_elements; ++i) {
        i64 buffer_size = input_columns[out_idx][i].size;
        u8* buffer = input_columns[out_idx][i].buffer;
        s_write(output_file, buffer, buffer_size);
        size_written += buffer_size;
      }
    }

    profiler_->increment("io_write", size_written);
  }
  profiler_->add_interval("column_sink:write", write_start, now());
}

void ColumnSink::new_task(i32 table_id, i32 task_id,
                          std::vector<ColumnType> column_types) {
  output_.clear();
  output_metadata_.clear();
  video_metadata_.clear();

  column_types_ = column_types;
  for (size_t out_idx = 0; out_idx < column_types.size(); ++out_idx) {
    const std::string output_path =
        table_item_output_path(table_id, out_idx, task_id);
    const std::string output_metadata_path =
        table_item_metadata_path(table_id, out_idx, task_id);

    WriteFile* output_file = nullptr;
    BACKOFF_FAIL(storage_->make_write_file(output_path, output_file),
                 "while trying to make write file for " + output_path);
    output_.emplace_back(output_file);

    WriteFile* output_metadata_file = nullptr;
    BACKOFF_FAIL(
        storage_->make_write_file(output_metadata_path, output_metadata_file),
        "while trying to make write file for " + output_metadata_path);
    output_metadata_.emplace_back(output_metadata_file);

    if (column_types[out_idx] == ColumnType::Video) {
      video_metadata_.emplace_back();

      VideoMetadata& video_meta = video_metadata_.back();
      proto::VideoDescriptor& video_descriptor = video_meta.get_descriptor();
      video_descriptor.set_table_id(table_id);
      video_descriptor.set_column_id(out_idx);
      video_descriptor.set_item_id(task_id);
    }
  }
}

void ColumnSink::finished() {
  auto finished_start = now();
  auto save_file = [&](std::unique_ptr<storehouse::WriteFile>& file) {
    BACKOFF_FAIL(file->save(), "while trying to save " + file->path());
  };

  std::vector<std::future<void>> futures;
  for (auto& file : output_) {
    futures.push_back(thread_pool_.enqueue(save_file, std::ref(file)));
  }
  for (auto& file : output_metadata_) {
    futures.push_back(thread_pool_.enqueue(save_file, std::ref(file)));
  }

  auto save_metadata = [&](VideoMetadata& meta) {
    write_video_metadata(storage_.get(), meta);
  };
  for (auto& meta : video_metadata_) {
    futures.push_back(thread_pool_.enqueue(save_metadata, std::ref(meta)));
  }

  for (auto& future : futures) {
    future.wait();
  }

  profiler_->add_interval("column_sink:finished", finished_start, now());
}

void ColumnSink::provide_column_info(const std::vector<bool>& compressed,
                                     const std::vector<FrameInfo>& frame_info) {
  compressed_ = compressed;
  frame_info_ = frame_info;
}

REGISTER_SINK(Column, ColumnSink)
    .variadic_inputs()
    .per_element_output()
    .protobuf_name("ColumnSinkArgs");

REGISTER_SINK(FrameColumn, ColumnSink)
    .variadic_inputs()
    .per_element_output()
    .protobuf_name("ColumnSinkArgs");
;
}
}  // namespace scanner

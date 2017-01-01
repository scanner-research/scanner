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

#include "scanner/evaluators/video/decoder_evaluator.h"
#include "scanner/evaluators/serialize.h"
#include "scanner/metadata.pb.h"

#include "scanner/util/memory.h"

namespace scanner {

DecoderEvaluator::DecoderEvaluator(const EvaluatorConfig& config,
                                   DeviceType device_type,
                                   VideoDecoderType decoder_type,
                                   i32 extra_outputs,
                                   i32 num_devices)
    : device_type_(device_type),
      device_id_(config.device_ids[0]),
      decoder_type_(decoder_type),
      needs_warmup_(false),
      discontinuity_(false),
      extra_outputs_(extra_outputs) {
}

void DecoderEvaluator::configure(const BatchConfig& config) {
  config_ = config;
  for (const InputFormat& m : config.formats) {
    frame_sizes_.push_back(m.width() * m.height() * 3);
  }
  for (size_t i = 0; i < config.formats.size(); ++i) {
    if (decoders_.size() < metdata.size()) {
      VideoDecoder* decoder = VideoDecoder::make_from_config(
          device_type_, device_id_, decoder_type_, device_type_, num_devices);
      assert(decoder);
      decoders_.emplace_back(decoder);
    }
    decoders_[i]->configure(config.formats[i]);
  }
  i32 out_col_idx = 0;
  for (size_t i = 0; i < config.input_columns.size(); ++i) {
    if (config.input_columns[i] == base_column_name()) {
      video_column_idxs.push_back({static_cast<i32>(i), out_col_idx++});
    } else if (config.input_columns[i] != base_column_args_name()) {
      regular_column_idxs.push_back({static_cast<i32>(i), out_col_idx++});
    }
  }
}

void DecoderEvaluator::reset() {
  needs_warmup_ = true;
  discontinuity_ = true;
}

void DecoderEvaluator::evaluate(const BatchedColumns& input_columns,
                                BatchedColumns& output_columns) {
  assert(input_columns.size() ==
         video_column_idxs.size() * 2 + regular_column_idxs.size());

  auto start = now();

  i64 total_frames_decoded = 0;
  i64 total_frames_used = 0;

  i32 video_num = 0;
  for (std::tuple<i32, i32> idxs : video_column_idxs) {
    assert(video_num < frame_sizes_.size());
    size_t frame_size = frame_sizes_.at(video_num);
    VideoDecoder* decoder = decoders_.at(video_num).get();
    i64 frames_passed = 0;

    i32 col_idx;
    i32 out_col_idx;
    std::tie(col_idx, out_col_idx) = idxs;

    const Column& frame_col = input_columns[col_idx];
    const Column& args_col = input_columns[col_idx + 1];
    size_t num_inputs = frame_col.rows.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      const u8* in_encoded_buffer = frame_col.rows[i].buffer;
      size_t in_encoded_buffer_size = frame_col.rows[i].size;

      u8* decode_args_buffer = args_col.rows[i].buffer;
      size_t decode_args_buffer_size = args_col.rows[i].size;

      DecodeArgs args;
      const u8* encoded_buffer;
      size_t encoded_buffer_size = in_encoded_buffer_size;
      if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
        u8* buffer = new u8[decode_args_buffer_size];
        memcpy_buffer(buffer, CPU_DEVICE, decode_args_buffer,
                      {DeviceType::GPU, device_id_}, decode_args_buffer_size);
        args = deserialize_decode_args(buffer, decode_args_buffer_size);
        delete[] buffer;

        buffer = new u8[encoded_buffer_size];
        memcpy_buffer(buffer, CPU_DEVICE, in_encoded_buffer,
                      {DeviceType::GPU, device_id_}, encoded_buffer_size);
        encoded_buffer = buffer;
#else
        LOG(FATAL) << "Not built with cuda support!";
#endif
      } else {
        args = deserialize_decode_args(decode_args_buffer,
                                       decode_args_buffer_size);
        encoded_buffer = in_encoded_buffer;
      }

      std::vector<i32> valid_frames;
      {
        i32 s = 0;
        if (!needs_warmup_) {
          s += std::min(args.warmup_count() - frames_passed,
                        static_cast<i32>(args.rows_from_start()));
        }
        frames_passed += static_cast<i32>(args.valid_frames.size());
        valid_frames.insert(args.valid_frames.begin() + s,
                            args.valid_frames.end());
      }
      // HACK(apoms): just always force discontinuity for now instead of
      //  properly figuring out if the previous frame was abut
      discontinuity_ = true;

      i32 total_output_frames = static_cast<i32>(valid_frames.size());

      u8* output_block = new_block_buffer({device_type_, device_id_},
                                          total_output_frames * frame_size,
                                          total_output_frames);

      size_t encoded_buffer_offset = 0;
      i32 current_frame = args.start_keyframe();
      i32 valid_index = 0;
      while (valid_index < total_output_frames) {
        auto video_start = now();

        i32 encoded_packet_size = 0;
        const u8* encoded_packet = NULL;
        if (encoded_buffer_offset < encoded_buffer_size) {
          encoded_packet_size = *reinterpret_cast<const i32*>(
              encoded_buffer + encoded_buffer_offset);
          encoded_buffer_offset += sizeof(i32);
          encoded_packet = encoded_buffer + encoded_buffer_offset;
          encoded_buffer_offset += encoded_packet_size;
        }

        if (decoder->feed(encoded_packet, encoded_packet_size,
                           discontinuity_)) {
          // New frames
          bool more_frames = true;
          while (more_frames && valid_index < total_output_frames) {
            if (current_frame == valid_frames[valid_index]) {
              u8* decoded_buffer = output_block + valid_index * frame_size;
              more_frames = decoder->get_frame(decoded_buffer, frame_size);
              output_columns[out_col_idx].rows.push_back(
                  Row{decoded_buffer, frame_size});
              valid_index++;
              total_frames_used++;
            } else {
              more_frames = decoder->discard_frame();
            }
            current_frame++;
            total_frames_decoded++;
          }
        }
        // Set a discontinuity if we sent an empty packet to reset
        // the stream next time
        discontinuity_ = (encoded_packet_size == 0);
      }
      // Wait on all memcpys from frames to be done
      decoder->wait_until_frames_copied();

      if (decoder->decoded_frames_buffered() > 0) {
        while (decoder->discard_frame()) {
          total_frames_decoded++;
        };
      }

      if (device_type_ == DeviceType::GPU) {
        delete[] encoded_buffer;
      }
    }
    video_num++;
  }

  // Forward all inputs
  for (std::tuple<i32, i32> idxs : regular_column_idxs) {
    i32 col_idx;
    i32 out_col_idx;
    std::tie(col_idx, out_col_idx) = idxs;

    size_t s = 0;
    if (!needs_warmup_) {
      s += std::min(args.warmup_count(),
                    static_cast<i32>(args.rows_from_start()));
    }
    output_columns[out_col_idx].rows.insert(
        input_columns[col_idx].rows.begin() + s,
        input_columns[col_idx].rows.end());
  }
  // All warmed up
  needs_warmup_ = false;

  if (profiler_) {
    profiler_->add_interval("decode", start, now());
    profiler_->increment("effective_frames", total_frames_used);
    profiler_->increment("decoded_frames", total_frames_decoded);
  }
}

DecoderEvaluatorFactory::DecoderEvaluatorFactory(DeviceType device_type,
                                                 VideoDecoderType decoder_type,
                                                 i32 extra_outputs)
    : device_type_(device_type),
      decoder_type_(decoder_type),
      extra_outputs_(extra_outputs) {
  num_devices_ = device_type_ == DeviceType::GPU ? 1 : 8;
}

EvaluatorCapabilities DecoderEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  caps.max_devices = num_devices_;
  caps.warmup_size = 0;
  caps.can_overlap = true;
  return caps;
}

std::vector<std::string> DecoderEvaluatorFactory::get_output_names(
    const std::vector<std::string>& input_columns) {
  std::vector<std::string> output_columns;
  for (const std::string& col : input_columns) {
    if (col != base_column_args_name()) {
      outputs.push_back(col);
    }
  }
  return outputs;
}

Evaluator* DecoderEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new DecoderEvaluator(config, device_type_, decoder_type_,
                              extra_outputs_, num_devices_);
}
}

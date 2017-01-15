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

#pragma once

#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/video/video_decoder.h"

#include <memory>

namespace scanner {

class DecoderEvaluator : public Evaluator {
 public:
  DecoderEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                   VideoDecoderType decoder_type, i32 num_devices);
  ~DecoderEvaluator();

  void configure(const BatchConfig& info) override;

  void reset() override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 private:
  struct CachedEncodedVideo {
    const u8* encoded_buffer = nullptr;
    size_t encoded_buffer_size = 0;
    i64 current_start_keyframe = -1;
    i64 current_end_keyframe = -1;
    size_t encoded_buffer_offset = 0;
    i32 current_frame = -1;
  };

  DeviceType device_type_;
  i32 device_id_;
  VideoDecoderType decoder_type_;
  i32 num_devices_;

  std::vector<std::tuple<i32, i32>> video_column_idxs_;
  std::vector<size_t> frame_sizes_;
  std::vector<std::unique_ptr<VideoDecoder>> decoders_;
  std::vector<CachedEncodedVideo> cached_video_;
  std::vector<bool> discontinuity_;
  std::vector<std::tuple<i32, i32>> regular_column_idxs_;
};

class DecoderEvaluatorFactory : public EvaluatorFactory {
 public:
  DecoderEvaluatorFactory(DeviceType device_type,
                          VideoDecoderType decoder_type);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  VideoDecoderType decoder_type_;
  i32 num_devices_;
};
}

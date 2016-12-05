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

#include "scanner/engine.h"
#include "scanner/eval/evaluator.h"
#include "scanner/eval/evaluator_factory.h"
#include "scanner/video/video_decoder.h"

#include <memory>

namespace scanner {

class DecoderEvaluator : public Evaluator {
 public:
  DecoderEvaluator(const EvaluatorConfig& config, DeviceType device_type,
                   VideoDecoderType decoder_type, i32 extra_outputs,
                   i32 pu_count);

  void configure(const InputFormat& metadata) override;

  void reset() override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

 private:
  DeviceType device_type_;
  i32 device_id_;
  VideoDecoderType decoder_type_;
  i32 extra_outputs_;
  InputFormat metadata_;
  size_t frame_size_;
  std::unique_ptr<VideoDecoder> decoder_;
  bool needs_warmup_;
  bool discontinuity_;
};

class DecoderEvaluatorFactory : public EvaluatorFactory {
 public:
  DecoderEvaluatorFactory(DeviceType device_type, VideoDecoderType decoder_type,
                          i32 extra_outputs = 0);

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_names() override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  VideoDecoderType decoder_type_;
  i32 extra_outputs_;
  i32 pu_count_;
};
}

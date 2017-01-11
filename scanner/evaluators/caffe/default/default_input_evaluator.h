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
#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/util/opencv.h"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

#include "caffe_input_transformer_gpu/caffe_input_transformer_gpu.h"
#include "caffe_input_transformer_cpu/caffe_input_transformer_cpu.h"

#ifdef HAVE_CUDA
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include "scanner/util/cuda.h"
#endif

namespace scanner {

using InputLayerBuilder =
    std::function<void(u8*&, size_t&, const InputFormat&)>;

class DefaultInputEvaluator : public Evaluator {
 public:
  DefaultInputEvaluator(DeviceType device_type, i32 device_id,
                        const NetDescriptor& descriptor, i32 batch_size,
                        std::vector<InputLayerBuilder> input_layer_builders,
                        const EvaluatorConfig& config);
  ~DefaultInputEvaluator();

  void configure(const BatchConfig& config) override;

  void evaluate(const BatchedColumns& input_columns,
                BatchedColumns& output_columns) override;

  i32 net_input_width_;
  i32 net_input_height_;
  EvaluatorConfig eval_config_;
  CUcontext context_;

 private:
  void set_halide_buf(buffer_t& halide_buf, u8* buf, size_t size);
  void unset_halide_buf(buffer_t& halide_buf);
  void transform_halide(u8* input_buffer, u8* output_buffer);
  void transform_caffe(u8* input_buffer, u8* output_buffer);

  DeviceType device_type_;
  i32 device_id_;
  NetDescriptor descriptor_;
  InputFormat metadata_;
  i32 batch_size_;

#ifdef HAVE_CUDA
  i32 num_cuda_streams_;
#endif

  std::vector<InputLayerBuilder> input_layer_builders_;
};

class DefaultInputEvaluatorFactory : public EvaluatorFactory {
 public:
  DefaultInputEvaluatorFactory(
      DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size,
      std::vector<InputLayerBuilder> input_layer_builders = {});

  EvaluatorCapabilities get_capabilities() override;

  std::vector<std::string> get_output_columns(
      const std::vector<std::string>& input_columns) override;

  Evaluator* new_evaluator(const EvaluatorConfig& config) override;

 private:
  DeviceType device_type_;
  NetDescriptor net_descriptor_;
  i32 batch_size_;
  std::vector<InputLayerBuilder> input_layer_builders_;
};
}

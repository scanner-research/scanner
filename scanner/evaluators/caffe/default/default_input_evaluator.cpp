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

#include "scanner/evaluators/caffe/default/default_input_evaluator.h"
#include "scanner/util/memory.h"

#include "HalideRuntimeCuda.h"

#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace scanner {

DefaultInputEvaluator::DefaultInputEvaluator(
    DeviceType device_type, i32 device_id, const NetDescriptor& descriptor,
    i32 batch_size, std::vector<InputLayerBuilder> input_layer_builders,
    const EvaluatorConfig& config)
    : device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size),
      input_layer_builders_(input_layer_builders),
      eval_config_(config)
{
  if (descriptor_.input_width != -1) {
    net_input_width_ = descriptor_.input_width;
    net_input_height_ = descriptor_.input_height;
  } else {
    net_input_width_ = -1;
    net_input_height_ = -1;
  }
}

void DefaultInputEvaluator::configure(const BatchConfig& config) {
  config_ = config;
  assert(config.formats.size() == 1);
  metadata_ = config.formats[0];

  i32 width = metadata_.width();
  i32 height = metadata_.height();
  if (net_input_width_ == -1) {
    net_input_width_ = width;
    net_input_height_ = height;
  }
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);
#else
  LOG(FATAL) << "Not built with Cuda support.";
#endif
  }
}

void DefaultInputEvaluator::set_halide_buf(buffer_t& halide_buf, u8* buf,
                                           size_t size) {
  if (device_type_ == DeviceType::GPU) {
    halide_buf.dev = (uintptr_t) nullptr;

    // "You likely want to set the dev_dirty flag for correctness. (It will
    // not matter if all the code runs on the GPU.)"
    halide_buf.dev_dirty = true;

    i32 err = halide_cuda_wrap_device_ptr(nullptr, &halide_buf, (uintptr_t)buf);
    LOG_IF(FATAL, err != 0) << "Halide wrap device ptr failed";

    // "You'll need to set the host field of the buffer_t structs to
    // something other than nullptr as that is used to indicate bounds query
    // calls" - Zalman Stern
    halide_buf.host = (u8*)0xdeadbeef;

  } else {
    halide_buf.host = buf;
  }
}

void DefaultInputEvaluator::unset_halide_buf(buffer_t& halide_buf) {
  if (device_type_ == DeviceType::GPU) {
    halide_cuda_detach_device_ptr(nullptr, &halide_buf);
  }
}

void DefaultInputEvaluator::transform_halide(u8* input_buffer,
                                             u8* output_buffer) {
  i32 frame_width = metadata_.width();
  i32 frame_height = metadata_.height();
  size_t net_input_size =
      net_input_width_ * net_input_height_ * 3 * sizeof(float);

  buffer_t input_buf = {0}, output_buf = {0};

  set_halide_buf(input_buf, input_buffer, frame_width * frame_height * 3);
  set_halide_buf(output_buf, output_buffer, net_input_size);

  // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
  // input_buf.host = input_buffer;
  input_buf.stride[0] = 3;
  input_buf.stride[1] = metadata_.width() * 3;
  input_buf.stride[2] = 1;
  input_buf.extent[0] = metadata_.width();
  input_buf.extent[1] = metadata_.height();
  input_buf.extent[2] = 3;
  input_buf.elem_size = 1;

  // Halide conveniently defaults to a planar format, which is what Caffe
  // expects
  output_buf.host = output_buffer;
  output_buf.stride[0] = 1;
  output_buf.stride[1] = net_input_width_;
  output_buf.stride[2] = net_input_width_ * net_input_height_;
  output_buf.extent[0] = net_input_width_;
  output_buf.extent[1] = net_input_height_;
  output_buf.extent[2] = 3;
  output_buf.elem_size = 4;

  auto func = device_type_ == DeviceType::GPU ? caffe_input_transformer_gpu
                                              : caffe_input_transformer_cpu;
  int error = func(&input_buf, metadata_.width(), metadata_.height(),
                   net_input_width_, net_input_height_, descriptor_.normalize,
                   descriptor_.mean_colors[2], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[0], true, &output_buf);
  LOG_IF(FATAL, error != 0) << "Halide error " << error;

  unset_halide_buf(input_buf);
  unset_halide_buf(output_buf);
}

void DefaultInputEvaluator::transform_caffe(u8* input_buffer,
                                            u8* output_buffer) {
  i32 frame_width = metadata_.width();
  i32 frame_height = metadata_.height();
  size_t net_input_size =
      net_input_width_ * net_input_height_ * 3 * sizeof(float);

  cv::Mat input_mat(frame_height, frame_width, CV_8UC3, input_buffer);
  cv::Mat resized_input;

  cv::resize(input_mat, resized_input,
             cv::Size(net_input_width_, net_input_height_), 0, 0,
             cv::INTER_LINEAR);
  cv::cvtColor(resized_input, resized_input, CV_RGB2BGR);
  std::vector<cv::Mat> input_mats = {resized_input};

  caffe::Blob<f32> output_blob;
  output_blob.Reshape(1, 3, net_input_height_, net_input_width_);
  output_blob.set_cpu_data((f32*)output_buffer);

  caffe::TransformationParameter param;
  std::vector<float>& mean_colors = descriptor_.mean_colors;
  param.set_force_color(true);
  if (descriptor_.normalize) {
    param.set_scale(1.0 / 255.0);
  }
  for (i32 i = 0; i < mean_colors.size(); i++) {
    param.add_mean_value(mean_colors[i]);
  }

  caffe::DataTransformer<f32> transformer(param, caffe::TEST);
  transformer.Transform(input_mats, &output_blob);
}

void DefaultInputEvaluator::evaluate(const BatchedColumns& input_columns,
                                     BatchedColumns& output_columns) {
  auto eval_start = now();
  i32 input_count = input_columns[0].rows.size();
  size_t net_input_size =
      net_input_width_ * net_input_height_ * 3 * sizeof(float);

  for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
    output_columns[0].rows.push_back(input_columns[0].rows[i]);
  }

  u8* output_block = new_block_buffer(
      {device_type_, device_id_}, net_input_size * input_count, input_count);

  for (i32 frame = 0; frame < input_count; frame++) {
    u8* input_buffer = input_columns[0].rows[frame].buffer;
    u8* output_buffer = output_block + frame * net_input_size;

    transform_halide(input_buffer, output_buffer);

    INSERT_ROW(output_columns[1], output_buffer, net_input_size);
  }

  for (i32 l = 0; l < input_layer_builders_.size(); ++l) {
    std::vector<u8*> bufs;
    std::vector<size_t> sizes;
    size_t total_size = 0;
    for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
      u8* buffer;
      size_t size;
      input_layer_builders_[l](buffer, size, metadata_);
      total_size += size;
      bufs.push_back(buffer);
      sizes.push_back(size);
    }

    u8* column_block = new_block_buffer({device_type_, device_id_}, total_size,
                                        input_columns[0].rows.size());
    for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
      memcpy_buffer(column_block, {device_type_, device_id_}, bufs[i],
                    {device_type_, device_id_}, sizes[i]);
      output_columns[l + 2].rows.push_back(Row{column_block, sizes[i]});
      column_block += sizes[i];
    }

    for (auto buf : bufs) {
      delete_buffer({device_type_, device_id_}, buf);
    }
  }

  if (profiler_) {
    profiler_->add_interval("caffe:transform_input", eval_start, now());
  }
}

DefaultInputEvaluatorFactory::DefaultInputEvaluatorFactory(
    DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size,
    std::vector<InputLayerBuilder> input_layer_builders)
    : device_type_(device_type),
      net_descriptor_(descriptor),
      batch_size_(batch_size),
      input_layer_builders_(input_layer_builders) {}

EvaluatorCapabilities DefaultInputEvaluatorFactory::get_capabilities() {
  EvaluatorCapabilities caps;
  caps.device_type = device_type_;
  if (device_type_ == DeviceType::GPU) {
    caps.max_devices = 1;
  } else {
    caps.max_devices = EvaluatorCapabilities::UnlimitedDevices;
  }
  caps.warmup_size = 0;
  return caps;
}

std::vector<std::string> DefaultInputEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  std::vector<std::string> output_names = {"frame"};
  for (std::string& name : net_descriptor_.input_layer_names) {
    output_names.emplace_back(name);
  }
  assert(input_columns[0] == "frame");
  for (size_t i = 1; i < input_columns.size(); ++i) {
    output_names.push_back(input_columns[i]);
  }
  assert(net_descriptor_.input_layer_names.size() > 0);
  assert(input_layer_builders_.size() ==
         net_descriptor_.input_layer_names.size() - 1);
  return output_names;
}

Evaluator* DefaultInputEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new DefaultInputEvaluator(device_type_, config.device_ids[0],
                                   net_descriptor_, batch_size_,
                                   input_layer_builders_,
                                   config);
}
}

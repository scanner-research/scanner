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
#include "caffe_input_transformer_gpu/caffe_input_transformer_gpu.h"
#include "caffe_input_transformer_cpu/caffe_input_transformer_cpu.h"

#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace scanner {

DefaultInputEvaluator::DefaultInputEvaluator(
    DeviceType device_type, i32 device_id, const NetDescriptor& descriptor,
    i32 batch_size, std::vector<InputLayerBuilder> input_layer_builders)
    : device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size),
      input_layer_builders_(input_layer_builders)
#ifdef HAVE_CUDA
      ,
      num_cuda_streams_(32),
      streams_(num_cuda_streams_)
#endif

{
  if (descriptor_.input_width != -1) {
    net_input_width_ = descriptor_.input_width;
    net_input_height_ = descriptor_.input_height;
  } else {
    net_input_width_ = -1;
    net_input_height_ = -1;
  }

  if (device_type_ == DeviceType::CPU) {
    std::vector<float>& mean_colors = descriptor_.mean_colors;
    caffe::TransformationParameter param;
    param.set_force_color(true);
    if (descriptor_.normalize) {
      param.set_scale(1.0 / 255.0);
    }
    for (i32 i = 0; i < mean_colors.size(); i++) {
      param.add_mean_value(mean_colors[i]);
    }
    transformer_.reset(new caffe::DataTransformer<f32>(param, caffe::TEST));
  }
}

void DefaultInputEvaluator::configure(const InputFormat& metadata) {
  metadata_ = metadata;

  i32 width = metadata.width();
  i32 height = metadata.height();
  if (net_input_width_ == -1) {
    net_input_width_ = width;
    net_input_height_ = height;
  }
  output_blob_.Reshape(batch_size_, 3, net_input_height_, net_input_width_);

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    mean_mat_g_ = cv::cuda::GpuMat(
        net_input_height_, net_input_width_, CV_32FC3,

        cv::Scalar(descriptor_.mean_colors[0], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[2]));

    frame_input_g_.clear();
    float_input_g_.clear();
    normalized_input_g_.clear();
    meanshifted_input_g_.clear();
    input_planes_g_.clear();
    planar_input_g_.clear();
    for (size_t i = 0; i < 512; ++i) {
      frame_input_g_.push_back(
          cv::cuda::GpuMat(metadata.height(), metadata.width(), CV_8UC3));
    }
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      resized_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_8UC3));
      float_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      meanshifted_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      normalized_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      std::vector<cv::cuda::GpuMat> planes1;
      std::vector<cv::cuda::GpuMat> planes2;
      for (i32 i = 0; i < 3; ++i) {
        planes1.push_back(
            cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
        planes2.push_back(
            cv::cuda::GpuMat(net_input_width_, net_input_height_, CV_32FC1));
      }
      input_planes_g_.push_back(planes1);
      flipped_planes_g_.push_back(planes2);
      planar_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_ * 3, net_input_width_, CV_32FC1));
    }
#else
    LOG(FATAL) << "Not built with Cuda support.";
#endif
  } else {
    resized_input_c_ = cv::Mat(net_input_height_, net_input_width_, CV_8UC3);
    for (i32 i = 0; i < batch_size_; ++i) {
      input_mats_c_.emplace_back(
          cv::Mat(net_input_height_, net_input_width_, CV_8UC3));
    }
  }
}

void DefaultInputEvaluator::evaluate(const BatchedColumns& input_columns,
                                     BatchedColumns& output_columns) {
  auto eval_start = now();

  size_t net_input_size = net_input_width_ * net_input_height_ * 3 * sizeof(float);
  i32 input_count = input_columns[0].rows.size();


  for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
    output_columns[0].rows.push_back(input_columns[0].rows[i]);
  }

  i32 frame_width = metadata_.width();
  i32 frame_height = metadata_.height();

  for (i32 frame = 0; frame < input_count; frame++) {
    u8* input_buffer = input_columns[0].rows[frame].buffer;
    if (device_type_ == DeviceType::GPU) {
      size_t size = input_columns[0].rows[frame].size;
      u8* cpu_buffer = new u8[size];
      memcpy_buffer(cpu_buffer, DeviceType::CPU, 0,
                    input_buffer, DeviceType::GPU, device_id_,
                    size);
      input_buffer = cpu_buffer;
    }

    buffer_t input_buf = {0}, output_buf = {0};
    // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
    input_buf.host = input_buffer;
    input_buf.stride[0] = 3;
    input_buf.stride[1] = metadata_.width() * 3;
    input_buf.stride[2] = 1;
    input_buf.extent[0] = metadata_.width();
    input_buf.extent[1] = metadata_.height();
    input_buf.extent[2] = 3;
    input_buf.elem_size = 1;

    // Halide conveniently defaults to a planar format, which is what Caffe expects
    u8* output_buffer = new u8[net_input_size];
    output_buf.host = output_buffer;
    output_buf.stride[0] = 1;
    output_buf.stride[1] = net_input_width_;
    output_buf.stride[2] = net_input_width_ * net_input_height_;
    output_buf.extent[0] = net_input_width_;
    output_buf.extent[1] = net_input_height_;
    output_buf.extent[2] = 3;
    output_buf.elem_size = 4;

    auto func = device_type_ == DeviceType::GPU ?
      caffe_input_transformer_gpu :
      caffe_input_transformer_cpu;
    int error = func(
      &input_buf,
      metadata_.width(), metadata_.height(),
      net_input_width_, net_input_height_,
      descriptor_.normalize,
      descriptor_.mean_colors[2],
      descriptor_.mean_colors[1],
      descriptor_.mean_colors[0],
      true,
      &output_buf);
    LOG_IF(FATAL, error != 0) << "Halide error " << error;

    if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
      u8* gpu_buffer = new_buffer(device_type_, device_id_, net_input_size);
      memcpy_buffer(gpu_buffer, DeviceType::GPU, device_id_,
                    output_buffer, DeviceType::CPU, 0,
                    net_input_size);
      delete output_buffer;
      output_buffer = gpu_buffer;
#else
      LOG(FATAL) << "Cuda not found.";
#endif
    }

    INSERT_ROW(output_columns[1], output_buffer, net_input_size);
  }

  for (i32 l = 0; l < input_layer_builders_.size(); ++l) {
    for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
      u8* buffer;
      size_t size;
      input_layer_builders_[l](buffer, size, metadata_);
      output_columns[l + 2].rows.push_back(Row{buffer, size});
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

std::vector<std::string> DefaultInputEvaluatorFactory::get_output_names() {
  std::vector<std::string> output_names = {"frame"};
  for (std::string& name : net_descriptor_.input_layer_names) {
    output_names.emplace_back(name);
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
                                   input_layer_builders_);
}
}

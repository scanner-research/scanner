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

namespace scanner {

DefaultInputEvaluator::DefaultInputEvaluator(DeviceType device_type,
                                             i32 device_id,
                                             const NetDescriptor& descriptor,
                                             i32 batch_size)
    : device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size) {
  if (descriptor_.input_width != -1) {
    net_input_width_ = descriptor_.input_width;
    net_input_height_ = descriptor_.input_height;
  } else {
    net_input_width_ = -1;
    net_input_height_ = -1;
  }

  std::vector<float>& mean_colors = descriptor_.mean_colors;
  caffe::TransformationParameter param;
  param.set_force_color(true);
  param.set_scale(1.0 / 255.0);
  for (i32 i = 0; i < mean_colors.size(); i++) {
    param.add_mean_value(mean_colors[i]);
  }
  transformer =
      std::make_unique<caffe::DataTransformer<f32>>(param, caffe::TEST);

  resized_input = cv::Mat(net_input_height_, net_input_width_, CV_8UC3);
  for (i32 i = 0; i < batch_size_; ++i) {
    input_mats.emplace_back(
        cv::Mat(net_input_height_, net_input_width_, CV_8UC3));
  }
}

void DefaultInputEvaluator::configure(const VideoMetadata& metadata) {
  metadata_ = metadata;

  i32 width = metadata.width();
  i32 height = metadata.height();
  if (net_input_width_ == -1) {
    net_input_width_ = width;
    net_input_height_ = height;
  }

  output_blob.Reshape(batch_size_, 3, net_input_height_, net_input_width_);
}

void DefaultInputEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  auto eval_start = now();

  size_t frame_size = net_input_width_ * net_input_height_ * 3 * sizeof(float);
  i32 input_count = input_buffers[0].size();

  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "Default input transformer doesn't support GPU mode.";
  } else {
    for (i32 frame = 0; frame < input_count; frame += batch_size_) {
      i32 batch_count = std::min(input_count - frame, batch_size_);

      i32 frame_width = metadata_.width();
      i32 frame_height = metadata_.height();

      for (i32 i = 0; i < batch_count; ++i) {
        u8* buffer = input_buffers[0][frame + i];
        cv::Mat input_mat(frame_height, frame_width, CV_8UC3, buffer);

        cv::resize(input_mat, resized_input,
                   cv::Size(net_input_width_, net_input_height_), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(resized_input, input_mats[i], CV_RGB2BGR);
      }

      std::vector<cv::Mat> input_mats_slice(input_mats.begin(),
                                            input_mats.begin() + batch_count);

      u8* net_input = new u8[frame_size * batch_count];
      output_blob.set_cpu_data((f32*)net_input);
      output_blob.Reshape(input_mats_slice.size(), output_blob.shape(1),
                          output_blob.shape(2), output_blob.shape(3));
      transformer->Transform(input_mats_slice, &output_blob);

      output_buffers[0].push_back(net_input);
      output_sizes[0].push_back(frame_size * batch_count);
    }

    i32 num_batches = output_buffers[0].size();
    for (i32 i = 0; i < input_buffers[0].size() - num_batches; ++i) {
      output_buffers[0].push_back(new u8[1]);
      output_sizes[0].push_back(1);
    }
  }

  for (i32 i = 0; i < input_buffers[0].size(); ++i) {
    size_t size = input_sizes[0][i];
    u8* buffer = new_buffer(device_type_, device_id_, size);
    memcpy_buffer(buffer, device_type_, device_id_, input_buffers[0][i],
                  device_type_, device_id_, size);
    output_buffers[1].push_back(buffer);
    output_sizes[1].push_back(size);
  }

  if (profiler_) {
    profiler_->add_interval("caffe:transform_input", eval_start, now());
  }
}

DefaultInputEvaluatorFactory::DefaultInputEvaluatorFactory(
    DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size)
    : device_type_(device_type),
      net_descriptor_(descriptor),
      batch_size_(batch_size) {}

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
  return {"net_input", "frame"};
}

Evaluator* DefaultInputEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new DefaultInputEvaluator(device_type_, config.device_ids[0],
                                   net_descriptor_, batch_size_);
}
}

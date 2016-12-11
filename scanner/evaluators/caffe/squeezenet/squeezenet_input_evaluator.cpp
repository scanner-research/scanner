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

#include "scanner/evaluators/caffe/squeezenet/squeezenet_input_evaluator.h"
#include "scanner/util/memory.h"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

namespace scanner {

SqueezeNetInputEvaluator::SqueezeNetInputEvaluator(
    DeviceType device_type, i32 device_id, const NetDescriptor& descriptor,
    i32 batch_size)
    : device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size) {
  // Resize into
  std::vector<float>& mean_colors = descriptor_.mean_colors;
  cv::Mat unsized_mean_mat(
      NET_INPUT_HEIGHT * 3, NET_INPUT_WIDTH, CV_32FC1,
      cv::Scalar((mean_colors[0] + mean_colors[1] + mean_colors[2]) / 3.0));
  // HACK(apoms): Resizing the mean like this is not likely to produce a correct
  //              result because we are resizing a planar BGR layout which is
  //              represented in OpenCV as a single channel image with a height
  //              three times as high. Thus resizing is going to blur the
  //              borders slightly where the channels touch.
  cv::resize(unsized_mean_mat, mean_mat,
             cv::Size(NET_INPUT_WIDTH, NET_INPUT_HEIGHT * 3));

  resized_input = cv::Mat(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC3);
  bgr_input = cv::Mat(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC3);
  for (i32 i = 0; i < 3; ++i) {
    input_planes.push_back(cv::Mat(NET_INPUT_HEIGHT, NET_INPUT_WIDTH, CV_8UC1));
  }
  planar_input = cv::Mat(NET_INPUT_HEIGHT * 3, NET_INPUT_WIDTH, CV_8UC1);
  float_input = cv::Mat(NET_INPUT_HEIGHT * 3, NET_INPUT_WIDTH, CV_32FC1);
  normalized_input = cv::Mat(NET_INPUT_HEIGHT * 3, NET_INPUT_WIDTH, CV_32FC1);
}

void SqueezeNetInputEvaluator::configure(const InputFormat& metadata) {
  metadata_ = metadata;
}

void SqueezeNetInputEvaluator::evaluate(const BatchedColumns& input_columns,
                                        BatchedColumns& output_columns) {
  auto eval_start = now();

  size_t frame_size = NET_INPUT_WIDTH * NET_INPUT_HEIGHT * 3 * sizeof(float);
  i32 input_count = input_columns[0].rows.size();

  if (device_type_ == DeviceType::GPU) {
    LOG(FATAL) << "SqueezeNet input transformer doesn't support GPU mode.";
  } else {
    for (i32 frame = 0; frame < input_count; frame += batch_size_) {
      i32 batch_count = std::min(input_count - frame, batch_size_);
      u8* net_input = new u8[frame_size * batch_count];

      i32 frame_width = metadata_.width();
      i32 frame_height = metadata_.height();

      std::vector<cv::Mat> input_mats;
      for (i32 i = 0; i < batch_count; ++i) {
        u8* buffer = input_columns[0].rows[frame + i].buffer;
        cv::Mat input_mat(frame_height, frame_width, CV_8UC3, buffer);

        cv::Mat resized, bgr;

        cv::resize(input_mat, resized,
                   cv::Size(NET_INPUT_WIDTH, NET_INPUT_HEIGHT), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(resized, bgr, CV_RGB2BGR);
        input_mats.emplace_back(bgr);
      }

      caffe::TransformationParameter param;
      param.set_scale(1.0 / 255.0);
      caffe::DataTransformer<f32> transformer(param, caffe::TEST);
      caffe::Blob<f32> blob;
      blob.Reshape(input_mats.size(), input_mats[0].channels(),
                   NET_INPUT_HEIGHT, NET_INPUT_WIDTH);
      transformer.Transform(input_mats, &blob);

      std::memcpy(net_input, blob.cpu_data(), frame_size * batch_count);

      output_columns[0].rows.push_back(
          Row{(u8*)net_input, frame_size * batch_count});
    }

    i32 num_batches = output_columns[0].rows.size();
    for (i32 i = 0; i < input_columns[0].rows.size() - num_batches; ++i) {
      INSERT_ROW(output_columns[0], new u8[1], 1);
    }
  }

  for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
    size_t size = ROW_SIZE(input_columns[0], i);
    u8* buffer = new_buffer({device_type_, device_id_}, size);
    memcpy_buffer(buffer, {device_type_, device_id_},
                  ROW_BUFFER(input_columns[0], i), {device_type_, device_id_},
                  size);
    INSERT_ROW(output_columns[1], buffer, size);
  }

  if (profiler_) {
    profiler_->add_interval("caffe:transform_input", eval_start, now());
  }
}

SqueezeNetInputEvaluatorFactory::SqueezeNetInputEvaluatorFactory(
    DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size)
    : device_type_(device_type),
      net_descriptor_(descriptor),
      batch_size_(batch_size) {}

EvaluatorCapabilities SqueezeNetInputEvaluatorFactory::get_capabilities() {
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

std::vector<std::string> SqueezeNetInputEvaluatorFactory::get_output_names() {
  return {"net_input", "frame"};
}

Evaluator* SqueezeNetInputEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new SqueezeNetInputEvaluator(device_type_, config.device_ids[0],
                                      net_descriptor_, batch_size_);
}
}

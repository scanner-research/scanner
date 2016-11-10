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

#include "scanner/evaluators/caffe/caffe_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/memory.h"
#include "scanner/util/util.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda.hpp>
#endif
#endif

namespace scanner {

CaffeEvaluator::CaffeEvaluator(const EvaluatorConfig& config,
                               DeviceType device_type, i32 device_id,
                               const NetDescriptor& descriptor, i32 batch_size,
                               bool forward_input)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size),
      forward_input_(forward_input) {
  if (device_type_ == DeviceType::GPU) {
    assert(GPUS_PER_NODE > 0);
    device_id = device_id % GPUS_PER_NODE;
    device_id_ = device_id;
  }
  set_device();
  // Initialize our network
  net_.reset(new caffe::Net<float>(descriptor_.model_path, caffe::TEST));
  net_->CopyTrainedLayersFrom(descriptor_.model_weights_path);
}

void CaffeEvaluator::configure(const VideoMetadata& metadata) {
  metadata_ = metadata;

  set_device();

  assert(descriptor_.input_layer_names.size() > 0);
  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name(descriptor_.input_layer_names[0])};
  if (input_blob->shape(0) != batch_size_) {
    input_blob->Reshape(
        {batch_size_, 3, input_blob->shape(2), input_blob->shape(3)});
  }

  i32 width = metadata.width();
  i32 height = metadata.height();
  if (descriptor_.input_width != -1) {
    width = descriptor_.input_width;
    height = descriptor_.input_height;
  }

  input_blob->Reshape(
      {input_blob->shape(0), input_blob->shape(1), height, width});
}

void CaffeEvaluator::evaluate(
    const std::vector<std::vector<u8*>>& input_buffers,
    const std::vector<std::vector<size_t>>& input_sizes,
    std::vector<std::vector<u8*>>& output_buffers,
    std::vector<std::vector<size_t>>& output_sizes) {
  set_device();

  std::vector<boost::shared_ptr<caffe::Blob<float>>> input_blobs;
  for (std::string& name : descriptor_.input_layer_names) {
    input_blobs.emplace_back(net_->blob_by_name(name));
  }
  assert(input_blobs.size() > 0);

  i32 input_count = (i32)input_buffers[0].size();

  i32 batch_id = 0;
  for (i32 frame = 0; frame < input_count; frame += batch_size_) {
    i32 batch_count = std::min(input_count - frame, batch_size_);
    if (input_blobs[0]->shape(0) != batch_count) {
      input_blobs[0]->Reshape(
          {batch_count, 3, input_blobs[0]->shape(2), input_blobs[0]->shape(3)});
    }

    for (i32 i = 0; i < input_blobs.size(); ++i) {
      f32* net_input_buffer = nullptr;

      if (device_type_ == DeviceType::GPU) {
        net_input_buffer = input_blobs[i]->mutable_gpu_data();
      } else {
        net_input_buffer = input_blobs[i]->mutable_cpu_data();
      }

      memcpy_buffer((u8*)net_input_buffer, device_type_, device_id_,
                    input_buffers[i + 1][batch_id], device_type_, device_id_,
                    input_sizes[i + 1][batch_id]);
    }

    // Compute features
    auto net_start = now();
    net_->ForwardPrefilled();
    if (profiler_) {
      profiler_->add_interval("caffe:net", net_start, now());
    }

    // Save batch of frames
    size_t num_outputs = descriptor_.output_layer_names.size();
    i32 output_offset = 0;
    if (forward_input_) {
      output_offset++;
      for (i32 b = 0; b < batch_count; ++b) {
        output_buffers[0].push_back(input_buffers[1][frame + b]);
        output_sizes[0].push_back(input_sizes[1][frame + b]);
      }
    }

    for (size_t i = 0; i < num_outputs; ++i) {
      const std::string& output_layer_name = descriptor_.output_layer_names[i];
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
          net_->blob_by_name(output_layer_name)};
      size_t output_length = output_blob->count() / batch_count;
      size_t output_size = output_length * sizeof(float);
      for (i32 b = 0; b < batch_count; ++b) {
        u8* buffer = nullptr;
        if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
          cudaMalloc((void**)&buffer, output_size);
          cudaMemcpy(buffer, output_blob->gpu_data() + b * output_length,
                     output_size, cudaMemcpyDefault);
#else
          LOG(FATAL) << "Not built with CUDA support.";
#endif
        } else {
          buffer = new u8[output_size];
          memcpy(buffer, output_blob->cpu_data() + b * output_length,
                 output_size);
        }
        assert(buffer != nullptr);
        output_buffers[output_offset + i].push_back(buffer);
        output_sizes[output_offset + i].push_back(output_size);
      }
    }

    ++batch_id;
  }
}

void CaffeEvaluator::set_device() {
  // Setup correct device for Caffe
  caffe::Caffe::set_mode(device_type_to_caffe_mode(device_type_));
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(device_id_));
    caffe::Caffe::SetDevice(device_id_);
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  }
}

CaffeEvaluatorFactory::CaffeEvaluatorFactory(
    DeviceType device_type, const NetDescriptor& net_descriptor, i32 batch_size,
    bool forward_input)
    : device_type_(device_type),
      net_descriptor_(net_descriptor),
      batch_size_(batch_size),
      forward_input_(forward_input) {}

EvaluatorCapabilities CaffeEvaluatorFactory::get_capabilities() {
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

std::vector<std::string> CaffeEvaluatorFactory::get_output_names() {
  std::vector<std::string> output_names;
  if (forward_input_) {
    output_names.push_back("frame");
  }
  const std::vector<std::string>& layer_names =
      net_descriptor_.output_layer_names;
  output_names.insert(output_names.end(), layer_names.begin(),
                      layer_names.end());

  return output_names;
}

Evaluator* CaffeEvaluatorFactory::new_evaluator(const EvaluatorConfig& config) {
  return new CaffeEvaluator(config, device_type_, config.device_ids[0],
                            net_descriptor_, batch_size_, forward_input_);
}
}

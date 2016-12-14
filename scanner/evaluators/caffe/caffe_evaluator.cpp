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
                               bool forward_input,
                               CustomNetConfiguration net_config)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size),
      forward_input_(forward_input),
      net_config_(net_config) {
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

void CaffeEvaluator::configure(const InputFormat& metadata) {
  metadata_ = metadata;

  set_device();

  assert(descriptor_.input_layer_names.size() > 0);
  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name(descriptor_.input_layer_names[0])};
  if (input_blob->shape(0) != batch_size_) {
    input_blob->Reshape({batch_size_, input_blob->shape(1),
                         input_blob->shape(2), input_blob->shape(3)});
  }

  if (net_config_) {
    net_config_(metadata, net_.get());
  } else {
    i32 width, height;
    if (descriptor_.transpose) {
      width = metadata.height();
      height = metadata.width();
    } else {
      width = metadata.width();
      height = metadata.height();
    }
    if (descriptor_.preserve_aspect_ratio) {
      if (descriptor_.input_width != -1) {
        width = descriptor_.input_width;
        f32 scale = static_cast<f32>(descriptor_.input_width) / width;
        width = width * scale;
        height = height * scale;
      } else if (descriptor_.input_height != -1) {
        f32 scale = static_cast<f32>(descriptor_.input_height) / height;
        width = width * scale;
        height = height * scale;
      }
    } else if (descriptor_.input_width != -1) {
      width = descriptor_.input_width;
      height = descriptor_.input_height;
    }

    if (descriptor_.pad_mod != -1) {
      i32 pad = descriptor_.pad_mod;
      width += (width % pad) ? pad - (width % pad) : 0;
      height += (height % pad) ? pad - (height % pad) : 0;
    }

    input_blob->Reshape(
        {input_blob->shape(0), input_blob->shape(1), height, width});
  }
}

void CaffeEvaluator::evaluate(const BatchedColumns& input_columns,
                              BatchedColumns& output_columns) {
  set_device();

  std::vector<boost::shared_ptr<caffe::Blob<float>>> input_blobs;
  for (std::string& name : descriptor_.input_layer_names) {
    input_blobs.emplace_back(net_->blob_by_name(name));
  }
  assert(input_blobs.size() > 0);

  i32 input_count = (i32)input_columns[1].rows.size();

  i32 output_offset = (forward_input_ ? 1 : 0);
  if (forward_input_) {
    for (i32 b = 0; b < (i32)input_columns[0].rows.size(); ++b) {
      output_columns[0].rows.push_back(input_columns[0].rows[b]);
    }
  }

  for (i32 frame = 0; frame < input_count; frame += batch_size_) {
    i32 batch_count = std::min(input_count - frame, batch_size_);
    if (input_blobs[0]->shape(0) != batch_count) {
      input_blobs[0]->Reshape({batch_count, input_blobs[0]->shape(1),
                               input_blobs[0]->shape(2),
                               input_blobs[0]->shape(3)});
    }

    for (i32 i = 0; i < input_blobs.size(); ++i) {
      f32* net_input_buffer = nullptr;
      if (device_type_ == DeviceType::GPU) {
        net_input_buffer = input_blobs[i]->mutable_gpu_data();
      } else {
        net_input_buffer = input_blobs[i]->mutable_cpu_data();
      }

      size_t offset = 0;
      for (i32 j = 0; j < batch_count; ++j) {
        memcpy_buffer((u8*)net_input_buffer + offset, {device_type_, device_id_},
                      input_columns[i + 1].rows[frame + j].buffer,
                      {device_type_, device_id_},
                      input_columns[i + 1].rows[frame + j].size);
        offset += input_columns[i + 1].rows[frame + j].size;
      }
    }

    // Compute features
    auto net_start = now();
    net_->ForwardPrefilled();
    if (profiler_) {
      cudaDeviceSynchronize();
      profiler_->add_interval("caffe:net", net_start, now());
    }

    // Save batch of frames
    size_t num_outputs = descriptor_.output_layer_names.size();
    size_t total_size = 0;
    i32 total_rows = num_outputs * batch_count;
    for (size_t i = 0; i < num_outputs; ++i) {
      const std::string& output_layer_name = descriptor_.output_layer_names[i];
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
          net_->blob_by_name(output_layer_name)};
      size_t output_length = output_blob->count() / batch_count;
      size_t output_size = output_length * sizeof(float);
      total_size += output_size * batch_count;
    }

    u8* output_block = new_block_buffer({device_type_, device_id_}, total_size,
                                        total_rows);
    std::vector<u8*> dest_buffers, src_buffers;
    std::vector<size_t> sizes;
    for (size_t i = 0; i < num_outputs; ++i) {
      const std::string& output_layer_name = descriptor_.output_layer_names[i];
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
          net_->blob_by_name(output_layer_name)};
      size_t output_length = output_blob->count() / batch_count;
      size_t output_size = output_length * sizeof(float);
      dest_buffers.push_back(output_block);
      src_buffers.push_back(
        (u8*) (device_type_ == DeviceType::CPU
               ? output_blob->cpu_data()
               : output_blob->gpu_data()));
      sizes.push_back(output_size * batch_count);
      for (i32 b = 0; b < batch_count; b++) {
        output_columns[output_offset + i].rows.push_back(
          Row{output_block, output_size});
        output_block += output_size;
      }
    }

    memcpy_vec(dest_buffers, {device_type_, device_id_},
               src_buffers, {device_type_, device_id_},
               sizes);
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
    DeviceType device_type, const NetDescriptor &net_descriptor, i32 batch_size,
    bool forward_input, CustomNetConfiguration net_config)
    : device_type_(device_type), net_descriptor_(net_descriptor),
      batch_size_(batch_size), forward_input_(forward_input),
      net_config_(net_config) {}

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
  if (false) {
    output_names.push_back("frame");
  }

  return output_names;
}

Evaluator* CaffeEvaluatorFactory::new_evaluator(const EvaluatorConfig& config) {
  return new CaffeEvaluator(config, device_type_, config.device_ids[0],
                            net_descriptor_, batch_size_, forward_input_,
                            net_config_);
}
}

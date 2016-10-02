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
#include "scanner/util/util.h"

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda.hpp>
#endif
#endif

namespace scanner {

CaffeEvaluator::CaffeEvaluator(const EvaluatorConfig& config,
                               DeviceType device_type,
                               const NetDescriptor& descriptor,
                               CaffeInputTransformer* transformer,
                               i32 device_id)
    : config_(config),
      device_type_(device_type),
      descriptor_(descriptor),
      transformer_(transformer),
      device_id_(device_id) {
  caffe::Caffe::set_mode(device_type_to_caffe_mode(device_type));
  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(device_id));
    caffe::Caffe::SetDevice(device_id);
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  }
  // Initialize our network
  net_.reset(new caffe::Net<float>(descriptor_.model_path, caffe::TEST));
  net_->CopyTrainedLayersFrom(descriptor_.model_weights_path);
}

void CaffeEvaluator::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name(descriptor_.input_layer_name)};
  if (input_blob->shape(0) != config_.max_input_count) {
    input_blob->Reshape({config_.max_input_count, 3, input_blob->shape(2),
                         input_blob->shape(3)});
  }

  transformer_->configure(metadata, net_.get());
}

void CaffeEvaluator::evaluate(
    const std::vector<std::vector<u8 *>> &input_buffers,
    const std::vector<std::vector<size_t>> &input_sizes,
    std::vector<std::vector<u8 *>> &output_buffers,
    std::vector<std::vector<size_t>> &output_sizes) {
  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net_->blob_by_name(descriptor_.input_layer_name)};

  i32 input_count = (i32)input_buffers[0].size();

  if (input_blob->shape(0) != input_count) {
    input_blob->Reshape(
        {input_count, 3, input_blob->shape(2), input_blob->shape(3)});
  }

  f32* net_input_buffer = nullptr;

  caffe::Caffe::set_mode(device_type_to_caffe_mode(device_type_));
  if (device_type_ == DeviceType::GPU) {
#if defined(HAVE_OPENCV) && defined(HAVE_CUDA)
    cv::cuda::setDevice(device_id_);
#endif
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(device_id_));
    caffe::Caffe::SetDevice(device_id_);
    net_input_buffer = input_blob->mutable_gpu_data();
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    net_input_buffer = input_blob->mutable_cpu_data();
  }

  // Process batch of frames
  // HACK(apoms): We know that the frames from the video decoder are store
  // contiguously so we can just use the first pointer as a wokaround for
  // redoing the input transformer interface
  u8* input_buffer = input_buffers[0][0];
  auto cv_start = now();
  transformer_->transform_input(input_count, input_buffer, net_input_buffer);
  if (profiler_) {
    profiler_->add_interval("caffe:transform_input", cv_start, now());
  }

  // Compute features
  auto net_start = now();
  net_->Forward();
  if (profiler_) {
    profiler_->add_interval("caffe:net", net_start, now());
  }

  // Save batch of frames
  size_t num_outputs = descriptor_.output_layer_names.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    const std::string& output_layer_name = descriptor_.output_layer_names[i];
    const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net_->blob_by_name(output_layer_name)};
    size_t output_length = output_blob->count(1);
    size_t output_size = output_length * sizeof(float);
    for (i32 b = 0; b < input_count; ++b) {
      if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
        u8 *buffer;
        cudaMalloc((void**)&buffer, output_size);;
        cudaMemcpy(buffer, output_blob->gpu_data() + b * output_length,
                   output_size, cudaMemcpyDefault);
        output_buffers[i].push_back(buffer);
        output_sizes[i].push_back(output_size);
#else
        LOG(FATAL) << "Not built with CUDA support.";
#endif
      } else {
        u8 *buffer = new u8[output_size];
        memcpy(buffer, output_blob->cpu_data() + b * output_length,
               output_size);
        output_buffers[i].push_back(buffer);
        output_sizes[i].push_back(output_size);
      }
    }
  }
}

CaffeEvaluatorFactory::CaffeEvaluatorFactory(
    DeviceType device_type, const NetDescriptor &net_descriptor,
    CaffeInputTransformerFactory *transformer_factory)
    : device_type_(device_type), net_descriptor_(net_descriptor),
      transformer_factory_(transformer_factory) {}

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
    return net_descriptor_.output_layer_names;
  }

  Evaluator *CaffeEvaluatorFactory::new_evaluator(
      const EvaluatorConfig &config) {
    CaffeInputTransformer *transformer =
        transformer_factory_->construct(config, net_descriptor_);
    return new CaffeEvaluator(config, device_type_, net_descriptor_,
                              transformer, 0);
  }
}

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

#include "scanner/evaluators/caffe/caffe_cpu_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/util.h"

namespace scanner {

CaffeCPUEvaluator::CaffeCPUEvaluator(
  const EvaluatorConfig& config,
  const NetDescriptor& descriptor,
  CaffeInputTransformer* transformer,
  i32 device_id)
  : config_(config),
    descriptor_(descriptor),
    transformer_(transformer),
    device_id_(device_id)
{
  caffe::Caffe::set_mode(device_type_to_caffe_mode(DeviceType::CPU));

  // Initialize our network
  net_.reset(new caffe::Net<float>(descriptor_.model_path, caffe::TEST));
  net_->CopyTrainedLayersFrom(descriptor_.model_weights_path);
}

void CaffeCPUEvaluator::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net_->blob_by_name(descriptor_.input_layer_name)};
  if (input_blob->shape(0) != config_.max_batch_size) {
    input_blob->Reshape(
      {config_.max_batch_size, 3, input_blob->shape(2), input_blob->shape(3)});
  }

  transformer_->configure(metadata, net_.get());
}

void CaffeCPUEvaluator::evaluate(
  u8* input_buffer,
  std::vector<std::vector<u8*>>& output_buffers,
  std::vector<std::vector<size_t>>& output_sizes,
  i32 batch_size)
{
  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net_->blob_by_name(descriptor_.input_layer_name)};

  if (input_blob->shape(0) != batch_size) {
    input_blob->Reshape({
        batch_size, 3, input_blob->shape(2), input_blob->shape(3)});
  }

  f32* net_input_buffer = input_blob->mutable_cpu_data();

  // Process batch of frames
  auto cv_start = now();
  transformer_->transform_input(input_buffer, net_input_buffer, batch_size);
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
    for (i32 b = 0; b < batch_size; ++b) {
      u8* buffer = new u8[output_size];
      memcpy(buffer, output_blob->cpu_data() + b * output_length, output_size);
      output_buffers[i].push_back(buffer);
      output_sizes[i].push_back(output_size);
    }
  }
}

CaffeCPUEvaluatorConstructor::CaffeCPUEvaluatorConstructor(
  const NetDescriptor& net_descriptor,
  CaffeInputTransformerFactory* transformer_factory)
  : net_descriptor_(net_descriptor),
    transformer_factory_(transformer_factory)
{
}

i32 CaffeCPUEvaluatorConstructor::get_number_of_devices() {
  return 1;
}

DeviceType CaffeCPUEvaluatorConstructor::get_input_buffer_type() {
  return DeviceType::CPU;
}

DeviceType CaffeCPUEvaluatorConstructor::get_output_buffer_type() {
  return DeviceType::CPU;
}

i32 CaffeCPUEvaluatorConstructor::get_number_of_outputs() {
  return static_cast<i32>(net_descriptor_.output_layer_names.size());
}

std::vector<std::string> CaffeCPUEvaluatorConstructor::get_output_names() {
  return net_descriptor_.output_layer_names;
}

u8* CaffeCPUEvaluatorConstructor::new_input_buffer(
  const EvaluatorConfig& config)
{
  return new u8[
    config.max_batch_size *
    config.max_frame_width *
    config.max_frame_height *
    3 *
    sizeof(u8)];
}

void CaffeCPUEvaluatorConstructor::delete_input_buffer(
  const EvaluatorConfig& config,
  u8* buffer)
{
  delete[] buffer;
}

void CaffeCPUEvaluatorConstructor::delete_output_buffer(
  const EvaluatorConfig& config,
  u8* buffer)
{
  delete[] buffer;
}

Evaluator* CaffeCPUEvaluatorConstructor::new_evaluator(
  const EvaluatorConfig& config)
{
  CaffeInputTransformer* transformer =
    transformer_factory_->construct(config, net_descriptor_);
  return new CaffeCPUEvaluator(config, net_descriptor_, transformer, 0);
}

}

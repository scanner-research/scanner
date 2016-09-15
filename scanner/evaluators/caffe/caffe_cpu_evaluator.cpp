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

#include "scanner/eval/caffe/caffe_cpu_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/opencv.h"
#include "scanner/util/util.h"

namespace scanner {

CaffeCPUEvaluator::CaffeCPUEvaluator(
  const EvaluatorConfig& config,
  const NetDescriptor& descriptor,
  int device_id)
  : descriptor_(descriptor),
    device_id_(device_id)
{
  caffe::Caffe::set_mode(device_type_to_caffe_mode(DeviceType::CPU));

  // Initialize our network
  net_.reset(new Net<float>(descriptor_.model_path, caffe::TEST));
  net_->CopyTrainedLayersFrom(descriptor_.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net_.blob_by_name(descriptor.input_layer_name)};

  // Get output blobs that we will extract net evaluation results from
  for (const std::string& output_layer_name : descriptor.output_layer_names)
  {
    const boost::shared_ptr<caffe::Blob<float>> output_blob{
      net_.blob_by_name(output_layer_name)};
    size_t output_size_per_frame = output_blob->count(1) * sizeof(float);
    output_sizes_.push_back(output_size_per_frame);
  }

  // Dimensions of network input image
  int input_height = input_blob->shape(2);
  int input_width = input_blob->shape(3);

}

CaffeCPUEvaluator::~CaffeCPUEvaluator() {
}

void CaffeCPUEvaluator::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;

  int frame_width = metadata.width;
  int frame_height = metadata.height;

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net_.blob_by_name(descriptor.input_layer_name)};

  // Dimensions of network input image
  int net_input_height = input_blob->shape(2);
  int net_input_width = input_blob->shape(3);

  // Resize into
  std::vector<float> mean_image = descriptor_.mean_image;
  cv::Mat unsized_mean_mat(
    descriptor_.mean_height * 3,
    descriptor_.mean_width,
    CV_32FC1,
    mean_image.data());
  // HACK(apoms): Resizing the mean like this is not likely to produce a correct
  //              result because we are resizing where the color channels are
  //              considered individual pixels.
  cv::resize(unsized_mean_mat, mean_mat_,
             cv::Size(net_input_width, net_input_height * 3));

  input_mat = cv::Mat(frame_height, frame_width, CV_8UC3);

  transformer_->configure(metadata);
}

void CaffeCPUEvaluator::evaluate(
  char* input_buffer,
  std::vector<char*> output_buffers,
  int batch_size)
{
  size_t frame_size =
    av_image_get_buffer_size(AV_PIX_FMT_NV12,
                             metadata.width,
                             metadata.height,
                             1);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net_.blob_by_name(descriptor_.input_layer_name)};

  // Dimensions of network input image
  int input_height = input_blob->shape(2);
  int input_width = input_blob->shape(3);

  if (input_blob->shape(0) != batch_size) {
    input_blob->Reshape({batch_size, 3, input_height, input_width});
  }

  float* net_input_buffer = input_blob->mutable_gpu_data();

  // Process batch of frames
  auto cv_start = now();
  args.profiler.add_interval("cv", cv_start, now());

  // Compute features
  auto net_start = now();
  net.Forward();
  args.profiler.add_interval("net", net_start, now());

  // Save batch of frames
  for (size_t i = 0; i < output_buffer_sizes.size(); ++i) {
    const std::string& output_layer_name =
      args.net_descriptor.output_layer_names[i];
    const boost::shared_ptr<caffe::Blob<float>> output_blob{
      net.blob_by_name(output_layer_name)};
    CU_CHECK(cudaMemcpy(
               output_buffers[i],
               output_blob->gpu_data(),
               batch_size * output_sizes[i],
               cudaMemcpyDeviceToHost));
  }
}

CaffeCPUEvaluatorConstructor::CaffeCPUEvaluatorConstructor(
  NetDescriptor net_descriptor,
  CaffeInputTransformerFactory* transformer_factory)
  : net_descriptor_(net_descriptor),
    transformer_factory_(transformer_factory)
{
}

CaffeCPUEvaluatorConstructor::~CaffeCPUEvaluatorConstructor() {
}

int CaffeCPUEvaluatorConstructor::get_number_of_devices() {
}

DeviceType CaffeCPUEvaluatorConstructor::get_input_buffer_type() {
}

DeviceType CaffeCPUEvaluatorConstructor::get_output_buffer_type() {
}

int CaffeCPUEvaluatorConstructor::get_number_of_outputs() {
}

std::vector<std::string> CaffeCPUEvaluatorConstructor::get_output_names() {
}

std::vector<size_t> CaffeCPUEvaluatorConstructor::get_output_element_sizes() {
}

char* CaffeCPUEvaluatorConstructor::new_input_buffer(
  const EvaluatorConfig& config)
{
}

void CaffeCPUEvaluatorConstructor::delete_input_buffer(
  const EvaluatorConfig& config,
  char* buffer)
{
}

std::vector<char*> CaffeCPUEvaluatorConstructor::new_output_buffers(
  const EvaluatorConfig& config,
  int num_inputs)
{
}

void CaffeCPUEvaluatorConstructor::delete_output_buffers(
  const EvaluatorConfig& config,
  std::vector<char*> buffers)
{
}

Evaluator* CaffeCPUEvaluatorConstructor::new_evaluator(
  const EvaluatorConfig& config)
{
  CaffeInputTransformer* transformer = transformer_factory_->construct(config);
  return new CaffeCPUEvaluator(config, descriptor, 
}

}

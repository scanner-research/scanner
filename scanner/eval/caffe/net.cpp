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

#include "scanner/eval/caffe/net.h"

namespace scanner {

CaffeEvaluator::CaffeEvaluator() {}

virtual int get_max_batch_size() override;

virtual int get_num_staging_buffers() override;

virtual char* get_staging_buffer(int index) override;

virtual void evaluate(
  int staging_buffer_index,
  char* output,
  int batch_size) override;

class CaffeEvaluatorConstructor : public EvaluatorConstructor {
public:
  CaffeEvaluatorConstructor::CaffeEvaluatorConstructor() {
  }
  virutal ~CaffeEvaluatorConstructor();

  virtual BufferType get_input_buffer_type() override;

  virtual BufferType get_output_buffer_type() override;

  virtual int get_number_of_devices() override;

  virtual Evaluator* new_evaluator(
    int device_id,
    int max_batch_size,
    int num_staging_buffers,
    size_t staging_buffer_size) override;
};

}

  CU_CHECK(cudaSetDevice(args.gpu_device_id));
  cv::cuda::setDevice(args.gpu_device_id);

  // Setup caffe net
  NetBundle net_bundle{args.net_descriptor, args.pu_device_id};

  caffe::Net<float>& net = net_bundle.get_net();

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net.blob_by_name(args.net_descriptor.input_layer_name)};

  // Get output blobs that we will extract net evaluation results from
  std::vector<size_t> output_sizes;
  for (const std::string& output_layer_name :
         args.net_descriptor.output_layer_names)
  {
    const boost::shared_ptr<caffe::Blob<float>> output_blob{
      net.blob_by_name(output_layer_name)};
    size_t output_size_per_frame = output_blob->count(1) * sizeof(float);
    output_sizes.push_back(output_size_per_frame);
  }

  // Dimensions of network input image
  int inputHeight = input_blob->shape(2);
  int inputWidth = input_blob->shape(3);

  // Resize into
  std::vector<float> mean_image = args.net_descriptor.mean_image;
  cv::Mat cpu_mean_mat(
    args.net_descriptor.mean_height * 3,
    args.net_descriptor.mean_width,
    CV_32FC1,
    mean_image.data());
  cv::cuda::GpuMat unsized_mean_mat(cpu_mean_mat);
  cv::cuda::GpuMat mean_mat;
  // HACK(apoms): Resizing the mean like this is not likely to produce a correct
  //              result because we are resizing where the color channels are
  //              considered individual pixels.
  cv::cuda::resize(unsized_mean_mat, mean_mat,
                   cv::Size(inputWidth, inputHeight * 3));

  // OpenCV matrices
  std::vector<cv::cuda::Stream> cv_streams(NUM_CUDA_STREAMS);

  std::vector<cv::cuda::GpuMat> input_mats;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    input_mats.push_back(
      cv::cuda::GpuMat(args.metadata[0].height + args.metadata[0].height / 2,
                       args.metadata[0].width,
                       CV_8UC1));
  }

  std::vector<cv::cuda::GpuMat> rgba_mat;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    rgba_mat.push_back(
      cv::cuda::GpuMat(args.metadata[0].height,
                       args.metadata[0].width,
                       CV_8UC4));
  }

  std::vector<cv::cuda::GpuMat> rgb_mat;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    rgb_mat.push_back(
      cv::cuda::GpuMat(args.metadata[0].height,
                       args.metadata[0].width,
                       CV_8UC3));
  }

  std::vector<cv::cuda::GpuMat> conv_input;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    conv_input.push_back(
      cv::cuda::GpuMat(inputHeight, inputWidth, CV_8UC3));
  }

  std::vector<cv::cuda::GpuMat> conv_planar_input;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    conv_planar_input.push_back(
      cv::cuda::GpuMat(inputHeight * 3, inputWidth, CV_8UC1));
  }

  std::vector<cv::cuda::GpuMat> float_conv_input;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    float_conv_input.push_back(
      cv::cuda::GpuMat(inputHeight * 3, inputWidth, CV_32FC1));
  }

  std::vector<cv::cuda::GpuMat> normed_input;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    normed_input.push_back(
      cv::cuda::GpuMat(inputHeight * 3, inputWidth, CV_32FC1));
  }

  std::vector<cv::cuda::GpuMat> scaled_input;
  for (size_t i = 0; i < NUM_CUDA_STREAMS; ++i) {
    scaled_input.push_back(
      cv::cuda::GpuMat(inputHeight * 3, inputWidth, CV_32FC1));
  }



class CaffeEvaluatorConstructor : public EvaluatorConstructor {
public:
  CaffeEvaluatorConstructor();
  virutal ~CaffeEvaluatorConstructor();

  virtual BufferType get_input_buffer_type() override;

  virtual BufferType get_output_buffer_type() override;

  virtual int get_number_of_devices() override;

  virtual Evaluator* new_evaluator(
    int device_id,
    int max_batch_size,
    int num_staging_buffers,
    size_t staging_buffer_size) override;
};

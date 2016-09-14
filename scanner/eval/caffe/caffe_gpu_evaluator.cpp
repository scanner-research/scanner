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

#include "scanner/eval/caffe/caffe_gpu_evaluator.h"

#include "scanner/util/common.h"
#include "scanner/util/cuda.h"
#include "scanner/util/opencv.h"
#include "scanner/util/util.h"

namespace scanner {

CaffeGPUEvaluator::CaffeGPUEvaluator(
  EvaluatorConfig config,
  NetDescriptor descriptor,
  int device_id)
  : CaffeEvaluator(),
    descriptor_(descriptor),
    device_id_(device_id),
    cv_streams(NUM_CUDA_STREAMS)
{
  CU_CHECK(cudaSetDevice(device_id));
  caffe::Caffe::set_mode(device_type_to_caffe_mode(DeviceType::GPU));
  caffe::Caffe::SetDevice(device_id);
  cv::cuda::setDevice(device_id);

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
  int inputHeight = input_blob->shape(2);
  int inputWidth = input_blob->shape(3);

  // Resize into
  std::vector<float> mean_image = descriptor.mean_image;
  cv::Mat cpu_mean_mat(
    descriptor.mean_height * 3,
    descriptor.mean_width,
    CV_32FC1,
    mean_image.data());
  cv::cuda::GpuMat unsized_mean_mat(cpu_mean_mat);
  cv::cuda::GpuMat mean_mat;
  // HACK(apoms): Resizing the mean like this is not likely to produce a correct
  //              result because we are resizing where the color channels are
  //              considered individual pixels.
  cv::cuda::resize(unsized_mean_mat, mean_mat_,
                   cv::Size(inputWidth, inputHeight * 3));

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
}

CaffeGPUEvaluator::~CaffeGPUEvaluator() {
}

void CaffeGPUEvaluator::evaluate(
  const DatasetItemMetadata& metadata,
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
  for (int i = 0; i < batch_size; ++i) {
    int sid = i % NUM_CUDA_STREAMS;
    cv::cuda::Stream& cv_stream = cv_streams[sid];

    char* buffer = frame_buffer + frame_size * (i + frame_offset);

    input_mats[sid] =
      cv::cuda::GpuMat(
        metadata.height + metadata.height / 2,
        metadata.width,
        CV_8UC1,
        buffer);

    convertNV12toRGBA(input_mats[sid], rgba_mat[sid],
                      metadata.width, metadata.height,
                      cv_stream);
    // BGR -> RGB for helnet
    cv::cuda::cvtColor(rgba_mat[sid], rgb_mat[sid], CV_BGRA2RGB, 0,
                       cv_stream);
    cv::cuda::resize(rgb_mat[sid], conv_input[sid],
                     cv::Size(inputWidth, inputHeight),
                     0, 0, cv::INTER_LINEAR, cv_stream);
    // Changed from interleaved BGR to planar BGR
    convertRGBInterleavedToPlanar(conv_input[sid], conv_planar_input[sid],
                                  inputWidth, inputHeight,
                                  cv_stream);
    conv_planar_input[sid].convertTo(
      float_conv_input[sid], CV_32FC1, cv_stream);
    cv::cuda::subtract(float_conv_input[sid], mean_mat, normed_input[sid],
                       cv::noArray(), -1, cv_stream);
    // For helnet, we need to transpose so width is fasting moving dim
    // and normalize to 0 - 1
    cv::cuda::divide(normed_input[sid], 255.0f, scaled_input[sid],
                     1, -1, cv_stream);
    cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
    CU_CHECK(cudaMemcpy2DAsync(
               net_input_buffer + i * (inputWidth * inputHeight * 3),
               inputWidth * sizeof(float),
               scaled_input[sid].data,
               scaled_input[sid].step,
               inputWidth * sizeof(float),
               inputHeight * 3,
               cudaMemcpyDeviceToDevice,
               s));

    // For checking for proper encoding
    if (false && ((current_frame + i) % 512) == 0) {
      CU_CHECK(cudaDeviceSynchronize());
      size_t image_size = metadata.width * metadata.height * 3;
      uint8_t* image_buff = new uint8_t[image_size];

      for (int i = 0; i < rgb_mat[sid].rows; ++i) {
        CU_CHECK(cudaMemcpy(image_buff + metadata.width * 3 * i,
                            rgb_mat[sid].ptr<uint8_t>(i),
                            metadata.width * 3,
                            cudaMemcpyDeviceToHost));
      }
      JPEGWriter writer;
      writer.header(metadata.width, metadata.height, 3, JPEG::COLOR_RGB);
      std::vector<uint8_t*> rows(metadata.height);
      for (int i = 0; i < metadata.height; ++i) {
        rows[i] = image_buff + metadata.width * 3 * i;
      }
      std::string image_path =
        "frame" + std::to_string(current_frame + i) + ".jpg";
      writer.write(image_path, rows.begin());
      delete[] image_buff;
    }
  }
  CU_CHECK(cudaDeviceSynchronize());
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

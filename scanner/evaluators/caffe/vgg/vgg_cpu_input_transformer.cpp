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

#pragma once

#include "scanner/evaluators/caffe/vgg/vgg_input_transformer.h"

namespace scanner {

void VGGCPUInputTransformer::configure(const DatasetItemMetadata& metadata) {
  metadata_ = metadata;

  int frame_width = metadata.width;
  int frame_height = metadata.height;

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
             cv::Size(NET_INPUT_WIDTH, NET_INPUT_HEIGHT * 3));

  input_mat = cv::Mat(frame_height, frame_width, CV_8UC3);
  conv_input = cv::Mat(input_height, input_width, CV_8UC3);
  conv_planar_input = cv::Mat(input_height * 3, input_width, CV_8UC1);
  float_conv_input = cv::Mat(input_height * 3, input_width, CV_32FC1);
  normed_input = cv::Mat(input_height * 3, input_width, CV_32FC1);
  scaled_input = cv::Mat(input_height * 3, input_width, CV_32FC1);
}

void VGGInputTransformer::transform_input(
  char* input_buffer,
  float* net_input,
  int batch_size) 
{
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
                     cv::Size(input_width, input_height),
                     0, 0, cv::INTER_LINEAR, cv_stream);
    // Changed from interleaved BGR to planar BGR
    convertRGBInterleavedToPlanar(conv_input[sid], conv_planar_input[sid],
                                  input_width, input_height,
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
               net_input_buffer + i * (input_width * input_height * 3),
               input_width * sizeof(float),
               scaled_input[sid].data,
               scaled_input[sid].step,
               input_width * sizeof(float),
               input_height * 3,
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
}

CaffeInputTransformer* VGGCPUInputTransformerFactory::construct(
  const EvaluatorConfig& config)
{
  return new VGGInputTransformer();
}

}

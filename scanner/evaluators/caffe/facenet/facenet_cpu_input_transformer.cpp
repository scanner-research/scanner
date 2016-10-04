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

#include "scanner/util/common.h"
#include "scanner/evaluators/caffe/facenet/facenet_cpu_input_transformer.h"

namespace scanner {

FacenetCPUInputTransformer::FacenetCPUInputTransformer(
    const NetDescriptor& descriptor)
    : descriptor_(descriptor), net_input_width_(224), net_input_height_(224) {}

void FacenetCPUInputTransformer::configure(const DatasetItemMetadata& metadata,
                                           caffe::Net<float>* net) {
  metadata_ = metadata;

  net_input_width_ = metadata.width;
  net_input_height_ = metadata.height;

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net->blob_by_name(descriptor_.input_layer_name)};

  if (input_blob->shape(2) != metadata.width ||
      input_blob->shape(3) != metadata.height) {
    input_blob->Reshape({input_blob->shape(0), input_blob->shape(1),
                         metadata.width, metadata.height});

    mean_mat_ = cv::Mat(
        net_input_height_, net_input_width_, CV_32FC3,
        cv::Scalar(descriptor_.mean_colors[0], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[2]));

    float_input = cv::Mat(net_input_height_, net_input_width_, CV_32FC3);
    normalized_input = cv::Mat(net_input_height_, net_input_width_, CV_32FC3);
    flipped_input = cv::Mat(net_input_width_, net_input_height_, CV_32FC3);
    for (i32 i = 0; i < 3; ++i) {
      input_planes.push_back(
          cv::Mat(net_input_width_, net_input_height_, CV_32FC1));
    }
    planar_input = cv::Mat(net_input_width_ * 3, net_input_height_, CV_32FC1);
  }
}

void FacenetCPUInputTransformer::transform_input(i32 input_count,
                                                 u8* input_buffer,
                                                 f32* net_input) {
  size_t frame_size = net_input_width_ * net_input_height_ * 3 * sizeof(u8);
  for (i32 i = 0; i < input_count; ++i) {
    u8* buffer = input_buffer + frame_size * i;

    cv::Mat input_mat =
        cv::Mat(net_input_height_, net_input_width_, CV_8UC3, buffer);

    // Changed from interleaved RGB to planar RGB
    input_mat.convertTo(float_input, CV_32FC3);
    cv::subtract(float_input, mean_mat_, normalized_input);
    cv::transpose(normalized_input, flipped_input);
    cv::split(flipped_input, input_planes);
    cv::vconcat(input_planes, planar_input);
    assert(planar_input.cols == net_input_height_);
    for (i32 r = 0; r < planar_input.rows; ++r) {
      u8* mat_pos = planar_input.data + r * planar_input.step;
      u8* input_pos = reinterpret_cast<u8*>(
          net_input + i * (net_input_width_ * net_input_height_ * 3) +
          r * net_input_height_);
      memcpy(input_pos, mat_pos, planar_input.cols * sizeof(float));
    }
  }
}

CaffeInputTransformer* FacenetCPUInputTransformerFactory::construct(
    const EvaluatorConfig& config, const NetDescriptor& descriptor) {
  return new FacenetCPUInputTransformer(descriptor);
}
}

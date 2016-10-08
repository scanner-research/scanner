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

#include "scanner/evaluators/caffe/yolo/yolo_cpu_input_transformer.h"

namespace scanner {

YoloCPUInputTransformer::YoloCPUInputTransformer(
    const NetDescriptor& descriptor)
    : descriptor_(descriptor) {
  // Resize into
  std::vector<float> mean_image = descriptor_.mean_image;
  cv::Mat unsized_mean_mat(descriptor_.mean_height * 3, descriptor_.mean_width,
                           CV_32FC1, mean_image.data());
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

void YoloCPUInputTransformer::configure(const VideoMetadata& metadata,
                                        caffe::Net<float>* net) {
  metadata_ = metadata;
}

void YoloCPUInputTransformer::transform_input(i32 input_count, u8* input_buffer,
                                              f32* net_input) {
  i32 frame_width = metadata_.width();
  i32 frame_height = metadata_.height();
  size_t frame_size = frame_width * frame_height * 3 * sizeof(u8);
  for (i32 i = 0; i < input_count; ++i) {
    u8* buffer = input_buffer + frame_size * i;

    cv::Mat input_mat = cv::Mat(frame_height, frame_width, CV_8UC3, buffer);

    cv::resize(input_mat, resized_input,
               cv::Size(NET_INPUT_WIDTH, NET_INPUT_HEIGHT), 0, 0,
               cv::INTER_LINEAR);
    cv::cvtColor(resized_input, bgr_input, CV_RGB2BGR);
    // Changed from interleaved BGR to planar BGR
    cv::split(bgr_input, input_planes);
    cv::vconcat(input_planes, planar_input);
    planar_input.convertTo(float_input, CV_32FC1);
    cv::subtract(float_input, mean_mat, normalized_input);
    for (i32 r = 0; r < normalized_input.rows; ++r) {
      u8* mat_pos = normalized_input.data + r * normalized_input.step;
      u8* input_pos = reinterpret_cast<u8*>(
          net_input + i * (NET_INPUT_WIDTH * NET_INPUT_HEIGHT * 3) +
          r * NET_INPUT_WIDTH);
      memcpy(input_pos, mat_pos, normalized_input.cols * sizeof(float));
    }
  }
}

CaffeInputTransformer* YoloCPUInputTransformerFactory::construct(
    const EvaluatorConfig& config, const NetDescriptor& descriptor) {
  return new YoloCPUInputTransformer(descriptor);
}
}

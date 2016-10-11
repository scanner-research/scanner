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

#include "scanner/evaluators/caffe/facenet/facenet_gpu_input_transformer.h"

namespace scanner {

FacenetGPUInputTransformer::FacenetGPUInputTransformer(
    const NetDescriptor& descriptor)
    : descriptor_(descriptor),
      num_cuda_streams_(32),
      initialized_(false),
      net_input_width_(224),
      net_input_height_(224),
      streams_(num_cuda_streams_) {
}

void FacenetGPUInputTransformer::configure(const VideoMetadata& metadata,
                                           caffe::Net<float>* net) {
  metadata_ = metadata;

  net_input_width_ = metadata.width();
  net_input_height_ = metadata.height();

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net->blob_by_name(descriptor_.input_layer_name)};
  if (!initialized_ ||
      input_blob->shape(2) != metadata.width() ||
      input_blob->shape(3) != metadata.height()) {
    initialize(net);
  }
}

void FacenetGPUInputTransformer::transform_input(i32 input_count,
                                                 u8* input_buffer,
                                                 f32* net_input) {
  size_t frame_size = net_input_width_ * net_input_height_ * 3 * sizeof(u8);
  for (i32 i = 0; i < input_count; ++i) {
    int sid = i % num_cuda_streams_;
    cv::cuda::Stream& cv_stream = streams_[sid];

    u8* buffer = input_buffer + frame_size * i;
    frame_input_[sid] =
        cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_8UC3, buffer);
    frame_input_[sid].convertTo(float_input_[sid], CV_32FC3, cv_stream);
    cv::cuda::subtract(float_input_[sid], mean_mat_, normalized_input_[sid],
                       cv::noArray(), -1, cv_stream);
    // Changed from interleaved RGB to planar RGB
    cv::cuda::split(normalized_input_[sid], input_planes_[sid], cv_stream);
    cv::cuda::transpose(input_planes_[sid][0], flipped_planes_[sid][0],
                        cv_stream);
    cv::cuda::transpose(input_planes_[sid][1], flipped_planes_[sid][1],
                        cv_stream);
    cv::cuda::transpose(input_planes_[sid][2], flipped_planes_[sid][2],
                        cv_stream);
    auto& plane1 = flipped_planes_[sid][0];
    auto& plane2 = flipped_planes_[sid][1];
    auto& plane3 = flipped_planes_[sid][2];
    auto& planar_input = planar_input_[sid];
    plane1.copyTo(planar_input(cv::Rect(0, net_input_width_ * 0,
                                        net_input_height_, net_input_width_)));
    plane2.copyTo(planar_input(cv::Rect(0, net_input_width_ * 1,
                                        net_input_height_, net_input_width_)));
    plane3.copyTo(planar_input(cv::Rect(0, net_input_width_ * 2,
                                        net_input_height_, net_input_width_)));
    assert(planar_input.cols == net_input_height_);
    cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
    CU_CHECK(cudaMemcpy2DAsync(
        net_input + i * (net_input_width_ * net_input_height_ * 3),
        net_input_height_ * sizeof(float), planar_input.data, planar_input.step,
        net_input_height_ * sizeof(float), net_input_width_ * 3,
        cudaMemcpyDeviceToDevice, s));
  }
  for (cv::cuda::Stream& s : streams_) {
    s.waitForCompletion();
  }
}

void FacenetGPUInputTransformer::initialize(caffe::Net<float>* net) {
  initialized_ = true;

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net->blob_by_name(descriptor_.input_layer_name)};

  input_blob->Reshape({input_blob->shape(0), input_blob->shape(1),
                       net_input_width_, net_input_height_});

  mean_mat_ = cv::cuda::GpuMat(
      net_input_height_, net_input_width_, CV_32FC3,
      cv::Scalar(descriptor_.mean_colors[0], descriptor_.mean_colors[1],
                 descriptor_.mean_colors[2]));

  frame_input_.clear();
  float_input_.clear();
  normalized_input_.clear();
  input_planes_.clear();
  flipped_planes_.clear();
  planar_input_.clear();
  for (size_t i = 0; i < num_cuda_streams_; ++i) {
    frame_input_.push_back(
        cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
    float_input_.push_back(
        cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
    normalized_input_.push_back(
        cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
    std::vector<cv::cuda::GpuMat> planes;
    std::vector<cv::cuda::GpuMat> flipped_planes;
    for (i32 i = 0; i < 3; ++i) {
      planes.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
      flipped_planes.push_back(
          cv::cuda::GpuMat(net_input_width_, net_input_height_, CV_32FC1));
    }
    input_planes_.push_back(planes);
    flipped_planes_.push_back(flipped_planes);
    planar_input_.push_back(
        cv::cuda::GpuMat(net_input_width_ * 3, net_input_height_, CV_32FC1));
    }
  }
}

CaffeInputTransformer* FacenetGPUInputTransformerFactory::construct(
    const EvaluatorConfig& config, const NetDescriptor& descriptor) {
  return new FacenetGPUInputTransformer(descriptor);
}
}

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

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"
#include "scanner/util/opencv.h"
#include "stdlib/stdlib.pb.h"

#include <opencv2/core/cuda_stream_accessor.hpp>

namespace scanner {

class CPM2InputKernel : public VideoKernel {
 public:
  CPM2InputKernel(const Kernel::Config& config)
      : VideoKernel(config),
        device_(config.devices[0])
#ifdef HAVE_CUDA
        ,
        num_cuda_streams_(32),
        streams_(num_cuda_streams_)
#endif
  {
    proto::CPM2Args args;
    args.ParseFromArray(config.args.data(), config.args.size());
    args_.CopyFrom(args.caffe_args());
    scale_ = args.scale();
  }

  void new_frame_info() override {
    frame_width_ = frame_info_.width();
    frame_height_ = frame_info_.height();

    resize_width_ = frame_width_ * scale_;
    resize_height_ = frame_height_ * scale_;

    width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;
    height_padding_ = (resize_height_ % 8) ? 8 - (resize_height_ % 8) : 0;

    net_input_width_ = resize_width_ + width_padding_;
    net_input_height_ = resize_height_ + height_padding_;

    cv::cuda::setDevice(device_.id);
    cudaSetDevice(device_.id);

    frame_input_.clear();
    bgr_input_.clear();
    resized_input_.clear();
    padded_input_.clear();
    float_input_.clear();
    input_planes_.clear();
    flipped_planes_.clear();
    planar_input_.clear();
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      frame_input_.push_back(
          cv::cuda::GpuMat(frame_height_, frame_width_, CV_8UC3));
      bgr_input_.push_back(
          cv::cuda::GpuMat(frame_height_, frame_width_, CV_8UC3));
      resized_input_.push_back(
          cv::cuda::GpuMat(resize_height_, resize_width_, CV_8UC3));
      padded_input_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_8UC3));
      float_input_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      std::vector<cv::cuda::GpuMat> planes;
      for (i32 i = 0; i < 3; ++i) {
        planes.push_back(
            cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
      }
      input_planes_.push_back(planes);
      planar_input_.push_back(
          cv::cuda::GpuMat(net_input_height_ * 3, net_input_width_, CV_32FC1));
    }
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    auto eval_start = now();

    auto& frame_col = input_columns[0];
    auto& frame_info_col = input_columns[1];
    check_frame_info(device_, frame_info_col);

    i32 input_count = frame_col.rows.size();

    streams_.resize(0);
    streams_.resize(num_cuda_streams_);

    size_t net_input_size =
        net_input_width_ * net_input_height_ * 3 * sizeof(f32);
    u8* output_block =
        new_block_buffer(device_, input_count * net_input_size, input_count);

    for (i32 i = 0; i < input_count; ++i) {
      f32* net_input =
          reinterpret_cast<f32*>(output_block + net_input_size * i);

      int sid = i % num_cuda_streams_;
      cv::cuda::Stream& cv_stream = streams_[sid];

      u8* buffer = frame_col.rows[i].buffer;
      assert(frame_col.rows[i].size == frame_height_ * frame_width_ * 3);
      frame_input_[sid] =
          cv::cuda::GpuMat(frame_height_, frame_width_, CV_8UC3, buffer);
      cv::cuda::cvtColor(frame_input_[sid], bgr_input_[sid], cv::COLOR_RGB2BGR,
                         0, cv_stream);
      cv::cuda::resize(bgr_input_[sid], resized_input_[sid],
                       cv::Size(resize_width_, resize_height_), 0, 0,
                       cv::INTER_CUBIC, cv_stream);
      cv::cuda::copyMakeBorder(resized_input_[sid], padded_input_[sid], 0,
                               height_padding_, 0, width_padding_,
                               cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128),
                               cv_stream);
      padded_input_[sid].convertTo(float_input_[sid], CV_32FC3, (1.0f / 256.0f),
                                   -0.5f, cv_stream);
      // Changed from interleaved BGR to planar RGB
      cv::cuda::split(float_input_[sid], input_planes_[sid], cv_stream);
      auto& plane1 = input_planes_[sid][0];
      auto& plane2 = input_planes_[sid][1];
      auto& plane3 = input_planes_[sid][2];
      auto& planar_input = planar_input_[sid];
      plane1.copyTo(planar_input(cv::Rect(
          0, net_input_height_ * 0, net_input_width_, net_input_height_)));
      plane2.copyTo(planar_input(cv::Rect(
          0, net_input_height_ * 1, net_input_width_, net_input_height_)));
      plane3.copyTo(planar_input(cv::Rect(
          0, net_input_height_ * 2, net_input_width_, net_input_height_)));
      assert(planar_input.cols == net_input_width_);
      cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
      CU_CHECK(cudaMemcpy2DAsync(
          net_input, net_input_width_ * sizeof(float), planar_input.data,
          planar_input.step, net_input_width_ * sizeof(float),
          net_input_height_ * 3, cudaMemcpyDeviceToDevice, s));

      INSERT_ROW(output_columns[0], (u8*)net_input, net_input_size);
    }
    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }

    if (profiler_) {
      profiler_->add_interval("cpm2_input", eval_start, now());
    }
  }

 private:
  DeviceHandle device_;
  proto::CaffeArgs args_;
  f32 scale_;

  i32 frame_width_;
  i32 frame_height_;
  i32 resize_width_;
  i32 resize_height_;
  i32 width_padding_;
  i32 height_padding_;
  i32 net_input_width_;
  i32 net_input_height_;

  i32 num_cuda_streams_;
  std::vector<cv::cuda::Stream> streams_;
  std::vector<cv::cuda::GpuMat> frame_input_;
  std::vector<cv::cuda::GpuMat> bgr_input_;
  std::vector<cv::cuda::GpuMat> resized_input_;
  std::vector<cv::cuda::GpuMat> padded_input_;
  std::vector<cv::cuda::GpuMat> float_input_;
  std::vector<std::vector<cv::cuda::GpuMat>> input_planes_;
  std::vector<std::vector<cv::cuda::GpuMat>> flipped_planes_;
  std::vector<cv::cuda::GpuMat> planar_input_;
};

REGISTER_OP(CPM2Input).inputs({"frame", "frame_info"}).outputs({"cpm2_input"});
REGISTER_KERNEL(CPM2Input, CPM2InputKernel)
    .device(DeviceType::GPU)
    .num_devices(1);
}

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

#include "scanner/evaluators/caffe/cpm/cpm_input_evaluator.h"
#include "scanner/evaluators/serialize.h"
#include "scanner/evaluators/types.pb.h"
#include "scanner/util/memory.h"

#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace scanner {

CPMInputEvaluator::CPMInputEvaluator(DeviceType device_type, i32 device_id,
                                     const NetDescriptor& descriptor,
                                     i32 batch_size)
    : device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      batch_size_(batch_size)
#ifdef HAVE_CUDA
      ,
      num_cuda_streams_(32),
      streams_(num_cuda_streams_)
#endif
{
}

void CPMInputEvaluator::configure(const InputFormat& metadata) {
  metadata_ = metadata;

  f32 scale = static_cast<f32>(box_size_) / metadata.height();
  // Calculate width by scaling by box size
  resize_width_ = metadata.width() * scale;
  resize_height_ = metadata.height() * scale;

  width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;
  padded_width_ = resize_width_ + width_padding_;

  net_input_width_ = box_size_;
  net_input_height_ = box_size_;

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    streams_.resize(0);
    streams_.resize(num_cuda_streams_);

    frame_input_g_.clear();
    bgr_input_g_.clear();
    resized_input_g_.clear();
    padded_input_g_.clear();
    cropped_input_g_.clear();
    float_input_g_.clear();
    input_planes_g_.clear();
    flipped_planes_g_.clear();
    planar_input_g_.clear();
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      frame_input_g_.push_back(
          cv::cuda::GpuMat(metadata.height(), metadata.width(), CV_8UC3));
      bgr_input_g_.push_back(
          cv::cuda::GpuMat(metadata.height(), metadata.width(), CV_8UC3));
      resized_input_g_.push_back(
          cv::cuda::GpuMat(resize_height_, resize_width_, CV_8UC3));
      padded_input_g_.push_back(
          cv::cuda::GpuMat(resize_height_, padded_width_, CV_8UC3));
      cropped_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_8UC3));
      float_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      std::vector<cv::cuda::GpuMat> planes;
      for (i32 i = 0; i < 3; ++i) {
        planes.push_back(
            cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
      }
      input_planes_g_.push_back(planes);
      planar_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_ * 4, net_input_width_, CV_32FC1));
    }

    i32 k_size = box_size_ - 1 + (box_size_ % 2);
    cv::Mat gaussian = cv::getGaussianKernel(k_size, 21.0, CV_32F);
    gaussian_channel_g_ = cv::cuda::GpuMat(gaussian * gaussian.t());
    double max;
    cv::cuda::minMax(gaussian_channel_g_, NULL, &max);
    cv::cuda::multiply(gaussian_channel_g_, 1 / max, gaussian_channel_g_);
    assert(gaussian_channel_g_.cols == k_size);
    assert(gaussian_channel_g_.rows == k_size);
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    LOG(FATAL) << "CPU not implemented yet.";
    mean_mat_c_ = cv::Mat(
        net_input_height_, net_input_width_, CV_32FC3,
        cv::Scalar(descriptor_.mean_colors[0], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[2]));

    float_input_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC3);
    normalized_input_c_ =
        cv::Mat(net_input_height_, net_input_width_, CV_32FC3);
    flipped_input_c_ = cv::Mat(net_input_width_, net_input_height_, CV_32FC3);
    for (i32 i = 0; i < 3; ++i) {
      input_planes_c_.push_back(
          cv::Mat(net_input_width_, net_input_height_, CV_32FC1));
    }
    planar_input_c_ =
        cv::Mat(net_input_width_ * 3, net_input_height_, CV_32FC1);
  }
}

void CPMInputEvaluator::evaluate(const BatchedColumns& input_columns,
                                 BatchedColumns& output_columns) {
  auto eval_start = now();

  size_t frame_size = net_input_width_ * net_input_height_ * 3;
  i32 input_count = input_columns[0].rows.size();

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    i32 sid = 0;
    for (i32 i = 0; i < input_count; ++i) {
      u8* buffer = input_columns[0].rows[i].buffer;
      assert(input_columns[0].rows[i].size ==
             metadata_.height() * metadata_.width() * 3);

      cv::cuda::Stream& cv_stream = streams_[sid];
      cudaStream_t cuda_s = cv::cuda::StreamAccessor::getStream(cv_stream);

      frame_input_g_[sid] = cv::cuda::GpuMat(
          metadata_.height(), metadata_.width(), CV_8UC3, buffer);
      cv::cuda::cvtColor(frame_input_g_[sid], bgr_input_g_[sid],
                         cv::COLOR_RGB2BGR, 0, cv_stream);
      cv::cuda::resize(bgr_input_g_[sid], resized_input_g_[sid],
                       cv::Size(resize_width_, resize_height_), 0, 0,
                       cv::INTER_CUBIC, cv_stream);
      cv::cuda::copyMakeBorder(resized_input_g_[sid], padded_input_g_[sid], 0,
                               0, 0, width_padding_, cv::BORDER_CONSTANT,
                               cv::Scalar(128, 128, 128), cv_stream);

      std::vector<scanner::Point> center_points;
      {
        u8* center_points_buffer = input_columns[1].rows[i].buffer;
        size_t center_points_size = input_columns[1].rows[i].size;
        u8* cpu_buffer = new u8[center_points_size];
        memcpy_buffer(cpu_buffer, DeviceType::CPU, 0, center_points_buffer,
                      device_type_, device_id_, center_points_size);
        center_points = deserialize_proto_vector<scanner::Point>(
            cpu_buffer, center_points_size);
        delete[] cpu_buffer;
      }
      for (const scanner::Point& pt : center_points) {
        i32 net_input_size =
            (frame_size + (net_input_width_ * net_input_height_)) * sizeof(f32);
        f32* net_input = nullptr;
        CU_CHECK(cudaMalloc((void**)&net_input, net_input_size));

        // Get center location
        f32 center_x = pt.x();
        f32 center_y = pt.y();

        f32 x1 = center_x - box_size_ / 2;
        f32 y1 = center_y - box_size_ / 2;
        f32 x2 = center_x + box_size_ / 2;
        f32 y2 = center_y + box_size_ / 2;
        f32 x1_adjustment = (x1 > 0 ? 0 : -x1);
        f32 y1_adjustment = (y1 > 0 ? 0 : -y1);
        f32 x2_adjustment = (x2 < padded_width_ ? 0 : padded_width_ - x2);
        f32 y2_adjustment = (y2 < resize_height_ ? 0 : resize_height_ - y2);
        cv::Range crop_src_width(x1 + x1_adjustment, x2 + x2_adjustment);
        cv::Range crop_src_height(y1 + y1_adjustment, y2 + y2_adjustment);
        cv::Range crop_dest_width(x1_adjustment,
                                  net_input_width_ + x2_adjustment);
        cv::Range crop_dest_height(y1_adjustment,
                                   net_input_height_ + y2_adjustment);

        cropped_input_g_[sid].setTo(cv::Scalar(128, 128, 128), cv_stream);
        cv::cuda::GpuMat crop_src_window =
            padded_input_g_[sid](crop_src_height, crop_src_width);
        cv::cuda::GpuMat crop_dest_window =
            cropped_input_g_[sid](crop_dest_height, crop_dest_width);
        crop_src_window.copyTo(crop_dest_window, cv_stream);
        cropped_input_g_[sid].convertTo(float_input_g_[sid], CV_32FC3,
                                        (1.0f / 256.0f), -0.5f, cv_stream);
        // Changed from interleaved BGR to planar RGB
        cv::cuda::split(float_input_g_[sid], input_planes_g_[sid], cv_stream);
        auto& plane1 = input_planes_g_[sid][0];
        auto& plane2 = input_planes_g_[sid][1];
        auto& plane3 = input_planes_g_[sid][2];
        auto& planar_input = planar_input_g_[sid];
        planar_input.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), cv_stream);
        plane1.copyTo(
            planar_input(cv::Rect(0, net_input_height_ * 0, net_input_width_,
                                  net_input_height_)),
            cv_stream);
        plane2.copyTo(
            planar_input(cv::Rect(0, net_input_height_ * 1, net_input_width_,
                                  net_input_height_)),
            cv_stream);
        plane3.copyTo(
            planar_input(cv::Rect(0, net_input_height_ * 2, net_input_width_,
                                  net_input_height_)),
            cv_stream);
        gaussian_channel_g_.copyTo(
            planar_input(cv::Rect(0, net_input_height_ * 3,
                                  gaussian_channel_g_.cols,
                                  gaussian_channel_g_.rows)),
            cv_stream);
        assert(planar_input.cols == net_input_width_);
        CU_CHECK(cudaMemcpy2DAsync(
            net_input, net_input_width_ * sizeof(float), planar_input.data,
            planar_input.step, net_input_width_ * sizeof(float),
            net_input_height_ * 4, cudaMemcpyDeviceToDevice, cuda_s));

        output_columns[1].rows.push_back(Row{(u8*)net_input, net_input_size});
      }
      sid += 1;
      sid %= num_cuda_streams_;
    }
    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    assert(false);
    // for (i32 frame = 0; frame < input_count; frame += batch_size_) {
    //   i32 batch_count = std::min(input_count - frame, batch_size_);
    //   f32* net_input = reinterpret_cast<f32*>(
    //       new u8[frame_size * batch_count * sizeof(f32)]);

    //   for (i32 i = 0; i < batch_count; ++i) {
    //     u8* buffer = input_buffers[0][frame + i];

    //     cv::Mat input_mat =
    //         cv::Mat(net_input_height_, net_input_width_, CV_8UC3, buffer);

    //     // Changed from interleaved RGB to planar RGB
    //     input_mat.convertTo(float_input_c_, CV_32FC3);
    //     cv::subtract(float_input_c_, mean_mat_c_, normalized_input_c_);
    //     cv::transpose(normalized_input_c_, flipped_input_c_);
    //     cv::split(flipped_input_c_, input_planes_c_);
    //     cv::vconcat(input_planes_c_, planar_input_c_);
    //     assert(planar_input_c_.cols == net_input_height_);
    //     for (i32 r = 0; r < planar_input_c_.rows; ++r) {
    //       u8* mat_pos = planar_input_c_.data + r * planar_input_c_.step;
    //       u8* input_pos = reinterpret_cast<u8*>(
    //           net_input + i * (net_input_width_ * net_input_height_ * 3) +
    //           r * net_input_height_);
    //       std::memcpy(input_pos, mat_pos, planar_input_c_.cols *
    //       sizeof(float));
    //     }
    //   }

    //   output_buffers[0].push_back((u8*)net_input);
    //   output_sizes[0].push_back(frame_size * batch_count * sizeof(f32));
    // }

    // i32 num_batches = output_buffers[0].size();
    // for (i32 i = 0; i < input_buffers[0].size() - num_batches; ++i) {
    //   output_buffers[0].push_back(new u8[1]);
    //   output_sizes[0].push_back(1);
    // }
  }

  for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
    output_columns[0].rows.push_back(input_columns[0].rows[i]);
  }

  if (profiler_) {
    profiler_->add_interval("cpm_input", eval_start, now());
  }
}

CPMInputEvaluatorFactory::CPMInputEvaluatorFactory(
    DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size)
    : device_type_(device_type),
      net_descriptor_(descriptor),
      batch_size_(batch_size) {}

EvaluatorCapabilities CPMInputEvaluatorFactory::get_capabilities() {
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

std::vector<std::string> CPMInputEvaluatorFactory::get_output_names() {
  return {"frame", "net_input"};
}

Evaluator* CPMInputEvaluatorFactory::new_evaluator(
    const EvaluatorConfig& config) {
  return new CPMInputEvaluator(device_type_, config.device_ids[0],
                               net_descriptor_, batch_size_);
}
}

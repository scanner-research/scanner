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

#include "scanner/evaluators/caffe/facenet/facenet_input_evaluator.h"
#include "scanner/util/memory.h"

namespace scanner {

FacenetInputEvaluator::FacenetInputEvaluator(DeviceType device_type,
                                             i32 device_id,
                                             const NetDescriptor& descriptor,
                                             i32 batch_size,
                                             f32 scale)
    : device_type_(device_type),
      device_id_(device_id),
      descriptor_(descriptor),
      scale_(scale),
      batch_size_(batch_size),
      net_input_width_(224),
      net_input_height_(224)
#ifdef HAVE_CUDA
      ,
      num_cuda_streams_(32),
      streams_(num_cuda_streams_)
#endif
{
}

void FacenetInputEvaluator::configure(const BatchConfig& config) {
  config_ = config;
  assert(config.formats.size() == 1);
  metadata_ = config.formats[0];

  net_input_width_ = metadata_.width() * scale_;
  net_input_height_ = metadata_.height() * scale_;

  net_input_width_ += (net_input_width_ % 8);
  net_input_height_ += (net_input_height_ % 8);

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    mean_mat_g_ = cv::cuda::GpuMat(
        net_input_height_, net_input_width_, CV_32FC3,
        cv::Scalar(descriptor_.mean_colors[0], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[2]));

    frame_input_g_.clear();
    float_input_g_.clear();
    normalized_input_g_.clear();
    input_planes_g_.clear();
    planar_input_g_.clear();
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      frame_input_g_.push_back(
          cv::cuda::GpuMat(metadata_.height(), metadata_.width(), CV_8UC3));
      resized_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_8UC3));
      float_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      normalized_input_g_.push_back(
          cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC3));
      std::vector<cv::cuda::GpuMat> planes;
      std::vector<cv::cuda::GpuMat> flipped_planes;
      for (i32 i = 0; i < 3; ++i) {
        planes.push_back(
            cv::cuda::GpuMat(net_input_height_, net_input_width_, CV_32FC1));
        flipped_planes.push_back(
            cv::cuda::GpuMat(net_input_width_, net_input_height_, CV_32FC1));
      }
      input_planes_g_.push_back(planes);
      flipped_planes_g_.push_back(flipped_planes);
      planar_input_g_.push_back(
          cv::cuda::GpuMat(net_input_width_ * 3, net_input_height_, CV_32FC1));
    }
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    mean_mat_c_ = cv::Mat(
        net_input_height_, net_input_width_, CV_32FC3,
        cv::Scalar(descriptor_.mean_colors[0], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[2]));

    float_input_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC3);
    normalized_input_c_ =
        cv::Mat(net_input_height_, net_input_width_, CV_32FC3);
    for (i32 i = 0; i < 3; ++i) {
      input_planes_c_.push_back(
          cv::Mat(net_input_height_, net_input_width_, CV_32FC1));
    }
    planar_input_c_ =
        cv::Mat(net_input_height_ * 3, net_input_width_, CV_32FC1);
  }
}

void FacenetInputEvaluator::evaluate(const BatchedColumns& input_columns,
                                     BatchedColumns& output_columns) {
  auto eval_start = now();

  size_t frame_size = net_input_width_ * net_input_height_ * 3;
  i32 input_count = input_columns[0].rows.size();

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    streams_.resize(0);
    streams_.resize(num_cuda_streams_);

    for (i32 i = 0; i < input_count; ++i) {
      int sid = i % num_cuda_streams_;
      cv::cuda::Stream& cv_stream = streams_[sid];

      f32* net_input = nullptr;
      i32 net_input_size = frame_size * sizeof(f32);
      cudaMalloc((void**)&net_input, net_input_size);

      u8 *buffer = input_columns[0].rows[i].buffer;
      frame_input_g_[sid] = cv::cuda::GpuMat(
          metadata_.height(), metadata_.width(), CV_8UC3, buffer);
      cv::cuda::resize(frame_input_g_[sid], resized_input_g_[sid],
                       cv::Size(net_input_width_, net_input_height_), 0, 0,
                       cv::INTER_LINEAR, cv_stream);
      resized_input_g_[sid].convertTo(float_input_g_[sid], CV_32FC3, cv_stream);
      cv::cuda::subtract(float_input_g_[sid], mean_mat_g_,
                         normalized_input_g_[sid], cv::noArray(), -1,
                         cv_stream);
      // Changed from interleaved RGB to planar RGB
      cv::cuda::split(normalized_input_g_[sid], input_planes_g_[sid],
                      cv_stream);
      cv::cuda::transpose(input_planes_g_[sid][0], flipped_planes_g_[sid][0],
                          cv_stream);
      cv::cuda::transpose(input_planes_g_[sid][1], flipped_planes_g_[sid][1],
                          cv_stream);
      cv::cuda::transpose(input_planes_g_[sid][2], flipped_planes_g_[sid][2],
                          cv_stream);
      auto& plane1 = flipped_planes_g_[sid][0];
      auto& plane2 = flipped_planes_g_[sid][1];
      auto& plane3 = flipped_planes_g_[sid][2];
      auto& planar_input = planar_input_g_[sid];
      plane1.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 0, net_input_height_, net_input_width_)));
      plane2.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 1, net_input_height_, net_input_width_)));
      plane3.copyTo(planar_input(cv::Rect(
          0, net_input_width_ * 2, net_input_height_, net_input_width_)));
      assert(planar_input.cols == net_input_height_);
      cudaStream_t s = cv::cuda::StreamAccessor::getStream(cv_stream);
      CU_CHECK(cudaMemcpy2DAsync(
          net_input, net_input_height_ * sizeof(float),
          planar_input.data, planar_input.step,
          net_input_height_ * sizeof(float),
          net_input_width_ * 3, cudaMemcpyDeviceToDevice, s));

      INSERT_ROW(output_columns[1], (u8*)net_input, net_input_size);
    }
    for (cv::cuda::Stream& s : streams_) {
      s.waitForCompletion();
    }
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    for (i32 i = 0; i < input_count; ++i) {
      u8* buffer = input_columns[0].rows[i].buffer;
      f32* net_input = reinterpret_cast<f32*>(new u8[frame_size * sizeof(f32)]);

      cv::Mat input_mat =
          cv::Mat(net_input_height_, net_input_width_, CV_8UC3, buffer);

      // Changed from interleaved RGB to planar RGB
      input_mat.convertTo(float_input_c_, CV_32FC3);
      cv::subtract(float_input_c_, mean_mat_c_, normalized_input_c_);
      cv::split(normalized_input_c_, input_planes_c_);
      cv::vconcat(input_planes_c_, planar_input_c_);
      assert(planar_input_c_.cols == net_input_height_);
      for (i32 r = 0; r < planar_input_c_.rows; ++r) {
        u8* mat_pos = planar_input_c_.data + r * planar_input_c_.step;
        u8* input_pos = reinterpret_cast<u8*>(
            net_input + i * (net_input_width_ * net_input_height_ * 3) +
            r * net_input_width_);
        std::memcpy(input_pos, mat_pos, planar_input_c_.cols * sizeof(float));
      }

      INSERT_ROW(output_columns[1], (u8*)net_input, frame_size * sizeof(f32));
    }
  }

  for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
    INSERT_ROW(output_columns[0], input_columns[0].rows[i].buffer,
               input_columns[0].rows[i].size);
  }

  if (profiler_) {
    profiler_->add_interval("facenet_input", eval_start, now());
  }
}

FacenetInputEvaluatorFactory::FacenetInputEvaluatorFactory(
    DeviceType device_type, const NetDescriptor& descriptor, i32 batch_size,
    f32 scale)
    : device_type_(device_type),
      net_descriptor_(descriptor),
      batch_size_(batch_size),
      scale_(scale) {}

EvaluatorCapabilities FacenetInputEvaluatorFactory::get_capabilities() {
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

std::vector<std::string> FacenetInputEvaluatorFactory::get_output_columns(
    const std::vector<std::string>& input_columns) {
  return {"frame", "net_input"};
}

Evaluator *
FacenetInputEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new FacenetInputEvaluator(device_type_, config.device_ids[0],
                                   net_descriptor_, batch_size_, scale_);
}
}

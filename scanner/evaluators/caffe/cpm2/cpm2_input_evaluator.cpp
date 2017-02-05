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

#include "scanner/evaluators/caffe/cpm2/cpm2_input_evaluator.h"
#include "caffe/layers/imresize_layer.hpp"
#include "scanner/util/memory.h"

#ifdef HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace scanner {

void cpm2_net_config(f32 scale, const BatchConfig &config,
                     caffe::Net<float> *net) {
  assert(config.formats.size() == 1);
  const InputFormat &metadata = config.formats[0];
  // Calculate width by scaling by box size
  int resize_width = metadata.width() * scale;
  int resize_height = metadata.height() * scale;

  int width_padding = (resize_width % 8) ? 8 - (resize_width % 8) : 0;
  int height_padding = (resize_height % 8) ? 8 - (resize_height % 8) : 0;

  int net_input_width = resize_width + width_padding;
  int net_input_height = resize_height + height_padding;

  caffe::ImResizeLayer<float> *resize_layer =
      (caffe::ImResizeLayer<float> *)net->layer_by_name("resize").get();

  resize_layer->SetStartScale(1);
  resize_layer->SetScaleGap(0.1);
  resize_layer->setTargetDimenions(net_input_width, net_input_height);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net->blob_by_name("image")};
  input_blob->Reshape({input_blob->shape(0), input_blob->shape(1),
                       net_input_height, net_input_width});
}

CPM2InputEvaluator::CPM2InputEvaluator(DeviceType device_type, i32 device_id,
                                       const NetDescriptor &descriptor,
                                       i32 batch_size, f32 scale)
    : device_type_(device_type), device_id_(device_id), descriptor_(descriptor),
      batch_size_(batch_size), net_input_width_(368), net_input_height_(368),
      scale_(scale)
#ifdef HAVE_CUDA
      ,
      num_cuda_streams_(32), streams_(num_cuda_streams_)
#endif
{
}

void CPM2InputEvaluator::configure(const BatchConfig &config) {
  config_ = config;
  assert(config.formats.size() == 1);

  frame_width_ = config.formats[0].width();
  frame_height_ = config.formats[0].height();

  resize_width_ = frame_width_ * scale_;
  resize_height_ = frame_height_ * scale_;

  width_padding_ = (resize_width_ % 8) ? 8 - (resize_width_ % 8) : 0;
  height_padding_ = (resize_height_ % 8) ? 8 - (resize_height_ % 8) : 0;

  net_input_width_ = resize_width_ + width_padding_;
  net_input_height_ = resize_height_ + height_padding_;

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    cv::cuda::setDevice(device_id_);
    cudaSetDevice(device_id_);

    frame_input_g_.clear();
    bgr_input_g_.clear();
    resized_input_g_.clear();
    padded_input_g_.clear();
    float_input_g_.clear();
    input_planes_g_.clear();
    flipped_planes_g_.clear();
    planar_input_g_.clear();
    for (size_t i = 0; i < num_cuda_streams_; ++i) {
      frame_input_g_.push_back(
          cv::cuda::GpuMat(frame_height_, frame_width_, CV_8UC3));
      bgr_input_g_.push_back(
          cv::cuda::GpuMat(frame_height_, frame_width_, CV_8UC3));
      resized_input_g_.push_back(
          cv::cuda::GpuMat(resize_height_, resize_width_, CV_8UC3));
      padded_input_g_.push_back(
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
          cv::cuda::GpuMat(net_input_height_ * 3, net_input_width_, CV_32FC1));
    }
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    LOG(FATAL) << "CPU not implemented yet.";
    mean_mat_c_ = cv::Mat(net_input_height_, net_input_width_, CV_32FC3,
                          cv::Scalar(descriptor_.mean_colors[0],
                                     descriptor_.mean_colors[1],
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

void CPM2InputEvaluator::evaluate(const BatchedColumns &input_columns,
                                  BatchedColumns &output_columns) {
  auto eval_start = now();

  size_t frame_size = net_input_width_ * net_input_height_ * 3;
  i32 input_count = input_columns[0].rows.size();

  if (device_type_ == DeviceType::GPU) {
#ifdef HAVE_CUDA
    streams_.resize(0);
    streams_.resize(num_cuda_streams_);

    for (i32 i = 0; i < input_count; ++i) {
      f32 *net_input = nullptr;
      i32 net_input_size = frame_size * sizeof(f32);
      cudaMalloc((void **)&net_input, net_input_size);

      int sid = i % num_cuda_streams_;
      cv::cuda::Stream &cv_stream = streams_[sid];

      u8 *buffer = input_columns[0].rows[i].buffer;
      assert(input_columns[0].rows[i].size == frame_height_ * frame_width_ * 3);
      frame_input_g_[sid] =
          cv::cuda::GpuMat(frame_height_, frame_width_, CV_8UC3, buffer);
      cv::cuda::cvtColor(frame_input_g_[sid], bgr_input_g_[sid],
                         cv::COLOR_RGB2BGR, 0, cv_stream);
      cv::cuda::resize(bgr_input_g_[sid], resized_input_g_[sid],
                       cv::Size(resize_width_, resize_height_), 0, 0,
                       cv::INTER_CUBIC, cv_stream);
      cv::cuda::copyMakeBorder(resized_input_g_[sid], padded_input_g_[sid], 0,
                               height_padding_, 0, width_padding_,
                               cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128),
                               cv_stream);
      padded_input_g_[sid].convertTo(float_input_g_[sid], CV_32FC3,
                                     (1.0f / 256.0f), -0.5f, cv_stream);
      // Changed from interleaved BGR to planar RGB
      cv::cuda::split(float_input_g_[sid], input_planes_g_[sid], cv_stream);
      auto &plane1 = input_planes_g_[sid][0];
      auto &plane2 = input_planes_g_[sid][1];
      auto &plane3 = input_planes_g_[sid][2];
      auto &planar_input = planar_input_g_[sid];
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

      output_columns[1].rows.push_back(Row{(u8 *)net_input, net_input_size});
    }
    for (cv::cuda::Stream &s : streams_) {
      s.waitForCompletion();
    }
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  } else {
    LOG(FATAL) << "CPU support not implemented yet.";
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
  }

  for (i32 i = 0; i < input_columns[0].rows.size(); ++i) {
    size_t size = input_columns[0].rows[i].size;
    u8 *buffer = new_buffer({device_type_, device_id_}, size);
    memcpy_buffer(buffer, {device_type_, device_id_},
                  input_columns[0].rows[i].buffer, {device_type_, device_id_},
                  size);
    output_columns[0].rows.push_back(Row{buffer, size});
  }

  if (profiler_) {
    profiler_->add_interval("cpm_person_input", eval_start, now());
  }
}

CPM2InputEvaluatorFactory::CPM2InputEvaluatorFactory(
    DeviceType device_type, const NetDescriptor &descriptor, i32 batch_size,
    f32 scale)
    : device_type_(device_type), net_descriptor_(descriptor),
      batch_size_(batch_size), scale_(scale) {}

EvaluatorCapabilities CPM2InputEvaluatorFactory::get_capabilities() {
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

std::vector<std::string> CPM2InputEvaluatorFactory::get_output_columns(
    const std::vector<std::string> &input_columns) {
  return {"frame", "net_input"};
}

Evaluator *
CPM2InputEvaluatorFactory::new_evaluator(const EvaluatorConfig &config) {
  return new CPM2InputEvaluator(device_type_, config.device_ids[0],
                                net_descriptor_, batch_size_, scale_);
}
}

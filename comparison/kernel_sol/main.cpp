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

#include "scanner/util/queue.h"
#include "scanner/util/util.h"

#include "scanner/evaluators/caffe/net_descriptor.h"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"

#include "caffe_input_transformer_gpu/caffe_input_transformer_gpu.h"
#include "HalideRuntimeCuda.h"
#include "scanner/engine/halide_context.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#ifdef HAVE_CUDA
#include "scanner/util/cuda.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

#include <iostream>
#include <fstream>
#include <thread>

namespace po = boost::program_options;
namespace cvc = cv::cuda;

using namespace scanner;

const std::string NET_PATH = "features/googlenet.toml";
const int NET_BATCH_SIZE = 96;      // Batch size for network
const int FLOW_WORK_REDUCTION = 20;
const int BINS = 16;
const int HISTO_ITERS = 50;
const int FLOW_ITERS = 5;
const int CAFFE_ITERS = 30;
int width;
int height;

std::string PATH;

std::map<std::string, double> TIMINGS;

void histogram(std::vector<cvc::GpuMat>& frames) {
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(0);
  int width = frames[0].cols;
  int height = frames[0].rows;
  assert(width != 0 && height != 0);
  int frame_size = width * height * 3 * sizeof(u8);

  std::vector<cvc::GpuMat> planes;
  for (int i = 0; i < 3; ++i) {
    planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
  }
  cvc::GpuMat hist = cvc::GpuMat(1, BINS, CV_32S);
  cvc::GpuMat out_gpu = cvc::GpuMat(1, BINS * 3, CV_32S);

  auto total_start = scanner::now();
  for (int it = 0; it < HISTO_ITERS; ++it) {
    for (int i = 0; i < frames.size(); ++i) {
      cvc::GpuMat &image = frames[i];
      auto histo_start = scanner::now();
      cvc::split(image, planes);
      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)));
      }
    }
  }
  cudaDeviceSynchronize();
  TIMINGS["total"] = scanner::nano_since(total_start);
}

void flow(std::vector<cvc::GpuMat>& frames) {
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(0);

  int width = frames[0].cols;
  int height = frames[0].rows;
  assert(width != 0 && height != 0);
  int frame_size = width * height * 3 * sizeof(u8);

  std::vector<cvc::GpuMat> inputs;
  for (int i = 0; i < 2; ++i) {
    inputs.emplace_back(height, width, CV_8UC3);
  }
  std::vector<cvc::GpuMat> gray;
  for (int i = 0; i < 2; ++i) {
    gray.emplace_back(height, width, CV_8UC1);
  }
  cvc::GpuMat output_flow_gpu(height, width, CV_32FC2);

  cv::Ptr<cvc::DenseOpticalFlow> flow = cvc::FarnebackOpticalFlow::create();

  auto total_start = scanner::now();

  for (int it = 0; it < FLOW_ITERS; ++it) {
    cvc::cvtColor(frames[0], gray[0], CV_BGR2GRAY);
    bool done = false;
    for (int i = 1; i < frames.size(); ++i) {
      int curr_idx = i % 2;
      int prev_idx = (i - 1) % 2;

      auto eval_start = scanner::now();
      cvc::GpuMat &image = frames[i];
      cvc::cvtColor(image, gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu);
    }
  }
  cudaDeviceSynchronize();
  TIMINGS["total"] = scanner::nano_since(total_start);
}

void set_halide_buf(buffer_t &halide_buf, u8 *buf, size_t size) {
  halide_buf.dev = (uintptr_t) nullptr;

  // "You likely want to set the dev_dirty flag for correctness. (It will
  // not matter if all the code runs on the GPU.)"
  halide_buf.dev_dirty = true;

  i32 err = halide_cuda_wrap_device_ptr(nullptr, &halide_buf, (uintptr_t)buf);
  assert(err == 0);

  // "You'll need to set the host field of the buffer_t structs to
  // something other than nullptr as that is used to indicate bounds query
  // calls" - Zalman Stern
  halide_buf.host = (u8 *)0xdeadbeef;
}

void unset_halide_buf(buffer_t &halide_buf) {
  halide_cuda_detach_device_ptr(nullptr, &halide_buf);
}

void transform_halide(const NetDescriptor& descriptor_,
                      i32 net_width, i32 net_height,
                      u8* input_buffer, size_t stride,
                      u8* output_buffer) {
  i32 net_input_width_ = net_width;
  i32 net_input_height_ = net_height;
  i32 frame_width = width;
  i32 frame_height = height;
  size_t net_input_size =
      net_input_width_ * net_input_height_ * 3 * sizeof(float);

  buffer_t input_buf = {0}, output_buf = {0};

  set_halide_buf(input_buf, input_buffer, frame_width * frame_height * 3);
  set_halide_buf(output_buf, output_buffer, net_input_size);

  // Halide has the input format x * stride[0] + y * stride[1] + c * stride[2]
  // input_buf.host = input_buffer;
  input_buf.stride[0] = 3;
  input_buf.stride[1] = stride;
  input_buf.stride[2] = 1;
  input_buf.extent[0] = frame_width;
  input_buf.extent[1] = frame_height;
  input_buf.extent[2] = 3;
  input_buf.elem_size = 1;

  // Halide conveniently defaults to a planar format, which is what Caffe
  // expects
  output_buf.host = output_buffer;
  output_buf.stride[0] = 1;
  output_buf.stride[1] = net_input_width_;
  output_buf.stride[2] = net_input_width_ * net_input_height_;
  output_buf.extent[0] = net_input_width_;
  output_buf.extent[1] = net_input_height_;
  output_buf.extent[2] = 3;
  output_buf.elem_size = 4;

  auto func = caffe_input_transformer_gpu;
  int error = func(&input_buf, frame_width, frame_height,
                   net_input_width_, net_input_height_, descriptor_.normalize,
                   descriptor_.mean_colors[2], descriptor_.mean_colors[1],
                   descriptor_.mean_colors[0], &output_buf);
  LOG_IF(FATAL, error != 0) << "Halide error " << error;

  unset_halide_buf(input_buf);
  unset_halide_buf(output_buf);
}

void caffe_fn(std::vector<cvc::GpuMat>& frames) {
  width = frames[0].cols;
  height = frames[0].rows;
  int frame_size = width * height * 3 * sizeof(u8);

  NetDescriptor descriptor;
  {
    std::ifstream net_file{NET_PATH};
    descriptor = scanner::descriptor_from_net_file(net_file);
  }

  // Set ourselves to the correct GPU
  int gpu_device_id = 0;
  CUcontext cuda_context;
  CUD_CHECK(cuDevicePrimaryCtxRetain(&cuda_context, gpu_device_id));
  Halide::Runtime::Internal::Cuda::context = cuda_context;
  halide_set_gpu_device(gpu_device_id);
  cv::cuda::setDevice(gpu_device_id);
  CU_CHECK(cudaSetDevice(gpu_device_id));
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_device_id);
  std::unique_ptr<caffe::Net<float>> net;
  net.reset(new caffe::Net<float>(descriptor.model_path, caffe::TEST));
  net->CopyTrainedLayersFrom(descriptor.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net->blob_by_name(descriptor.input_layer_names[0])};

  input_blob->Reshape({NET_BATCH_SIZE, input_blob->shape(1),
                       input_blob->shape(2), input_blob->shape(3)});

  int net_input_width = input_blob->shape(2); // width
  int net_input_height = input_blob->shape(3); // height
  size_t net_input_size =
      net_input_width * net_input_height * 3 * sizeof(float);

  auto start = scanner::now();
  for (int it = 0; it < CAFFE_ITERS; ++it) {
    int64_t frame = 0;
    while (frame < frames.size()) {
      int batch = std::min((int)(frames.size() - frame), (int)NET_BATCH_SIZE);
      if (batch != NET_BATCH_SIZE) {
        input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
                             input_blob->shape(3)});
      }
      for (int i = 0; i < batch; i++) {
        u8 *input = frames[frame + i].data;
        u8 *output =
            ((u8 *)input_blob->mutable_gpu_data()) + i * net_input_size;
        transform_halide(descriptor, net_input_width, net_input_height, input,
                         frames[frame + i].step, output);
      }
      net->Forward();
      frame += batch;
    }
  }
  cudaDeviceSynchronize();
  Halide::Runtime::Internal::Cuda::context = 0;
  TIMINGS["total"] = scanner::nano_since(start);
}


int main(int argc, char** argv) {
  std::string operation;
  int num_frames;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(
        "operation", po::value<std::string>()->required(),
        "histogram, flow, caffe")(

        "frames", po::value<int>()->required(), "")(

        "path", po::value<std::string>()->required(), "");
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      operation = vm["operation"].as<std::string>();
      num_frames = vm["frames"].as<int>();
      PATH = vm["path"].as<std::string>();
    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }

  // Load frames into memory
  cv::cuda::setDevice(0);
  cv::Ptr<cv::cudacodec::VideoReader> video;
  const std::string &path = PATH;
  video = cv::cudacodec::createVideoReader(path);
  int width = video->format().width;
  int height = video->format().height;
  assert(width != 0 && height != 0);

  std::vector<cvc::GpuMat> images;
  auto total_start = scanner::now();
  for (int64_t frame = 0; frame < num_frames; ++frame) {
    cvc::GpuMat image(height, width, CV_8UC4);
    bool valid_frame = video->nextFrame(image);
    assert(valid_frame);
    assert(image.data != nullptr);
    //cv::Mat temp(image);
    cvc::cvtColor(image, image, CV_RGBA2RGB);
    //cvc::GpuMat temp2(image);
    images.push_back(image);
  }

  // Run kernel on frames
  if (operation == "histogram") {
    histogram(images);
  } else if (operation == "flow") {
    flow(images);
  } else if (operation == "caffe") {
    caffe_fn(images);
  }

  for (auto& kv : TIMINGS) {
    printf("TIMING: %s,%.2f\n", kv.first.c_str(), kv.second / 1000000000.0);
  }
}

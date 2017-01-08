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

using scanner::NetDescriptor;
using scanner::Queue;

using WorkerFn= std::function<void(int, Queue<int64_t>&)>;

enum InputType {
  BMP,
  JPG,
  H264
};

enum OpType {
  Histogram,
  Flow,
  Caffe
};

const int BATCH_SIZE = 96;      // Batch size for network
const int FLOW_WORK_REDUCTION = 20;
const int BINS = 16;
const std::string NET_PATH = "features/googlenet.toml";

InputType INPUT;
std::vector<std::string> PATHS;
int GPUS_PER_NODE = 1;           // GPUs to use per node

std::map<std::string, double> TIMINGS;

std::string meta_path(std::string path) {
  return path + "/meta.txt";
}

std::string bmp_path(std::string path, int64_t frame) {
  char image_suffix[256];
  snprintf(image_suffix, sizeof(image_suffix), "frame%07ld", frame);
  std::string ext;
  switch (INPUT) {
    case BMP:
      ext = ".bmp";
      break;
    case JPG:
      ext = ".jpg";
      break;
    default:
      assert(false);
  }
  return path + "/" + std::string(image_suffix) + ext;
}

std::string output_path(int64_t idx) {
  return "outputs/videos" + std::to_string(idx) + ".bin";
}

void image_histogram_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  //cvc::GpuMat hist = cvc::GpuMat(1, BINS, CV_32S);
  cv::Mat hist;
  cv::Mat hist_32s;
  cv::Mat out(1, BINS * 3, CV_32S);
  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    const std::string& path = PATHS[work_item_index];
    // Read meta file to determine number of
    std::ifstream meta_file(meta_path(path));
    assert(meta_file.good());
    int64_t num_images = 0;
    int width = 0;
    int height = 0;
    meta_file >> num_images;
    meta_file >> width;
    meta_file >> height;

    // std::vector<cvc::GpuMat> planes;
    // for (int i = 0; i < 3; ++i) {
    //   planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
    // }
    std::vector<cv::Mat> planes;
    for (int i = 0; i < 3; ++i) {
      planes.push_back(cv::Mat(height, width, CV_8UC1));
    }
    cvc::GpuMat image_gpu(height, width, CV_8UC1);

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());

    // for (int64_t frame = 0; frame < num_images; ++frame) {
    //   auto load_start = scanner::now();
    //   cv::Mat image = cv::imread(bmp_path(path, frame));
    //   if (image.data == nullptr) {
    //     std::cout << bmp_path(path, frame) << std::endl;
    //     assert(image.data != nullptr);
    //   }
    //   load_time += scanner::nano_since(load_start);
    //   auto histo_start_ = scanner::now();
    //   image_gpu.upload(image);
    //   cvc::split(image_gpu, planes);

    //   for (int j = 0; j < 3; ++j) {
    //     cvc::histEven(planes[j], hist, BINS, 0, 256);
    //     hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)));
    //   }
    //   out_gpu.download(out);
    //   histo_time += scanner::nano_since(load_start);
    //   // Save outputs
    //   auto save_start = scanner::now();
    //   outfile.write((char*)out.data, BINS * 3 * sizeof(float));
    //   save_time += scanner::nano_since(save_start);
    // }
    for (int64_t frame = 0; frame < num_images; ++frame) {
      auto load_start = scanner::now();
      cv::Mat image = cv::imread(bmp_path(path, frame));
      if (image.data == nullptr) {
        std::cout << bmp_path(path, frame) << std::endl;
        assert(image.data != nullptr);
      }
      load_time += scanner::nano_since(load_start);
      auto histo_start = scanner::now();
      cv::split(image, planes);

      float range[] = {0, 256};
      const float* histRange = {range};
      int channels[] = {0};
      for (int j = 0; j < 3; ++j) {
        cv::calcHist(&planes[j], 1, channels, cv::Mat(), hist, 1, &BINS,
                     &histRange);
        hist.convertTo(hist_32s, CV_32S);
        cv::transpose(hist_32s, hist_32s);
        hist_32s.copyTo(out(cv::Rect(j * BINS, 0, BINS, 1)));
      }
      histo_time += scanner::nano_since(histo_start);
      // Save outputs
      auto save_start = scanner::now();
      outfile.write((char*)out.data, BINS * 3 * sizeof(float));
      save_time += scanner::nano_since(save_start);
    }
  }
  TIMINGS["load"] = load_time;
  TIMINGS["eval"] = histo_time;
  TIMINGS["save"] = save_time;
}

void video_histogram_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  double setup_time = 0;
  double load_time = 0;
  double histo_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  cvc::GpuMat hist = cvc::GpuMat(1, BINS, CV_32S);
  cvc::GpuMat out_gpu = cvc::GpuMat(1, BINS * 3, CV_32S);
  cv::Mat out = cv::Mat(1, BINS * 3, CV_32S);

  cv::Ptr<cv::cudacodec::VideoReader> video;
  cvc::GpuMat frame;
  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    const std::string& path = PATHS[work_item_index];
    // Read video file to determine number of

    auto setup_start = scanner::now();
    video = cv::cudacodec::createVideoReader(path);
    int width = video->format().width;
    int height = video->format().height;
    assert(width != 0 && height != 0);

    std::vector<cvc::GpuMat> planes;
    for (int i = 0; i < 3; ++i) {
      planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
    }

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());
    cvc::GpuMat image(height, width, CV_8UC3);
    setup_time += scanner::nano_since(setup_start);

    bool done = false;
    int64_t frame = 0;
    while (!done) {
      auto load_start = scanner::now();
      bool valid_frame = video->nextFrame(image);
      load_time += scanner::nano_since(load_start);
      if (!valid_frame) {
        done = true;
      }
      if (image.data == nullptr) {
        break;
      }
      auto histo_start = scanner::now();
      cvc::split(image, planes);
      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)));
      }
      out_gpu.download(out);
      histo_time += scanner::nano_since(histo_start);
      // Save outputs
      auto save_start = scanner::now();
      outfile.write((char*)out.data, BINS * 3 * sizeof(float));
      save_time += scanner::nano_since(save_start);
      frame++;
    }
  }
  TIMINGS["setup"] = setup_time;
  TIMINGS["load"] = load_time;
  TIMINGS["eval"] = histo_time;
  TIMINGS["save"] = save_time;
}

void image_flow_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  double setup_time = 0;
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    auto setup_start = scanner::now();
    const std::string& path = PATHS[work_item_index];
    // Read meta file to determine number of
    std::ifstream meta_file(meta_path(path));
    int64_t num_images = 0;
    int width = 0;
    int height = 0;
    meta_file >> num_images;
    meta_file >> width;
    meta_file >> height;

    // Flow makes much longer than other pipelines
    num_images /= FLOW_WORK_REDUCTION;

    // Flow intermediates
    std::vector<cvc::GpuMat> inputs;
    for (int i = 0; i < 2; ++i) {
      inputs.emplace_back(height, width, CV_8UC3);
    }
    std::vector<cvc::GpuMat> gray;
    for (int i = 0; i < 2; ++i) {
      gray.emplace_back(height, width, CV_8UC1);
    }
    cvc::GpuMat output_flow_gpu(height, width, CV_32FC2);
    cv::Mat output_flow(height, width, CV_32FC2);
    cv::Ptr<cvc::DenseOpticalFlow> flow =
      cvc::FarnebackOpticalFlow::create();

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());
    setup_time += scanner::nano_since(setup_start);

    // Load the first frame
    assert(num_images > 0);
    auto load_first = scanner::now();
    cv::Mat image = cv::imread(bmp_path(path, 0));
    load_time += scanner::nano_since(load_first);
    auto eval_first = scanner::now();
    inputs[0].upload(image);
    cvc::cvtColor(inputs[0], gray[0], CV_BGR2GRAY);
    eval_time += scanner::nano_since(eval_first);
    for (int64_t frame = 1; frame < num_images; ++frame) {
      int curr_idx = frame % 2;
      int prev_idx = (frame - 1) % 2;

      auto load_start = scanner::now();
      image = cv::imread(bmp_path(path, frame));
      load_time += scanner::nano_since(load_start);
      auto eval_start = scanner::now();
      inputs[curr_idx].upload(image);
      cvc::cvtColor(inputs[curr_idx], gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu);
      output_flow_gpu.download(output_flow);
      eval_time += scanner::nano_since(eval_start);

      // Save outputs
      auto save_start = scanner::now();
      for (size_t i = 0; i < height; ++i) {
        outfile.write((char*)output_flow.data + output_flow.step * i,
                      width * sizeof(float) * 2);
      }
      save_time += scanner::nano_since(save_start);
    }
  }
  TIMINGS["setup"] = setup_time;
  TIMINGS["load"] = load_time;
  TIMINGS["eval"] = eval_time;
  TIMINGS["save"] = save_time;
}

void video_flow_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  double setup_time = 0;
  double load_time = 0;
  double eval_time = 0;
  double save_time = 0;
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  cv::Ptr<cv::cudacodec::VideoReader> video;
  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    auto startup_start = scanner::now();
    const std::string& path = PATHS[work_item_index];
    int64_t num_frames;
    {
      cv::VideoCapture video;
      video.open(path);
      assert(video.isOpened());
      num_frames = (int64_t)video.get(CV_CAP_PROP_FRAME_COUNT);
    }
    video = cv::cudacodec::createVideoReader(path);
    int width = video->format().width;
    int height = video->format().height;
    assert(width != 0 && height != 0);

    num_frames /= FLOW_WORK_REDUCTION;

    // Flow intermediates
    std::vector<cvc::GpuMat> inputs;
    for (int i = 0; i < 2; ++i) {
      inputs.emplace_back(height, width, CV_8UC4);
    }
    std::vector<cvc::GpuMat> gray;
    for (int i = 0; i < 2; ++i) {
      gray.emplace_back(height, width, CV_8UC1);
    }
    cvc::GpuMat output_flow_gpu(height, width, CV_32FC2);
    cv::Mat output_flow(height, width, CV_32FC2);
    cv::Ptr<cvc::DenseOpticalFlow> flow = cvc::FarnebackOpticalFlow::create();

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());

    setup_time = scanner::nano_since(startup_start);

    // Load the first frame
    auto load_first = scanner::now();
    if (!video->nextFrame(inputs[0])) {
      assert(false);
    }
    load_time += scanner::nano_since(load_first);
    auto eval_first = scanner::now();
    cvc::cvtColor(inputs[0], gray[0], CV_BGRA2GRAY);
    eval_time += scanner::nano_since(eval_first);
    bool done = false;
    for (int64_t frame = 1; frame < num_frames; ++frame) {
      int curr_idx = frame % 2;
      int prev_idx = (frame - 1) % 2;

      auto load_start = scanner::now();
      bool valid_frame = video->nextFrame(inputs[curr_idx]);
      load_time += scanner::nano_since(load_start);
      assert(valid_frame);
      if (inputs[curr_idx].data == nullptr) {
        break;
      }

      auto eval_start = scanner::now();
      cvc::cvtColor(inputs[curr_idx], gray[curr_idx], CV_BGRA2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu);
      output_flow_gpu.download(output_flow);
      eval_time += scanner::nano_since(eval_start);

      // Save outputs
      auto save_start = scanner::now();
      for (size_t i = 0; i < height; ++i) {
        outfile.write((char*)output_flow.data + output_flow.step * i,
                      width * sizeof(float) * 2);
      }
      save_time += scanner::nano_since(save_start);
    }
  }
  TIMINGS["setup"] = setup_time;
  TIMINGS["load"] = load_time;
  TIMINGS["eval"] = eval_time;
  TIMINGS["save"] = save_time;
}

void image_caffe_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  double load_time = 0;
  double transform_time = 0;
  double net_time = 0;
  double eval_time = 0;
  double save_time = 0;
  NetDescriptor descriptor;
  {
    std::ifstream net_file{NET_PATH};
    descriptor = scanner::descriptor_from_net_file(net_file);
  }

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);
  CU_CHECK(cudaSetDevice(gpu_device_id));
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_device_id);
  std::unique_ptr<caffe::Net<float>> net;
  net.reset(new caffe::Net<float>(descriptor.model_path, caffe::TEST));
  net->CopyTrainedLayersFrom(descriptor.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net->blob_by_name(descriptor.input_layer_names[0])};

  // const boost::shared_ptr<caffe::Blob<float>> output_blob{
  //   net.blob_by_name(net_descriptor.output_layer_name)};

  int net_input_width = input_blob->shape(2); // width
  int net_input_height = input_blob->shape(3); // height

  // Setup caffe batch transformer
  caffe::TransformationParameter param;
  std::vector<float>& mean_colors = descriptor.mean_colors;
  param.set_force_color(true);
  if (descriptor.normalize) {
    param.set_scale(1.0 / 255.0);
  }
  for (int i = 0; i < mean_colors.size(); i++) {
    param.add_mean_value(mean_colors[i]);
  }
  caffe::DataTransformer<float> transformer(param, caffe::TEST);

  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    const std::string& path = PATHS[work_item_index];
    // Read meta file to determine number of
    std::ifstream meta_file(meta_path(path));
    int64_t num_images = 0;
    int width = 0;
    int height = 0;
    meta_file >> num_images;
    meta_file >> width;
    meta_file >> height;

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());

    // Load the first frame
    assert(num_images > 0);
    cv::Mat input;
    std::vector<cv::Mat> images(BATCH_SIZE);
    for (int64_t frame = 0; frame < num_images; frame += BATCH_SIZE) {
      int batch = std::min((int64_t)BATCH_SIZE, num_images - frame);
      images.resize(batch);
      input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
                           input_blob->shape(3)});
      for (int b = 0; b < batch; ++b) {
        auto load_start = scanner::now();
        input = cv::imread(bmp_path(path, frame + b));
        load_time += scanner::nano_since(load_start);
        auto transform_start = scanner::now();
        cv::resize(input, images[b],
                   cv::Size(net_input_width, net_input_height), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(images[b], images[b], CV_RGB2BGR);
        transform_time += scanner::nano_since(transform_start);
        eval_time += scanner::nano_since(transform_start);
      }
      auto transform_start = scanner::now();
      transformer.Transform(images, input_blob.get());
      transform_time += scanner::nano_since(transform_start);
      eval_time += scanner::nano_since(transform_start);

      auto net_start = scanner::now();
      net->Forward();
      net_time += scanner::nano_since(net_start);
      eval_time += scanner::nano_since(net_start);

      // Save outputs
      auto save_start = scanner::now();
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net->blob_by_name(descriptor.output_layer_names[0])};
      outfile.write((char*)output_blob->cpu_data(),
                    output_blob->count() * sizeof(float));
      save_time += scanner::nano_since(save_start);
    }
  }
  TIMINGS["load"] = load_time;
  TIMINGS["transform"] = transform_time;
  TIMINGS["net"] = net_time;
  TIMINGS["eval"] = eval_time;
  TIMINGS["save"] = save_time;
}

void video_caffe_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  double load_time = 0;
  double transform_time = 0;
  double net_time = 0;
  double eval_time = 0;
  double save_time = 0;

  NetDescriptor descriptor;
  {
    std::ifstream net_file{NET_PATH};
    descriptor = scanner::descriptor_from_net_file(net_file);
  }

  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);
  CU_CHECK(cudaSetDevice(gpu_device_id));
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_device_id);
  std::unique_ptr<caffe::Net<float>> net;
  net.reset(new caffe::Net<float>(descriptor.model_path, caffe::TEST));
  net->CopyTrainedLayersFrom(descriptor.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
    net->blob_by_name(descriptor.input_layer_names[0])};

  // const boost::shared_ptr<caffe::Blob<float>> output_blob{
  //   net.blob_by_name(net_descriptor.output_layer_name)};

  int net_input_width = input_blob->shape(2); // width
  int net_input_height = input_blob->shape(3); // height

  // Setup caffe batch transformer
  caffe::TransformationParameter param;
  std::vector<float>& mean_colors = descriptor.mean_colors;
  param.set_force_color(true);
  if (descriptor.normalize) {
    param.set_scale(1.0 / 255.0);
  }
  for (int i = 0; i < mean_colors.size(); i++) {
    param.add_mean_value(mean_colors[i]);
  }
  caffe::DataTransformer<float> transformer(param, caffe::TEST);

  cv::VideoCapture video;
  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    const std::string& path = PATHS[work_item_index];
    video.open(path);
    assert(video.isOpened());
    int width = (int)video.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)video.get(CV_CAP_PROP_FRAME_HEIGHT);
    assert(width != 0 && height != 0);

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());

    // Load the first frame
    cv::Mat input;
    std::vector<cv::Mat> images(BATCH_SIZE);
    int64_t frame = 0;
    bool done = false;
    while (!done) {
      int b;
      images.resize(BATCH_SIZE);
      for (b = 0; b < BATCH_SIZE; b++) {
        auto load_start = scanner::now();
        bool valid_frame = video.read(input);
        load_time += scanner::nano_since(load_start);
        if (!valid_frame) {
          done = true;
          break;
        }
        auto transform_start = scanner::now();
        cv::resize(input, images[b],
                   cv::Size(net_input_width, net_input_height), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(images[b], images[b], CV_RGB2BGR);
        transform_time += scanner::nano_since(transform_start);
        eval_time += scanner::nano_since(transform_start);
      }

      int batch = b;
      images.resize(batch);
      input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
                           input_blob->shape(3)});

      auto transform_start = scanner::now();
      transformer.Transform(images, input_blob.get());
      transform_time += scanner::nano_since(transform_start);
      eval_time += scanner::nano_since(transform_start);

      auto net_start = scanner::now();
      net->Forward();
      net_time += scanner::nano_since(net_start);
      eval_time += scanner::nano_since(net_start);

      // Save outputs
      auto save_start = scanner::now();
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net->blob_by_name(descriptor.output_layer_names[0])};
      outfile.write((char*)output_blob->cpu_data(),
                    output_blob->count() * sizeof(float));

      frame += batch;
      save_time += scanner::nano_since(save_start);
    }
  }
  TIMINGS["load"] = load_time;
  TIMINGS["transform"] = transform_time;
  TIMINGS["net"] = net_time;
  TIMINGS["eval"] = eval_time;
  TIMINGS["save"] = save_time;
}

int main(int argc, char** argv) {
  std::string input_type;
  std::string paths_file;
  std::string operation;
  std::string gpus_per_node;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(
        "input_type", po::value<std::string>()->required(),
        "bmp, jpg, or mp4")(

        "paths_file", po::value<std::string>()->required(),
        "File which contains paths to videos or folders of images")(

        "operation", po::value<std::string>()->required(),
        "histogram, flow, or caffe")(

        "gpus_per_node", po::value<int>(), "GPUs to use per node");
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      input_type = vm["input_type"].as<std::string>();

      paths_file = vm["paths_file"].as<std::string>();

      operation = vm["operation"].as<std::string>();

      if (vm.count("gpus_per_node")) {
        GPUS_PER_NODE = vm["gpus_per_node"].as<int>();
      }
    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }
  std::cout << "input " << input_type << ", paths " << paths_file
            << ", operation " << operation << std::endl;

  if (input_type == "bmp") {
    INPUT = BMP;
  } else if (input_type == "jpg") {
    INPUT = JPG;
  } else if (input_type == "mp4") {
    INPUT = H264;
  }

  WorkerFn worker_fn;
  if (operation == "histogram") {
    if (INPUT == H264) {
      worker_fn = video_histogram_worker;
    } else {
      worker_fn = image_histogram_worker;
    }
  } else if (operation == "flow") {
    if (INPUT == H264) {
      worker_fn = video_flow_worker;
    } else {
      worker_fn = image_flow_worker;
    }
  } else if (operation == "caffe") {
    if (INPUT == H264) {
      worker_fn = video_caffe_worker;
    } else {
      worker_fn = image_caffe_worker;
    }
  }

  // Read in list of video paths
  {
    std::fstream fs(paths_file, std::fstream::in);
    while (fs) {
      std::string path;
      fs >> path;
      if (path.empty()) continue;
      PATHS.push_back(path);
    }
  }
  // Setup queue of work to distribute to threads
  Queue<int64_t> work_items;
  for (size_t i = 0; i < PATHS.size(); ++i) {
    work_items.push(i);
  }

  // Start up workers to process videos
  std::vector<std::thread> workers;
  for (int gpu = 0; gpu < GPUS_PER_NODE; ++gpu) {
    workers.emplace_back(worker_fn, gpu, std::ref(work_items));
  }
  // Place sentinel values to end workers
  for (size_t i = 0; i < GPUS_PER_NODE; ++i) {
    work_items.push(-1);
  }
  // Wait for workers to finish
  for (int gpu = 0; gpu < GPUS_PER_NODE; ++gpu) {
    workers[gpu].join();
  }
  for (auto& kv : TIMINGS) {
    printf("TIMING: %s,%.2f\n", kv.first.c_str(), kv.second / 1000000000.0);
  }
}

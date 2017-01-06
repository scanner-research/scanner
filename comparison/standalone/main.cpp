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

//#define USE_OFDIS
#ifdef USE_OFDIS
#include "oflow.h"
#endif

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

InputType INPUT;
std::vector<std::string> PATHS;
int GPUS_PER_NODE = 1;           // GPUs to use per node
const int BATCH_SIZE = 96;      // Batch size for network

const int BINS = 16;
const std::string NET_PATH = "features/googlenet.toml";

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
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  cvc::GpuMat hist = cvc::GpuMat(1, BINS, CV_32S);
  cvc::GpuMat out_gpu = cvc::GpuMat(1, BINS * 3, CV_32S);
  cv::Mat out = cv::Mat(1, BINS * 3, CV_32S);
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

    // Read first image in directory to figure out width and height
    std::vector<cvc::GpuMat> planes;
    for (int i = 0; i < 3; ++i) {
      planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
    }
    cvc::GpuMat image_gpu(height, width, CV_8UC1);

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());
    for (int64_t frame = 0; frame < num_images; ++frame) {
      cv::Mat image = cv::imread(bmp_path(path, frame));
      assert(image.data != nullptr);
      image_gpu.upload(image);
      cvc::split(image_gpu, planes);

      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)));
      }
      out_gpu.download(out);
      // Save outputs
      outfile.write((char*)out.data, BINS * 3 * sizeof(float));
    }
  }
}

void video_histogram_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

  cvc::GpuMat hist = cvc::GpuMat(1, BINS, CV_32S);
  cvc::GpuMat out_gpu = cvc::GpuMat(1, BINS * 3, CV_32S);
  cv::Mat out = cv::Mat(1, BINS * 3, CV_32S);

  cv::VideoCapture video;
  cv::Mat frame;
  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    const std::string& path = PATHS[work_item_index];
    // Read video file to determine number of
    video.open(path);
    assert(video.isOpened());
    int width = (int)video.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)video.get(CV_CAP_PROP_FRAME_HEIGHT);
    assert(width != 0 && height != 0);

    std::vector<cvc::GpuMat> planes;
    for (int i = 0; i < 3; ++i) {
      planes.push_back(cvc::GpuMat(height, width, CV_8UC1));
    }
    cvc::GpuMat image_gpu(height, width, CV_8UC3);

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());
    cv::Mat image;
    bool done = false;
    int64_t frame = 0;
    while (!done) {
      bool valid_frame = video.read(image);
      if (!valid_frame) {
        done = true;
      }
      if (image.data == nullptr) {
        break;
      }
      image_gpu.upload(image);
      cvc::split(image_gpu, planes);

      for (int j = 0; j < 3; ++j) {
        cvc::histEven(planes[j], hist, BINS, 0, 256);
        hist.copyTo(out_gpu(cv::Rect(j * BINS, 0, BINS, 1)));
      }
      out_gpu.download(out);
      // Save outputs
      outfile.write((char*)out.data, BINS * 3 * sizeof(float));
      frame++;
    }
  }
}

void image_flow_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

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
    cv::Ptr<cvc::DenseOpticalFlow> flow = cvc::OpticalFlowDual_TVL1::create();

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());

    // Load the first frame
    assert(num_images > 0);
    cv::Mat image = cv::imread(bmp_path(path, 0));
    inputs[0].upload(image);
    cvc::cvtColor(inputs[0], gray[0], CV_BGR2GRAY);
    for (int64_t frame = 1; frame < num_images; ++frame) {
      int curr_idx = frame % 2;
      int prev_idx = (frame - 1) % 2;

      image = cv::imread(bmp_path(path, frame));
      inputs[curr_idx].upload(image);
      cvc::cvtColor(inputs[curr_idx], gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu);
      output_flow_gpu.download(output_flow);

      // Save outputs
      for (size_t i = 0; i < height; ++i) {
        outfile.write((char*)output_flow.data + output_flow.step * i,
                      width * sizeof(float) * 2);
      }
    }
  }
}

void video_flow_worker(int gpu_device_id, Queue<int64_t>& work_items) {
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(gpu_device_id);

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
    cv::Ptr<cvc::DenseOpticalFlow> flow = cvc::OpticalFlowDual_TVL1::create();

    std::ofstream outfile(output_path(work_item_index),
                          std::fstream::binary | std::fstream::trunc);
    assert(outfile.good());

    // Load the first frame
    cv::Mat image;
    if (!video.read(image)) {
      assert(false);
    }
    inputs[0].upload(image);
    cvc::cvtColor(inputs[0], gray[0], CV_BGR2GRAY);
    bool done = false;
    int64_t frame = 1;
    while (!done) {
      int curr_idx = frame % 2;
      int prev_idx = (frame - 1) % 2;

      bool valid_frame = video.read(image);
      if (!valid_frame) {
        done = true;
      }
      if (image.data == nullptr) {
        break;
      }

      inputs[curr_idx].upload(image);
      cvc::cvtColor(inputs[curr_idx], gray[curr_idx], CV_BGR2GRAY);
      flow->calc(gray[prev_idx], gray[curr_idx], output_flow_gpu);
      output_flow_gpu.download(output_flow);

      // Save outputs
      for (size_t i = 0; i < height; ++i) {
        outfile.write((char*)output_flow.data + output_flow.step * i,
                      width * sizeof(float) * 2);
      }
      frame++;
    }
  }
}

void image_caffe_worker(int gpu_device_id, Queue<int64_t>& work_items) {
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
        input = cv::imread(bmp_path(path, frame + b));
        cv::resize(input, images[b],
                   cv::Size(net_input_width, net_input_height), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(images[b], images[b], CV_RGB2BGR);
      }
      transformer.Transform(images, input_blob.get());
      net->Forward();
      // Save outputs
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net->blob_by_name(descriptor.output_layer_names[0])};
      outfile.write((char*)output_blob->cpu_data(),
                    output_blob->count() * sizeof(float));
    }
  }
}

void video_caffe_worker(int gpu_device_id, Queue<int64_t>& work_items) {
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
        bool valid_frame = video.read(input);
        if (!valid_frame) {
          done = true;
          break;
        }
        cv::resize(input, images[b],
                   cv::Size(net_input_width, net_input_height), 0, 0,
                   cv::INTER_LINEAR);
        cv::cvtColor(images[b], images[b], CV_RGB2BGR);
      }

      int batch = b;
      images.resize(batch);
      input_blob->Reshape({batch, input_blob->shape(1), input_blob->shape(2),
                           input_blob->shape(3)});
      transformer.Transform(images, input_blob.get());
      net->Forward();
      // Save outputs
      const boost::shared_ptr<caffe::Blob<float>> output_blob{
        net->blob_by_name(descriptor.output_layer_names[0])};
      outfile.write((char*)output_blob->cpu_data(),
                    output_blob->count() * sizeof(float));

      frame += batch;
    }
  }
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
}

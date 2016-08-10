/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
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

#include "lightscan/util/caffe.h"
#include "lightscan/util/queue.h"
#include "lightscan/util/cuda.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <fstream>
#include <thread>

namespace po = boost::program_options;

int GPUS_PER_NODE = 1;           // GPUs to use per node
int GLOBAL_BATCH_SIZE = 64;      // Batch size for network

const std::string KCAM_DIRECTORY = "/Users/abpoms/kcam";

void worker(
  int gpu_device_id,
  std::vector<std::string>& video_paths,
  Queue<int64_t>& work_items)
{
  // Set ourselves to the correct GPU
  CU_CHECK(cudaSetDevice(gpu_device_id));

  // Setup network to use for evaluation
  lightscan::NetInfo net_info =
    load_neural_net(lightscan::NetType::VGG, gpu_device_id);
  caffe::Net<float>* net = net_info.net;

  int dim = net_info.input_size;

  cv::Mat base_mean_frame(
    net_info.mean_width, net_info.mean_height, CV_32FC3, net_info.mean_image);

  cv::Mat mean_frame;
  cv::resize(base_mean_frame, mean_frame, cv::Size(dim, dim));

  const boost::shared_ptr<caffe::Blob<float>> data_blob{
    net->blob_by_name("data")};

  // Setup net input blob to feed frames into network
  caffe::Blob<float> net_input{GLOBAL_BATCH_SIZE, 3, dim, dim};

  cv::VideoCapture video;
  cv::Mat frame;
  cv::Mat resize_frame;
  cv::Mat float_frame;
  cv::Mat input_frame;

  while (true) {
    int64_t work_item_index;
    work_items.pop(work_item_index);

    if (work_item_index == -1) {
      break;
    }

    const std::string& video_path = video_paths[work_item_index];

    video.open(KCAM_DIRECTORY + "/" + video_path);

    bool done = false;
    int frame_index = 0;
    while (!done) {
      int batch_size = GLOBAL_BATCH_SIZE;
      float* net_input_buffer = net_input.mutable_cpu_data();
      // Get batch of frames and convert into proper net input format
      int i = 0;
      for (; i < batch_size; ++i) {
        bool valid_frame = video.read(frame);
        if (!valid_frame) {
          done = true;
          break;
        }
        (void) valid_frame; assert(valid_frame);

        // Resize frame to net input, convert to float, and subract mean image
        cv::resize(frame, resize_frame, cv::Size(dim, dim));
        resize_frame.convertTo(float_frame, CV_32FC3);
        cv::subtract(float_frame, mean_frame, input_frame);

        memcpy(net_input_buffer + i * (dim * dim * 3),
               input_frame.data,
               dim * dim * 3 * sizeof(float));

        frame_index += 1;
      }
      batch_size = i;

      if (data_blob->shape(0) != batch_size) {
        data_blob->Reshape({batch_size, 3, dim, dim});
        net_input.Reshape({batch_size, 3, dim, dim});
      }

      // Evaluate net on batch of frames
      net->Forward({&net_input});
    }
  }
}

int main(int argc, char** argv) {
  std::string video_paths_file;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Produce help message")
      ("video_paths_file", po::value<std::string>()->required(),
       "File which contains paths to video files to process")
      ("batch_size", po::value<int>(), "Neural Net input batch size")
      ("gpus_per_node", po::value<int>(), "GPUs to use per node");
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      if (vm.count("batch_size")) {
        GLOBAL_BATCH_SIZE = vm["batch_size"].as<int>();
      }

      if (vm.count("gpus_per_node")) {
        GPUS_PER_NODE = vm["gpus_per_node"].as<int>();
      }

      video_paths_file = vm["video_paths_file"].as<std::string>();

    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }

  // Read in list of video paths
  std::vector<std::string> video_paths;
  {
    std::fstream fs(video_paths_file, std::fstream::in);
    while (fs) {
      std::string path;
      fs >> path;
      if (path.empty()) continue;
      video_paths.push_back(path);
    }
  }

  // Setup queue of work to distribute to threads
  Queue<int64_t> work_items;
  for (size_t i = 0; i < video_paths.size(); ++i) {
    work_items.push(i);
  }

  // Start up workers to process videos
  std::vector<std::thread> workers;
  for (int gpu = 0; gpu < GPUS_PER_NODE; ++gpu) {
    workers.emplace_back(
      worker,
      gpu,
      std::ref(video_paths),
      std::ref(work_items));
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

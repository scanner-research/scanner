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

namespace po = boost::program_options;

int GLOBAL_BATCH_SIZE = 64;      // Batch size for network

const std::string KCAM_DIRECTORY = "/Users/abpoms/kcam";

int main(int argc, char** argv) {
  std::string video_paths_file;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Produce help message")
      ("video_paths_file", po::value<std::string>()->required(),
       "File which contains paths to video files to process")
      ("batch_size", po::value<int>(), "Neural Net input batch size");
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

  // Setup caffe
  NetInfo net_info = load_neural_net(NetType::ALEX_NET, 0);
  caffe::Net<float>* net = net_info.net;

  int dim = net_info.input_size;

  const boost::shared_ptr<caffe::Blob<float>> data_blob{
    net->blob_by_name("data")};

  // Setup net input blob to feed frames into network
  caffe::Blob<float> net_input{GLOBAL_BATCH_SIZE, 3, dim, dim};

  cv::VideoCapture video;
  cv::Mat frame;
  cv::Mat resize_frame;
  cv::Mat float_frame;
  cv::Mat input_frame;
  for (size_t video_index = 0;
       video_index < video_paths.size();
       ++video_index)
  {
    video.open(KCAM_DIRECTORY + "/" + video_paths[video_index]);
    int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

    int frame_index = 0;
    while (frame_index < total_frames) {
      int batch_size = std::min(GLOBAL_BATCH_SIZE, total_frames - frame_index);
      if (data_blob->shape(0) != batch_size) {
        data_blob->Reshape({GLOBAL_BATCH_SIZE, 3, dim, dim});
        net_input.Reshape({GLOBAL_BATCH_SIZE, 3, dim, dim});
      }

      // Get batch of frames and convert into proper net input format
      for (int i = 0; i < GLOBAL_BATCH_SIZE; ++i) {
        bool valid_frame = video.read(frame);
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
    }

    // Evaluate net on batch of frames
    net->Forward({&net_input});
  }
}

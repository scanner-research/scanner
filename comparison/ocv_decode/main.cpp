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

std::string PATH;

std::map<std::string, double> TIMINGS;

void cpu_decode() {
  // Set ourselves to the correct GPU
  cv::VideoCapture video;

  const std::string &path = PATH;

  video.open(path);
  assert(video.isOpened());
  int width = (int)video.get(CV_CAP_PROP_FRAME_WIDTH);
  int height = (int)video.get(CV_CAP_PROP_FRAME_HEIGHT);
  assert(width != 0 && height != 0);

  cv::Mat input(height, width, CV_8UC3);
  auto total_start = scanner::now();
  int64_t frame = 0;
  while (true) {
    bool valid_frame = video.read(input);
    if (!valid_frame) {
      break;
    }
    assert(input.data != nullptr);
    frame++;
  }
  TIMINGS["total"] = scanner::nano_since(total_start);
}

void gpu_decode() {
  // Set ourselves to the correct GPU
  cv::cuda::setDevice(0);
  cv::Ptr<cv::cudacodec::VideoReader> video;

  const std::string &path = PATH;

  video = cv::cudacodec::createVideoReader(path);
  int width = video->format().width;
  int height = video->format().height;
  assert(width != 0 && height != 0);

  cvc::GpuMat image(height, width, CV_8UC4);
  auto total_start = scanner::now();
  int64_t frame = 0;
  while (true) {
    bool valid_frame = video->nextFrame(image);
    if (!valid_frame) {
      break;
    }
    assert(image.data != nullptr);
    frame++;
  }
  TIMINGS["total"] = scanner::nano_since(total_start);
}

int main(int argc, char** argv) {
  std::string decoder;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(
        "decoder", po::value<std::string>()->required(), "")(

        "path", po::value<std::string>()->required(), "");
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      decoder = vm["decoder"].as<std::string>();
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

  if (decoder == "cpu") {
    cpu_decode();
  } else if (decoder == "gpu") {
    gpu_decode();
  }

  for (auto& kv : TIMINGS) {
    printf("TIMING: %s,%.2f\n", kv.first.c_str(), kv.second / 1000000000.0);
  }
}

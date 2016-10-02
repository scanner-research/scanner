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

#include "scanner/evaluators/caffe/net_descriptor.h"
#include "scanner/util/common.h"
#include "scanner/util/queue.h"

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

#include <fstream>
#include <thread>

namespace po = boost::program_options;
using namespace scanner;

namespace {

i32 BATCH_SIZE = 8; // Batch size for network
scanner::DeviceType DEVICE_TYPE = scanner::DeviceType::CPU;
i32 NET_INPUT_WIDTH = -1;
i32 NET_INPUT_HEIGHT = -1;
}

void worker(const NetDescriptor &net_descriptor, i32 num_batches) {
  caffe::Caffe::set_mode(scanner::device_type_to_caffe_mode(DEVICE_TYPE));
  if (DEVICE_TYPE == DeviceType::GPU) {
#ifdef HAVE_CUDA
    CU_CHECK(cudaSetDevice(0));
    caffe::Caffe::SetDevice(0);
#else
    LOG(FATAL) << "Not built with CUDA support.";
#endif
  }
  // Initialize our network
  std::unique_ptr<caffe::Net<float>> net{
      new caffe::Net<float>(net_descriptor.model_path, caffe::TEST)};
  net->CopyTrainedLayersFrom(net_descriptor.model_weights_path);

  const boost::shared_ptr<caffe::Blob<float>> input_blob{
      net->blob_by_name(net_descriptor.input_layer_name)};
  i32 input_width =
      NET_INPUT_WIDTH == -1 ? input_blob->shape(2) : NET_INPUT_WIDTH;
  i32 input_height =
      NET_INPUT_HEIGHT == -1 ? input_blob->shape(3) : NET_INPUT_HEIGHT;
  input_blob->Reshape(
      {BATCH_SIZE, 3, input_height, input_width});

  for (i32 b = 0; b < num_batches; ++b) {
    // Evaluate net on batch of frames
    net->Forward();
  }
}

int main(int argc, char** argv) {
  i32 num_batches;
  std::string net_descriptor_file;
  {
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message")(

        "batch_size", po::value<int>(), "Neural Net input batch size")(

        "num_elements", po::value<int>(), "Number of \"elements\" to execute. "
                                          "The number of batches run is equal "
                                          "to (num_elements / batch_size)")(

        "device_type", po::value<std::string>(), "CPU or GPU")(

        "net_input_width", po::value<int>(),
        "Size to reshape width of network input to")(

        "net_input_height", po::value<int>(),
        "Size to reshape height of network input to")(

        "net_descriptor_file", po::value<std::string>()->required(),
        "File which contains a description of the net to use");

    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      }

      if (vm.count("batch_size")) {
        BATCH_SIZE = vm["batch_size"].as<i32>();
      }

      if (vm.count("num_elements")) {
        num_batches = vm["num_elements"].as<i32>() / BATCH_SIZE;
      } else {
        num_batches = 128 / BATCH_SIZE;
      }

      if (vm.count("device_type")) {
        std::string device = vm["device_type"].as<std::string>();
        if (device == "CPU" || device == "cpu") {
          DEVICE_TYPE = DeviceType::CPU;
        } else if (device == "GPU" || device == "gpu") {
          DEVICE_TYPE = DeviceType::GPU;
        } else {
          LOG(FATAL) << "device_type must be either CPU or GPU";
        }
      }

      if (vm.count("net_input_width")) {
        NET_INPUT_WIDTH = vm["net_input_width"].as<i32>();
      }

      if (vm.count("net_input_height")) {
        NET_INPUT_HEIGHT = vm["net_input_height"].as<i32>();
      }

      net_descriptor_file = vm["net_descriptor_file"].as<std::string>();

    } catch (const po::required_option& e) {
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
      } else {
        throw e;
      }
    }
  }

  // Load net descriptor for specifying target network
  NetDescriptor net_descriptor;
  {
    std::ifstream s{net_descriptor_file};
    net_descriptor = scanner::descriptor_from_net_file(s);
  }

  worker(net_descriptor, num_batches);
}

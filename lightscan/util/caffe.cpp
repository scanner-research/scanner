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

#include "toml/toml.h"

#include <opencv2/opencv.hpp>

#include <cstdlib>


using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::Net;
using boost::shared_ptr;
using std::string;

namespace lightscan {

NetDescriptor descriptor_from_net_file(std::ifstream& net_file) {
  toml::ParseResult pr = toml::parse(net_file);
  if (!pr.valid()) {
    std::cout << pr.errorReason << std::endl;
    exit(EXIT_FAILURE);
  }
  const toml::Value& root = pr.value;

  NetDescriptor descriptor;

  auto net = root.find("net");
  if (!net) {
    std::cout << "Missing 'net': net description map" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto model_path = net->find("model");
  if (!model_path) {
    std::cout << "Missing 'net.model': path to model" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto weights_path = net->find("weights");
  if (!weights_path) {
    std::cout << "Missing 'net.weights': path to model weights" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto input_layer = net->find("input_layer");
  if (!input_layer) {
    std::cout << "Missing 'net.input_layer': name of input layer" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto output_layer = net->find("output_layer");
  if (!output_layer) {
    std::cout << "Missing 'net.output_layer': name of output layer "
              << std::endl;
    exit(EXIT_FAILURE);
  }

  descriptor.model_path = model_path->as<std::string>();
  descriptor.model_weights_path = weights_path->as<std::string>();
  descriptor.input_layer = input_layer->as<std::string>();
  descriptor.output_layer = output_layer->as<std::string>();

  auto mean_image = root.find("mean-image");
  if (!mean_image) {
    std::cout << "Missing 'mean-image': mean image descripton map" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto mean_image_width = mean_image->find("width");
  if (!mean_image_width) {
    std::cout << "Missing 'mean-image.width': width of mean" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto mean_image_height = mean_image->find("height");
  if (!mean_image_height) {
    std::cout << "Missing 'mean-image.height': height of mean" << std::endl;
    exit(EXIT_FAILURE);
  }

  descriptor.mean_width = mean_image_width->as<int>();
  descriptor.mean_height = mean_image_height->as<int>();

  int mean_size = descriptor.mean_width * descriptor.mean_height;
  descriptor.mean_image.resize(mean_size * 3);

  if (mean_image->has("colors")) {
    auto mean_blue = mean_image->find("colors.blue");
    if (!mean_blue) {
      std::cout << "Missing 'mean-image.colors.blue'" << std::endl;
      exit(EXIT_FAILURE);
    }
    auto mean_green = mean_image->find("colors.green");
    if (!mean_green) {
      std::cout << "Missing 'mean-image.colors.green'" << std::endl;
      exit(EXIT_FAILURE);
    }
    auto mean_red = mean_image->find("colors.red");
    if (!mean_red) {
      std::cout << "Missing 'mean-image.colors.red'" << std::endl;
      exit(EXIT_FAILURE);
    }

    float blue = mean_blue->as<float>();
    float green = mean_green->as<float>();
    float red = mean_red->as<float>();

    for (int i = 0; i < mean_size; ++i) {
      size_t offset = i * 3;
      descriptor.mean_image[offset + 0] = blue;
      descriptor.mean_image[offset + 1] = green;
      descriptor.mean_image[offset + 2] = red;
    }
  } else if (mean_image->has("path")) {
    std::string mean_path = mean_image->get<std::string>("path");

    // Load mean image
    Blob<float> data_mean;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_path, &blob_proto);
    data_mean.FromProto(blob_proto);

    memcpy(descriptor.mean_image.data(),
           data_mean.cpu_data(),
           sizeof(float) * mean_size * 3);
  } else {
    std::cout << "Missing 'mean-image.{colors,path}': must specify " <<
              << "color channel values or path of mean image file"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  return descriptor;
}

//////////////////////////////////////////////////////////////////////
/// NetBundle
NetBundle::NetBundle(const NetDescriptor& descriptor, int gpu_device_id)
  : descriptor_(descriptor),
    gpu_device_id_(gpu_device_id)
{
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(gpu_device_id);

  // Initialize our network
  net_.reset(new Net<float>(descriptor_.model_path, caffe::TEST));
  net_->CopyTrainedLayersFrom(descriptor_.model_weights_path);
}

NetBundle::~NetBundle() {
}

const NetDescriptor& NetBundle::get_descriptor() {
  return descriptor_;
}

caffe::Net<float>& NetBundle::get_net() {
  return *net_.get();
}

}

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

#include "toml/toml.h"

#include <cstdlib>

using caffe::Blob;
using caffe::BlobProto;
using caffe::Caffe;
using caffe::Net;
using boost::shared_ptr;
using std::string;

namespace scanner {

//////////////////////////////////////////////////////////////////////
/// NetDescriptor
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
  auto output_layers = net->find("output_layers");
  if (!output_layers) {
    std::cout << "Missing 'net.output_layers': name of output layers "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto input_format = net->find("input");
  if (!input_format) {
    std::cout << "Missing 'net.input': description of net input format "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto dimensions_ordering = input_format->find("dimensions");
  if (!dimensions_ordering) {
    std::cout << "Missing 'net.input.dimensions': ordering of dimensions "
              << "for input format "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto channel_ordering = input_format->find("channel_ordering");
  if (!channel_ordering) {
    std::cout << "Missing 'net.input.channel_ordering': ordering of channels "
              << "for input format "
              << std::endl;
    exit(EXIT_FAILURE);
  }

  descriptor.model_path = model_path->as<std::string>();
  descriptor.model_weights_path = weights_path->as<std::string>();
  descriptor.input_layer_name = input_layer->as<std::string>();
  for (const toml::Value& v : output_layers->as<toml::Array>()) {
    descriptor.output_layer_names.push_back(v.as<std::string>());
  }

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

    float blue = mean_blue->as<double>();
    float green = mean_green->as<double>();
    float red = mean_red->as<double>();

    std::vector<float>& mean_colors = descriptor.mean_colors;
    for (const toml::Value& v : channel_ordering->as<toml::Array>()) {
      std::string color = v.as<std::string>();
      if (color == "red") {
        mean_colors.push_back(red);
      } else if (color == "green") {
        mean_colors.push_back(green);
      } else if (color == "blue") {
        mean_colors.push_back(blue);
      }
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
  } else if (!mean_image->has("empty")) {
    std::cout << "Missing 'mean-image.{colors,path,empty}': must specify "
              << "color channel values or path of mean image file or that "
              << "there is no mean"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  return descriptor;
}

//////////////////////////////////////////////////////////////////////
/// Utils
caffe::Caffe::Brew device_type_to_caffe_mode(DeviceType type) {
  caffe::Caffe::Brew caffe_type;

  switch (type) {
  case DeviceType::GPU:
    caffe_type = caffe::Caffe::GPU;
    break;
  case DeviceType::CPU:
    caffe_type = caffe::Caffe::CPU;
    break;
  default:
    // TODO(apoms): error message
    exit(EXIT_FAILURE);
    break;
  }

  return caffe_type;
}

}

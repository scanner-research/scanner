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

#pragma once

#include "scanner/eval/evaluator.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include <memory>

namespace scanner {

//////////////////////////////////////////////////////////////////////
/// NetDescriptor
struct NetDescriptor {
  std::string model_path;
  std::string model_weights_path;

  std::vector<std::string> input_layer_names;
  std::vector<std::string> output_layer_names;

  int input_width;
  int input_height;

  std::vector<float> mean_colors;
  // or
  std::vector<float> mean_image;
  int mean_width;
  int mean_height;

  bool normalize;
  bool preserve_aspect_ratio;
  i32 pad_mod;
};

NetDescriptor descriptor_from_net_file(std::ifstream& net_file);

//////////////////////////////////////////////////////////////////////
/// Utils
caffe::Caffe::Brew device_type_to_caffe_mode(DeviceType type);
}

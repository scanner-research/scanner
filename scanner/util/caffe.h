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

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include <memory>

namespace scanner {

struct NetDescriptor {
  std::string model_path;
  std::string model_weights_path;

  std::string input_layer_name;
  std::vector<std::string> output_layer_names;

  int input_width;
  int input_height;

  std::vector<float> mean_image;
  int mean_width;
  int mean_height;
};

NetDescriptor descriptor_from_net_file(std::ifstream& net_file);

//////////////////////////////////////////////////////////////////////
/// NetBundle
class NetBundle {
public:
  NetBundle(const NetDescriptor& descriptor, int gpu_device_id);
  virtual ~NetBundle();

  const NetDescriptor& get_descriptor();

  caffe::Net<float>& get_net();

private:
  NetDescriptor descriptor_;
  int gpu_device_id_;
  std::unique_ptr<caffe::Net<float>> net_;
};

}

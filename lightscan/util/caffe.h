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

namespace lightscan {

enum class NetType {
  ALEX_NET,
  VGG,
  VGG_FACE,
};

struct NetInfo {
  caffe::Net<float>* net;
  int input_size;
  float* mean_image;
  int mean_width;
  int mean_height;

  std::string input_layer_name;
  std::string output_layer_name;
};

NetInfo load_neural_net(NetType type, int gpu_id);

}

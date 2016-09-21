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

#pragma once

#include "scanner/util/common.h"

#include "caffe/net.hpp"

namespace scanner {

class CaffeInputTransformer {
public:
  virtual ~CaffeInputTransformer() {};

  virtual void configure(
    const DatasetItemMetadata& metadata,
    caffe::Net<float>* net) = 0;

  virtual void transform_input(
    u8* input_buffer,
    f32* net_input,
    i32 batch_size) = 0;
};

}

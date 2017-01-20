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

#include "scanner/api/kernel.h"
#include "scanner/engine/kernel_factory.h"

#include "scanner/util/common.h"

#include <map>

namespace scanner {

class KernelRegistry {
 public:
   void add_kernel(const std::string &name, KernelFactory *factory);

   KernelFactory *get_kernel(const std::string &name, DeviceType device_type);

 protected:
  static std::string factory_name(const std::string &name, DeviceType type);

 private:

  std::map<std::string, KernelFactory*> factories_;
};

KernelRegistry* get_kernel_registry();

}

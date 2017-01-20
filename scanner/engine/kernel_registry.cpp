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

#include "scanner/engine/kernel_registry.h"

namespace scanner {

void KernelRegistry::add_kernel(const std::string &name,
                                KernelFactory *factory) {
  DeviceType type = factory->get_device_type();
  factories_.insert({factory_name(name, type), factory});
}

KernelFactory *KernelRegistry::get_kernel(const std::string &name,
                                          DeviceType type) {
  return factories_.at(factory_name(name, type));
}

std::string KernelRegistry::factory_name(const std::string &name,
                                         DeviceType type) {
  return name + (type == DeviceType::CPU) ? "_cpu" : "_gpu";
}

KernelRegistry* get_kernel_registry() {
  static KernelRegistry* registry = new KernelRegistry;
  return registry;
}
}

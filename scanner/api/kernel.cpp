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

#include "scanner/api/kernel.h"
#include "scanner/engine/kernel_factory.h"
#include "scanner/engine/kernel_registry.h"

namespace scanner {

Kernel::Kernel(const Config& config) {
}

namespace internal {
KernelRegistration::KernelRegistration(const KernelBuilder& builder) {

  const std::string &name = builder.name_;
  DeviceType type = builder.device_type_;
  i32 num_devices = builder.num_devices_;
  KernelConstructor constructor = builder.constructor_;
  internal::KernelFactory *factory =
    new internal::KernelFactory(name, type, num_devices, 0, constructor);
  internal::KernelRegistry *registry = internal::get_kernel_registry();
  registry->add_kernel(name, factory);
}
}

}

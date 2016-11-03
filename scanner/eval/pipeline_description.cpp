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

#include "scanner/eval/pipeline_description.h"

namespace scanner {
namespace {

std::map<std::string, std::function<PipelineDescription(void)>> pipeline_fns;
}

bool add_pipeline(std::string name,
                  std::function<PipelineDescription(void)> fn) {
  LOG_IF(FATAL, pipeline_fns.count(name) > 0)
      << "Pipeline with name " << name << " has already been registered!";
  pipeline_fns.insert({name, fn});
  printf("insert pipeline %s\n", name.c_str());
  return true;
}

std::function<PipelineDescription(void)> get_pipeline(const std::string& name) {
  return pipeline_fns.at(name);
}
}

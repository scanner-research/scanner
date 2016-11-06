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

std::map<std::string, PipelineGeneratorFn> pipeline_fns;
}

DatasetItemMetadata::DatasetItemMetadata(i32 frames, i32 width, i32 height)
    : frames_(frames), width_(width), height_(height) {}

i32 DatasetItemMetadata::frames() const { return frames_; }

i32 DatasetItemMetadata::width() const { return width_; }

i32 DatasetItemMetadata::height() const { return height_; }

bool add_pipeline(std::string name, PipelineGeneratorFn fn) {
  LOG_IF(FATAL, pipeline_fns.count(name) > 0)
      << "Pipeline with name " << name << " has already been registered!";
  pipeline_fns.insert({name, fn});
  return true;
}

PipelineGeneratorFn get_pipeline(const std::string& name) {
  if (pipeline_fns.count(name) == 0) {
    std::string current_names;
    for (auto& entry : pipeline_fns) {
      current_names += entry.first + " ";
    }

    LOG(FATAL) << "Pipeline with name " << name << " has not been registered. "
               << "Valid pipelines are: " << current_names;
  }

  return pipeline_fns.at(name);
}
}

# Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(SOURCE_FILES
  common.cpp
  memory.cpp
  profiler.cpp
  util.cpp
  opencv.cpp
  jpeg/JPEGReader.cpp
  jpeg/JPEGWriter.cpp)

add_library(util OBJECT
  ${SOURCE_FILES})

if (CUDA_FOUND)
  cuda_add_library(util_cuda)
endif()

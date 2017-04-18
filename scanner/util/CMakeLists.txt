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
  fs.cpp
  bbox.cpp
  progress_bar.cpp)

message("${OPENCV_FOUND}")
if (OPENCV_FOUND)
  list(APPEND SOURCE_FILES opencv.cpp)
endif()

add_library(util OBJECT
  ${SOURCE_FILES})

if (BUILD_CUDA)
  cuda_add_library(util_cuda
    image.cu)
endif()

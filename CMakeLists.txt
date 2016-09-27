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

cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

project(Scanner)

###### Config options #####
option(BUILD_IMAGE_PROCESSING_EVALUATORS "" ON)
option(BUILD_CAFFE_EVALUATORS "" OFF)
option(BUILD_CAFFE_INPUT_TRANSFORMERS "" OFF)
option(BUILD_MOVIE_ANALYSIS_EVALUATORS "" OFF)
option(BUILD_SERVER "" OFF)

enable_testing()

###### Setup #########
# Verify C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
if(COMPILER_SUPPORTS_CXX1Y)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
# if(COMPILER_SUPPORTS_CXX11)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++1y support.")
endif()

# Include our custom cmake modules for finding packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(GLOBAL_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(THIRDPARTY_SOURCE_DIR "${CMAKE_SOURCE_DIR}/thirdparty")

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(NOT APPLE AND UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -pthread -ldl -lrt")
endif()

###### Optional Dependencies #######
find_package(CUDA)
if (CUDA_FOUND)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (BUILD_CAFFE_EVALUATORS)
  find_package(Caffe REQUIRED)
  add_definitions(-DHAVE_CAFFE)
  include_directories(${CAFFE_INCLUDE_DIRS})

  if (NOT CUDA_FOUND)
    add_definitions(-DCPU_ONLY)
  endif()

  if (BUILD_CAFFE_INPUT_TRANSFORMERS)
    # Requires opencv3
    set(OPENCV_DESIRED_COMPONENTS core highgui imgproc imgcodecs)
    if (CUDA_FOUND)
      list(APPEND OPENCV_DESIRED_COMPONENTS
        cudawarping cudaimgproc cudafilters cudaarithm)
    endif()

    find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_DESIRED_COMPONENTS})
    include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
    add_definitions(-DUSE_OPENCV -DHAVE_OPENCV)
  endif()
endif()

if (BUILD_SERVER)
  find_package(Proxygen REQUIRED)
  find_package(Folly REQUIRED)
  add_definitions(-DHAVE_SERVER)
  include_directories(
    "${PROXYGEN_INCLUDE_DIRS}"
    "${FOLLY_INCLUDE_DIRS}")
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  set(OPENCV_DESIRED_COMPONENTS core highgui imgproc objdetect)
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_DESIRED_COMPONENTS})
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
endif()

###### Required Dependencies #######

find_package(FFmpeg REQUIRED)
find_package(MPI REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(BZip2 REQUIRED)
find_package(Boost COMPONENTS thread program_options regex REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(CURL REQUIRED)
find_package(Iconv REQUIRED)
find_package(Storehouse REQUIRED CONFIG
  PATHS "thirdparty/build/bin/storehouse")
find_package(GoogleTest REQUIRED)
find_package(TinyToml REQUIRED)
find_package(JsonCpp REQUIRED)

set(GTEST_INCLUDE_DIRS
  "${GLOBAL_OUTPUT_PATH}/googletest/include")
set(GTEST_LIBRARIES
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest.a")
set(GTEST_LIB_MAIN
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest_main.a")

set(SCANNER_LIBRARIES
  "${STOREHOUSE_LIBRARIES}"
  "${JSONCPP_LIBRARIES}"
  "${FFMPEG_LIBRARIES}"
  "-lswscale"
  "${MPI_LIBRARIES}"
  "${CAFFE_LIBRARIES}"
  "${OpenCV_LIBRARIES}"
  "${LIBLZMA_LIBRARIES}"
  "${OPENSSL_LIBRARIES}"
  "${BZIP2_LIBRARIES}"
  "${PROXYGEN_LIBRARIES}"
  "${FOLLY_LIBRARIES}"
  "${Boost_LIBRARIES}"
  "${GFLAGS_LIBRARIES}"
  "${GLOG_LIBRARIES}"
  "${CURL_LIBRARIES}"
  "${ICONV_LIBRARIES}"
  "${SCANNER_LIBRARIES}"
  "-ljpeg"
  "-lz")

include_directories(
  "."
  "${FFMPEG_INCLUDE_DIR}"
  "${TINYTOML_INCLUDE_DIR}"
  "${JSONCPP_INCLUDE_DIR}"
  "${STOREHOUSE_INCLUDE_DIRS}"
  "${OPENSSL_INCLUDE_DIR}"
  "${Boost_INCLUDE_DIRS}"
  "${MPI_INCLUDE_PATH}"
  "${GLOG_INCLUDE_DIRS}"
  "${LIBLZMA_INCLUDE_DIRS}"
  "${GOOGLETEST_INCLUDE_DIRS}")

if (CUDA_FOUND)
  list(APPEND SCANNER_LIBRARIES
    util_cuda
    "${CUDA_LIBRARIES}"
    "-lcuda"
    "-lcudart"
    "-lnvcuvid")
endif()

if (APPLE)
  include_directories(
    "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
elseif()
endif()

###### Project code #######

add_subdirectory(scanner)
#add_subdirectory(comparison)

add_executable(scanner_server
  scanner/main.cpp
  scanner/ingest.cpp
  scanner/engine.cpp
  $<TARGET_OBJECTS:video>
  $<TARGET_OBJECTS:eval>
  $<TARGET_OBJECTS:evaluators>
  $<TARGET_OBJECTS:parsers>
  $<TARGET_OBJECTS:server>
  $<TARGET_OBJECTS:util>)

if(APPLE)
  set(PLATFORM_LINK_FLAGS
    "-framework CoreFoundation"
    "-framework CoreMedia"
    "-framework CoreVideo"
    "-framework Security"
    "-framework VideoDecodeAcceleration"
    "-framework VideoToolbox"
    "-framework Accelerate"
    "-undefined dynamic_lookup"
    )
elseif(UNIX)
  set(PLATFORM_LINK_FLAGS "-pthread -ldl -lrt")
endif()

target_link_libraries(scanner_server
  ${SCANNER_LIBRARIES}
  "${PLATFORM_LINK_FLAGS}")

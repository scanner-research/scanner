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

cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)

project(Lightscan)

enable_testing()

include(ExternalProject)

set(ENABLE_HARDWARE_DECODE ON CACHE BOOL "Turn on hardware decode")
set(ENABLE_H264_ENCODE OFF CACHE BOOL "Turn on video encoding using libx264")

######################
###     Setup     ####
######################
# Verify C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(GLOBAL_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(THIRDPARTY_SOURCE_DIR "${CMAKE_SOURCE_DIR}/thirdparty")

###### DEPENDENCIES #######

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(NOT APPLE AND UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -pthread -ldl -lrt")
endif()

ExternalProject_Add(JsonCpp
  GIT_REPOSITORY "https://github.com/open-source-parsers/jsoncpp.git"
  GIT_TAG "1.7.2"
  #URL_HASH SHA256=32b84c47910c36137db198a9c5396dd0a87c29ad72f2040493aa9e8b2a24f929
  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/jsoncpp"
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/jsoncpp
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
)

# Google API cpp client
ExternalProject_Add(GoogleAPICppClient
  DEPENDS JsonCpp
  GIT_REPOSITORY "https://github.com/abpoms/google-api-cpp-client.git"
  GIT_TAG "0f6cd03b33ed6a863a6d8d1a30eade4b7766835a"

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/google-api-cpp-client"
  INSTALL_DIR ${GLOBAL_OUTPUT_PATH}/google-api-cpp-client
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/google-api-cpp-client
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
  CMAKE_CACHE_ARGS
      -DJSONCPP_INCLUDE_DIRS:FILEPATH=${THIRDPARTY_SOURCE_DIR}/jsoncpp/include
      -DJSONCPP_LIBRARY:FILEPATH=${GLOBAL_OUTPUT_PATH}/jsoncpp/lib/libjsoncpp.a
      -DGLOG_ROOT_DIR:FILEPATH=${GLOG_ROOT_DIR}
      -DGFLAGS_ROOT_DIR:FILEPATH=${GFLAGS_ROOT_DIR}
)

set(FFMPEG_CFLAGS
  -fPIC)

if(ENABLE_H264_ENCODE)
  set(FFMPEG_BUILD_FLAGS
    --enable-libx264
    --enable-gpl)
endif()

if (ENABLE_HARDWARE_DECODE)
  set(FFMPEG_BUILD_FLAGS
    ${FFMPEG_BUILD_FLAGS}
    --enable-nonfree
    --enable-cuda
    --enable-cuvid)

  add_definitions(-DHARDWARE_DECODE)
endif()

# ffmpeg
ExternalProject_Add(Ffmpeg
  URL "http://ffmpeg.org/releases/ffmpeg-3.1.1.tar.gz"
  #URL_HASH SHA256=
  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/ffmpeg"
  CONFIGURE_COMMAND "${THIRDPARTY_SOURCE_DIR}/ffmpeg/configure"
      --disable-programs
      --disable-doc
      --enable-shared
      "${FFMPEG_BUILD_FLAGS}"
      --prefix=${GLOBAL_OUTPUT_PATH}/ffmpeg

  BUILD_COMMAND make -j4
  INSTALL_COMMAND make install
)

# googletest
ExternalProject_Add(GoogleTest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG 0a439623f75c029912728d80cb7f1b8b48739ca4

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/googletest"

  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/googletest

  INSTALL_DIR "${GLOBAL_OUTPUT_PATH}/googletest"
)

set(GTEST_INCLUDE_DIRS
  "${GLOBAL_OUTPUT_PATH}/googletest/include")
set(GTEST_LIBRARIES
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest.a")
set(GTEST_LIB_MAIN
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest_main.a")

###### PROJECT LIBRARIES #######

# Require opencv3
find_package(OpenCV REQUIRED COMPONENTS
  core highgui imgproc imgcodecs
  cudawarping cudaimgproc cudafilters cudaarithm)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
add_definitions(-DUSE_OPENCV)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(Caffe REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(BZip2 REQUIRED)
find_package(Boost COMPONENTS thread program_options REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(CURL REQUIRED)
find_package(Iconv REQUIRED)

set(PUBLIC_LIBRARIES
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogle_storage_api.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_oauth2.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_jsoncpp.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_json.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_http.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_curl_http.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_utils.a"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/lib/libgoogleapis_internal.a"
  "${GLOBAL_OUTPUT_PATH}/jsoncpp/lib/libjsoncpp.a"
  "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libavformat.a"
  "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libavcodec.a"
  "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libswscale.a"
  "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libswresample.a"
  "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libavutil.a")

set(LIGHTSCAN_LIBRARIES
  ${PUBLIC_LIBRARIES}
  "${MPI_LIBRARIES}"
  "${CUDA_LIBRARIES}"
  "-lcuda"
  "-lnvcuvid"
  "${CAFFE_LIBRARIES}"
  "${OpenCV_LIBRARIES}"
  "${LIBLZMA_LIBRARIES}"
  "${OPENSSL_LIBRARIES}"
  "${BZIP2_LIBRARIES}"
  "${Boost_LIBRARIES}"
  "${GFLAGS_LIBRARIES}"
  "${GLOG_LIBRARIES}"
  "${CURL_LIBRARIES}"
  "${ICONV_LIBRARIES}"
  "${LIGHTSCAN_LIBRARIES}"
  "-ljpeg"
  "-lz")

if (ENABLE_H264_ENCODE)
  set(LIGHTSCAN_LIBRARIES
    "${LIGHTSCAN_LIBRARIES}"
    "-lx264")
  add_definitions(-DHAVE_X264_ENCODER)
endif()

include_directories(
  "."
  ${PUBLIC_INCLUDE_DIRECTORIES}
  "${THIRDPARTY_SOURCE_DIR}/ffmpeg/"
  "${GLOBAL_OUTPUT_PATH}/ffmpeg/include"
  "${GLOBAL_OUTPUT_PATH}/jsoncpp/include/"
  "${GLOBAL_OUTPUT_PATH}/google-api-cpp-client/include"
  "${CUDA_INCLUDE_DIRS}"
  "${MPI_INCLUDE_PATH}"
  "${CAFFE_INCLUDE_DIRS}"
  "${OpenCV_INCLUDE_DIRS}"
  "${GLOG_INCLUDE_DIRS}"
  "${LIBLZMA_INCLUDE_DIRS}"
  "${GTEST_INCLUDE_DIRS}"
  )

if (APPLE)
  include_directories(
    "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
endif()

add_subdirectory(lightscan)

add_executable(lightscanner
  lightscan/main.cpp
  $<TARGET_OBJECTS:storage>
  $<TARGET_OBJECTS:util>)

add_dependencies(lightscanner
  GoogleAPICppClient
  Ffmpeg)

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

target_link_libraries(lightscanner
  ${LIGHTSCAN_LIBRARIES}
  util_cuda
  "${PLATFORM_LINK_FLAGS}")

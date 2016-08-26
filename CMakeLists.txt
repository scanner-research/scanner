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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")


set(GLOBAL_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(THIRDPARTY_SOURCE_DIR "${CMAKE_SOURCE_DIR}/thirdparty")

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()


###### DEPENDENCIES #######

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(NOT APPLE AND UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -pthread -ldl -lrt")
endif()

###### PROJECT LIBRARIES #######

# Require opencv3
find_package(OpenCV REQUIRED COMPONENTS
  core highgui imgproc imgcodecs
  cudawarping cudaimgproc cudafilters cudaarithm)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
add_definitions(-DUSE_OPENCV)

find_package(FFmpeg REQUIRED)
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

ExternalProject_Add(TinyToml
  GIT_REPOSITORY "https://github.com/mayah/tinytoml.git"
  GIT_TAG "3559856002eee57693349b8a2d8a0cf6250d269c"

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/tinytoml"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""

  INSTALL_COMMAND ""
)

ExternalProject_Add(JsonCpp
  GIT_REPOSITORY "https://github.com/open-source-parsers/jsoncpp.git"
  GIT_TAG "1.7.2"

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/jsoncpp"
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/jsoncpp
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
)

# storage
if (NOT OPENSSL_ROOT_DIR)
  set(OPENSSL_ROOT_DIR $ENV{OPENSSL_ROOT_DIR})
endif()

ExternalProject_Add(Storage
  GIT_REPOSITORY "https://github.com/abpoms/storage"
  #GIT_TAG 0a439623f75c029912728d80cb7f1b8b48739ca4

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/storage"

  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/storage
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DBZIP2_INCLUDE_DIR=${BZIP2_INCLUDE_DIR}
      -DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}
      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
      -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}

  CMAKE_CACHE_ARGS
      -DGLOG_ROOT_DIR:FILEPATH=${GLOG_ROOT_DIR}
      -DGFLAGS_ROOT_DIR:FILEPATH=${GFLAGS_ROOT_DIR}
      -DBOOST_ROOT:FILEPATH=${BOOST_ROOT}
      -DCURL_INCLUDE_DIR:FILEPATH=${CURL_INCLUDE_DIR}
      -DCURL_LIBRARY:FILEPATH=${CURL_LIBRARY}
      -DCURL_LIBRARIES:FILEPATH=${CURL_LIBRARIES}

  INSTALL_DIR "${GLOBAL_OUTPUT_PATH}/storage"
)


# googletest
ExternalProject_Add(GoogleTest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG 0a439623f75c029912728d80cb7f1b8b48739ca4

  UPDATE_COMMAND ""

  SOURCE_DIR "${THIRDPARTY_SOURCE_DIR}/googletest"

  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOBAL_OUTPUT_PATH}/googletest
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}

  INSTALL_DIR "${GLOBAL_OUTPUT_PATH}/googletest"
)


set(GTEST_INCLUDE_DIRS
  "${GLOBAL_OUTPUT_PATH}/googletest/include")
set(GTEST_LIBRARIES
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest.a")
set(GTEST_LIB_MAIN
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest_main.a")

set(LIGHTSCAN_LIBRARIES
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libstorage.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogle_storage_api.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_oauth2.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_jsoncpp.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_json.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_http.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_curl_http.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_utils.a"
  "${GLOBAL_OUTPUT_PATH}/storage/lib/libgoogleapis_internal.a"
  "${GLOBAL_OUTPUT_PATH}/jsoncpp/lib/libjsoncpp.a"
  # "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libavformat.a"
  # "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libavcodec.a"
  # "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libswscale.a"
  # "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libswresample.a"
  # "${GLOBAL_OUTPUT_PATH}/ffmpeg/lib/libavutil.a"
  "${FFMPEG_LIBRARIES}"
  "${MPI_LIBRARIES}"
  "${CUDA_LIBRARIES}"
  "-lcuda"
  "-lcudart"
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

include_directories(
  "."
  "${FFMPEG_INCLUDE_DIR}"
  "${THIRDPARTY_SOURCE_DIR}/tinytoml/include"
  "${GLOBAL_OUTPUT_PATH}/jsoncpp/include"
  "${GLOBAL_OUTPUT_PATH}/storage/include"
  "${Boost_INCLUDE_DIRS}"
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
elseif()
endif()

add_subdirectory(lightscan)
add_subdirectory(comparison)

add_executable(lightscanner
  lightscan/main.cpp
  lightscan/ingest.cpp
  lightscan/engine.cpp
  $<TARGET_OBJECTS:util>)

add_dependencies(lightscanner
  TinyToml
  JsonCpp)

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

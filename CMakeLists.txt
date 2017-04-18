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
option(BUILD_CUDA "" ON)
option(BUILD_TESTS "" ON)
option(BUILD_SERVER "" OFF)
option(ENABLE_PROFILING "" OFF)

if (BUILD_TESTS)
  enable_testing()
endif()

###### Setup #########
# Verify C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
if(COMPILER_SUPPORTS_CXX1Y)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
# if(COMPILER_SUPPORTS_CXX11)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++1y support.")
endif()

# Include our custom cmake modules for finding packages
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(GLOBAL_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(THIRDPARTY_SOURCE_DIR "${CMAKE_SOURCE_DIR}/thirdparty")
set(THIRDPARTY_OUTPUT_PATH "${THIRDPARTY_SOURCE_DIR}/build/bin")

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(NOT APPLE AND UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -pthread -ldl -lrt")
endif()

if (ENABLE_PROFILING)
  add_definitions(-DSCANNER_PROFILING)
endif()

###### Optional Dependencies #######
if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
endif()

if (BUILD_CUDA)
  add_library(scanner_halide scanner/util/halide_context.cpp)
endif()

###### Required Dependencies #######
find_package(SaneProtobuf REQUIRED)
find_package(FFmpeg REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(BZip2 REQUIRED)
find_package(Boost COMPONENTS thread program_options regex python REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(GoogleTest REQUIRED)
find_package(CURL REQUIRED)
find_package(Iconv REQUIRED)
find_package(Storehouse REQUIRED CONFIG
  PATHS "thirdparty/build/bin/storehouse")
find_package(TinyToml REQUIRED)
find_package(PythonLibs 2.7 EXACT REQUIRED)
find_package(OpenCV COMPONENTS core highgui imgproc cudaimgproc cudaarithm)

set(GTEST_INCLUDE_DIRS
  "${THIRDPARTY_OUTPUT_PATH}/googletest/include")
set(GTEST_LIBRARIES
  "${THIRDPARTY_OUTPUT_PATH}/googletest/lib/libgtest.a")
set(GTEST_LIB_MAIN
  "${THIRDPARTY_OUTPUT_PATH}/googletest/lib/libgtest_main.a")

set(SCANNER_LIBRARIES
  "${PROTOBUF_LIBRARY}"
  "${STOREHOUSE_LIBRARIES}"
  "${FFMPEG_LIBRARIES}"
  "-L/opt/ffmpeg-3.2.2/lib"
  "-lswscale"
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
  "${PYTHON_LIBRARIES}"
  "-ljpeg"
  "-lz"
  "-ldl"
  "-lgrpc++_unsecure -lgrpc -lgpr")

include_directories(
  "."
  "${CMAKE_CURRENT_BINARY_DIR}" # for protobuf generated files
  "${PROTOBUF_INCLUDE_DIRS}"
  "${FFMPEG_INCLUDE_DIR}"
  "${TINYTOML_INCLUDE_DIR}"
  "${STOREHOUSE_INCLUDE_DIRS}"
  "${OPENSSL_INCLUDE_DIR}"
  "${Boost_INCLUDE_DIRS}"
  "${GLOG_INCLUDE_DIRS}"
  "${LIBLZMA_INCLUDE_DIRS}"
  "${PYTHON_INCLUDE_DIRS}")

if (OPENCV_FOUND)
  list(APPEND SCANNER_LIBRARIES ${OpenCV_LIBRARIES})
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if (BUILD_TESTS)
  include_directories("${GTEST_INCLUDE_DIRS}")
endif()

if (BUILD_CUDA)
  list(APPEND SCANNER_LIBRARIES
    util_cuda
    "${CUDA_LIBRARIES}"
    "/usr/lib/x86_64-linux-gnu/libnvcuvid.so"
    "-lcuda")
endif()

if (APPLE)
  include_directories(
    "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
elseif()
endif()

###### Project code #######
set(PROTO_FILES
  scanner/metadata.proto
  scanner/types.proto)

set(GRPC_PROTO_FILES
  scanner/engine/rpc.proto)

protobuf_generate_cpp(
  PROTO_SRCS PROTO_HDRS OFF ${PROTO_FILES} ${GRPC_PROTO_FILES})
protobuf_generate_cpp(
  GRPC_PROTO_SRCS GRPC_PROTO_HDRS ON ${GRPC_PROTO_FILES})

protobuf_generate_python(PROTO_PY OFF ${PROTO_FILES})
protobuf_generate_python(GRPC_PROTO_PY ON ${GRPC_PROTO_FILES})

add_custom_target(proto_files DEPENDS
  ${PROTO_HDRS} ${PROTO_PY} ${GRPC_PROTO_HDRS} ${GRPC_PROTO_PY})

macro(add_deps)
  add_dependencies(${targetName} proto_files)
endmacro()

function(add_library targetName)
  _add_library(${targetName} ${ARGN})
  add_deps()
endfunction()

function(add_executable targetName)
  _add_executable(${targetName} ${ARGN})
  add_deps()
endfunction()

set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG")

add_subdirectory(scanner)

add_library(scanner SHARED
  $<TARGET_OBJECTS:api>
  $<TARGET_OBJECTS:engine>
  $<TARGET_OBJECTS:video>
  $<TARGET_OBJECTS:util>
  ${PROTO_SRCS}
  ${GRPC_PROTO_SRCS}
  ${STRUCK_SOURCES}
  ${OFDIS_SOURCES})

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

target_link_libraries(scanner PUBLIC
  ${SCANNER_LIBRARIES}
  "${PLATFORM_LINK_FLAGS}")

set(PYDIR ${CMAKE_CURRENT_BINARY_DIR})

# Make init files so python code can import from subdirectories
foreach(FIL ${PROTO_FILES} ${GRPC_PROTO_FILES})
  get_filename_component(DIR_FIL ${FIL} DIRECTORY)
  get_filename_component(FIL_WE ${FIL} NAME_WE)
  add_custom_command(TARGET proto_files POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch ${PYDIR}/${DIR_FIL}/__init__.py)
endforeach()

add_subdirectory(stdlib)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

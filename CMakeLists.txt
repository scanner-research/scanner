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

include(cmake/Dependencies.cmake)

###### Project code #######
set(PROTO_FILES
  scanner/metadata.proto
  scanner/source_args.proto
  scanner/sampler_args.proto
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
  scanner/util/halide_context.cpp
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

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
option(BUILD_VIDEO_EVALUATORS "" ON)
option(BUILD_IMAGE_EVALUATORS "" OFF)
option(BUILD_IMAGE_PROCESSING_EVALUATORS "" ON)
option(BUILD_CAFFE_EVALUATORS "" OFF)
option(BUILD_CAFFE_INPUT_EVALUATORS "" OFF)
option(BUILD_CPM2_EVALUATORS "" OFF)
option(BUILD_MOVIE_ANALYSIS_EVALUATORS "" OFF)
option(BUILD_TRACKER_EVALUATORS "" OFF)
option(BUILD_UTIL_EVALUATORS "" OFF)
option(BUILD_MEDIAN_EVALUATORS "" OFF)
option(BUILD_SERVER "" OFF)
option(BUILD_COMPARISON "" OFF)
option(BUILD_EXAMPLES "" ON)
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
set(OPENCV_DESIRED_COMPONENTS)
set(OPENCV_MAJOR_VERSION 3)

set(HALIDE_TARGETS)
macro(add_halide_target SRC TARGET)
  if (NOT HALIDE_FOUND)
    find_package(Halide REQUIRED)
    include(${CMAKE_SOURCE_DIR}/cmake/Util/HalideGenerator.cmake)
    include_directories("${HALIDE_INCLUDE_DIR}")
  endif()

  get_filename_component(NAME ${SRC} NAME_WE)
  set(GENERATOR ${NAME}.generator)
  halide_add_generator(${GENERATOR}
    SRCS ${SRC})
  halide_add_aot_library(${NAME}
    GENERATOR_TARGET ${GENERATOR}
    GENERATOR_ARGS target=${TARGET})
  list(APPEND HALIDE_TARGETS ${NAME})
endmacro()

if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
endif()

if (BUILD_VIDEO_EVALUATORS)
  list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video videoio)

  # Check for intel HW decode
  #find_package(MFX)
endif()

if (BUILD_CUDA)
  add_library(scanner_halide scanner/engine/halide_context.cpp)
endif()

if (BUILD_CAFFE_EVALUATORS)
  find_package(Caffe REQUIRED)
  add_definitions(-DHAVE_CAFFE)
  include_directories(${CAFFE_INCLUDE_DIRS})

  if (NOT BUILD_CUDA)
    add_definitions(-DCPU_ONLY)
  endif()

  add_halide_target(scanner/evaluators/caffe/default/caffe_input_transformer_cpu.cpp host)
  if (BUILD_CUDA)
    add_halide_target(scanner/evaluators/caffe/default/caffe_input_transformer_gpu.cpp cuda)
  endif()

  if (BUILD_CAFFE_INPUT_EVALUATORS)
    # Requires opencv3
    list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc imgcodecs)
    if (BUILD_CUDA)
      list(APPEND OPENCV_DESIRED_COMPONENTS
        cudawarping cudaimgproc cudafilters cudaarithm)
    endif()
  endif()
endif()

if (BUILD_TRACKER_EVALUATORS)
  find_package(Eigen REQUIRED)
  find_package(Struck REQUIRED CONFIG
    PATHS "thirdparty/build/bin/struck")
  include_directories(
    "${EIGEN_INCLUDE_DIRS}"
    "${STRUCK_INCLUDE_DIRS}")
endif()

if (BUILD_SERVER)
  find_package(Proxygen REQUIRED)
  find_package(Folly REQUIRED)
  add_definitions(-DHAVE_SERVER)
  include_directories(
    "${PROXYGEN_INCLUDE_DIRS}"
    "${FOLLY_INCLUDE_DIRS}")
endif()

if (BUILD_UTIL_EVALUATORS)
  if (OPENCV_MAJOR_VERSION EQUAL 3)
     list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video videoio)
  else()
     list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video)
  endif()
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  set(DOPPIA_DIR /homes/wcrichto/rodrigob-doppia-2f93f2657960/src)
  set(OFDIS_DIR /homes/wcrichto/OF_DIS)
  if (OPENCV_MAJOR_VERSION EQUAL 3)
      list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc objdetect video)
      if (BUILD_CUDA)
          list(APPEND OPENCV_DESIRED_COMPONENTS cudaoptflow cudacodec)
      endif()
  else()
      list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc objdetect video)
  endif()
  find_package(Eigen REQUIRED)
  include_directories(
    "${EIGEN_INCLUDE_DIRS}"
    "${OFDIS_DIR}"
    "${DOPPIA_DIR}")
  file(GLOB OFDIS_SOURCES
    ${OFDIS_DIR}/FDF1.0.1/*.c
    ${OFDIS_DIR}/oflow.cpp
    ${OFDIS_DIR}/patchgrid.cpp
    ${OFDIS_DIR}/patch.cpp
    ${OFDIS_DIR}/refine_variational.cpp)
  add_definitions(-DWITH_OPENMP=true)
  FIND_PACKAGE( OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  #link_directories(${DOPPIA_DIR}/applications/objects_detection_lib/build)
  set(SCANNER_LIBRARIES
    #"-lmonocular_objects_detection"
    #"-liomp5"
    "${SCANNER_LIBRARIES}")
endif()

if (BUILD_MEDIAN_EVALUATORS)
  find_package(Eigen)
endif()

if (OPENCV_DESIRED_COMPONENTS)
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_DESIRED_COMPONENTS})
  include_directories(SYSTEM
    ${OpenCV_INCLUDE_DIRS})
  add_definitions(-DUSE_OPENCV -DHAVE_OPENCV)
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
  "${STRUCK_LIBRARIES}"
  "${OpenCV_LIBRARIES}"
  "${CAFFE_LIBRARIES}"
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
  "-lz"
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
  "${LIBLZMA_INCLUDE_DIRS}")

if (BUILD_TESTS)
  include_directories("${GTEST_INCLUDE_DIRS}")
endif()

if (BUILD_CUDA)
  list(APPEND SCANNER_LIBRARIES
    util_cuda
    "${CUDA_LIBRARIES}"
    "-lcuda"
    "-L/usr/lib/nvidia-367"
    "-lnvcuvid")
endif()

if (APPLE)
  include_directories(
    "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
elseif()
endif()

###### Project code #######
set(PROTO_FILES
  scanner/metadata.proto
  scanner/kernels/args.proto
  scanner/kernels/types.proto)

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
  if (HALIDE_FOUND)
    foreach(HALIDE_TARGET ${HALIDE_TARGETS})
      add_dependencies(${targetName} "${HALIDE_TARGET}.exec_generator")
      target_include_directories(${targetName} PUBLIC
        "${PROJECT_BINARY_DIR}/generator_genfiles")
    endforeach()
    if (BUILD_CUDA)
      add_dependencies(${targetName} scanner_halide)
    endif()
  endif()
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

if (BUILD_COMPARISON)
  add_subdirectory(comparison)
endif()

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

add_library(scanner SHARED
  $<TARGET_OBJECTS:api>
  $<TARGET_OBJECTS:engine>
  $<TARGET_OBJECTS:video>
  $<TARGET_OBJECTS:kernels>
  $<TARGET_OBJECTS:util>
  ${PROTO_SRCS}
  ${GRPC_PROTO_SRCS}
  ${STRUCK_SOURCES}
  ${OFDIS_SOURCES})

add_executable(scanner_server
  scanner/main.cpp
  scanner)

configure_file(scanner/engine/db.in.cpp scanner/engine/db.cpp)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
configure_file(scanner/engine/python.in.cpp scanner/engine/python.cpp)
add_library(scanner_bindings SHARED scanner/engine/python.cpp)
set_target_properties(scanner_bindings PROPERTIES PREFIX "")
target_include_directories(scanner_bindings PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(scanner_bindings PUBLIC
  scanner
  ${Boost_LIBRARIES}
  ${PythonLibs_LIBRARIES})

if (HALIDE_FOUND)
  foreach(TARGET ${HALIDE_TARGETS})
    halide_add_aot_library_dependency(scanner ${TARGET})
    if (BUILD_CUDA)
      target_link_libraries(scanner PRIVATE scanner_halide)
    endif()
  endforeach()
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  set_target_properties(scanner PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
  set_property(TARGET scanner APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1")
endif()


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

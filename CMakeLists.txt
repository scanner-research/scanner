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
option(BUILD_VIDEO_EVALUATORS "" ON)
option(BUILD_IMAGE_EVALUATORS "" ON)
option(BUILD_IMAGE_PROCESSING_EVALUATORS "" ON)
option(BUILD_CAFFE_EVALUATORS "" OFF)
option(BUILD_CAFFE_INPUT_EVALUATORS "" OFF)
option(BUILD_MOVIE_ANALYSIS_EVALUATORS "" OFF)
option(BUILD_TRACKER_EVALUATORS "" OFF)
option(BUILD_UTIL_EVALUATORS "" OFF)
option(BUILD_MEDIAN_EVALUATORS "" OFF)
option(BUILD_SERVER "" OFF)
option(BUILD_COMPARISON "" OFF)

enable_testing()

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

###### Optional Dependencies #######
set(OPENCV_DESIRED_COMPONENTS)
set(OPENCV_MAJOR_VERSION 3)

find_package(CUDA)
if (CUDA_FOUND)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
endif()

if (BUILD_VIDEO_EVALUATORS)
  list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video videoio)
endif()

if (BUILD_CAFFE_EVALUATORS)
  find_package(Caffe REQUIRED)
  add_definitions(-DHAVE_CAFFE)
  include_directories(${CAFFE_INCLUDE_DIRS})

  if (NOT CUDA_FOUND)
    add_definitions(-DCPU_ONLY)
  endif()

  if (BUILD_CAFFE_INPUT_EVALUATORS)
    # Requires opencv3
    list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc imgcodecs)
    if (CUDA_FOUND)
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
      list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc objdetect video xfeatures2d)
      if (CUDA_FOUND)
          list(APPEND OPENCV_DESIRED_COMPONENTS cudaoptflow)
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
find_package(JPEGWrapper REQUIRED CONFIG
  PATHS "thirdparty/build/bin/jpegwrapper")
find_package(LodePNG REQUIRED)

set(GTEST_INCLUDE_DIRS
  "${GLOBAL_OUTPUT_PATH}/googletest/include")
set(GTEST_LIBRARIES
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest.a")
set(GTEST_LIB_MAIN
  "${GLOBAL_OUTPUT_PATH}/googletest/lib/libgtest_main.a")

set(SCANNER_LIBRARIES
  "${PROTOBUF_LIBRARY}"
  "${STOREHOUSE_LIBRARIES}"
  "${JSONCPP_LIBRARIES}"
  "${FFMPEG_LIBRARIES}"
  "-lswscale"
  "${MPI_LIBRARIES}"
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
  "${JPEGWRAPPER_LIBRARIES}"
  "${SCANNER_LIBRARIES}"
  "-ljpeg"
  "-lz")

include_directories(
  "."
  "${CMAKE_CURRENT_BINARY_DIR}" # for protobuf generated files
  "${PROTOBUF_INCLUDE_DIRS}"
  "${FFMPEG_INCLUDE_DIR}"
  "${TINYTOML_INCLUDE_DIR}"
  "${JSONCPP_INCLUDE_DIR}"
  "${STOREHOUSE_INCLUDE_DIRS}"
  "${JPEGWRAPPER_INCLUDE_DIRS}"
  "${LODEPNG_INCLUDE_DIRS}"
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
  scanner/evaluators/types.proto)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

protobuf_generate_python(PROTO_PY ${PROTO_FILES})

add_custom_target(proto_files DEPENDS ${PROTO_HDRS} ${PROTO_PY})

function(add_library targetName)
  _add_library(${targetName} ${ARGN})
  add_dependencies(${targetName} proto_files)
endfunction()

function(add_executable targetName)
  _add_executable(${targetName} ${ARGN})
  add_dependencies(${targetName} proto_files)
endfunction()

add_subdirectory(scanner)

if (BUILD_COMPARISON)
  add_subdirectory(comparison)
endif()

add_executable(scanner_server
  scanner/main.cpp
  scanner/ingest.cpp
  scanner/engine.cpp
  $<TARGET_OBJECTS:video>
  $<TARGET_OBJECTS:eval>
  $<TARGET_OBJECTS:evaluators>
  $<TARGET_OBJECTS:parsers>
  $<TARGET_OBJECTS:server>
  $<TARGET_OBJECTS:pipelines>
  $<TARGET_OBJECTS:util>
  ${PROTO_SRCS}
  ${LODEPNG_SOURCES}
  ${STRUCK_SOURCES}
  ${OFDIS_SOURCES})

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  set_target_properties(scanner_server PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
  set_property(TARGET scanner_server APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1")
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

target_link_libraries(scanner_server
  ${SCANNER_LIBRARIES}
  "${PLATFORM_LINK_FLAGS}")

set(PYDIR ${CMAKE_CURRENT_BINARY_DIR}/scannerpy)

add_custom_command(TARGET proto_files POST_BUILD
  COMMAND rm -rf ${PYDIR})

# Make init files so python code can import from subdirectories
foreach(FIL ${PROTO_FILES})
  get_filename_component(DIR_FIL ${FIL} DIRECTORY)
  get_filename_component(FIL_WE ${FIL} NAME_WE)
  add_custom_command(TARGET proto_files POST_BUILD
    COMMAND
    ${CMAKE_COMMAND} -E
    make_directory ${PYDIR}/${DIR_FIL} &&
    ${CMAKE_COMMAND} -E
    touch ${PYDIR}/${DIR_FIL}/__init__.py &&
    ${CMAKE_COMMAND} -E
    copy ${DIR_FIL}/${FIL_WE}_pb2.py ${PYDIR}/${DIR_FIL})
endforeach()

add_custom_command(TARGET proto_files POST_BUILD
  COMMAND mv -f ${PYDIR}/scanner/* ${PYDIR} && rm -r ${PYDIR}/scanner)

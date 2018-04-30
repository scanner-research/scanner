set(SCANNER_PATH "${CMAKE_SOURCE_DIR}")
include(${SCANNER_PATH}/cmake/Util/Op.cmake)

option(BUILD_IMGPROC_OPS "" ON)
option(BUILD_CAFFE_OPS "" ON)
option(BUILD_OPENPOSE_OPS "" ON)
option(BUILD_MOTION_OPS "" ON)
option(BUILD_VIZ_OPS "" ON)
option(BUILD_OPENFACE_OPS "" OFF)
option(BUILD_GIPUMA_OPS "" OFF)

set(STDLIB_LIBRARIES)
set(OPENCV_MAJOR_VERSION 3)
set(OPENCV_COMPONENTS)

# Can't build OpenPose if we don't have GPUs
if (NOT ${BUILD_CUDA})
  set(BUILD_OPENPOSE_OPS OFF)
endif()

find_package(Halide REQUIRED)
set(HALIDE_DISTRIB_DIR "${Halide_DIR}")
set(HALIDE_DISTRIB_USE_STATIC_LIBRARY TRUE)
include(${Halide_DIR}/halide.cmake)
#include_directories("${HALIDE_INCLUDE_DIR}")
#include_directories("${PROJECT_BINARY_DIR}/generator_genfiles")


if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
  list(APPEND STDLIB_LIBRARIES "${CUDA_LIBRARIES}")
else()
  add_definitions(-DCPU_ONLY)
endif()

if (BUILD_OPENPOSE_OPS)
  find_package(OpenPose REQUIRED)
  include_directories(${OPENPOSE_INCLUDE_DIRS})
  list(APPEND STDLIB_LIBRARIES "${OPENPOSE_LIBRARIES}")
endif()

set(TARGETS)
if (BUILD_CAFFE_OPS)
  add_subdirectory(caffe)
  list(APPEND TARGETS caffe)
  #list(APPEND STDLIB_LIBRARIES
  #  ${PROJECT_BINARY_DIR}/genfiles/caffe_input_transformer_cpu/caffe_input_transformer_cpu.a)
  #if (BUILD_CUDA)
  #  list(APPEND STDLIB_LIBRARIES
  #    ${PROJECT_BINARY_DIR}/genfiles/caffe_input_transformer_gpu/caffe_input_transformer_gpu.a)
  #endif()
endif()

if (BUILD_IMGPROC_OPS)
  add_subdirectory(imgproc)
  list(APPEND TARGETS imgproc)
endif()

if (BUILD_OPENFACE_OPS)
  add_subdirectory(openface)
  list(APPEND TARGETS openface)
endif()

if (BUILD_GIPUMA_OPS)
  add_subdirectory(gipuma)
  list(APPEND TARGETS gipuma)
endif()

if (BUILD_MOTION_OPS)
  add_subdirectory(motion)
  list(APPEND TARGETS motion)
endif()

if (BUILD_VIZ_OPS)
  add_subdirectory(viz)
  list(APPEND TARGETS viz)
endif()

add_subdirectory(misc)
list(APPEND TARGETS misc)

set(SOURCES)
foreach (TARGET ${TARGETS})
  list(APPEND SOURCES $<TARGET_OBJECTS:${TARGET}>)
endforeach()

build_op(
  LIB_NAME stdlib
  CPP_SRCS ${SOURCES}
  PROTO_SRC stdlib.proto
  NO_FLAGS ON)
add_dependencies(stdlib scanner)

add_custom_command(TARGET stdlib POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)

find_package(OpenCV REQUIRED COMPONENTS "${OPENCV_COMPONENTS}")
list(APPEND STDLIB_LIBRARIES "${OpenCV_LIBRARIES}")

foreach (TARGET ${TARGETS})
  target_include_directories(${TARGET} PUBLIC ${OpenCV_INCLUDE_DIRS})
  add_dependencies(${TARGET} stdlib_proto_files)
endforeach()

target_link_libraries(stdlib PUBLIC scanner
  ${STDLIB_LIBRARIES}
  ${HALIDE_LIBRARY}
  ${Boost_LIBRARIES})

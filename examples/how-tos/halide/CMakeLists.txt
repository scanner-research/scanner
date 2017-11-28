cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

# Uncomment the line below and change the Scanner path to the repo you cloned:
# set(SCANNER_PATH /path/to/scanner)
if(NOT SCANNER_PATH)
  message(FATAL_ERROR "You need to update the SCANNER_PATH in halide/CMakeLists.txt first.")
endif()
include(${SCANNER_PATH}/cmake/Util/Op.cmake)

find_package(Halide REQUIRED)
include(${SCANNER_PATH}/cmake/Util/HalideGenerator.cmake)

halide_add_generator(halide_resize.generator SRCS halide_resize.cpp)
halide_add_aot_library(halide_resize
  GENERATOR_TARGET halide_resize.generator
  GENERATOR_ARGS target=cuda)

build_op(
  LIB_NAME resize
  CPP_SRCS resize_op.cpp
  PROTO_SRC resize.proto)
target_compile_definitions(resize PUBLIC -DHAVE_CUDA)

add_dependencies(resize halide_resize.exec_generator)
target_include_directories(resize PUBLIC "${HALIDE_INCLUDE_DIR}")
target_include_directories(resize PUBLIC "${PROJECT_BINARY_DIR}/generator_genfiles")
halide_add_aot_library_dependency(resize halide_resize)

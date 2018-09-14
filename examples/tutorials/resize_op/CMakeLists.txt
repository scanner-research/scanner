# To build your custom op, you can either use our CMake convenience methods
# or do it the slightly harder way in normal Make (see the Makefile).

cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.stdlib.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

# build_op will create a shared library called lib${LIB_NAME}.so that builds
# from all of the CPP_SRCS. You can also optionally specify a PROTO_SRC that
# points to a Protobuf file and will generate the C++ and Python bindings.
build_op(
  LIB_NAME resize_op
  CPP_SRCS resize_op.cpp
  PROTO_SRC resize.proto)

# The library specified in build_op is a normal CMake target, so you can use all
# the normal CMake functions with it.
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
target_link_libraries(resize_op PUBLIC "${OpenCV_LIBRARIES}")

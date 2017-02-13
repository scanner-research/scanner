cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

set(SCANNER_PATH /h/wcrichto/scanner)
include(${SCANNER_PATH}/cmake/Util/Op.cmake)

build_op(
  LIB_NAME resize_op
  CPP_SRCS resize_op.cpp
  PROTO_SRC args.proto)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
target_link_libraries(resize_op PUBLIC "${OpenCV_LIBRARIES}")

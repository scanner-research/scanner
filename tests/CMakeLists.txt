include(${CMAKE_SOURCE_DIR}/cmake/Util/Op.cmake)
build_op(LIB_NAME scanner_tests CPP_SRCS test_ops.cpp PROTO_SRC test_ops.proto NO_FLAGS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui video)
target_link_libraries(scanner_tests PUBLIC ${OpenCV_LIBRARIES} scanner)

add_test(
  NAME PythonTests
  COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -x -vv)

if (NO_FFMPEG STREQUAL "false")
  add_executable(FfmpegTest ffmpeg_test.cpp)
  target_link_libraries(FfmpegTest ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN} scanner)
  add_test(FfmpegTests FfmpegTest)
endif()

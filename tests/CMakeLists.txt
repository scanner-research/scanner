include(${CMAKE_SOURCE_DIR}/cmake/Util/Op.cmake)
build_op(LIB_NAME scanner_tests CPP_SRCS test_ops.cpp)

add_test(
  NAME PythonTests
  COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -x -vv)

if (NO_FFMPEG STREQUAL "false")
  add_executable(FfmpegTest ffmpeg_test.cpp)
  target_link_libraries(FfmpegTest ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN} scanner)
  add_test(FfmpegTests FfmpegTest)
endif()

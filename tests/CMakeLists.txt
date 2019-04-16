find_package(OpenCV COMPONENTS core imgproc highgui video)

if (OpenCV_FOUND)
  include(${CMAKE_SOURCE_DIR}/cmake/Util/Op.cmake)
  build_op(LIB_NAME scanner_tests CPP_SRCS test_ops.cpp PROTO_SRC test_ops.proto NO_FLAGS ON)

  target_link_libraries(scanner_tests PUBLIC ${OpenCV_LIBRARIES} scanner)
  target_include_directories(scanner_tests PUBLIC ${OpenCV_INCLUDE_DIR})

  add_test(
    NAME PythonTests
    COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -x -vv)
else()
  message("Scanner tests are disabled. OpenCV not found which is required for building the tests.")
endif()

if (NO_FFMPEG STREQUAL "false")
  add_executable(FfmpegTest ffmpeg_test.cpp)
  target_link_libraries(FfmpegTest ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN} scanner)
  add_test(FfmpegTests FfmpegTest)
endif()

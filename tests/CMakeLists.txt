add_test(
  NAME PythonTests
  COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -x -vv)

add_executable(FfmpegTest ffmpeg_test.cpp)
target_link_libraries(FfmpegTest ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN} scanner stdlib)
add_test(FfmpegTests FfmpegTest)

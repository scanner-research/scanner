set(SOURCE_FILES
  info_from_frame_kernel.cpp
  discard_kernel.cpp
  sleep_kernel.cpp
  test_increment_kernel.cpp
  packed_file_source.cpp
  pass_kernel.cpp
  files_source.cpp)

add_library(misc OBJECT ${SOURCE_FILES})

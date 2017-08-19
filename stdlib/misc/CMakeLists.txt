set(SOURCE_FILES
  info_from_frame_kernel.cpp
  discard_kernel.cpp
  sleep_kernel.cpp)

add_library(misc OBJECT ${SOURCE_FILES})

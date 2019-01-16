set(SOURCE_FILES
  info_from_frame_kernel.cpp
  discard_kernel.cpp
  sleep_kernel.cpp
  test_increment_kernel.cpp
  packed_file_source.cpp
  pass_kernel.cpp
  python_source.cpp
  files_source.cpp
  files_sink.cpp
  captions_source.cpp)

if (NO_FFMPEG STREQUAL "false")
  list(APPEND SOURCE_FILES
    audio_source.cpp)
endif()

if (NO_LIBPQXX STREQUAL "false")
  list(APPEND SOURCE_FILES
    sql_source.cpp
    sql_sink.cpp
    sql.cpp)
endif()

add_library(misc OBJECT ${SOURCE_FILES})

set(SOURCE_FILES)

if (BUILD_SERVER)
  list(APPEND SOURCE_FILES
    video_handler_factory.cpp
    video_handler.cpp
    request_utils.cpp)
endif()

add_library(server OBJECT
  ${SOURCE_FILES})

set(SOURCE_FILES)

if (BUILD_SERVER)
  list(APPEND SOURCE_FILES
    imagenet_parser.cpp
    yolo_parser.cpp
    facenet_parser.cpp
    bbox_parser.cpp)
endif()

add_library(parsers OBJECT
  ${SOURCE_FILES})

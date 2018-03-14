set(SOURCE_FILES
  frame.cpp
  kernel.cpp
  op.cpp
  source.cpp
  sink.cpp
  enumerator.cpp
  database.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

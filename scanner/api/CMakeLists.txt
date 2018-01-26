set(SOURCE_FILES
  frame.cpp
  kernel.cpp
  op.cpp
  database.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

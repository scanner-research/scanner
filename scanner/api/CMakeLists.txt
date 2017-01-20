set(SOURCE_FILES
  kernel.cpp
  evaluator.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

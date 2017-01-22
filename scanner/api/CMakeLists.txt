set(SOURCE_FILES
  kernel.cpp
  evaluator.cpp
  ingest.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

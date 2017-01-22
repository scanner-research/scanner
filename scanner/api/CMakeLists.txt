set(SOURCE_FILES
  kernel.cpp
  evaluator.cpp
  run.cpp
  ingest.cpp
  task_set.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

set(SOURCE_FILES
  kernel.cpp
  evaluator.cpp
  commands.cpp
  task_set.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

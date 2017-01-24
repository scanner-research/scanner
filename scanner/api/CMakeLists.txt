set(SOURCE_FILES
  kernel.cpp
  evaluator.cpp
  commands.cpp
  user_function.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

set(SOURCE_FILES
  kernel.cpp
  op.cpp
  database.cpp
  user_function.cpp)

add_library(api OBJECT
  ${SOURCE_FILES})

set(SOURCE_FILES
  runtime.cpp
  load_worker.cpp
  evaluate_worker.cpp
  save_worker.cpp
  sampling.cpp
  db.cpp)

add_library(engine OBJECT
  ${SOURCE_FILES})

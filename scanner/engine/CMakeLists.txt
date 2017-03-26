get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
configure_file(python.in.cpp python.cpp)

set(SOURCE_FILES
  runtime.cpp
  master.cpp
  worker.cpp
  ingest.cpp
  load_worker.cpp
  evaluate_worker.cpp
  save_worker.cpp
  sampler.cpp
  metadata.cpp
  kernel_registry.cpp
  op_registry.cpp
  python.cpp)

add_library(engine OBJECT
  ${SOURCE_FILES})

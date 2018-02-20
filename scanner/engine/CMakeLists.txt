get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
configure_file(build_flags.in.cpp build_flags.cpp)

add_executable(build_flags build_flags.cpp)

set(SOURCE_FILES
  runtime.cpp
  master.cpp
  worker.cpp
  ingest.cpp
  video_index_entry.cpp
  load_worker.cpp
  evaluate_worker.cpp
  save_worker.cpp
  sampler.cpp
  dag_analysis.cpp
  metadata.cpp
  kernel_registry.cpp
  op_registry.cpp
  source_registry.cpp
  column_source.cpp
  column_enumerator.cpp
  enumerator_registry.cpp
  table_meta_cache.cpp
  python.cpp
  python_kernel.cpp
  sample_op.cpp
  space_op.cpp
  slice_op.cpp
  unslice_op.cpp)

add_library(engine OBJECT
  ${SOURCE_FILES})

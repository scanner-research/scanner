add_subdirectory(caffe)

add_library(eval OBJECT
  $<TARGET_OBJECTS:eval_caffe>)

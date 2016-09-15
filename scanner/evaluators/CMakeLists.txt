set(SOURCE_FILES)

if (CAFFE_FOUND)
  add_subdirectory(caffe)
  list(APPEND SOURCE_FILES
    caffe/net_descriptor.cpp
    caffe/caffe_gpu_evaluator.cpp
    caffe/caffe_cpu_evaluator.cpp)
endif()

list(APPEND SOURCE_FILES
  image_processing/blur_evaluator.cpp)

add_library(evaluators OBJECT
  ${SOURCE_FILES})

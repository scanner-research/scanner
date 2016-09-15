set(SOURCE_FILES)

if (CAFFE_FOUND AND BUILD_CAFFE_EVALUATORS)
  add_subdirectory(caffe)
  list(APPEND SOURCE_FILES
    caffe/net_descriptor.cpp
    caffe/caffe_gpu_evaluator.cpp
    caffe/caffe_cpu_evaluator.cpp)
endif()

if (BUILD_IMAGE_PROCESSING_EVALUATORS)
  list(APPEND SOURCE_FILES
    image_processing/blur_evaluator.cpp)
endif()

add_library(evaluators OBJECT
  ${SOURCE_FILES})

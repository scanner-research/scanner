set(SOURCE_FILES)

if (CAFFE_FOUND AND BUILD_CAFFE_EVALUATORS)
  list(APPEND SOURCE_FILES
    caffe/net_descriptor.cpp
    caffe/caffe_cpu_evaluator.cpp
    caffe/vgg/vgg_cpu_input_transformer.cpp)
  if (CUDA_FOUND)
    list(APPEND SOURCE_FILES
      caffe/caffe_gpu_evaluator.cpp
      caffe/vgg/vgg_gpu_input_transformer.cpp)
  endif()
endif()

if (BUILD_IMAGE_PROCESSING_EVALUATORS)
  list(APPEND SOURCE_FILES
    image_processing/blur_evaluator.cpp)
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  list(APPEND SOURCE FILES
    movie_analysis/histogram_evaluator.cpp)
endif()

add_library(evaluators OBJECT
  ${SOURCE_FILES})

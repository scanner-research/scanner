set(SOURCE_FILES)

if (BUILD_CAFFE_EVALUATORS)
  list(APPEND SOURCE_FILES
    caffe/net_descriptor.cpp
    caffe/caffe_evaluator.cpp)
  if (BUILD_CAFFE_INPUT_TRANSFORMERS)
    list(APPEND SOURCE_FILES
      caffe/vgg/vgg_cpu_input_transformer.cpp
      caffe/yolo/yolo_cpu_input_transformer.cpp
      caffe/facenet/facenet_cpu_input_transformer.cpp
      caffe/facenet/facenet_parser_evaluator.cpp)
    if (CUDA_FOUND)
      list(APPEND SOURCE_FILES
        caffe/facenet/facenet_gpu_input_transformer.cpp)
    endif()
  endif()
endif()

if (BUILD_IMAGE_PROCESSING_EVALUATORS)
  list(APPEND SOURCE_FILES
    image_processing/blur_evaluator.cpp)
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  list(APPEND SOURCE_FILES
    movie_analysis/movie_evaluator.cpp)
endif()

if (BUILD_TRACKER_EVALUATORS)
  list(APPEND SOURCE_FILES
    tracker/tracker_evaluator.cpp)
endif()

add_library(evaluators OBJECT
  ${SOURCE_FILES})

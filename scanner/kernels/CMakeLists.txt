set(SOURCE_FILES
  blur_kernel.cpp)

if (BUILD_IMAGE_EVALUATORS)
  list(APPEND SOURCE_FILES
    image_encoder_kernel.cpp)
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  list(APPEND SOURCE_FILES
    histogram_kernel.cpp
    optical_flow_kernel.cpp)
endif()

if (BUILD_CAFFE_EVALUATORS)
  list(APPEND SOURCE_FILES
    caffe_kernel.cpp
    caffe_input_kernel.cpp
    facenet_input_kernel.cpp
    facenet_output_kernel.cpp)
endif()

add_library(kernels OBJECT
  ${SOURCE_FILES})

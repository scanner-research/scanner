set(SOURCE_FILES
  net_descriptor.cpp)

if (CUDA_FOUND)
  append(SOURCE_FILES
    caffe_gpu_evaluator.cpp)
endif()

add_library(eval_caffe OBJECT ${SOURCE_FILES})

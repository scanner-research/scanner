set(SOURCE_FILES
  blur_kernel_cpu.cpp)

if (BUILD_IMAGE_OPS)
  list(APPEND SOURCE_FILES
    image_encoder_kernel_cpu.cpp)
endif()

if (BUILD_MOVIE_ANALYSIS_OPS)
  list(APPEND SOURCE_FILES
    histogram_kernel_cpu.cpp
    optical_flow_kernel_cpu.cpp)
  if (BUILD_CUDA)
    list(APPEND SOURCE_FILES
      histogram_kernel_gpu.cpp
      optical_flow_kernel_gpu.cpp)
  endif()
endif()

if (BUILD_CAFFE_OPS)
  list(APPEND SOURCE_FILES
    caffe_kernel.cpp
    caffe_kernel_cpu.cpp
    caffe_input_kernel.cpp
    caffe_input_kernel_cpu.cpp
    facenet_output_kernel_cpu.cpp)
  if (BUILD_CUDA)
    list(APPEND SOURCE_FILES
      caffe_kernel_gpu.cpp
      caffe_input_kernel_gpu.cpp
      facenet_input_kernel_gpu.cpp
      facenet_kernel.cpp)
  endif()
endif()

if (BUILD_CPM2_OPS)
  list(APPEND SOURCE_FILES
    cpm2_output_kernel_cpu.cpp)
  if (BUILD_CUDA)
    list(APPEND SOURCE_FILES
      cpm2_input_kernel_gpu.cpp
      cpm2_kernel.cpp)
  endif()
endif()

add_library(kernels OBJECT
  ${SOURCE_FILES})

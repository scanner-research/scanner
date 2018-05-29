set(SOURCE_FILES
  caffe_kernel.cpp
  caffe_kernel_cpu.cpp
  # Removed until halide '_halide_downgrade_buffer_t' symbol issue resolved
  # caffe_input_kernel.cpp
  # caffe_input_kernel_cpu.cpp
  facenet_input_kernel_cpu.cpp
  facenet_kernel.cpp
  facenet_output_kernel_cpu.cpp
  yolo_output_kernel_cpu.cpp
  faster_rcnn_kernel.cpp
  faster_rcnn_output_kernel_cpu.cpp)
if (BUILD_CUDA)
  list(APPEND SOURCE_FILES
    caffe_kernel_gpu.cpp
    # Removed until halide '_halide_downgrade_buffer_t' symbol issue resolved
    #caffe_input_kernel_gpu.cpp
    facenet_input_kernel_gpu.cpp)
endif()

if (BUILD_OPENPOSE_OPS)
  list(APPEND SOURCE_FILES
    openpose_kernel.cpp)
endif()

add_library(caffe OBJECT ${SOURCE_FILES})

#halide_library(caffe_input_transformer_cpu
#  SRCS caffe_input_transformer_cpu.cpp)
#add_dependencies(caffe caffe_input_transformer_cpu)
#include_directories(${CMAKE_BINARY_DIR}/genfiles/caffe_input_transformer_cpu)
if (BUILD_CUDA)
  # halide_library(caffe_input_transformer_gpu
  #   SRCS caffe_input_transformer_gpu.cpp
  #   HALIDE_TARGET cuda)
  # include_directories(${CMAKE_BINARY_DIR}/genfiles/caffe_input_transformer_gpu)
  # add_dependencies(caffe caffe_input_transformer_gpu)
endif()

add_definitions(-DUSE_OPENCV)


list(APPEND OPENCV_COMPONENTS core highgui imgproc video videoio)
if (BUILD_CUDA)
  list(APPEND OPENCV_COMPONENTS cudaarithm cudawarping cudaimgproc)
endif()
set(OPENCV_COMPONENTS ${OPENCV_COMPONENTS} PARENT_SCOPE)

find_package(Caffe REQUIRED)
target_compile_definitions(caffe PUBLIC -DHAVE_CAFFE)
target_include_directories(caffe PUBLIC "${CAFFE_INCLUDE_DIRS}")
list(APPEND STDLIB_LIBRARIES "${CAFFE_LIBRARIES}")

set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)

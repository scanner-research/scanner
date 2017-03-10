set(SOURCE_FILES
  caffe_kernel.cpp
  caffe_kernel_cpu.cpp
  caffe_input_kernel.cpp
  caffe_input_kernel_cpu.cpp
  facenet_output_kernel_cpu.cpp
  yolo_output_kernel_cpu.cpp
  faster_rcnn_kernel.cpp
  faster_rcnn_output_kernel_cpu.cpp)
if (BUILD_CUDA)
  list(APPEND SOURCE_FILES
    caffe_kernel_gpu.cpp
    caffe_input_kernel_gpu.cpp
    facenet_input_kernel_gpu.cpp
    facenet_kernel.cpp)
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

add_halide_target(caffe_input_transformer_cpu.cpp host)
if (BUILD_CUDA)
  add_halide_target(caffe_input_transformer_gpu.cpp cuda)
endif()

add_definitions(-DUSE_OPENCV)

add_library(caffe OBJECT ${SOURCE_FILES})

# TODO(wcrichto): auto add these dependencies?
add_dependencies(caffe
  caffe_input_transformer_cpu.exec_generator)
if (BUILD_CUDA)
  add_dependencies(caffe
    caffe_input_transformer_gpu.exec_generator)
endif()

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

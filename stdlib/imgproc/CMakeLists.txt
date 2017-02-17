set(SOURCE_FILES
  blur_kernel_cpu.cpp
  histogram_kernel_cpu.cpp
  image_encoder_kernel_cpu.cpp)

if (BUILD_CUDA)
  list(APPEND SOURCE_FILES
    histogram_kernel_gpu.cpp)
endif()

add_library(imgproc OBJECT ${SOURCE_FILES})

list(APPEND OPENCV_COMPONENTS core highgui imgproc video videoio)
set(OPENCV_COMPONENTS ${OPENCV_COMPONENTS} PARENT_SCOPE)

set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)

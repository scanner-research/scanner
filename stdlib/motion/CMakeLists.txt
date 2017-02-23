# TODO(wcrichto): fill this out once stenciling is implemented

# if (BUILD_TRACKER_OPS)
#   find_package(Eigen REQUIRED)
#   find_package(Struck REQUIRED CONFIG
#     PATHS "thirdparty/build/bin/struck")
#   include_directories(
#     "${EIGEN_INCLUDE_DIRS}"
#     "${STRUCK_INCLUDE_DIRS}")
# endif()

set(SOURCE_FILES
  optical_flow_kernel_cpu.cpp)

if(BUILD_CUDA)
  list(APPEND SOURCE_FILES optical_flow_kernel_gpu.cpp)
  list(APPEND OPENCV_COMPONENTS cudaoptflow)
endif()

add_library(motion OBJECT ${SOURCE_FILES})

list(APPEND OPENCV_COMPONENTS core video)
set(OPENCV_COMPONENTS ${OPENCV_COMPONENTS} PARENT_SCOPE)

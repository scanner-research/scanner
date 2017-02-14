set(SCANNER_PATH "${CMAKE_SOURCE_DIR}")
include(${SCANNER_PATH}/cmake/Util/Op.cmake)

option(BUILD_IMAGE_OPS "" ON)
option(BUILD_CAFFE_OPS "" ON)
option(BUILD_CAFFE_INPUT_OPS "" ON)
option(BUILD_CPM2_OPS "" OFF)
option(BUILD_MOVIE_ANALYSIS_OPS "" ON)
option(BUILD_TRACKER_OPS "" OFF)
option(BUILD_UTIL_OPS "" OFF)
option(BUILD_MEDIAN_OPS "" OFF)

set(OPENCV_DESIRED_COMPONENTS)
set(OPENCV_MAJOR_VERSION 3)

set(KERNEL_LIBRARIES)

set(HALIDE_TARGETS)
macro(add_halide_target SRC TARGET)
  if (NOT HALIDE_FOUND)
    find_package(Halide REQUIRED)
    include(${CMAKE_SOURCE_DIR}/cmake/Util/HalideGenerator.cmake)
    include_directories("${HALIDE_INCLUDE_DIR}")
  endif()

  get_filename_component(NAME ${SRC} NAME_WE)
  set(GENERATOR ${NAME}.generator)
  halide_add_generator(${GENERATOR}
    SRCS ${SRC})
  halide_add_aot_library(${NAME}
    GENERATOR_TARGET ${GENERATOR}
    GENERATOR_ARGS target=${TARGET})
  list(APPEND HALIDE_TARGETS ${NAME})
endmacro()

if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
  list(APPEND KERNEL_LIBRARIES "${CUDA_LIBRARIES}")
endif()

if (BUILD_CAFFE_OPS)
  find_package(Caffe REQUIRED)
  add_definitions(-DHAVE_CAFFE)
  include_directories(${CAFFE_INCLUDE_DIRS})

  if (NOT BUILD_CUDA)
    add_definitions(-DCPU_ONLY)
  endif()

  add_halide_target(caffe_input_transformer_cpu.cpp host)
  if (BUILD_CUDA)
    add_halide_target(caffe_input_transformer_gpu.cpp cuda)
  endif()

  if (BUILD_CAFFE_INPUT_OPS)
    # Requires opencv3
    list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc imgcodecs)
    if (BUILD_CUDA)
      list(APPEND OPENCV_DESIRED_COMPONENTS
        cudawarping cudaimgproc cudafilters cudaarithm)
    endif()
  endif()

  list(APPEND KERNEL_LIBRARIES "${CAFFE_LIBRARIES}")
endif()

if (BUILD_TRACKER_OPS)
  find_package(Eigen REQUIRED)
  find_package(Struck REQUIRED CONFIG
    PATHS "thirdparty/build/bin/struck")
  include_directories(
    "${EIGEN_INCLUDE_DIRS}"
    "${STRUCK_INCLUDE_DIRS}")
endif()

if (BUILD_SERVER)
  find_package(Proxygen REQUIRED)
  find_package(Folly REQUIRED)
  add_definitions(-DHAVE_SERVER)
  include_directories(
    "${PROXYGEN_INCLUDE_DIRS}"
    "${FOLLY_INCLUDE_DIRS}")
endif()

if (BUILD_UTIL_OPS)
  if (OPENCV_MAJOR_VERSION EQUAL 3)
     list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video videoio)
  else()
     list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video)
  endif()
endif()

if (BUILD_MOVIE_ANALYSIS_OPS)
  set(DOPPIA_DIR /homes/wcrichto/rodrigob-doppia-2f93f2657960/src)
  set(OFDIS_DIR /homes/wcrichto/OF_DIS)
  if (OPENCV_MAJOR_VERSION EQUAL 3)
      list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc objdetect video)
      if (BUILD_CUDA)
          list(APPEND OPENCV_DESIRED_COMPONENTS cudaoptflow cudacodec)
      endif()
  else()
      list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc objdetect video)
  endif()
  find_package(Eigen REQUIRED)
  include_directories(
    "${EIGEN_INCLUDE_DIRS}"
    "${OFDIS_DIR}"
    "${DOPPIA_DIR}")
  file(GLOB OFDIS_SOURCES
    ${OFDIS_DIR}/FDF1.0.1/*.c
    ${OFDIS_DIR}/oflow.cpp
    ${OFDIS_DIR}/patchgrid.cpp
    ${OFDIS_DIR}/patch.cpp
    ${OFDIS_DIR}/refine_variational.cpp)
  add_definitions(-DWITH_OPENMP=true)
  FIND_PACKAGE( OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  #link_directories(${DOPPIA_DIR}/applications/objects_detection_lib/build)
  set(SCANNER_LIBRARIES
    #"-lmonocular_objects_detection"
    #"-liomp5"
    "${SCANNER_LIBRARIES}")
endif()

list(APPEND OPENCV_DESIRED_COMPONENTS core highgui imgproc video videoio)

if (BUILD_MEDIAN_OPS)
  find_package(Eigen)
endif()

if (OPENCV_DESIRED_COMPONENTS)
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_DESIRED_COMPONENTS})
  include_directories(SYSTEM
    ${OpenCV_INCLUDE_DIRS})
  add_definitions(-DUSE_OPENCV -DHAVE_OPENCV)
  list(APPEND KERNEL_LIBRARIES "${OpenCV_LIBRARIES}")
endif()

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

build_op(
  LIB_NAME kernels
  CPP_SRCS ${SOURCE_FILES}
  PROTO_SRC args.proto
  NO_FLAGS ON)
add_dependencies(kernels scanner)

target_link_libraries(kernels PUBLIC "${KERNEL_LIBRARIES}")

if (HALIDE_FOUND)
  foreach(HALIDE_TARGET ${HALIDE_TARGETS})
    add_dependencies(kernels "${HALIDE_TARGET}.exec_generator")
    target_include_directories(kernels PUBLIC
      "${PROJECT_BINARY_DIR}/generator_genfiles")
    halide_add_aot_library_dependency(kernels ${HALIDE_TARGET})
  endforeach()
  if (BUILD_CUDA)
    add_dependencies(kernels scanner_halide)
    target_link_libraries(kernels PRIVATE scanner_halide)
  endif()
endif()

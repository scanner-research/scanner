set(SCANNER_PATH "${CMAKE_SOURCE_DIR}")
include(${SCANNER_PATH}/cmake/Util/Op.cmake)

option(BUILD_IMGPROC_OPS "" ON)
option(BUILD_CAFFE_OPS "" ON)
option(BUILD_CPM2_OPS "" OFF)
option(BUILD_MOTION_OPS "" ON)
option(BUILD_OPENFACE_OPS "" OFF)

set(STDLIB_LIBRARIES)
set(OPENCV_MAJOR_VERSION 3)
set(OPENCV_COMPONENTS)

set(HALIDE_TARGETS)
macro(add_halide_target SRC TARGET)
  if (NOT HALIDE_FOUND)
    find_package(Halide REQUIRED)
    include(${CMAKE_SOURCE_DIR}/cmake/Util/HalideGenerator.cmake)
    include_directories("${HALIDE_INCLUDE_DIR}")
    include_directories("${PROJECT_BINARY_DIR}/generator_genfiles")
  endif()

  get_filename_component(NAME ${SRC} NAME_WE)
  set(GENERATOR ${NAME}.generator)
  halide_add_generator(${GENERATOR}
    SRCS ${SRC})
  halide_add_aot_library(${NAME}
    GENERATOR_TARGET ${GENERATOR}
    GENERATOR_ARGS target=${TARGET})
  list(APPEND HALIDE_TARGETS ${NAME})
  set(HALIDE_TARGETS ${HALIDE_TARGETS} PARENT_SCOPE)
endmacro()

if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
  list(APPEND STDLIB_LIBRARIES "${CUDA_LIBRARIES}")
else()
  add_definitions(-DCPU_ONLY)
endif()

set(SOURCES)
if (BUILD_CAFFE_OPS)
  add_subdirectory(caffe)
  list(APPEND SOURCES $<TARGET_OBJECTS:caffe>)
endif()

if (BUILD_IMGPROC_OPS)
  add_subdirectory(imgproc)
  list(APPEND SOURCES $<TARGET_OBJECTS:imgproc>)
endif()

if (BUILD_OPENFACE_OPS)
  add_subdirectory(openface)
  list(APPEND SOURCES $<TARGET_OBJECTS:openface>)
endif()

if (BUILD_MOTION_OPS)
  add_subdirectory(motion)
  list(APPEND SOURCES $<TARGET_OBJECTS:motion>)
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

if (BUILD_MEDIAN_OPS)
  find_package(Eigen)
endif()

build_op(
  LIB_NAME stdlib
  CPP_SRCS ${SOURCES}
  PROTO_SRC stdlib.proto
  NO_FLAGS ON)
add_dependencies(stdlib scanner)

find_package(OpenCV REQUIRED COMPONENTS "${OPENCV_COMPONENTS}")
if (BUILD_CAFFE_OPS)
  target_include_directories(caffe PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
if (BUILD_IMGPROC_OPS)
  target_include_directories(imgproc PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
if (BUILD_OPENFACE_OPS)
  target_include_directories(openface PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
if (BUILD_MOTION_OPS)
  target_include_directories(motion PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
list(APPEND STDLIB_LIBRARIES "${OpenCV_LIBRARIES}")

target_link_libraries(stdlib PUBLIC "${STDLIB_LIBRARIES}")

if (HALIDE_TARGETS)
  foreach(HALIDE_TARGET ${HALIDE_TARGETS})
    add_dependencies(stdlib "${HALIDE_TARGET}.exec_generator")
    halide_add_aot_library_dependency(stdlib ${HALIDE_TARGET})
  endforeach()
  if (BUILD_CUDA)
    add_dependencies(stdlib scanner_halide)
    target_link_libraries(stdlib PRIVATE scanner_halide)
  endif()
endif()

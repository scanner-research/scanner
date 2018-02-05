###### Parse dependency file #######
file(STRINGS ${CMAKE_SOURCE_DIR}/dependencies.txt ConfigContents)
foreach(NameAndValue ${ConfigContents})
  # Strip leading spaces
  string(REGEX REPLACE "^[ ]+" "" NameAndValue ${NameAndValue})
  # Find variable name
  string(REGEX MATCH "^[^=]+" Name ${NameAndValue})
  # Find the value
  string(REPLACE "${Name}=" "" Value ${NameAndValue})
  # Set the variable
  set(${Name} "${Value}")
endforeach()

list(APPEND CMAKE_PREFIX_PATH ${PROTOBUF_DIR})

# Disable cuda if nvidia-smi was not detected
if (${HAVE_GPU} STREQUAL "false")
  set(BUILD_CUDA OFF)
endif()

###### Optional Dependencies #######
if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DHAVE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  if(COMPILER_SUPPORTS_CXX1Y)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
  find_package(NVCUVID REQUIRED)
endif()

if (BUILD_CUDA)
  add_library(scanner_halide scanner/util/halide_context.cpp)
endif()

set(OPENCV_DESIRED_COMPONENTS core highgui imgproc)
if (BUILD_CUDA)
  list(APPEND OPENCV_DESIRED_COMPONENTS cudaimgproc cudaarithm)
endif()

###### Required Dependencies #######
find_package(SaneProtobuf REQUIRED)
find_package(GRPC REQUIRED)
find_package(FFmpeg REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(BZip2 REQUIRED)
find_package(Boost COMPONENTS thread program_options regex python REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(GoogleTest REQUIRED)
find_package(CURL REQUIRED)
find_package(Iconv REQUIRED)
find_package(Storehouse REQUIRED CONFIG
  PATHS "${CMAKE_SOURCE_DIR}/thirdparty/install")
find_package(Hwang REQUIRED)
find_package(TinyToml REQUIRED)
find_package(PythonLibs 2.7 EXACT REQUIRED)
find_package(OpenCV COMPONENTS ${OPENCV_DESIRED_COMPONENTS})
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(GTEST_INCLUDE_DIRS ${GOOGLETEST_INCLUDE_DIR})
set(GTEST_LIBRARIES ${GOOGLETEST_LIBRARIES})
set(GTEST_LIB_MAIN ${GOOGLETEST_MAIN})

set(SCANNER_LIBRARIES
  "${HWANG_LIBRARY}"
  "${PROTOBUF_LIBRARY}"
  "${GRPC_LIBRARIES}"
  "${FFMPEG_LIBRARIES}"
  "${LIBLZMA_LIBRARIES}"
  "${BZIP2_LIBRARIES}"
  "${GFLAGS_LIBRARIES}"
  "${GLOG_LIBRARIES}"
  "${CURL_LIBRARIES}"
  "${ICONV_LIBRARIES}"
  "${SCANNER_LIBRARIES}"
  "-lpython2.7"
  "${Boost_LIBRARIES}"
  "${Boost_LIBRARY_DIRS}/libboost_python.so"
  "${Boost_LIBRARY_DIRS}/libboost_numpy.so"
  "${STOREHOUSE_LIBRARIES}"
  "${OPENSSL_LIBRARIES}"
  "-ljpeg"
  "-lz"
  "-ldl"
  )

message(${SCANNER_LIBRARIES})

include_directories(
  "."
  "${CMAKE_CURRENT_BINARY_DIR}" # for protobuf generated files
  "${HWANG_INCLUDE_DIRS}"
  "${PROTOBUF_INCLUDE_DIRS}"
  "${GRPC_INCLUDE_DIRS}"
  "${FFMPEG_INCLUDE_DIR}"
  "${TINYTOML_INCLUDE_DIR}"
  "${STOREHOUSE_INCLUDE_DIRS}"
  "${OPENSSL_INCLUDE_DIR}"
  "${GLOG_INCLUDE_DIRS}"
  "${LIBLZMA_INCLUDE_DIRS}"
  "${PYTHON_INCLUDE_DIRS}"
  "${Boost_INCLUDE_DIRS}")

if (OpenCV_FOUND)
  list(APPEND SCANNER_LIBRARIES ${OpenCV_LIBRARIES})
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_definitions(-DHAVE_OPENCV)
endif()

if (BUILD_TESTS)
  include_directories("${GTEST_INCLUDE_DIRS}")
endif()

if (BUILD_CUDA)
  list(APPEND SCANNER_LIBRARIES
    util_cuda
    ${CUDA_LIBRARIES}
    ${NVCUVID_LIBRARIES}
    "-lcuda")
endif()

if (APPLE)
  include_directories(
    "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
elseif()
endif()

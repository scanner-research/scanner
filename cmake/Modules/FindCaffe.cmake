# - Try to find Caffe
#
# The following variables are optionally searched for defaults
#  CAFFE_ROOT_DIR:       Base directory where all Caffe components are found
#
# The following are set after configuration is done:
#  CAFFE_FOUND
#  CAFFE_INCLUDE_DIRS
#  CAFFE_LIBRARIES
#  CAFFE_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(CAFFE_ROOT_DIR "" CACHE PATH "Folder contains Caffe")

if (NOT "$ENV{Caffe_DIR}" STREQUAL "")
  set(CAFFE_ROOT_DIR $ENV{Caffe_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(CAFFE_INCLUDE_DIR caffe/caffe.hpp
    PATHS ${CAFFE_ROOT_DIR}/src/windows)
else()
  find_path(CAFFE_INCLUDE_DIR caffe/caffe.hpp
    PATHS ${CAFFE_ROOT_DIR}/include)
endif()

find_library(CAFFE_LIBRARY caffe PATHS ${CAFFE_ROOT_DIR}/lib)

find_package_handle_standard_args(CAFFE DEFAULT_MSG
    CAFFE_INCLUDE_DIR CAFFE_LIBRARY)

if(CAFFE_FOUND)
    set(CAFFE_INCLUDE_DIRS ${CAFFE_INCLUDE_DIR})
    set(CAFFE_LIBRARIES ${CAFFE_LIBRARY})
endif()

# - Try to find OpenPose
#
# The following variables are optionally searched for defaults
#  OPENPOSE_ROOT_DIR:       Base directory where all Caffe components are found
#
# The following are set after configuration is done:
#  OPENPOSE_FOUND
#  OPENPOSE_INCLUDE_DIRS
#  OPENPOSE_LIBRARIES
#  OPENPOSE_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(OPENPOSE_ROOT_DIR "" CACHE PATH "Folder contains OpenPose")

if (NOT "$ENV{OpenPose_DIR}" STREQUAL "")
  set(OPENPOSE_ROOT_DIR $ENV{OpenPose_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(OPENPOSE_INCLUDE_DIR openpose/headers.hpp
    PATHS ${OPENPOSE_ROOT_DIR}/src/windows)
else()
  find_path(OPENPOSE_INCLUDE_DIR openpose/headers.hpp
    PATHS ${OPENPOSE_ROOT_DIR}/include)
endif()

find_library(OPENPOSE_LIBRARY openpose PATHS ${OPENPOSE_ROOT_DIR}/lib)

find_package_handle_standard_args(OPENPOSE DEFAULT_MSG
    OPENPOSE_INCLUDE_DIR OPENPOSE_LIBRARY)

if(OPENPOSE_FOUND)
    set(OPENPOSE_INCLUDE_DIRS ${OPENPOSE_INCLUDE_DIR})
    set(OPENPOSE_LIBRARIES ${OPENPOSE_LIBRARY})
endif()

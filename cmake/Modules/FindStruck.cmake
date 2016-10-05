# - Try to find Struck
#
# The following variables are optionally searched for defaults
#  STRUCK_ROOT_DIR:       Base directory where all Struck components are found
#
# The following are set after configuration is done:
#  STRUCK_FOUND
#  STRUCK_INCLUDE_DIRS
#  STRUCK_LIBRARIES
#  STRUCK_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(STRUCK_ROOT_DIR "" CACHE PATH "Folder contains Struck")

if (NOT "$ENV{Struck_DIR}" STREQUAL "")
  set(STRUCK_ROOT_DIR $ENV{Struck_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(STRUCK_INCLUDE_DIR struck/tracker.h
    PATHS ${STRUCK_ROOT_DIR}/src/windows)
else()
  find_path(STRUCK_INCLUDE_DIR struck/tracker.h
    PATHS ${STRUCK_ROOT_DIR}/include)
endif()

find_library(STRUCK_LIBRARY caffe PATHS ${STRUCK_ROOT_DIR}/lib)

find_package_handle_standard_args(STRUCK DEFAULT_MSG
    STRUCK_INCLUDE_DIR STRUCK_LIBRARY)

if(STRUCK_FOUND)
    set(STRUCK_INCLUDE_DIRS ${STRUCK_INCLUDE_DIR})
    set(STRUCK_LIBRARIES ${STRUCK_LIBRARY})
endif()

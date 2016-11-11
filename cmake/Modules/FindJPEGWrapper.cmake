# - Try to find jpeg library
#
# The following variables are optionally searched for defaults
#  JPEGWRAPPER_ROOT_DIR:   Base directory where all Storage components are found
#
# The following are set after configuration is done:
#  JPEGWRAPPER_FOUND
#  JPEGWRAPPER_INCLUDE_DIRS

include(FindPackageHandleStandardArgs)

set(JPEGWRAPPER_ROOT_DIR "" CACHE PATH "Folder contains JPEGWrapper")

if (NOT "$ENV{JPEGWrapper_DIR}" STREQUAL "")
  set(JPEGWRAPPER_ROOT_DIR $ENV{JPEGWrapper_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(JPEGWRAPPER_INCLUDE_DIR jpegwrapper/JPEGReader.h
    PATHS ${JPEGWRAPPER_ROOT_DIR}/src/windows)
else()
  find_path(JPEGWRAPPER_INCLUDE_DIR jpegwrapper/JPEGReader.h
    PATHS ${JPEGWRAPPER_ROOT_DIR}/include)
endif()

find_library(JPEGWRAPPER_LIBRARY jpegwrapper PATHS ${JPEGWRAPPER_ROOT_DIR}/lib)

find_package_handle_standard_args(JPEGWRAPPER DEFAULT_MSG
    JPEGWRAPPER_INCLUDE_DIR JPEG_LIBRARIES)

if(JPEGWRAPPER_FOUND)
  set(JPEGWRAPPER_INCLUDE_DIRS ${JPEGWRAPPER_INCLUDE_DIR})
  set(JPEGWRAPPER_LIBRARIES ${JPEGWRAPPER_LIBRARIES})
endif()

# - Try to find Folly
#
# The following variables are optionally searched for defaults
#  FOLLY_ROOT_DIR:   Base directory where all folly components are found
#
# The following are set after configuration is done:
#  FOLLY_FOUND
#  FOLLY_INCLUDE_DIRS
#  FOLLY_LIBRARIES
#  FOLLY_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(FOLLY_ROOT_DIR "" CACHE PATH "Folder contains Folly")

if (NOT "$ENV{Folly_DIR}" STREQUAL "")
  set(FOLLY_ROOT_DIR $ENV{Folly_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(FOLLY_INCLUDE_DIR folly/FBVector.h
    PATHS ${FOLLY_ROOT_DIR}/src/windows)
else()
  find_path(FOLLY_INCLUDE_DIR folly/FBVector.h
    PATHS ${FOLLY_ROOT_DIR}/include)
endif()

find_library(FOLLY_LIBRARY folly
  PATHS ${FOLLY_ROOT_DIR}/lib)
find_library(FOLLY_BENCHMARK_LIBRARY follybenchmark
  PATHS ${FOLLY_ROOT_DIR}/lib)

find_package_handle_standard_args(FOLLY DEFAULT_MSG
    FOLLY_INCLUDE_DIR FOLLY_LIBRARY)

if(FOLLY_FOUND)
    set(FOLLY_INCLUDE_DIRS ${FOLLY_INCLUDE_DIR})
    set(FOLLY_LIBRARIES
      ${FOLLY_LIBRARY}
      ${FOLLY_BENCHMARK_LIBRARY})
endif()

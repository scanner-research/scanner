# - Try to find LodePNG
#
# The following variables are optionally searched for defaults
#  LODEPNG_ROOT_DIR:       Base directory where all LodePNG components are found
#
# The following are set after configuration is done:
#  LODEPNG_FOUND
#  LODEPNG_INCLUDE_DIRS
#  LODEPNG_SOURCES

include(FindPackageHandleStandardArgs)

set(LODEPNG_ROOT_DIR "" CACHE PATH "Folder contains LodePNG")

if (NOT "$ENV{LodePNG_DIR}" STREQUAL "")
  set(LODEPNG_ROOT_DIR $ENV{LodePNG_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(LODEPNG_INCLUDE_DIR struck/tracker.h
    PATHS ${LODEPNG_ROOT_DIR}/src/windows)
else()
  find_path(LODEPNG_INCLUDE_DIR lodepng/lodepng.h
    PATHS
    ${LODEPNG_ROOT_DIR}
    ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/lodepng)
endif()

find_file(LODEPNG_SRC lodepng/lodepng.cpp
  PATHS
  ${LODEPNG_ROOT_DIR}
  ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/lodepng)

find_file(LODEPNG_UTIL_SRC lodepng/lodepng_util.cpp
  PATHS
  ${LODEPNG_ROOT_DIR}
  ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/lodepng)

find_package_handle_standard_args(LODEPNG DEFAULT_MSG
  LODEPNG_INCLUDE_DIR LODEPNG_SRC LODEPNG_UTIL_SRC)

if(LODEPNG_FOUND)
    set(LODEPNG_INCLUDE_DIRS ${LODEPNG_INCLUDE_DIR})
    set(LODEPNG_SOURCES ${LODEPNG_SRC} ${LODEPNG_UTIL_SRC})
endif()

# - Try to find googletest library
#
# The following variables are optionally searched for defaults
#  GOOGLETEST_ROOT_DIR:   Base directory where all Storage components are found
#
# The following are set after configuration is done:
#  GOOGLETEST_FOUND
#  GOOGLETEST_INCLUDE_DIRS
#  GOOGLETEST_LIBRARIES
#  GOOGLETEST_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GOOGLETEST_ROOT_DIR "" CACHE PATH "Folder contains GoogleTest")

if (NOT "$ENV{Googletest_DIR}" STREQUAL "")
  set(GOOGLETEST_ROOT_DIR $ENV{GoogleTest_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(GOOGLETEST_INCLUDE_DIR gtest/gtest.h
    PATHS ${GOOGLETEST_ROOT_DIR}/src/windows)
else()
  find_path(GOOGLETEST_INCLUDE_DIR gtest/gtest.h
    PATHS
    ${GOOGLETEST_ROOT_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/googletest/include)
endif()

find_library(GOOGLETEST_LIBRARY gtest
  PATHS
  ${GOOGLETEST_ROOT_DIR}/lib
  ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/googletest/lib)

find_package_handle_standard_args(GOOGLETEST DEFAULT_MSG
    GOOGLETEST_INCLUDE_DIR GOOGLETEST_LIBRARY)

if(GOOGLETEST_FOUND)
    set(GOOGLETEST_INCLUDE_DIRS ${GOOGLETEST_INCLUDE_DIR})
    set(GOOGLETEST_LIBRARIES ${GOOGLETEST_LIBRARY})
endif()

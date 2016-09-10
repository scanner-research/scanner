# - Try to find storehouse library
#
# The following variables are optionally searched for defaults
#  STOREHOUSE_ROOT_DIR:   Base directory where all Storage components are found
#
# The following are set after configuration is done:
#  STOREHOUSE_FOUND
#  STOREHOUSE_INCLUDE_DIRS
#  STOREHOUSE_LIBRARIES
#  STOREHOUSE_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(STOREHOUSE_ROOT_DIR "" CACHE PATH "Folder contains Storehouse")

if (NOT "$ENV{Storehouse_DIR}" STREQUAL "")
  set(STOREHOUSE_ROOT_DIR $ENV{Storehouse_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(STOREHOUSE_INCLUDE_DIR folly/FBVector.h
    PATHS ${STOREHOUSE_ROOT_DIR}/src/windows)
else()
  find_path(STOREHOUSE_INCLUDE_DIR folly/FBVector.h
    PATHS ${STOREHOUSE_ROOT_DIR}/include)
endif()

find_library(STOREHOUSE_LIBRARY folly
  PATHS ${STOREHOUSE_ROOT_DIR}/lib)

find_package_handle_standard_args(STOREHOUSE DEFAULT_MSG
    STOREHOUSE_INCLUDE_DIR STOREHOUSE_LIBRARY)

if(STOREHOUSE_FOUND)
    set(STOREHOUSE_INCLUDE_DIRS ${STOREHOUSE_INCLUDE_DIR})
    set(STOREHOUSE_LIBRARIES
      ${STOREHOUSE_LIBRARY})
endif()

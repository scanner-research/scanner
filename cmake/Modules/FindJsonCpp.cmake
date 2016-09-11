# - Try to find jsoncpp library
#
# The following variables are optionally searched for defaults
#  JSONCPP_ROOT_DIR:   Base directory where all Storage components are found
#
# The following are set after configuration is done:
#  JSONCPP_FOUND
#  JSONCPP_INCLUDE_DIRS
#  JSONCPP_LIBRARIES
#  JSONCPP_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(JSONCPP_ROOT_DIR "" CACHE PATH "Folder contains JsonCpp")

if (NOT "$ENV{JsonCpp_DIR}" STREQUAL "")
  set(JSONCPP_ROOT_DIR $ENV{JsonCpp_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(JSONCPP_INCLUDE_DIR json/json.h
    PATHS ${JSONCPP_ROOT_DIR}/src/windows)
else()
  find_path(JSONCPP_INCLUDE_DIR json/json.h
    PATHS
    ${JSONCPP_ROOT_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/jsoncpp/include)
endif()

find_library(JSONCPP_LIBRARY jsoncpp
  PATHS
  ${JSONCPP_ROOT_DIR}/lib
  ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/jsoncpp/lib)

find_package_handle_standard_args(JSONCPP DEFAULT_MSG
    JSONCPP_INCLUDE_DIR JSONCPP_LIBRARY)

if(JSONCPP_FOUND)
    set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIR})
    set(JSONCPP_LIBRARIES ${JSONCPP_LIBRARY})
endif()

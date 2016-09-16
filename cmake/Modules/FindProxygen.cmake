# - Try to find Proxygen
#
# The following variables are optionally searched for defaults
#  PROXYGEN_ROOT_DIR:   Base directory where all proxygen components are found
#
# The following are set after configuration is done:
#  PROXYGEN_FOUND
#  PROXYGEN_INCLUDE_DIRS
#  PROXYGEN_LIBRARIES
#  PROXYGEN_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(PROXYGEN_ROOT_DIR "" CACHE PATH "Folder contains Proxygen")

if (NOT "$ENV{Proxygen_DIR}" STREQUAL "")
  set(PROXYGEN_ROOT_DIR $ENV{proxygen_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(PROXYGEN_INCLUDE_DIR proxygen/lib/http/HTTPHeaders.h
    PATHS ${PROXYGEN_ROOT_DIR}/src/windows)
else()
  find_path(PROXYGEN_INCLUDE_DIR proxygen/lib/http/HTTPHeaders.h
    PATHS ${PROXYGEN_ROOT_DIR}/include)
endif()

find_library(PROXYGEN_LIBRARY proxygenlib
  PATHS ${PROXYGEN_ROOT_DIR}/lib)
find_library(PROXYGEN_HTTP_SERVER_LIBRARY proxygenhttpserver
  PATHS ${PROXYGEN_ROOT_DIR}/lib)
find_library(PROXYGEN_CURL_LIBRARY proxygencurl
  PATHS ${PROXYGEN_ROOT_DIR}/lib)

find_package_handle_standard_args(PROXYGEN DEFAULT_MSG
    PROXYGEN_INCLUDE_DIR PROXYGEN_LIBRARY)

if(PROXYGEN_FOUND)
    set(PROXYGEN_INCLUDE_DIRS ${PROXYGEN_INCLUDE_DIR})
    set(PROXYGEN_LIBRARIES
      ${PROXYGEN_LIBRARY}
      ${PROXYGEN_HTTP_SERVER_LIBRARY})
endif()

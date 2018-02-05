# - Try to find NVCUVID
#
# The following variables are optionally searched for defaults
#  NVCUVID_DIR:       Base directory where NVCUVID can be found
#
# The following are set after configuration is done:
#  NVCUVID_FOUND
#  NVCUVID_LIBRARY

include(FindPackageHandleStandardArgs)

set(NVCUVID_ROOT_DIR "" CACHE PATH "Folder contains NVCUVID")

if (NOT "$ENV{NVCUVID_DIR}" STREQUAL "")
  set(NVCUVID_ROOT_DIR
    $ENV{NVCUVID_DIR} CACHE PATH "Folder contains NVCUVID" FORCE)
endif()

find_library(NVCUVID_LIBRARY nvcuvid
  PATHS
  ${NVCUVID_ROOT_DIR}/lib
  /usr/local/cuda/lib64)

find_package_handle_standard_args(NVCUVID DEFAULT_MSG NVCUVID_LIBRARY)

if(NVCUVID_FOUND)
  set(NVCUVID_LIBRARIES ${NVCUVID_LIBRARY})
endif()

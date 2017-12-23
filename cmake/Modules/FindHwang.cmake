# FindHwang.cmake

set(HWANG_ROOT_DIR "" CACHE PATH "Folder contains Hwang")

if (NOT "$ENV{Hwang_DIR}" STREQUAL "")
  set(HWANG_ROOT_DIR $ENV{Hwang_DIR} CACHE PATH "Folder contains Hwang"
    FORCE)
elseif (Hwang_DIR)
  set(HWANG_ROOT_DIR ${Hwang_DIR} CACHE PATH "Folder contains Hwang"
    FORCE)
endif()

find_library(HWANG_LIBRARIES
  NAMES hwang
  HINTS ${HWANG_ROOT_DIR}/lib
  )

find_path(HWANG_INCLUDE_DIR
  NAMES hwang/common.h 
  HINTS ${HWANG_ROOT_DIR}/include
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hwang DEFAULT_MSG
  HWANG_LIBRARIES
  HWANG_INCLUDE_DIR
  )

set(HWANG_LIBRARY ${HWANG_LIBRARIES})
set(HWANG_INCLUDE_DIRS ${HWANG_INCLUDE_DIR})

mark_as_advanced(
  HWANG_ROOT_DIR
  HWANG_LIBRARY
  HWANG_LIBRARIES
  HWANG_INCLUDE_DIR
  HWANG_INCLUDE_DIRS
  )

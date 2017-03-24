# - Try to find Gipuma
#
# The following variables are optionally searched for defaults
#  GIPUMA_ROOT_DIR:       Base directory where all Gipuma components are found
#
# The following are set after configuration is done:
#  GIPUMA_FOUND
#  GIPUMA_INCLUDE_DIRS

include(FindPackageHandleStandardArgs)

set(GIPUMA_ROOT_DIR "" CACHE PATH "Folder contains Gipuma")

if (NOT "$ENV{Gipuma_DIR}" STREQUAL "")
  set(GIPUMA_ROOT_DIR $ENV{Gipuma_DIR} CACHE PATH "Folder contains Gipuma" FORCE)
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(GIPUMA_INCLUDE_DIR gipuma.h
    PATHS ${GIPUMA_ROOT_DIR}/src/windows
    PATH_SUFFIXES gipuma)
else()
  find_path(GIPUMA_INCLUDE_DIR gipuma.h
    PATHS ${GIPUMA_ROOT_DIR}/include
    PATH_SUFFIXES gipuma)

endif()

find_library(GIPUMA_LIBRARY gipuma PATHS ${GIPUMA_ROOT_DIR}/lib)

find_package_handle_standard_args(GIPUMA DEFAULT_MSG GIPUMA_INCLUDE_DIR
  GIPUMA_LIBRARY)

if(GIPUMA_FOUND)
  set(GIPUMA_INCLUDE_DIRS ${GIPUMA_INCLUDE_DIR})
  set(GIPUMA_LIBRARIES ${GIPUMA_LIBRARY})
endif()

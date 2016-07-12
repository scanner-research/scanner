# - Try to find VDPAU
#
# The following variables are optionally searched for defaults
#  VDPAU_ROOT_DIR:       Base directory where all VDPAU components are found
#
# The following are set after configuration is done:
#  VDPAU_FOUND
#  VDPAU_INCLUDE_DIRS
#  VDPAU_LIBRARIES
#  VDPAU_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(VDPAU_ROOT_DIR "" CACHE PATH "Folder contains VDPAU")

if (NOT "$ENV{VDPAU_DIR}" STREQUAL "")
  set(VDPAU_ROOT_DIR $ENV{VDPAU_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(VDPAU_INCLUDE_DIR vdpau/vdpau.h
    PATHS ${VDPAU_ROOT_DIR}/src/windows)
else()
  find_path(VDPAU_INCLUDE_DIR vdpau/vdpau.h
    PATHS ${VDPAU_ROOT_DIR}/include)
endif()

find_library(VDPAU_LIBRARY vdpau PATHS ${VDPAU_ROOT_DIR}/lib)

find_package_handle_standard_args(VDPAU DEFAULT_MSG
    VDPAU_INCLUDE_DIR VDPAU_LIBRARY)

if(VDPAU_FOUND)
    set(VDPAU_INCLUDE_DIRS ${VDPAU_INCLUDE_DIR})
    set(VDPAU_LIBRARIES ${VDPAU_LIBRARY})
endif()

# - Try to find grpc library
#
# The following variables are optionally searched for defaults
#  GRPC_DIR:   Base directory where all components are found
#
# The following are set after configuration is done:
#  GRPC_FOUND
#  GRPC_INCLUDE_DIRS
#  GRPC_LIBRARIES
#  GRPC_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GRPC_ROOT_DIR "" CACHE PATH "Folder contains GRPC")

if (NOT "$ENV{GRPC_DIR}" STREQUAL "")
  set(GRPC_DIR $ENV{GRPC_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(GRPC_INCLUDE_DIR grpc/grpc.h
    PATHS ${GRPC_ROOT_DIR}/src/windows)
else()
  find_path(GRPC_INCLUDE_DIR grpc/grpc.h
    PATHS
    ${GRPC_DIR}/include)
endif()

find_library(GRPCPP_UNSECURE_LIBRARY grpc++_unsecure
  PATHS
  ${GRPC_DIR}/lib)

find_library(GRPC_LIBRARY grpc
  PATHS
  ${GRPC_DIR}/lib)

find_library(GPR_LIBRARY gpr
  PATHS
  ${GRPC_DIR}/lib)

find_package_handle_standard_args(GRPC DEFAULT_MSG
    GRPC_INCLUDE_DIR GRPC_LIBRARY)

if(GRPC_FOUND)
    set(GRPC_INCLUDE_DIRS ${GRPC_INCLUDE_DIR})
    set(GRPC_LIBRARIES ${GRPCP_UNSECURE_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
endif()

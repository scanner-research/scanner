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

# Get GRPC version info
set(GRPC_VERSION_PROG
"#include <grpcpp/grpcpp.h>

int main(void) {
  std::cout << grpc::Version();
}")
execute_process(COMMAND "echo" "${GRPC_VERSION_PROG}" OUTPUT_FILE "/tmp/test.cpp")
set(EX ${CMAKE_CXX_FLAGS})
separate_arguments(EX)
if(APPLE)
  set(EX ${EX} "-isysroot" ${CMAKE_OSX_SYSROOT})
endif()
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "/tmp/test.cpp"
  "-I${GRPC_INCLUDE_DIR}" "${GRPCPP_UNSECURE_LIBRARY}" ${EX} "-o" "/tmp/test.out")
execute_process(COMMAND "/tmp/test.out" OUTPUT_VARIABLE GRPC_VERSION)

if(GRPC_FOUND)
    set(GRPC_INCLUDE_DIRS ${GRPC_INCLUDE_DIR})
    set(GRPC_LIBRARIES ${GRPCPP_UNSECURE_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
endif()

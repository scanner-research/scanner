# Op.cmake should be included by a CMake script that will build a custom Scanner
# op. It sets a few default flags and exposes a function build_op for simplifying
# the build process. See examples/tutorial/04_custom_op.py for an example usage.

if(NOT SCANNER_PATH)
  message(FATAL_ERROR "Set SCANNER_PATH to the Scanner repo directory before including Op.cmake.")
endif()
list(APPEND CMAKE_MODULE_PATH "${SCANNER_PATH}/cmake/Modules/")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
if(NOT COMPILER_SUPPORTS_CXX1Y)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++1y support.")
endif()

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

function(build_op)
  set(options )
  set(oneValueArgs LIB_NAME PROTO_SRC NO_FLAGS)
  set(multiValueArgs CPP_SRCS)
  cmake_parse_arguments(args "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include_directories("${CMAKE_CURRENT_BINARY_DIR}")

  if(NOT("${args_PROTO_SRC}" STREQUAL ""))
    find_package(SaneProtobuf REQUIRED)
    set(PROTOBUF_IMPORT_DIRS "${SCANNER_PATH}")
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS OFF ${args_PROTO_SRC})
    protobuf_generate_python(PROTO_PY OFF ${args_PROTO_SRC})
    add_custom_target(${args_LIB_NAME}_proto_files DEPENDS ${PROTO_HDRS} ${PROTO_PY})
    add_library(${args_LIB_NAME} SHARED ${args_CPP_SRCS} ${PROTO_SRCS})
    add_dependencies(${args_LIB_NAME} ${args_LIB_NAME}_proto_files)
    target_link_libraries(${args_LIB_NAME} PUBLIC "${PROTOBUF_LIBRARY}")
  else()
    add_library(${args_LIB_NAME} SHARED ${args_CPP_SRCS})
  endif()

  target_link_libraries(${args_LIB_NAME} PUBLIC "${SCANNER_PATH}/build/libscanner.so")

  if("${args_NO_FLAGS}" STREQUAL "")
    execute_process(
      OUTPUT_VARIABLE BUILD_FLAGS
      COMMAND python -c "import scannerpy.stdlib.build_flags")
    set_target_properties(
      ${args_LIB_NAME} PROPERTIES
      COMPILE_FLAGS "${BUILD_FLAGS}")
  endif()
endfunction()

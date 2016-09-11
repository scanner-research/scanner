# - Try to find tinytoml library
#
# The following variables are optionally searched for defaults
#  TINYTOML_ROOT_DIR:   Base directory where all Storage components are found
#
# The following are set after configuration is done:
#  TINYTOML_FOUND
#  TINYTOML_INCLUDE_DIRS

include(FindPackageHandleStandardArgs)

set(TINYTOML_ROOT_DIR "" CACHE PATH "Folder contains TinyToml")

if (NOT "$ENV{TinyToml_DIR}" STREQUAL "")
  set(TINYTOML_ROOT_DIR $ENV{TinyToml_DIR})
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(TINYTOML_INCLUDE_DIR toml/toml.h
    PATHS ${TINYTOML_ROOT_DIR}/src/windows)
else()
  find_path(TINYTOML_INCLUDE_DIR toml/toml.h
    PATHS
    ${TINYTOML_ROOT_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/build/bin/tinytoml/include)
endif()

find_package_handle_standard_args(TINYTOML DEFAULT_MSG
    TINYTOML_INCLUDE_DIR)

if(TINYTOML_FOUND)
    set(TINYTOML_INCLUDE_DIRS ${TINYTOML_INCLUDE_DIR})
endif()

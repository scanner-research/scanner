# - Try to find Eigen
#
# The following variables are optionally searched for defaults
#  EIGEN_ROOT_DIR:       Base directory where all Eigen components are found
#
# The following are set after configuration is done:
#  EIGEN_FOUND
#  EIGEN_INCLUDE_DIRS

include(FindPackageHandleStandardArgs)

set(EIGEN_ROOT_DIR "" CACHE PATH "Folder contains Eigen")

if (NOT "$ENV{Eigen_DIR}" STREQUAL "")
  set(EIGEN_ROOT_DIR $ENV{Eigen_DIR} CACHE PATH "Folder contains Eigen" FORCE)
endif()

# We are testing only a couple of files in the include directories
if(WIN32)
  find_path(EIGEN_INCLUDE_DIR Eigen/Core
    PATHS ${EIGEN_ROOT_DIR}/src/windows
    PATH_SUFFIXES eigen3)
else()
  find_path(EIGEN_INCLUDE_DIR Eigen/Core
    PATHS ${EIGEN_ROOT_DIR}/include
    PATH_SUFFIXES eigen3)

endif()

find_package_handle_standard_args(EIGEN DEFAULT_MSG EIGEN_INCLUDE_DIR)

if(EIGEN_FOUND)
    set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
endif()

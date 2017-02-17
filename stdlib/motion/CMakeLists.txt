# TODO(wcrichto): fill this out once stenciling is implemented

if (BUILD_TRACKER_OPS)
  find_package(Eigen REQUIRED)
  find_package(Struck REQUIRED CONFIG
    PATHS "thirdparty/build/bin/struck")
  include_directories(
    "${EIGEN_INCLUDE_DIRS}"
    "${STRUCK_INCLUDE_DIRS}")
endif()

add_library(motion OBJECT)

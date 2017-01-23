set(SOURCE_FILES
  blur_kernel.cpp)

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  list(APPEND SOURCE_FILES
    histogram_kernel.cpp)
endif()


add_library(kernels OBJECT
  ${SOURCE_FILES})

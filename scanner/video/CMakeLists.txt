set(SOURCE_FILES
  video_decoder.cpp)

if (HAVE_CUDA)
  list(APPEND SOURCE_FILES
    nvidia/nvidia_video_decoder.cpp)
endif()

# list(APPEND SOURCE_FILES
#   intel/intel_video_decoder.cpp)
list(APPEND SOURCE_FILES
  software/software_video_decoder.cpp)

add_library(video OBJECT
  ${SOURCE_FILES})

set(SOURCE_FILES
  video_decoder.cpp)

if (CUDA_FOUND)
  add_definitions(-DHAVE_NVIDIA_VIDEO_HARDWARE)
  list(APPEND SOURCE_FILES
    nvidia/nvidia_video_decoder.cpp)
endif()

if (MFX_FOUND)
  add_definitions(-DHAVE_INTEL_VIDEO_HARDWARE)
  list(APPEND SOURCE_FILES
    intel/intel_video_decoder.cpp)
endif()

list(APPEND SOURCE_FILES
  software/software_video_decoder.cpp)

add_library(video OBJECT
  ${SOURCE_FILES})

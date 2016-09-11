add_subdirectory(nvidia)
add_subdirectory(intel)
add_subdirectory(software)

add_library(video OBJECT
  video_decoder.cpp
  video_separator.cpp
  $<TARGET_OBJECTS:video_nvidia>
  $<TARGET_OBJECTS:video_intel>
  $<TARGET_OBJECTS:video_software>)

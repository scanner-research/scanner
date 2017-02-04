set(SOURCE_FILES
  decoder_automata.cpp
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

add_executable(DecoderAutomataTest decoder_automata_test.cpp
  $<TARGET_OBJECTS:video>
  $<TARGET_OBJECTS:util>
  $<TARGET_OBJECTS:engine>
  ${PROTO_SRCS}
  ${GRPC_PROTO_SRCS})
target_link_libraries(DecoderAutomataTest ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN}
  ${SCANNER_LIBRARIES})
add_test(DecoderAutomataTest DecoderAutomataTest)

set(SOURCE_FILES
  decoder_automata.cpp
  video_decoder.cpp
  video_encoder.cpp)

if (BUILD_CUDA)
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
  software/software_video_decoder.cpp
  software/software_video_encoder.cpp)

add_library(video OBJECT
  ${SOURCE_FILES})

set_source_files_properties(${PROTO_SRCS} ${GRPC_PROTO_SRCS} PROPERTIES
  GENERATED TRUE)

add_executable(DecoderAutomataTest decoder_automata_test.cpp)
target_link_libraries(DecoderAutomataTest
  ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN}
  scanner)
add_test(DecoderAutomataTest DecoderAutomataTest)

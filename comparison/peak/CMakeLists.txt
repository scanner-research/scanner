set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

if(CUDA_FOUND AND OpenCV_FOUND)
  add_executable(peak_comparison
    main.cpp
    peak_video_decoder.cpp
    ../../../scanner/evaluators/caffe/net_descriptor.cpp
    ${PROTO_SRCS}
    $<TARGET_OBJECTS:util>)

  target_link_libraries(peak_comparison
    "${LIBRARIES}"
    "${PUBLIC_LIBRARIES}"
    "${SCANNER_LIBRARIES}"
    "${PLATFORM_LINK_FLAGS}")
endif()

set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

if(CUDA_FOUND AND OpenCV_FOUND)
  add_executable(standalone_comparison
    main.cpp
    ../../../scanner/evaluators/caffe/net_descriptor.cpp
    ${PROTO_SRCS}
    $<TARGET_OBJECTS:util>)

  target_link_libraries(standalone_comparison
    "${LIBRARIES}"
    "${PUBLIC_LIBRARIES}"
    "${SCANNER_LIBRARIES}"
    "${PLATFORM_LINK_FLAGS}")
endif()

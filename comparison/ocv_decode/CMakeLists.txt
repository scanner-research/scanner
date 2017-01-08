set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

if(CUDA_FOUND AND OpenCV_FOUND)
  add_executable(ocv_decode
    main.cpp
    ${PROTO_SRCS}
    $<TARGET_OBJECTS:util>)

  target_link_libraries(ocv_decode
    "${LIBRARIES}"
    "${PUBLIC_LIBRARIES}"
    "${SCANNER_LIBRARIES}"
    "${PLATFORM_LINK_FLAGS}")
endif()

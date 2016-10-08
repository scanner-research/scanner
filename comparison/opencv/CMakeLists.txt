set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

if(HAVE_CUDA AND HAVE_OPENCV) 
  add_executable(opencv_compare
    main.cpp
    ${PROTO_SRCS}
    $<TARGET_OBJECTS:util>)

  target_link_libraries(opencv_compare
    "${LIBRARIES}"
    "${PUBLIC_LIBRARIES}"
    "${SCANNER_LIBRARIES}"
    "${PLATFORM_LINK_FLAGS}")
endif()

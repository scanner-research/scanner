set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

add_executable(readwrite_test
  main.cpp
  ${PROTO_SRCS}
  $<TARGET_OBJECTS:util>)

target_link_libraries(readwrite_test
  "${LIBRARIES}"
  "${PUBLIC_LIBRARIES}"
  "${SCANNER_LIBRARIES}"
  "${PLATFORM_LINK_FLAGS}")

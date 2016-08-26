set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

add_executable(opencv_compare
  main.cpp
  $<TARGET_OBJECTS:util>)

target_link_libraries(opencv_compare
  "${LIBRARIES}"
  "${PUBLIC_LIBRARIES}"
  "${LIGHTSCAN_LIBRARIES}"
  "${PLATFORM_LINK_FLAGS}")

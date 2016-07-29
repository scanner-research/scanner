set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

add_executable(opencv_compare
  main.cpp
  $<TARGET_OBJECTS:storage>)

target_link_libraries(opencv_compare
  "${LIBRARIES}"
  "${PLATFORM_LINK_FLAGS}")

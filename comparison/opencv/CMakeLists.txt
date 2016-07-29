set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

add_executable(opencv_compare
  main.cpp
  $<TARGET_OBJECTS:storage>
  ${LIBRARIES})

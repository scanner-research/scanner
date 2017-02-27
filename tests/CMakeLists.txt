add_executable(CppTests cpp_test.cpp)
target_link_libraries(CppTests ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN} scanner stdlib)
add_test(CppTests CppTests)

if(NOT BUILD_CUDA)
  set(MARKERS "-m 'not gpu'")
endif()
add_test(
  NAME PythonTests
  COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR} -xv ${MARKERS})

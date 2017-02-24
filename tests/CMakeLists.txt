add_executable(EndToEndTests cpp_test.cpp)
target_link_libraries(EndToEndTests ${GTEST_LIBRARIES} ${GTEST_LIB_MAIN} scanner stdlib)
add_test(EndToEndTests EndToEndTests)

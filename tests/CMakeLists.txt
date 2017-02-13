add_executable(tests_decode decode.cpp)
target_link_libraries(tests_decode scanner stdlib)

add_test(TestsDecode tests_decode)

add_executable(tests_decode decode.cpp)
target_link_libraries(tests_decode scanner kernels)

add_test(TestsDecode tests_decode)

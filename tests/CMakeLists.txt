add_executable(tests_simple simple.cpp)
target_link_libraries(tests_simple scanner)

add_test(TestsSimple tests_simple)

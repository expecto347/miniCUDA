find_package(GTest REQUIRED)

add_executable(test_vector_add test_vector_add.cu)

target_link_libraries(test_vector_add PRIVATE
    ck
    GTest::gtest
    GTest::gtest_main
)

# 每个 TEST 注册成独立的 ctest 用例
include(GoogleTest)
gtest_discover_tests(test_vector_add)
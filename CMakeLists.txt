cmake_minimum_required(VERSION 3.18)

# ---- 目标 GPU 架构 ----
# 必须在 project() 之前设置：CUDA 编译器探测阶段会把 CMAKE_CUDA_ARCHITECTURES
# 初始化为默认值（CMake 3.22 + CUDA 12.8 下是 52），之后再设就不生效了。
# 默认 sm_89（本机 L20）。换卡时覆盖，例如：
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
# CMake >= 3.24 可直接用 -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
endif()

project(miniCUDA
    VERSION 0.1.0
    DESCRIPTION "A minimal CUDA kernel library"
    LANGUAGES CXX CUDA
)

# ---- 默认构建类型 ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- C++ / CUDA 标准 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- 核心库 ck ----
add_library(ck STATIC
    src/elementwise/add.cu
)

# include/ 作为公共头文件目录，下游 target_link_libraries(ck) 即可使用
# #include "ck/elementwise/add.cuh"
target_include_directories(ck PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 警告只加给 host 编译器，避免 nvcc 转发时的噪声
target_compile_options(ck PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# ---- 可选子目录 ----
# tests/ 与 benchmarks/ 放入各自的 CMakeLists.txt 后会自动纳入构建
enable_testing()

foreach(dir tests benchmarks)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
        add_subdirectory(${dir})
    endif()
endforeach()
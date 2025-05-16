cmake_minimum_required(VERSION 3.18)
project(square_add)

find_package(Torch REQUIRED)

add_library(square_add SHARED
    square_add/square_add_cpu.cpp
    square_add/square_add_cuda.cu
    square_add/bindings.cpp)

target_link_libraries(square_add "${TORCH_LIBRARIES}")
set_property(TARGET square_add PROPERTY CXX_STANDARD 14)
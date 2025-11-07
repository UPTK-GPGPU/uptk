cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(MyDemo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 75)  

add_executable(MyDemo
    MyDemo.cu
)

target_link_libraries(MyDemo cudart UPTKrt)
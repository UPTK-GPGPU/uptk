cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(HIP REQUIRED)

enable_language(HIP)

set(CMAKE_HIP_ARCHITECTURES "gfx906" CACHE STRING "AMD GPU architecture")

add_executable(demo 
    demo.hip
)

target_link_libraries(demo hip::host UPTKrt)

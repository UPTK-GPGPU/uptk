cmake_minimum_required(VERSION 3.17)
project(UPTK)
find_package(CUDAToolkit REQUIRED) 
enable_language(CUDA)
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CUDA_STANDARD 17 )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

set(CMAKE_C_COMPILER nvcc)
set(CMAKE_CXX_COMPILER nvcc)

add_compile_options(-xcu)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_BUILD_TYPE "Release")
    set(UPTK_ADD_DEBUGINFO "yes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_POSITION_INDEPENDENT_CODE ON)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ROCM_PATH}/include
    )
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
add_subdirectory(src)
install(FILES ${CMAKE_SOURCE_DIR}/env.sh
        DESTINATION .
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_EXECUTE)

option(UPTK_ENABLE_TESTS "Enable building and running UPTK test suite" OFF)
if(UPTK_ENABLE_TESTS)
    message(STATUS "UPTK tests enabled. Build with 'make' then run 'ctest' in the build directory.")
    add_subdirectory(tests)
endif()

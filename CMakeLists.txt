cmake_minimum_required(VERSION 3.17)
project(UPTK)
# Find 现代 CUDAToolkit 模块
find_package(CUDAToolkit REQUIRED) 
# Enable CUDA language support
enable_language(CUDA)
# #set(ROCM_PATH $ENV{ROCM_PATH})
# if (NOT DEFINED ROCM_PATH)
#     set(ROCM_PATH $ENV{ROCM_PATH})
# endif ()
# SET(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
# common configure
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CUDA_STANDARD 17 )
# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
# judge debug or release
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_BUILD_TYPE "Release")
    set(UPTK_ADD_DEBUGINFO "yes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()
# Set CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# Set CUDA position independent code
set(CMAKE_CUDA_POSITION_INDEPENDENT_CODE ON)
# Include UPTK and CUDA header files
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ROCM_PATH}/include
    )
# Link CUDA libraries
# link_directories(${CUDA_LIBRARY_DIRS})
# Add instructions for installing header files
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
# src
add_subdirectory(src)
install(FILES ${CMAKE_SOURCE_DIR}/env.sh
        DESTINATION .
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_EXECUTE)

cmake_minimum_required(VERSION 3.10)

project(UPTK)

#set(ROCM_PATH $ENV{ROCM_PATH})
if (NOT DEFINED ROCM_PATH)
    set(ROCM_PATH $ENV{ROCM_PATH})
endif ()

SET(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")

# common configure
set( CMAKE_CXX_STANDARD 17 )

list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH}/hip ${ROCM_PATH} )
list( APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake )
find_package(hip)

# judge debug or release
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_BUILD_TYPE "Release")
    set(UPTK_ADD_DEBUGINFO "yes")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-g")
endif()

#include UPTK and hip header files
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    )


# build all target
SET(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")


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













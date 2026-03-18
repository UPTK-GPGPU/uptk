# build target libcublas.so
add_library(UPTKrand SHARED rand_type_convert.cpp rand_fun_convert.cpp)
target_link_libraries(UPTKrand PRIVATE curand)


# add library version info
set ( CURAND_LIB_VERSION_STRING "1.0.0.0" )
set_property ( TARGET UPTKrand PROPERTY VERSION "${CURAND_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKrand PROPERTY SOVERSION "1" )


# filter exported symbols
set ( CURAND_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${CUDA_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(CURAND_SHARED_LINK_FLAGS "${CURAND_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKrand.so.def")
  
else()
    set(CURAND_SHARED_LINK_FLAGS "${CURAND_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKrand.so.def")
endif()
endif()

set_property ( TARGET UPTKrand PROPERTY LINK_FLAGS ${CURAND_SHARED_LINK_FLAGS})


# install target and needed file 
install(TARGETS UPTKrand
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
# build target libUPTKnccl.so
add_library(UPTKnccl SHARED nccl_type_convert.cpp nccl_fun_convert.cpp)
target_link_libraries(UPTKnccl PRIVATE nccl)


# add library version info
set ( UPTKNCCL_LIB_VERSION_STRING "1.0.0.0" )
set_property ( TARGET UPTKnccl PROPERTY VERSION "${UPTKNCCL_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKnccl PROPERTY SOVERSION "1" )


# filter exported symbols
set ( UPTKNCCL_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${CUDA_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(UPTKNCCL_SHARED_LINK_FLAGS "${UPTKNCCL_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKnccl.so.def")
  
else()
    set(UPTKNCCL_SHARED_LINK_FLAGS "${UPTKNCCL_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKnccl.so.def")
endif()
endif()

set_property ( TARGET UPTKnccl PROPERTY LINK_FLAGS ${UPTKNCCL_SHARED_LINK_FLAGS})


# install target and needed file 
install(TARGETS UPTKnccl
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
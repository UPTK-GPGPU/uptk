# build target libUPTKblas.so
add_library(UPTKblas SHARED blas_type_convert.cpp blas_fun_convert.cpp blasXt_fun_convert.cpp)
target_link_libraries(UPTKblas PRIVATE cublas)


# add library version info
set ( UPTKBLAS_LIB_VERSION_STRING "1.0.0.0" )
set_property ( TARGET UPTKblas PROPERTY VERSION "${UPTKBLAS_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKblas PROPERTY SOVERSION "1" )


# filter exported symbols
set ( UPTKBLAS_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${CUDA_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(UPTKBLAS_SHARED_LINK_FLAGS "${UPTKBLAS_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKblas.so.def")
  
else()
    set(UPTKBLAS_SHARED_LINK_FLAGS "${UPTKBLAS_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKblas.so.def")
endif()
endif()

set_property ( TARGET UPTKblas PROPERTY LINK_FLAGS ${UPTKBLAS_SHARED_LINK_FLAGS})


# install target and needed file 
install(TARGETS UPTKblas
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
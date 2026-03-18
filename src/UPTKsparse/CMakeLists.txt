# build target libUPTKsparse.so
add_library(UPTKsparse SHARED sparse_type_convert.cpp sparse_fun_convert.cpp)
target_link_libraries(UPTKsparse PRIVATE cusparse)


# add library version info
set ( CUSPARSE_LIB_VERSION_STRING "1.0.0.0" )
set_property ( TARGET UPTKsparse PROPERTY VERSION "${CUSPARSE_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKsparse PROPERTY SOVERSION "1" )


# filter exported symbols
set ( CUSPARSE_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${CUDA_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(CUSPARSE_SHARED_LINK_FLAGS "${CUSPARSE_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKsparse.so.def")
  
else()
    set(CUSPARSE_SHARED_LINK_FLAGS "${CUSPARSE_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKsparse.so.def")
endif()
endif()

set_property ( TARGET UPTKsparse PROPERTY LINK_FLAGS ${CUSPARSE_SHARED_LINK_FLAGS})


# install target and needed file 
install(TARGETS UPTKsparse
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
# build target libUPTKrt.so
add_library(UPTKrt SHARED runtime_type_convert.cpp runtime_fun_convert.cpp)


# add library version info
set ( UPTKRT_LIB_VERSION_STRING "1.0.0" )
set_property ( TARGET UPTKrt PROPERTY VERSION "${UPTKRT_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKrt PROPERTY SOVERSION "1" )


# filter exported symbols
set ( UPTKRT_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${UPTK_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(UPTKRT_SHARED_LINK_FLAGS "${UPTKRT_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKrt.so.def")
  
else()
    set(UPTKRT_SHARED_LINK_FLAGS "${UPTKRT_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKrt.so.def")
endif()
endif()

set_property ( TARGET UPTKrt PROPERTY LINK_FLAGS ${UPTKRT_SHARED_LINK_FLAGS})



# install target and needed file 
install(TARGETS UPTKrt
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

if(EXISTS GALAXY_STATIC_LIBRARY)
    install(FILES ${CMAKE_BINARY_DIR}/src/lib${UPTKRT_STATIC}.a DESTINATION ${CMAKE_INSTALL_PREFIX}/UPTK/lib)
endif()

# build target libcuda.so
add_library(UPTKrtc SHARED rtc_type_convert.cpp rtc_fun_convert.cpp)
target_link_libraries(UPTKrtc PRIVATE nvrtc)


# add library version info
set ( UPTKRTC_LIB_VERSION_STRING "11.8.89" )
set_property ( TARGET UPTKrtc PROPERTY VERSION "${UPTKRTC_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKrtc PROPERTY SOVERSION "11.2" )


# filter exported symbols
set ( UPTKRTC_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${CUDA_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(UPTKRTC_SHARED_LINK_FLAGS "${UPTKRTC_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKrtc.so.def")
  
else()
    set(UPTKRTC_SHARED_LINK_FLAGS "${UPTKRTC_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKrtc.so.def")
endif()
endif()

set_property ( TARGET UPTKrtc PROPERTY LINK_FLAGS ${UPTKRTC_SHARED_LINK_FLAGS})


# install target and needed file 
#install(TARGETS UPTKrtc LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/cuda/targets/x86_64-linux/lib)
install(TARGETS UPTKrtc
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

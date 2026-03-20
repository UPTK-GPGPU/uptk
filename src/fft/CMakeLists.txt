# build target libUPTKfft.so
add_library(UPTKfft SHARED fft_type_convert.cpp fft_fun_convert.cpp)
target_link_libraries(UPTKfft PRIVATE cufft)


# add library version info
set ( UPTKFFT_LIB_VERSION_STRING "1.0.0.0" )
set_property ( TARGET UPTKfft PROPERTY VERSION "${UPTKFFT_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKfft PROPERTY SOVERSION "1" )


# filter exported symbols
set ( UPTKFFT_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )

if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${CUDA_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(UPTKFFT_SHARED_LINK_FLAGS "${UPTKFFT_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKfft.so.def")
  
else()
    set(UPTKFFT_SHARED_LINK_FLAGS "${UPTKFFT_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKfft.so.def")
endif()
endif()

set_property ( TARGET UPTKfft PROPERTY LINK_FLAGS ${UPTKFFT_SHARED_LINK_FLAGS})


# install target and needed file 
install(TARGETS UPTKfft
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
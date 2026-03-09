set_source_files_properties(
    driver_type_convert.cpp
    driver_fun_convert.cpp
    PROPERTIES 
    LANGUAGE CUDA
)
# build target libUPTKdrt.so
add_library(UPTKdrt SHARED driver_type_convert.cpp driver_fun_convert.cpp)
set_target_properties(UPTKdrt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_POSITION_INDEPENDENT_CODE ON
)
# Link CUDA libraries using modern CMake targets
target_link_libraries(UPTKdrt 
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)
# add library version info
set ( UPTKDRT_LIB_VERSION_STRING "1.0.0" )
set_property ( TARGET UPTKdrt PROPERTY VERSION "${UPTKDRT_LIB_VERSION_STRING}" )
set_property ( TARGET UPTKdrt PROPERTY SOVERSION "1" )
# filter exported symbols
set ( UPTKDRT_SHARED_LINK_FLAGS "-Wl,-z,noexecstack -Wl,--enable-new-dtags" )
if(CMAKE_BUILD_TYPE MATCHES Release)
if ( NOT "${UPTKD_ADD_DEBUGINFO}" STREQUAL "yes" )
    set(UPTKDRT_SHARED_LINK_FLAGS "${UPTKDRT_SHARED_LINK_FLAGS} -s -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKdrt.so.def")
else()
    set(UPTKDRT_SHARED_LINK_FLAGS "${UPTKDRT_SHARED_LINK_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/packaging/libUPTKdrt.so.def")
endif()
endif()
set_property ( TARGET UPTKdrt PROPERTY LINK_FLAGS ${UPTKDRT_SHARED_LINK_FLAGS})
# install target and needed file 
install(TARGETS UPTKdrt
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
if(EXISTS GALAXY_STATIC_LIBRARY)
    install(FILES ${CMAKE_BINARY_DIR}/src/lib${UPTKDRT_STATIC}.a DESTINATION ${CMAKE_INSTALL_PREFIX}/UPTK/lib)
endif()

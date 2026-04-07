# UPTK Test Suite
# Controlled by UPTK_ENABLE_TESTS option in root CMakeLists.txt

enable_testing()

# Common include directories for all test targets
set(TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Helper function to add a test target
function(uptk_add_test TEST_NAME TEST_SOURCE LINK_LIBS)
    add_executable(${TEST_NAME} ${TEST_SOURCE})

    set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)

    target_include_directories(${TEST_NAME} PRIVATE
        ${TEST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${TEST_NAME} PRIVATE ${LINK_LIBS})

    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_POSITION_INDEPENDENT_CODE ON
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 120
    )
endfunction()

# ---- Test targets ----

# Runtime tests: link against UPTKrt and CUDA runtime
uptk_add_test(test_runtime
    test_runtime.cpp
    "UPTKrt;CUDA::cudart;CUDA::cuda_driver"
)

# BLAS tests: link against UPTKblas, UPTKrt, and cublas
uptk_add_test(test_blas
    test_blas.cpp
    "UPTKblas;UPTKrt;cublas;CUDA::cudart"
)

# FFT tests: link against UPTKfft, UPTKrt, and cufft
uptk_add_test(test_fft
    test_fft.cpp
    "UPTKfft;UPTKrt;cufft;CUDA::cudart"
)

# NCCL tests: link against UPTKnccl, UPTKrt, and nccl
uptk_add_test(test_nccl
    test_nccl.cpp
    "UPTKnccl;UPTKrt;nccl;CUDA::cudart"
)

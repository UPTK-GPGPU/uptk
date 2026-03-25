#include "blas.hpp"

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

    cublasStatus_t UPTKblasStatusTocublasStatus(UPTKblasStatus_t para)
    {
        switch (para)
        {
        case UPTKBLAS_STATUS_ALLOC_FAILED:
            return CUBLAS_STATUS_ALLOC_FAILED;
        case UPTKBLAS_STATUS_ARCH_MISMATCH:
            return CUBLAS_STATUS_ARCH_MISMATCH;
        case UPTKBLAS_STATUS_EXECUTION_FAILED:
            return CUBLAS_STATUS_EXECUTION_FAILED;
        case UPTKBLAS_STATUS_INTERNAL_ERROR:
            return CUBLAS_STATUS_INTERNAL_ERROR;
        case UPTKBLAS_STATUS_INVALID_VALUE:
            return CUBLAS_STATUS_INVALID_VALUE;
        case UPTKBLAS_STATUS_LICENSE_ERROR:
            return CUBLAS_STATUS_LICENSE_ERROR;
        case UPTKBLAS_STATUS_MAPPING_ERROR:
            return CUBLAS_STATUS_MAPPING_ERROR;
        case UPTKBLAS_STATUS_NOT_INITIALIZED:
            return CUBLAS_STATUS_NOT_INITIALIZED;
        case UPTKBLAS_STATUS_NOT_SUPPORTED:
            return CUBLAS_STATUS_NOT_SUPPORTED;
        case UPTKBLAS_STATUS_SUCCESS:
            return CUBLAS_STATUS_SUCCESS;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasStatus_t cublasStatusToUPTKblasStatus(cublasStatus_t para)
    {
        switch (para)
        {
        case CUBLAS_STATUS_ALLOC_FAILED:
            return UPTKBLAS_STATUS_ALLOC_FAILED;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return UPTKBLAS_STATUS_ARCH_MISMATCH;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return UPTKBLAS_STATUS_EXECUTION_FAILED;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return UPTKBLAS_STATUS_INTERNAL_ERROR;
        case CUBLAS_STATUS_INVALID_VALUE:
            return UPTKBLAS_STATUS_INVALID_VALUE;
        case CUBLAS_STATUS_LICENSE_ERROR:
            return UPTKBLAS_STATUS_LICENSE_ERROR;
        case CUBLAS_STATUS_MAPPING_ERROR:
            return UPTKBLAS_STATUS_MAPPING_ERROR;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return UPTKBLAS_STATUS_NOT_INITIALIZED;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return UPTKBLAS_STATUS_NOT_SUPPORTED;
        case CUBLAS_STATUS_SUCCESS:
            return UPTKBLAS_STATUS_SUCCESS;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasAtomicsMode_t UPTKblasAtomicsModeTocublasAtomicsMode(UPTKblasAtomicsMode_t para)
    {
        switch (para)
        {
        case UPTKBLAS_ATOMICS_ALLOWED:
            return CUBLAS_ATOMICS_ALLOWED;
        case UPTKBLAS_ATOMICS_NOT_ALLOWED:
            return CUBLAS_ATOMICS_NOT_ALLOWED;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasAtomicsMode_t cublasAtomicsModeToUPTKblasAtomicsMode(cublasAtomicsMode_t para)
    {
        switch (para)
        {
        case CUBLAS_ATOMICS_ALLOWED:
            return UPTKBLAS_ATOMICS_ALLOWED;
        case CUBLAS_ATOMICS_NOT_ALLOWED:
            return UPTKBLAS_ATOMICS_NOT_ALLOWED;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasAtomicsMode_t hipblasAtomicsModeTocublasAtomicsMode(cublasAtomicsMode_t para)
    {
        switch (para)
        {
        case CUBLAS_ATOMICS_ALLOWED:
            return UPTKBLAS_ATOMICS_ALLOWED;
        case CUBLAS_ATOMICS_NOT_ALLOWED:
            return UPTKBLAS_ATOMICS_NOT_ALLOWED;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    // HIPBLAS_XF32_XDL_MATH, /* equivalent to rocblas_xf32_xdl_math_op, not supported in cuBLAS */
    cublasMath_t UPTKblasMathTocublasMath(UPTKblasMath_t para)
    {
        switch (para)
        {
        case UPTKBLAS_DEFAULT_MATH:
            return CUBLAS_DEFAULT_MATH;
        case UPTKBLAS_TENSOR_OP_MATH:
            return CUBLAS_TENSOR_OP_MATH;
        case UPTKBLAS_PEDANTIC_MATH:
            return CUBLAS_PEDANTIC_MATH;
        case UPTKBLAS_TF32_TENSOR_OP_MATH:
            return CUBLAS_TF32_TENSOR_OP_MATH;
        case UPTKBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
            return CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasMath_t cublasMathToUPTKblasMath(cublasMath_t para)
    {
        switch (para)
        {
        case CUBLAS_DEFAULT_MATH:
            return UPTKBLAS_DEFAULT_MATH;
        case CUBLAS_TENSOR_OP_MATH:
            return UPTKBLAS_TENSOR_OP_MATH;
        case CUBLAS_PEDANTIC_MATH:
            return UPTKBLAS_PEDANTIC_MATH;
        case CUBLAS_TF32_TENSOR_OP_MATH:
            return UPTKBLAS_TF32_TENSOR_OP_MATH;
        case CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
            return UPTKBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasDiagType_t UPTKblasDiagTypeTocublasDiagType(UPTKblasDiagType_t para)
    {
        switch (para)
        {
        case UPTKBLAS_DIAG_NON_UNIT:
            return CUBLAS_DIAG_NON_UNIT;
        case UPTKBLAS_DIAG_UNIT:
            return CUBLAS_DIAG_UNIT;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasFillMode_t UPTKblasFillModeTocublasFillMode(UPTKblasFillMode_t para)
    {
        switch (para)
        {
        case UPTKBLAS_FILL_MODE_FULL:
            return CUBLAS_FILL_MODE_FULL;
        case UPTKBLAS_FILL_MODE_LOWER:
            return CUBLAS_FILL_MODE_LOWER;
        case UPTKBLAS_FILL_MODE_UPPER:
            return CUBLAS_FILL_MODE_UPPER;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasGemmAlgo_t UPTKblasGemmAlgoTocublasGemmAlgo(UPTKblasGemmAlgo_t para)
    {
        switch (para)
        {
        case UPTKBLAS_GEMM_DFALT:
            // case UPTKBLAS_GEMM_DEFAULT:
            return CUBLAS_GEMM_DEFAULT;
        case UPTKBLAS_GEMM_ALGO0:
            return CUBLAS_GEMM_ALGO0;
        case UPTKBLAS_GEMM_ALGO1:
            return CUBLAS_GEMM_ALGO1;
        case UPTKBLAS_GEMM_ALGO2:
            return CUBLAS_GEMM_ALGO2;
        case UPTKBLAS_GEMM_ALGO3:
            return CUBLAS_GEMM_ALGO3;
        case UPTKBLAS_GEMM_ALGO4:
            return CUBLAS_GEMM_ALGO4;
        case UPTKBLAS_GEMM_ALGO5:
            return CUBLAS_GEMM_ALGO5;
        case UPTKBLAS_GEMM_ALGO6:
            return CUBLAS_GEMM_ALGO6;
        case UPTKBLAS_GEMM_ALGO7:
            return CUBLAS_GEMM_ALGO7;
        case UPTKBLAS_GEMM_ALGO8:
            return CUBLAS_GEMM_ALGO8;
        case UPTKBLAS_GEMM_ALGO9:
            return CUBLAS_GEMM_ALGO9;
        case UPTKBLAS_GEMM_ALGO10:
            return CUBLAS_GEMM_ALGO10;
        case UPTKBLAS_GEMM_ALGO11:
            return CUBLAS_GEMM_ALGO11;
        case UPTKBLAS_GEMM_ALGO12:
            return CUBLAS_GEMM_ALGO12;
        case UPTKBLAS_GEMM_ALGO13:
            return CUBLAS_GEMM_ALGO13;
        case UPTKBLAS_GEMM_ALGO14:
            return CUBLAS_GEMM_ALGO14;
        case UPTKBLAS_GEMM_ALGO15:
            return CUBLAS_GEMM_ALGO15;
        case UPTKBLAS_GEMM_ALGO16:
            return CUBLAS_GEMM_ALGO16;
        case UPTKBLAS_GEMM_ALGO17:
            return CUBLAS_GEMM_ALGO17;
        case UPTKBLAS_GEMM_ALGO18:
            return CUBLAS_GEMM_ALGO18;
        case UPTKBLAS_GEMM_ALGO19:
            return CUBLAS_GEMM_ALGO19;
        case UPTKBLAS_GEMM_ALGO20:
            return CUBLAS_GEMM_ALGO20;
        case UPTKBLAS_GEMM_ALGO21:
            return CUBLAS_GEMM_ALGO21;
        case UPTKBLAS_GEMM_ALGO22:
            return CUBLAS_GEMM_ALGO22;
        case UPTKBLAS_GEMM_ALGO23:
            return CUBLAS_GEMM_ALGO23;

        case UPTKBLAS_GEMM_DEFAULT_TENSOR_OP:
            // case UPTKBLAS_GEMM_DFALT_TENSOR_OP:
            return CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        case UPTKBLAS_GEMM_ALGO0_TENSOR_OP:
            return CUBLAS_GEMM_ALGO0_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO1_TENSOR_OP:
            return CUBLAS_GEMM_ALGO1_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO2_TENSOR_OP:
            return CUBLAS_GEMM_ALGO2_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO3_TENSOR_OP:
            return CUBLAS_GEMM_ALGO3_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO4_TENSOR_OP:
            return CUBLAS_GEMM_ALGO4_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO5_TENSOR_OP:
            return CUBLAS_GEMM_ALGO5_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO6_TENSOR_OP:
            return CUBLAS_GEMM_ALGO6_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO7_TENSOR_OP:
            return CUBLAS_GEMM_ALGO7_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO8_TENSOR_OP:
            return CUBLAS_GEMM_ALGO8_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO9_TENSOR_OP:
            return CUBLAS_GEMM_ALGO9_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO10_TENSOR_OP:
            return CUBLAS_GEMM_ALGO10_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO11_TENSOR_OP:
            return CUBLAS_GEMM_ALGO11_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO12_TENSOR_OP:
            return CUBLAS_GEMM_ALGO12_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO13_TENSOR_OP:
            return CUBLAS_GEMM_ALGO13_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO14_TENSOR_OP:
            return CUBLAS_GEMM_ALGO14_TENSOR_OP;
        case UPTKBLAS_GEMM_ALGO15_TENSOR_OP:
            return CUBLAS_GEMM_ALGO15_TENSOR_OP;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasOperation_t UPTKblasOperationTocublasOperation(UPTKblasOperation_t para)
    {
        switch (para)
        {
        case UPTKBLAS_OP_C:
            // case UPTKBLAS_OP_HERMITAN:
            return CUBLAS_OP_C;
        case UPTKBLAS_OP_CONJG:
            return CUBLAS_OP_CONJG;
        case UPTKBLAS_OP_N:
            return CUBLAS_OP_N;
        case UPTKBLAS_OP_T:
            return CUBLAS_OP_T;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasOperation_t hipblasOperationTohcublasOperation(cublasOperation_t para)
    {
        switch (para)
        {
        case CUBLAS_OP_C:
            // case CUBLAS_OP_HERMITAN:
            return UPTKBLAS_OP_C;
        case CUBLAS_OP_CONJG:
            return UPTKBLAS_OP_CONJG;
        case CUBLAS_OP_N:
            return UPTKBLAS_OP_N;
        case CUBLAS_OP_T:
            return UPTKBLAS_OP_T;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasPointerMode_t UPTKblasPointerModeTocublasPointerMode(UPTKblasPointerMode_t para)
    {
        switch (para)
        {
        case UPTKBLAS_POINTER_MODE_DEVICE:
            return CUBLAS_POINTER_MODE_DEVICE;
        case UPTKBLAS_POINTER_MODE_HOST:
            return CUBLAS_POINTER_MODE_HOST;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasPointerMode_t cublasPointerModeToUPTKblasPointerMode(cublasPointerMode_t para)
    {
        switch (para)
        {
        case CUBLAS_POINTER_MODE_DEVICE:
            return UPTKBLAS_POINTER_MODE_DEVICE;
        case CUBLAS_POINTER_MODE_HOST:
            return UPTKBLAS_POINTER_MODE_HOST;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasSideMode_t UPTKblasSideModeTocublasSideMode(UPTKblasSideMode_t para)
    {
        switch (para)
        {
        case UPTKBLAS_SIDE_LEFT:
            return CUBLAS_SIDE_LEFT;
        case UPTKBLAS_SIDE_RIGHT:
            return CUBLAS_SIDE_RIGHT;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasStatus_t UPTKblasStatusStatusTocublasStatus(UPTKblasStatus_t para)
    {
        switch (para)
        {
        case UPTKBLAS_STATUS_ALLOC_FAILED:
            return CUBLAS_STATUS_ALLOC_FAILED;
        case UPTKBLAS_STATUS_ARCH_MISMATCH:
            return CUBLAS_STATUS_ARCH_MISMATCH;
        case UPTKBLAS_STATUS_EXECUTION_FAILED:
            return CUBLAS_STATUS_EXECUTION_FAILED;
        case UPTKBLAS_STATUS_INTERNAL_ERROR:
            return CUBLAS_STATUS_INTERNAL_ERROR;
        case UPTKBLAS_STATUS_INVALID_VALUE:
            return CUBLAS_STATUS_INVALID_VALUE;
        case UPTKBLAS_STATUS_LICENSE_ERROR:
            return CUBLAS_STATUS_LICENSE_ERROR;
        case UPTKBLAS_STATUS_MAPPING_ERROR:
            return CUBLAS_STATUS_MAPPING_ERROR;
        case UPTKBLAS_STATUS_NOT_INITIALIZED:
            return CUBLAS_STATUS_NOT_INITIALIZED;
        case UPTKBLAS_STATUS_NOT_SUPPORTED:
            return CUBLAS_STATUS_NOT_SUPPORTED;
        case UPTKBLAS_STATUS_SUCCESS:
            return CUBLAS_STATUS_SUCCESS;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasStatus_t hipblasStatusTocublasStatus(cublasStatus_t para)
    {
        switch (para)
        {
        case CUBLAS_STATUS_ALLOC_FAILED:
            return UPTKBLAS_STATUS_ALLOC_FAILED;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return UPTKBLAS_STATUS_ARCH_MISMATCH;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return UPTKBLAS_STATUS_EXECUTION_FAILED;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return UPTKBLAS_STATUS_INTERNAL_ERROR;
        case CUBLAS_STATUS_INVALID_VALUE:
            return UPTKBLAS_STATUS_INVALID_VALUE;
        case CUBLAS_STATUS_LICENSE_ERROR:
            return UPTKBLAS_STATUS_LICENSE_ERROR;
        case CUBLAS_STATUS_MAPPING_ERROR:
            return UPTKBLAS_STATUS_MAPPING_ERROR;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return UPTKBLAS_STATUS_NOT_INITIALIZED;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return UPTKBLAS_STATUS_NOT_SUPPORTED;
        case CUBLAS_STATUS_SUCCESS:
            return UPTKBLAS_STATUS_SUCCESS;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cudaDataType UPTKDataTypeTocudaDataType(UPTKDataType para)
    {
        switch (para)
        {
        case UPTK_R_16F:
            return CUDA_R_16F;
        case UPTK_C_16F:
            return CUDA_C_16F;
        case UPTK_R_16BF:
            return CUDA_R_16BF;
        case UPTK_C_16BF:
            return CUDA_C_16BF;
        case UPTK_R_32F:
            return CUDA_R_32F;
        case UPTK_C_32F:
            return CUDA_C_32F;
        case UPTK_R_64F:
            return CUDA_R_64F;
        case UPTK_C_64F:
            return CUDA_C_64F;
        case UPTK_R_4I:
            return CUDA_R_4I;
        case UPTK_C_4I:
            return CUDA_C_4I;
        case UPTK_R_4U:
            return CUDA_R_4U;
        case UPTK_C_4U:
            return CUDA_C_4U;
        case UPTK_R_8I:
            return CUDA_R_8I;
        case UPTK_C_8I:
            return CUDA_C_8I;
        case UPTK_R_8U:
            return CUDA_R_8U;
        case UPTK_C_8U:
            return CUDA_C_8U;
        case UPTK_R_16I:
            return CUDA_R_16I;
        case UPTK_C_16I:
            return CUDA_C_16I;
        case UPTK_R_16U:
            return CUDA_R_16U;
        case UPTK_C_16U:
            return CUDA_C_16U;
        case UPTK_R_32I:
            return CUDA_R_32I;
        case UPTK_C_32I:
            return CUDA_C_32I;
        case UPTK_R_32U:
            return CUDA_R_32U;
        case UPTK_C_32U:
            return CUDA_C_32U;
        case UPTK_R_64I:
            return CUDA_R_64I;
        case UPTK_C_64I:
            return CUDA_C_64I;
        case UPTK_R_64U:
            return CUDA_R_64U;
        case UPTK_C_64U:
            return CUDA_C_64U;
        case UPTK_R_8F_E4M3:
            return CUDA_R_8F_E4M3;
        case UPTK_R_8F_E5M2:
            return CUDA_R_8F_E5M2;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKDataType hipDataTypeTocudaDataType(cudaDataType para)
    {
        switch (para)
        {
        case CUDA_R_16F:
            return UPTK_R_16F;
        case CUDA_C_16F:
            return UPTK_C_16F;

        case CUDA_R_16BF:
            return UPTK_R_16BF;
        case CUDA_C_16BF:
            return UPTK_C_16BF;

        case CUDA_R_32F:
            return UPTK_R_32F;
        case CUDA_C_32F:
            return UPTK_C_32F;

        case CUDA_R_64F:
            return UPTK_R_64F;
        case CUDA_C_64F:
            return UPTK_C_64F;

        case CUDA_R_4I:
            return UPTK_R_4I;
        case CUDA_C_4I:
            return UPTK_C_4I;

        case CUDA_R_4U:
            return UPTK_R_4U;
        case CUDA_C_4U:
            return UPTK_C_4U;

        case CUDA_R_8I:
            return UPTK_R_8I;
        case CUDA_C_8I:
            return UPTK_C_8I;

        case CUDA_R_8U:
            return UPTK_R_8U;
        case CUDA_C_8U:
            return UPTK_C_8U;

        case CUDA_R_16I:
            return UPTK_R_16I;
        case CUDA_C_16I:
            return UPTK_C_16I;

        case CUDA_R_16U:
            return UPTK_R_16U;
        case CUDA_C_16U:
            return UPTK_C_16U;

        case CUDA_R_32I:
            return UPTK_R_32I;
        case CUDA_C_32I:
            return UPTK_C_32I;

        case CUDA_R_32U:
            return UPTK_R_32U;
        case CUDA_C_32U:
            return UPTK_C_32U;

        case CUDA_R_64I:
            return UPTK_R_64I;
        case CUDA_C_64I:
            return UPTK_C_64I;

        case CUDA_R_64U:
            return UPTK_R_64U;
        case CUDA_C_64U:
            return UPTK_C_64U;

        case CUDA_R_8F_E4M3:
            return UPTK_R_8F_E4M3;
        case CUDA_R_8F_E5M2:
            return UPTK_R_8F_E5M2;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    cublasComputeType_t UPTKblasComputeTypeTocublasComputeType(UPTKblasComputeType_t para)
    {
        switch (para)
        {
        case UPTKBLAS_COMPUTE_16F:
            return CUBLAS_COMPUTE_16F;
        case UPTKBLAS_COMPUTE_16F_PEDANTIC:
            return CUBLAS_COMPUTE_16F_PEDANTIC;
        case UPTKBLAS_COMPUTE_32F:
            return CUBLAS_COMPUTE_32F;
        case UPTKBLAS_COMPUTE_32F_PEDANTIC:
            return CUBLAS_COMPUTE_32F_PEDANTIC;
        case UPTKBLAS_COMPUTE_32F_FAST_16F:
            return CUBLAS_COMPUTE_32F_FAST_16F;
        case UPTKBLAS_COMPUTE_32F_FAST_16BF:
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        case UPTKBLAS_COMPUTE_32F_FAST_TF32:
            return CUBLAS_COMPUTE_32F_FAST_TF32;
        case UPTKBLAS_COMPUTE_64F:
            return CUBLAS_COMPUTE_64F;
        case UPTKBLAS_COMPUTE_64F_PEDANTIC:
            return CUBLAS_COMPUTE_64F_PEDANTIC;
        case UPTKBLAS_COMPUTE_32I:
            return CUBLAS_COMPUTE_32I;
        case UPTKBLAS_COMPUTE_32I_PEDANTIC:
            return CUBLAS_COMPUTE_32I_PEDANTIC;
        default:
            ERROR_INVALID_ENUM();
        }
    }

    UPTKblasComputeType_t hipblasComputeTypeModeTocublasComputeType(cublasComputeType_t para)
    {
        switch (para)
        {
        case CUBLAS_COMPUTE_16F:
            return UPTKBLAS_COMPUTE_16F;
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            return UPTKBLAS_COMPUTE_16F_PEDANTIC;
        case CUBLAS_COMPUTE_32F:
            return UPTKBLAS_COMPUTE_32F;
        case CUBLAS_COMPUTE_32F_PEDANTIC:
            return UPTKBLAS_COMPUTE_32F_PEDANTIC;
        case CUBLAS_COMPUTE_32F_FAST_16F:
            return UPTKBLAS_COMPUTE_32F_FAST_16F;
        case CUBLAS_COMPUTE_32F_FAST_16BF:
            return UPTKBLAS_COMPUTE_32F_FAST_16BF;
        case CUBLAS_COMPUTE_32F_FAST_TF32:
            return UPTKBLAS_COMPUTE_32F_FAST_TF32;
        case CUBLAS_COMPUTE_64F:
            return UPTKBLAS_COMPUTE_64F;
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            return UPTKBLAS_COMPUTE_64F_PEDANTIC;
        case CUBLAS_COMPUTE_32I:
            return UPTKBLAS_COMPUTE_32I;
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            return UPTKBLAS_COMPUTE_32I_PEDANTIC;
        default:
            ERROR_INVALID_ENUM();
        }
    }

#if defined(__cplusplus)
}
#endif /* __cplusplus */
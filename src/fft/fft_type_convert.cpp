#include "fft.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

cufftResult UPTKfftResultTocufftResult(UPTKfftResult para) {
    switch (para) {
        case UPTKFFT_ALLOC_FAILED:
            return CUFFT_ALLOC_FAILED;
        case UPTKFFT_EXEC_FAILED:
            return CUFFT_EXEC_FAILED;
        case UPTKFFT_INCOMPLETE_PARAMETER_LIST:
            return CUFFT_INCOMPLETE_PARAMETER_LIST;
        case UPTKFFT_INTERNAL_ERROR:
            return CUFFT_INTERNAL_ERROR;
        case UPTKFFT_INVALID_DEVICE:
            return CUFFT_INVALID_DEVICE;
        case UPTKFFT_INVALID_PLAN:
            return CUFFT_INVALID_PLAN;
        case UPTKFFT_INVALID_SIZE:
            return CUFFT_INVALID_SIZE;
        case UPTKFFT_INVALID_TYPE:
            return CUFFT_INVALID_TYPE;
        case UPTKFFT_INVALID_VALUE:
            return CUFFT_INVALID_VALUE;
        case UPTKFFT_LICENSE_ERROR:
            return CUFFT_LICENSE_ERROR;
        case UPTKFFT_NOT_IMPLEMENTED:
            return CUFFT_NOT_IMPLEMENTED;
        case UPTKFFT_NOT_SUPPORTED:
            return CUFFT_NOT_SUPPORTED;
        case UPTKFFT_NO_WORKSPACE:
            return CUFFT_NO_WORKSPACE;
        case UPTKFFT_PARSE_ERROR:
            return CUFFT_PARSE_ERROR;
        case UPTKFFT_SETUP_FAILED:
            return CUFFT_SETUP_FAILED;
        case UPTKFFT_SUCCESS:
            return CUFFT_SUCCESS;
        case UPTKFFT_UNALIGNED_DATA:
            return CUFFT_UNALIGNED_DATA;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKfftResult cufftResultToUPTKfftResult(cufftResult para) {
    switch (para) {
        case CUFFT_ALLOC_FAILED:
            return UPTKFFT_ALLOC_FAILED;
        case CUFFT_EXEC_FAILED:
            return UPTKFFT_EXEC_FAILED;
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return UPTKFFT_INCOMPLETE_PARAMETER_LIST;
        case CUFFT_INTERNAL_ERROR:
            return UPTKFFT_INTERNAL_ERROR;
        case CUFFT_INVALID_DEVICE:
            return UPTKFFT_INVALID_DEVICE;
        case CUFFT_INVALID_PLAN:
            return UPTKFFT_INVALID_PLAN;
        case CUFFT_INVALID_SIZE:
            return UPTKFFT_INVALID_SIZE;
        case CUFFT_INVALID_TYPE:
            return UPTKFFT_INVALID_TYPE;
        case CUFFT_INVALID_VALUE:
            return UPTKFFT_INVALID_VALUE;
        case CUFFT_LICENSE_ERROR:
            return UPTKFFT_LICENSE_ERROR;
        case CUFFT_NOT_IMPLEMENTED:
            return UPTKFFT_NOT_IMPLEMENTED;
        case CUFFT_NOT_SUPPORTED:
            return UPTKFFT_NOT_SUPPORTED;
        case CUFFT_NO_WORKSPACE:
            return UPTKFFT_NO_WORKSPACE;
        case CUFFT_PARSE_ERROR:
            return UPTKFFT_PARSE_ERROR;
        case CUFFT_SETUP_FAILED:
            return UPTKFFT_SETUP_FAILED;
        case CUFFT_SUCCESS:
            return UPTKFFT_SUCCESS;
        case CUFFT_UNALIGNED_DATA:
            return UPTKFFT_UNALIGNED_DATA;
        default:
            ERROR_INVALID_ENUM();
    }
}

cufftType UPTKfftTypeTocufftType(UPTKfftType para) {
    switch (para) {
        case UPTKFFT_C2C:
            return CUFFT_C2C;
        case UPTKFFT_C2R:
            return CUFFT_C2R;
        case UPTKFFT_D2Z:
            return CUFFT_D2Z;
        case UPTKFFT_R2C:
            return CUFFT_R2C;
        case UPTKFFT_Z2D:
            return CUFFT_Z2D;
        case UPTKFFT_Z2Z:
            return CUFFT_Z2Z;
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

cufftXtCallbackType UPTKfftXtCallbackTypeTocufftXtCallbackType(UPTKfftXtCallbackType para) {
    switch (para) {
        case UPTKFFT_CB_LD_COMPLEX:
            return CUFFT_CB_LD_COMPLEX;
        case UPTKFFT_CB_LD_COMPLEX_DOUBLE:
            return CUFFT_CB_LD_COMPLEX_DOUBLE;
        case UPTKFFT_CB_LD_REAL:
            return CUFFT_CB_LD_REAL;
        case UPTKFFT_CB_LD_REAL_DOUBLE:
            return CUFFT_CB_LD_REAL_DOUBLE;
        case UPTKFFT_CB_ST_COMPLEX:
            return CUFFT_CB_ST_COMPLEX;
        case UPTKFFT_CB_ST_COMPLEX_DOUBLE:
            return CUFFT_CB_ST_COMPLEX_DOUBLE;
        case UPTKFFT_CB_ST_REAL:
            return CUFFT_CB_ST_REAL;
        case UPTKFFT_CB_ST_REAL_DOUBLE:
            return CUFFT_CB_ST_REAL_DOUBLE;
        case UPTKFFT_CB_UNDEFINED:
            return CUFFT_CB_UNDEFINED;
        default:
            ERROR_INVALID_ENUM();
    }
}

cufftXtSubFormat UPTKfftXtSubFormatTocuftXtSubFormat(UPTKfftXtSubFormat para)
{
    switch (para)
    {
    case UPTKFFT_XT_FORMAT_INPUT:
        return CUFFT_XT_FORMAT_INPUT;
    case UPTKFFT_XT_FORMAT_OUTPUT:
        return CUFFT_XT_FORMAT_OUTPUT;
    case UPTKFFT_XT_FORMAT_INPLACE:
        return CUFFT_XT_FORMAT_INPLACE;
    case UPTKFFT_XT_FORMAT_INPLACE_SHUFFLED:
        return CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    case UPTKFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
        return CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED;
    case UPTKFFT_XT_FORMAT_DISTRIBUTED_INPUT:
        return CUFFT_XT_FORMAT_DISTRIBUTED_INPUT;
    case UPTKFFT_XT_FORMAT_DISTRIBUTED_OUTPUT:
        return CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT;
    case UPTKFFT_FORMAT_UNDEFINED:
        return CUFFT_FORMAT_UNDEFINED;
    default:
        ERROR_INVALID_ENUM();
    }
}

cufftXtCopyType UPTKfftXtCopyTypeTocufftXtCopyType(UPTKfftXtCopyType para)
{
    switch (para)
    {
    case UPTKFFT_COPY_HOST_TO_DEVICE:
        return CUFFT_COPY_HOST_TO_DEVICE;
    case UPTKFFT_COPY_DEVICE_TO_HOST:
        return CUFFT_COPY_DEVICE_TO_HOST;
    case UPTKFFT_COPY_DEVICE_TO_DEVICE:
        return CUFFT_COPY_DEVICE_TO_DEVICE;
    case UPTKFFT_COPY_UNDEFINED:
        return CUFFT_COPY_UNDEFINED;
    default:
        ERROR_INVALID_ENUM();
    }
}


cufftXtQueryType UPTKfftXtQueryTypeTohcufftXtQueryType(UPTKfftXtQueryType para)
{
    switch (para)
    {
    case UPTKFFT_QUERY_1D_FACTORS:
        return CUFFT_QUERY_1D_FACTORS;
        case UPTKFFT_QUERY_UNDEFINED:
    default:
        ERROR_INVALID_ENUM();
    }
}


cufftXtWorkAreaPolicy UPTKfftXtWorkAreaPolicyTocufftXtWorkAreaPolicy(UPTKfftXtWorkAreaPolicy para)
{
    switch (para)
    {
    case UPTKFFT_WORKAREA_MINIMAL:
        return CUFFT_WORKAREA_MINIMAL;
    case UPTKFFT_WORKAREA_USER:
        return CUFFT_WORKAREA_USER;
    case UPTKFFT_WORKAREA_PERFORMANCE:
        return CUFFT_WORKAREA_PERFORMANCE;
    default:
        ERROR_INVALID_ENUM();
    }
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
#include "sparse.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

cusparseAction_t UPTKsparseActionTocusparseAction(UPTKsparseAction_t para) {
    switch (para) {
        case UPTKSPARSE_ACTION_NUMERIC:
            return CUSPARSE_ACTION_NUMERIC;
        case UPTKSPARSE_ACTION_SYMBOLIC:
            return CUSPARSE_ACTION_SYMBOLIC;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseSpMMAlg_t UPTKsparseSpMMAlgTocusparseSpMMAlg(UPTKsparseSpMMAlg_t para) {
    switch (para) {
        case UPTKSPARSE_SPMM_ALG_DEFAULT:
            return CUSPARSE_SPMM_ALG_DEFAULT;
        case UPTKSPARSE_SPMM_COO_ALG1:
            return CUSPARSE_SPMM_COO_ALG1;
        case UPTKSPARSE_SPMM_COO_ALG2:
            return CUSPARSE_SPMM_COO_ALG2;
        case UPTKSPARSE_SPMM_COO_ALG3:
            return CUSPARSE_SPMM_COO_ALG3;
        case UPTKSPARSE_SPMM_COO_ALG4:
            return CUSPARSE_SPMM_COO_ALG4;
        case UPTKSPARSE_SPMM_CSR_ALG1:
            return CUSPARSE_SPMM_CSR_ALG1;
        case UPTKSPARSE_SPMM_CSR_ALG2:
            return CUSPARSE_SPMM_CSR_ALG2;
        case UPTKSPARSE_SPMM_CSR_ALG3:
            return CUSPARSE_SPMM_CSR_ALG3;
        case UPTKSPARSE_SPMM_BLOCKED_ELL_ALG1:
            return CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseSpMVAlg_t UPTKsparseSpMVAlgTocusparseSpMVAlg(UPTKsparseSpMVAlg_t para) {
    switch (para) {
        case UPTKSPARSE_SPMV_ALG_DEFAULT:
            return CUSPARSE_SPMV_ALG_DEFAULT;
        case UPTKSPARSE_SPMV_CSR_ALG1:
            return CUSPARSE_SPMV_CSR_ALG1;
        case UPTKSPARSE_SPMV_CSR_ALG2:
            return CUSPARSE_SPMV_CSR_ALG2;
        case UPTKSPARSE_SPMV_COO_ALG1:
            return CUSPARSE_SPMV_COO_ALG1;
        case UPTKSPARSE_SPMV_COO_ALG2:
            return CUSPARSE_SPMV_COO_ALG2;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseCsr2CscAlg_t UPTKsparseCsr2CscAlgTocusparseCsr2CscAlg(UPTKsparseCsr2CscAlg_t para) {
    switch (para) {
        case UPTKSPARSE_CSR2CSC_ALG1:
            return CUSPARSE_CSR2CSC_ALG1;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseDiagType_t UPTKsparseDiagTypeTocusparseDiagType(UPTKsparseDiagType_t para) {
    switch (para) {
        case UPTKSPARSE_DIAG_TYPE_NON_UNIT:
            return CUSPARSE_DIAG_TYPE_NON_UNIT;
        case UPTKSPARSE_DIAG_TYPE_UNIT:
            return CUSPARSE_DIAG_TYPE_UNIT;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseDiagType_t cusparseDiagTypeToUPTKsparseDiagType(cusparseDiagType_t para) {
    switch (para) {
        case CUSPARSE_DIAG_TYPE_NON_UNIT:
            return UPTKSPARSE_DIAG_TYPE_NON_UNIT;
        case CUSPARSE_DIAG_TYPE_UNIT:
            return UPTKSPARSE_DIAG_TYPE_UNIT;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseDirection_t UPTKsparseDirectionTocusparseDirection(UPTKsparseDirection_t para) {
    switch (para) {
        case UPTKSPARSE_DIRECTION_COLUMN:
            return CUSPARSE_DIRECTION_COLUMN;
        case UPTKSPARSE_DIRECTION_ROW:
            return CUSPARSE_DIRECTION_ROW;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseFillMode_t UPTKsparseFillModeTocusparseFillMode(UPTKsparseFillMode_t para) {
    switch (para) {
        case UPTKSPARSE_FILL_MODE_LOWER:
            return CUSPARSE_FILL_MODE_LOWER;
        case UPTKSPARSE_FILL_MODE_UPPER:
            return CUSPARSE_FILL_MODE_UPPER;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseFillMode_t cusparseFillModeToUPTKsparseFillMode(cusparseFillMode_t para) {
    switch (para) {
        case CUSPARSE_FILL_MODE_LOWER:
            return UPTKSPARSE_FILL_MODE_LOWER;
        case CUSPARSE_FILL_MODE_UPPER:
            return UPTKSPARSE_FILL_MODE_UPPER;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseFormat_t UPTKsparseFormatTocusparseFormat(UPTKsparseFormat_t para) {
    switch (para) {
         case UPTKSPARSE_FORMAT_CSR:
            return CUSPARSE_FORMAT_CSR;
        case UPTKSPARSE_FORMAT_CSC:
             return CUSPARSE_FORMAT_CSC;
        case UPTKSPARSE_FORMAT_COO:
            return CUSPARSE_FORMAT_COO;
        case UPTKSPARSE_FORMAT_BLOCKED_ELL:
            return CUSPARSE_FORMAT_BLOCKED_ELL;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseFormat_t cusparseFormatToUPTKsparseFormat(cusparseFormat_t para) {
    switch (para) {
        case CUSPARSE_FORMAT_COO:
            return UPTKSPARSE_FORMAT_COO;
        case CUSPARSE_FORMAT_CSC:
            return UPTKSPARSE_FORMAT_CSC;
        case CUSPARSE_FORMAT_CSR:
            return UPTKSPARSE_FORMAT_CSR;
        case CUSPARSE_FORMAT_BLOCKED_ELL:
            return UPTKSPARSE_FORMAT_BLOCKED_ELL;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseIndexType_t UPTKsparseIndexTypeTocusparseIndexType(UPTKsparseIndexType_t para) {
    switch (para) {
        case UPTKSPARSE_INDEX_16U:
            return CUSPARSE_INDEX_16U;
        case UPTKSPARSE_INDEX_32I:
            return CUSPARSE_INDEX_32I;
        case UPTKSPARSE_INDEX_64I:
            return CUSPARSE_INDEX_64I;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseIndexType_t cusparseIndexTypeToUPTKsparseIndexType(cusparseIndexType_t para) {
    switch (para) {
        case CUSPARSE_INDEX_16U:
            return UPTKSPARSE_INDEX_16U;
        case CUSPARSE_INDEX_32I:
            return UPTKSPARSE_INDEX_32I;
        case CUSPARSE_INDEX_64I:
            return UPTKSPARSE_INDEX_64I;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseIndexBase_t UPTKsparseIndexBaseTocusparseIndexBase(UPTKsparseIndexBase_t para) {
    switch (para) {
        case UPTKSPARSE_INDEX_BASE_ONE:
            return CUSPARSE_INDEX_BASE_ONE;
        case UPTKSPARSE_INDEX_BASE_ZERO:
            return CUSPARSE_INDEX_BASE_ZERO;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseIndexBase_t cusparseIndexBaseToUPTKsparseIndexBase(cusparseIndexBase_t para) {
    switch (para) {
        case CUSPARSE_INDEX_BASE_ONE:
            return UPTKSPARSE_INDEX_BASE_ONE;
        case CUSPARSE_INDEX_BASE_ZERO:
            return UPTKSPARSE_INDEX_BASE_ZERO;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseMatrixType_t UPTKsparseMatrixTypeTocusparseMatrixType(UPTKsparseMatrixType_t para) {
    switch (para) {
        case UPTKSPARSE_MATRIX_TYPE_GENERAL:
            return CUSPARSE_MATRIX_TYPE_GENERAL;
        case UPTKSPARSE_MATRIX_TYPE_HERMITIAN:
            return CUSPARSE_MATRIX_TYPE_HERMITIAN;
        case UPTKSPARSE_MATRIX_TYPE_SYMMETRIC:
            return CUSPARSE_MATRIX_TYPE_SYMMETRIC;
        case UPTKSPARSE_MATRIX_TYPE_TRIANGULAR:
            return CUSPARSE_MATRIX_TYPE_TRIANGULAR;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseMatrixType_t cusparseMatrixTypeToUPTKsparseMatrixType(cusparseMatrixType_t para) {
    switch (para) {
        case CUSPARSE_MATRIX_TYPE_GENERAL:
            return UPTKSPARSE_MATRIX_TYPE_GENERAL;
        case CUSPARSE_MATRIX_TYPE_HERMITIAN:
            return UPTKSPARSE_MATRIX_TYPE_HERMITIAN;
        case CUSPARSE_MATRIX_TYPE_SYMMETRIC:
            return UPTKSPARSE_MATRIX_TYPE_SYMMETRIC;
        case CUSPARSE_MATRIX_TYPE_TRIANGULAR:
            return UPTKSPARSE_MATRIX_TYPE_TRIANGULAR;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseOperation_t UPTKsparseOperationTocusparseOperation(UPTKsparseOperation_t para) {
    switch (para) {
        case UPTKSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        case UPTKSPARSE_OPERATION_NON_TRANSPOSE:
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        case UPTKSPARSE_OPERATION_TRANSPOSE:
            return CUSPARSE_OPERATION_TRANSPOSE;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseOrder_t UPTKsparseOrderTocusparseOrder(UPTKsparseOrder_t para) {
    switch (para) {
        case UPTKSPARSE_ORDER_COL:
            return CUSPARSE_ORDER_COL;
         case UPTKSPARSE_ORDER_ROW:
             return CUSPARSE_ORDER_ROW;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseOrder_t cusparseOrderToUPTKsparseOrder(cusparseOrder_t para) {
    switch (para) {
        case CUSPARSE_ORDER_COL:
            return UPTKSPARSE_ORDER_COL;
         case CUSPARSE_ORDER_ROW:
             return UPTKSPARSE_ORDER_ROW;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparsePointerMode_t UPTKsparsePointerModeTocusparsePointerMode(UPTKsparsePointerMode_t para) {
    switch (para) {
        case UPTKSPARSE_POINTER_MODE_DEVICE:
            return CUSPARSE_POINTER_MODE_DEVICE;
        case UPTKSPARSE_POINTER_MODE_HOST:
            return CUSPARSE_POINTER_MODE_HOST;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparsePointerMode_t cusparsePointerModeToUPTKsparsePointerMode(cusparsePointerMode_t para) {
    switch (para) {
        case CUSPARSE_POINTER_MODE_DEVICE:
            return UPTKSPARSE_POINTER_MODE_DEVICE;
        case CUSPARSE_POINTER_MODE_HOST:
            return UPTKSPARSE_POINTER_MODE_HOST;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseSolvePolicy_t UPTKsparseSolvePolicyTocusparseSolvePolicy(UPTKsparseSolvePolicy_t para) {
    switch (para) {
        case UPTKSPARSE_SOLVE_POLICY_NO_LEVEL:
            return CUSPARSE_SOLVE_POLICY_NO_LEVEL;
        case UPTKSPARSE_SOLVE_POLICY_USE_LEVEL:
            return CUSPARSE_SOLVE_POLICY_USE_LEVEL;
        default:
            ERROR_INVALID_ENUM();
    }
}

cusparseStatus_t UPTKsparseStatusTocusparseStatus(UPTKsparseStatus_t para) {
    switch (para) {
        case UPTKSPARSE_STATUS_ALLOC_FAILED:
            return CUSPARSE_STATUS_ALLOC_FAILED;
        case UPTKSPARSE_STATUS_ARCH_MISMATCH:
            return CUSPARSE_STATUS_ARCH_MISMATCH;
        case UPTKSPARSE_STATUS_EXECUTION_FAILED:
            return CUSPARSE_STATUS_EXECUTION_FAILED;
        case UPTKSPARSE_STATUS_INTERNAL_ERROR:
            return CUSPARSE_STATUS_INTERNAL_ERROR;
        case UPTKSPARSE_STATUS_INVALID_VALUE:
            return CUSPARSE_STATUS_INVALID_VALUE;
        case UPTKSPARSE_STATUS_MAPPING_ERROR:
            return CUSPARSE_STATUS_MAPPING_ERROR;
        case UPTKSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case UPTKSPARSE_STATUS_NOT_INITIALIZED:
            return CUSPARSE_STATUS_NOT_INITIALIZED;
        case UPTKSPARSE_STATUS_NOT_SUPPORTED:
            return CUSPARSE_STATUS_NOT_SUPPORTED;
        case UPTKSPARSE_STATUS_SUCCESS:
            return CUSPARSE_STATUS_SUCCESS;
        case UPTKSPARSE_STATUS_ZERO_PIVOT:
            return CUSPARSE_STATUS_ZERO_PIVOT;
        case UPTKSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKsparseStatus_t cusparseStatusToUPTKsparseStatus(cusparseStatus_t para) {
    switch (para) {
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return UPTKSPARSE_STATUS_ALLOC_FAILED;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return UPTKSPARSE_STATUS_ARCH_MISMATCH;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return UPTKSPARSE_STATUS_EXECUTION_FAILED;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return UPTKSPARSE_STATUS_INTERNAL_ERROR;
        case CUSPARSE_STATUS_INVALID_VALUE:
            return UPTKSPARSE_STATUS_INVALID_VALUE;
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return UPTKSPARSE_STATUS_MAPPING_ERROR;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return UPTKSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return UPTKSPARSE_STATUS_NOT_INITIALIZED;
        case CUSPARSE_STATUS_NOT_SUPPORTED:
            return UPTKSPARSE_STATUS_NOT_SUPPORTED;
        case CUSPARSE_STATUS_SUCCESS:
            return UPTKSPARSE_STATUS_SUCCESS;
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return UPTKSPARSE_STATUS_ZERO_PIVOT;
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return UPTKSPARSE_STATUS_INSUFFICIENT_RESOURCES;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaDataType UPTKDataTypeTocudaDataType(UPTKDataType_t para)
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

UPTKDataType_t cudaDataTypeToUPTKDataType(cudaDataType para)
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

cusparseSpMatAttribute_t UPTKsparseSpMatAttributeTocusparseSpMatAttribute(UPTKsparseSpMatAttribute_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SPMAT_FILL_MODE:
        return CUSPARSE_SPMAT_FILL_MODE;
    case UPTKSPARSE_SPMAT_DIAG_TYPE:
        return CUSPARSE_SPMAT_DIAG_TYPE;
    default:
       ERROR_INVALID_ENUM();
    }
}
UPTKsparseSpMatAttribute_t cusparseSpMatAttributeToUPTKsparseSpMatAttribute(cusparseSpMatAttribute_t para)
{
    switch (para)
    {
     case CUSPARSE_SPMAT_FILL_MODE:
        return UPTKSPARSE_SPMAT_FILL_MODE;
    case CUSPARSE_SPMAT_DIAG_TYPE:
        return UPTKSPARSE_SPMAT_DIAG_TYPE;
    default:
       ERROR_INVALID_ENUM();
    }
}

cusparseSparseToDenseAlg_t UPTKsparseSparseToDenseAlgTocusparseSparseToDenseAlg(UPTKsparseSparseToDenseAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SPARSETODENSE_ALG_DEFAULT:
        return CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
    default:
        ERROR_INVALID_ENUM();
    }
}

cusparseDenseToSparseAlg_t UPTKsparseDenseToSparseAlgTocusparseDenseToSparseAlg(UPTKsparseDenseToSparseAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_DENSETOSPARSE_ALG_DEFAULT:
        return CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    default:
        ERROR_INVALID_ENUM();
    }
}

cusparseSpSVAlg_t UPTKsparseSpSVAlgTocusparseSpSVAlg(UPTKsparseSpSVAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SPSV_ALG_DEFAULT:
        return CUSPARSE_SPSV_ALG_DEFAULT;
    default:
        ERROR_INVALID_ENUM();
    }
}

cusparseSpSMAlg_t UPTKsparseSpSMAlgTocusparseSpSMAlg(UPTKsparseSpSMAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SPSM_ALG_DEFAULT:
        return CUSPARSE_SPSM_ALG_DEFAULT;
    default:
        ERROR_INVALID_ENUM();
    }
}

cusparseSpGEMMAlg_t UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(UPTKsparseSpGEMMAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SPGEMM_DEFAULT:
        return CUSPARSE_SPGEMM_DEFAULT;
    case UPTKSPARSE_SPGEMM_CSR_ALG_DETERMINITIC:
        return CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
    case UPTKSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC:
        return CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
    default:
       ERROR_INVALID_ENUM();
    }
}

cusparseSDDMMAlg_t UPTKsparseSDDMMAlgTocusparseSDDMMAlg(UPTKsparseSDDMMAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SDDMM_ALG_DEFAULT:
        return CUSPARSE_SDDMM_ALG_DEFAULT;
    default:
        ERROR_INVALID_ENUM();
    }
}

cusparseSpMMOpAlg_t UPTKsparseSpMMOpAlgTocusparseSpMMOpAlg(UPTKsparseSpMMOpAlg_t para)
{
    switch (para)
    {
    case UPTKSPARSE_SPMM_OP_ALG_DEFAULT:
        return CUSPARSE_SPMM_OP_ALG_DEFAULT;
    default:
        ERROR_INVALID_ENUM();
    }
}
#if defined(__cplusplus)
}
#endif /* __cplusplus */

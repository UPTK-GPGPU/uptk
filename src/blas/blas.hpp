#ifndef __BLAS_HPP__
#define __BLAS_HPP__

#include "../runtime/runtime.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cublas_v2.h>
#include <cublasXt.h>
#include <UPTK_blas.h>
#include <UPTK_blasXt.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * cublas status convert function
 */
cublasStatus_t UPTKblasStatusStatusTocublasStatus(UPTKblasStatus_t para);
UPTKblasStatus_t cublasStatusToUPTKblasStatus(cublasStatus_t para);

/**
 * cublas data type convert function
 */
cublasAtomicsMode_t UPTKblasAtomicsModeTocublasAtomicsMode(UPTKblasAtomicsMode_t para);
UPTKblasAtomicsMode_t cublasAtomicsModeToUPTKblasAtomicsMode(cublasAtomicsMode_t para);

cublasMath_t UPTKblasMathTocublasMath(UPTKblasMath_t para);
UPTKblasMath_t cublasMathToUPTKblasMath(cublasMath_t para);

cublasDiagType_t UPTKblasDiagTypeTocublasDiagType(UPTKblasDiagType_t para);
cublasFillMode_t UPTKblasFillModeTocublasFillMode(UPTKblasFillMode_t para);
cublasGemmAlgo_t UPTKblasGemmAlgoTocublasGemmAlgo(UPTKblasGemmAlgo_t para);

cublasOperation_t UPTKblasOperationTocublasOperation(UPTKblasOperation_t para);
UPTKblasOperation_t hipblasOperationTohcublasOperation(cublasOperation_t para);

cublasPointerMode_t UPTKblasPointerModeTocublasPointerMode(UPTKblasPointerMode_t para);
UPTKblasPointerMode_t cublasPointerModeToUPTKblasPointerMode(cublasPointerMode_t para);

cublasSideMode_t UPTKblasSideModeTocublasSideMode(UPTKblasSideMode_t para);

cudaDataType UPTKDataTypeTocudaDataType(UPTKDataType para);
UPTKDataType hipDataTypeTocudaDataType(cudaDataType para);

cublasComputeType_t UPTKblasComputeTypeTocublasComputeType(UPTKblasComputeType_t para);
UPTKblasComputeType_t hipblasComputeTypeModeTocublasComputeType(cublasComputeType_t para);


#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __BLAS_HPP__
#include "blas.hpp"

#if defined(__cplusplus)

extern "C"
{
#endif /* __cplusplus */
    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetVersion(UPTKblasHandle_t handle, int *version)
    {
        if (nullptr == handle)
        {
            return UPTKBLAS_STATUS_NOT_INITIALIZED;
        }

        if (nullptr == version)
        {
            return UPTKBLAS_STATUS_INVALID_VALUE;
        }

        *version = UPTKBLAS_VERSION;

        return UPTKBLAS_STATUS_SUCCESS;
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasAxpyEx(UPTKblasHandle_t handle, int n, const void *alpha, UPTKDataType alphaType, const void *x, UPTKDataType xType, int incx, void *y, UPTKDataType yType, int incy, UPTKDataType executiontype)
    {
        cudaDataType_t cuda_alphaType = UPTKDataTypeTocudaDataType(alphaType);
        cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
        cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
        cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executiontype);
        cublasStatus_t cuda_res;
        cuda_res = cublasAxpyEx((cublasHandle_t)handle, n, alpha, cuda_alphaType, x, cuda_xType, incx, y, cuda_yType, incy, cuda_executionType);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCaxpy(UPTKblasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCaxpy_v2((cublasHandle_t)handle, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCcopy(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCcopy_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
        cublasStatus_t cuda_res;
        cuda_res = cublasCdgmm((cublasHandle_t)handle, cuda_side, m, n, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCdotc(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCdotc_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCdotu(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCdotu_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgbmv_v2((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)beta, (const cuComplex *)B, ldb, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuComplex *alpha,
                                                      const cuComplex *const Aarray[], int lda, const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, Aarray, lda, Barray, ldb, (const cuComplex *)beta, Carray, ldc, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (long long)strideA, (const cuComplex *)B, ldb, (long long)strideB, (const cuComplex *)beta, (cuComplex *)C, ldc, (long long)strideC, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemm_v2((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemv_v2((cublasHandle_t)handle, cuda_trans, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgeqrfBatched(UPTKblasHandle_t handle, int m, int n, cuComplex *const Aarray[], int lda, cuComplex *const TauArray[], int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasCgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgerc(UPTKblasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCgerc_v2((cublasHandle_t)handle, m, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgeru(UPTKblasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCgeru_v2((cublasHandle_t)handle, m, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgetrfBatched(UPTKblasHandle_t handle, int n, cuComplex *const A[], int lda, int *P, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasCgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgetriBatched(UPTKblasHandle_t handle, int n, const cuComplex *const A[], int lda, const int *P, cuComplex *const C[], int ldc, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasCgetriBatched((cublasHandle_t)handle, n, (cuComplex *const *)A, lda, (int *)P, C, ldc, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const cuComplex *const Aarray[], int lda, const int *devIpiv, cuComplex *const Barray[], int ldb, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasCgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (cuComplex *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasChbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasChbmv_v2((cublasHandle_t)handle, cuda_uplo, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasChemm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasChemm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasChemv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasChemv_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCher2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCher2_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCher2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCher2k_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCher(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCher_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuComplex *)x, incx, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCherk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCherk_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuComplex *)A, lda, beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCherkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCherkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasChpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasChpmv_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)AP, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasChpr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasChpr2_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasChpr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasChpr_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuComplex *)x, incx, (cuComplex *)AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCreate(UPTKblasHandle_t *handle)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCreate_v2((cublasHandle_t *)handle);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCrot(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCrot_v2((cublasHandle_t)handle, n, (cuComplex *)x, incx, (cuComplex *)y, incy, c, (const cuComplex *)s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCrotg(UPTKblasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCrotg_v2((cublasHandle_t)handle, (cuComplex *)a, (cuComplex *)b, c, (cuComplex *)s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCscal(UPTKblasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCscal_v2((cublasHandle_t)handle, n, (const cuComplex *)alpha, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsrot(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCsrot_v2((cublasHandle_t)handle, n, (cuComplex *)x, incx, (cuComplex *)y, incy, c, s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsscal(UPTKblasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCsscal_v2((cublasHandle_t)handle, n, alpha, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCswap(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCswap_v2((cublasHandle_t)handle, n, (cuComplex *)x, incx, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsymm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsymv_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyr2_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyr2k_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyr_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyrk_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtbmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuComplex *)A, lda, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtbsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuComplex *)A, lda, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtpmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)AP, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtpsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)AP, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // FIXME: num of cuda paras not equal to cuda
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrmm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc)
    {
        return cublasStatusToUPTKblasStatus(cublasCtrmm_v2(
            (cublasHandle_t)handle,
            UPTKblasSideModeTocublasSideMode(side),
            UPTKblasFillModeTocublasFillMode(uplo),
            UPTKblasOperationTocublasOperation(trans),
            UPTKblasDiagTypeTocublasDiagType(diag),
            m,
            n,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtrmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)A, lda, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *const A[], int lda, cuComplex *const B[], int ldb, int batchCount)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuComplex *)alpha, const_cast<cuComplex *const *>(A), lda, const_cast<cuComplex **>(B), ldb, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtrsm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuComplex *)alpha, (cuComplex *)A, lda, (cuComplex *)B, ldb);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtrsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)A, lda, (cuComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDasum(UPTKblasHandle_t handle, int n, const double *x, int incx, double *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDasum_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDaxpy(UPTKblasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDaxpy_v2((cublasHandle_t)handle, n, alpha, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDcopy(UPTKblasHandle_t handle, int n, const double *x, int incx, double *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDcopy_v2((cublasHandle_t)handle, n, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
        cublasStatus_t cuda_res;
        cuda_res = cublasDdgmm((cublasHandle_t)handle, cuda_side, m, n, A, lda, x, incx, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDdot(UPTKblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDdot_v2((cublasHandle_t)handle, n, x, incx, y, incy, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDestroy(UPTKblasHandle_t handle)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDestroy_v2((cublasHandle_t)handle);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasDgbmv_v2((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasDgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasDgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasDgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, (long long)strideA, B, ldb, (long long)strideB, beta, C, ldc, (long long)strideC, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasDgemm_v2((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasDgemv_v2((cublasHandle_t)handle, cuda_trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgeqrfBatched(UPTKblasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasDgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDger(UPTKblasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDger_v2((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgetrfBatched(UPTKblasHandle_t handle, int n, double *const A[], int lda, int *P, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasDgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgetriBatched(UPTKblasHandle_t handle, int n, const double *const A[], int lda, const int *P, double *const C[], int ldc, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasDgetriBatched((cublasHandle_t)handle, n, (double *const *)A, lda, (int *)P, C, ldc, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasDgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (double *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDnrm2(UPTKblasHandle_t handle, int n, const double *x, int incx, double *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDnrm2_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDotEx(UPTKblasHandle_t handle, int n, const void *x, UPTKDataType xType, int incx, const void *y, UPTKDataType yType, int incy, void *result, UPTKDataType resultType, UPTKDataType executionType)
    {
        cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
        cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
        cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
        cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
        cublasStatus_t cuda_res;
        cuda_res = cublasDotEx((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, result, cuda_resultType, cuda_executionType);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDotcEx(UPTKblasHandle_t handle, int n, const void *x, UPTKDataType xType, int incx, const void *y, UPTKDataType yType, int incy, void *result, UPTKDataType resultType, UPTKDataType executionType)
    {
        cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
        cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
        cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
        cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
        cublasStatus_t cuda_res;
        cuda_res = cublasDotcEx((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, result, cuda_resultType, cuda_executionType);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDrot(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDrot_v2((cublasHandle_t)handle, n, x, incx, y, incy, c, s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDrotg(UPTKblasHandle_t handle, double *a, double *b, double *c, double *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDrotg_v2((cublasHandle_t)handle, a, b, c, s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDrotm(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *param)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDrotm_v2((cublasHandle_t)handle, n, x, incx, y, incy, param);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDrotmg(UPTKblasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDrotmg_v2((cublasHandle_t)handle, d1, d2, x1, y1, param);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsbmv_v2((cublasHandle_t)handle, cuda_uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDscal(UPTKblasHandle_t handle, int n, const double *alpha, double *x, int incx)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDscal_v2((cublasHandle_t)handle, n, alpha, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDspmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDspmv_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, AP, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDspr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDspr2_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDspr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDspr_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDswap(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDswap_v2((cublasHandle_t)handle, n, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsymm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsymv_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsyr2_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsyr2k_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsyr_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsyrk_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasDsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtbmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtbsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *AP, double *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtpmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *AP, double *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtpsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // FIXME: num of cuda paras not equal to cuda
    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrmm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc)
    {
        return cublasStatusToUPTKblasStatus(cublasDtrmm_v2(
            (cublasHandle_t)handle,
            UPTKblasSideModeTocublasSideMode(side),
            UPTKblasFillModeTocublasFillMode(uplo),
            UPTKblasOperationTocublasOperation(trans),
            UPTKblasDiagTypeTocublasDiagType(diag),
            m,
            n,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtrmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<double *const *>(A), lda, const_cast<double **>(B), ldb, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtrsm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<double *>(A), lda, B, ldb);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtrsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDzasum(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDzasum_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDznrm2(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDznrm2_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGemmBatchedEx(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], UPTKDataType Atype, int lda, const void *const Barray[], UPTKDataType Btype, int ldb, const void *beta, void *const Carray[], UPTKDataType Ctype, int ldc, int batchCount, UPTKblasComputeType_t computeType, UPTKblasGemmAlgo_t algo)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
        cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
        cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
        cublasStatus_t cuda_res;

        cuda_res = cublasGemmBatchedEx((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, cuda_aType, lda, Barray, cuda_bType, ldb, beta, Carray, cuda_cType, ldc, batchCount, cuda_computeType, cuda_algo);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGemmEx(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, UPTKDataType Atype, int lda, const void *B, UPTKDataType Btype, int ldb, const void *beta, void *C, UPTKDataType Ctype, int ldc, UPTKblasComputeType_t computeType, UPTKblasGemmAlgo_t algo)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
        cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
        cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
        cublasStatus_t cuda_res;

        cuda_res = cublasGemmEx((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, cuda_aType, lda, B, cuda_bType, ldb, beta, C, cuda_cType, ldc, cuda_computeType, cuda_algo);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGemmStridedBatchedEx(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, UPTKDataType Atype, int lda, long long int strideA, const void *B, UPTKDataType Btype, int ldb, long long int strideB, const void *beta, void *C, UPTKDataType Ctype, int ldc, long long int strideC, int batchCount, UPTKblasComputeType_t computeType, UPTKblasGemmAlgo_t algo)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
        cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
        cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
        cublasStatus_t cuda_res;

        cuda_res = cublasGemmStridedBatchedEx((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, cuda_aType, lda, strideA, B, cuda_bType, ldb, strideB, beta, C, cuda_cType, ldc, strideC, batchCount, cuda_computeType, cuda_algo);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t *mode)
    {
        if (nullptr == mode)
        {
            return UPTKBLAS_STATUS_INVALID_VALUE;
        }

        cublasAtomicsMode_t cudaMode;
        cublasStatus_t cudaStatus = cublasGetAtomicsMode((cublasHandle_t)handle, &cudaMode);

        if (CUBLAS_STATUS_SUCCESS == cudaStatus)
        {
            *mode = cublasAtomicsModeToUPTKblasAtomicsMode(cudaMode);
        }
        return cublasStatusToUPTKblasStatus(cudaStatus);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, UPTKStream_t stream)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, (cudaStream_t)stream);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t *mode)
    {
        if (nullptr == mode)
        {
            return UPTKBLAS_STATUS_INVALID_VALUE;
        }

        cublasPointerMode_t cuda_mode;
        cublasStatus_t cuda_res;
        cuda_res = cublasGetPointerMode_v2((cublasHandle_t)handle, &cuda_mode);

        if (CUBLAS_STATUS_SUCCESS == cuda_res)
        {
            *mode = cublasPointerModeToUPTKblasPointerMode(cuda_mode);
        }
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetStream(UPTKblasHandle_t handle, UPTKStream_t *streamId)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasGetStream_v2((cublasHandle_t)handle, (cudaStream_t *)streamId);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasGetVector(n, elemSize, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, UPTKStream_t stream)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, (cudaStream_t)stream);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasHgemm((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *)A, lda, (const __half *)B, ldb, (const __half *)beta, (__half *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *const Aarray[], int lda, const __half *const Barray[], int ldb, const __half *beta, __half *const Carray[], int ldc, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasHgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *const *)Aarray, lda, (const __half *const *)Barray, ldb, (const __half *)beta, (__half *const *)Carray, ldc, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasHgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *)A, lda, (long long)strideA, (const __half *)B, ldb, (long long)strideB, (const __half *)beta, (__half *)C, ldc, (long long)strideC, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIcamax(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIcamax_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIcamin(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIcamin_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIdamax(UPTKblasHandle_t handle, int n, const double *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIdamax_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIdamin(UPTKblasHandle_t handle, int n, const double *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIdamin_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIsamax(UPTKblasHandle_t handle, int n, const float *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIsamax_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIsamin(UPTKblasHandle_t handle, int n, const float *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIsamin_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIzamax(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIzamax_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasIzamin(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasIzamin_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasNrm2Ex(UPTKblasHandle_t handle, int n, const void *x, UPTKDataType xType, int incx, void *result, UPTKDataType resultType, UPTKDataType executionType)
    {
        cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
        cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
        cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
        cublasStatus_t cuda_res;
        cuda_res = cublasNrm2Ex((cublasHandle_t)handle, n, x, cuda_xType, incx, result, cuda_resultType, cuda_executionType);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasRotEx(UPTKblasHandle_t handle, int n, void *x, UPTKDataType xType, int incx, void *y, UPTKDataType yType, int incy, const void *c, const void *s, UPTKDataType csType, UPTKDataType executiontype)
    {
        cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
        cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
        cudaDataType_t cuda_csType = UPTKDataTypeTocudaDataType(csType);
        cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executiontype);
        cublasStatus_t cuda_res;
        cuda_res = cublasRotEx((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, c, s, cuda_csType, cuda_executionType);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSasum(UPTKblasHandle_t handle, int n, const float *x, int incx, float *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSasum_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSaxpy(UPTKblasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSaxpy_v2((cublasHandle_t)handle, n, alpha, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasScalEx(UPTKblasHandle_t handle, int n, const void *alpha, UPTKDataType alphaType, void *x, UPTKDataType xType, int incx, UPTKDataType executionType)
    {
        cudaDataType_t cuda_alphaType = UPTKDataTypeTocudaDataType(alphaType);
        cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
        cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
        cublasStatus_t cuda_res;
        cuda_res = cublasScalEx((cublasHandle_t)handle, n, alpha, cuda_alphaType, x, cuda_xType, incx, cuda_executionType);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasScasum(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, float *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasScasum_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasScnrm2(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, float *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasScnrm2_v2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasScopy(UPTKblasHandle_t handle, int n, const float *x, int incx, float *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasScopy_v2((cublasHandle_t)handle, n, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
        cublasStatus_t cuda_res;
        cuda_res = cublasSdgmm((cublasHandle_t)handle, cuda_side, m, n, A, lda, x, incx, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSdot(UPTKblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSdot_v2((cublasHandle_t)handle, n, x, incx, y, incy, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t mode)
    {
        return cublasStatusToUPTKblasStatus(cublasSetAtomicsMode((cublasHandle_t)handle, UPTKblasAtomicsModeTocublasAtomicsMode(mode)));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t *mode)
    {
        if (nullptr == mode)
        {
            return UPTKBLAS_STATUS_INVALID_VALUE;
        }

        cublasStatus_t cuda_res;
        cublasMath_t cudaMode;

        cuda_res = cublasGetMathMode((cublasHandle_t)handle, &cudaMode);

        if (CUBLAS_STATUS_SUCCESS == cuda_res)
        {
            *mode = cublasMathToUPTKblasMath(cudaMode);
        }

        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t mode)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetMathMode((cublasHandle_t)handle, UPTKblasMathTocublasMath(mode));
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, UPTKStream_t stream)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, (cudaStream_t)stream);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t mode)
    {
        cublasPointerMode_t cuda_mode = UPTKblasPointerModeTocublasPointerMode(mode);
        cublasStatus_t cuda_res;
        cuda_res = cublasSetPointerMode_v2((cublasHandle_t)handle, cuda_mode);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetStream(UPTKblasHandle_t handle, UPTKStream_t streamId)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetStream_v2((cublasHandle_t)handle, (cudaStream_t)streamId);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetVector(n, elemSize, x, incx, devicePtr, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, UPTKStream_t stream)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, (cudaStream_t)stream);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgbmv_v2((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, (long long)strideA, B, ldb, (long long)strideB, beta, C, ldc, (long long)strideC, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgemm_v2((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgemv_v2((cublasHandle_t)handle, cuda_trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgeqrfBatched(UPTKblasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasSgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSger(UPTKblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSger_v2((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgetrfBatched(UPTKblasHandle_t handle, int n, float *const A[], int lda, int *P, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasSgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgetriBatched(UPTKblasHandle_t handle, int n, const float *const A[], int lda, const int *P, float *const C[], int ldc, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasSgetriBatched((cublasHandle_t)handle, n, (float *const *)A, lda, (int *)P, C, ldc, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasSgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (float *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSnrm2(UPTKblasHandle_t handle, int n, const float *x, int incx, float *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSnrm2_v2((cublasHandle_t)handle, n, x, incx, result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSrot(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSrot_v2((cublasHandle_t)handle, n, x, incx, y, incy, c, s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSrotg(UPTKblasHandle_t handle, float *a, float *b, float *c, float *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSrotg_v2((cublasHandle_t)handle, a, b, c, s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSrotm(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSrotm_v2((cublasHandle_t)handle, n, x, incx, y, incy, param);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSrotmg(UPTKblasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSrotmg_v2((cublasHandle_t)handle, d1, d2, x1, y1, param);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsbmv_v2((cublasHandle_t)handle, cuda_uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSscal(UPTKblasHandle_t handle, int n, const float *alpha, float *x, int incx)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSscal_v2((cublasHandle_t)handle, n, alpha, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSspmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSspmv_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, AP, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSspr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSspr2_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSspr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSspr_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSswap(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSswap_v2((cublasHandle_t)handle, n, x, incx, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsymm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsymv_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsyr2_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsyr2k_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsyr_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsyrk_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasSsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStbmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStbsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *AP, float *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStpmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *AP, float *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStpsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStrmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<float *const *>(A), lda, const_cast<float **>(B), ldb, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStrsm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<float *>(A), lda, B, ldb);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // FIXME: cuda have 14 arguments, but cuda have only 12 arguments
    UPTKblasStatus_t UPTKBLASAPI UPTKblasStrmm(UPTKblasHandle_t handle,
                                               UPTKblasSideMode_t side,
                                               UPTKblasFillMode_t uplo,
                                               UPTKblasOperation_t trans,
                                               UPTKblasDiagType_t diag,
                                               int m,
                                               int n,
                                               const float *alpha,
                                               const float *A,
                                               int lda,
                                               const float *B,
                                               int ldb,
                                               float *C,
                                               int ldc)
    {
        return cublasStatusToUPTKblasStatus(cublasStrmm_v2(
            (cublasHandle_t)handle,
            UPTKblasSideModeTocublasSideMode(side),
            UPTKblasFillModeTocublasFillMode(uplo),
            UPTKblasOperationTocublasOperation(trans),
            UPTKblasDiagTypeTocublasDiagType(diag),
            m,
            n,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasStrsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZaxpy(UPTKblasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZaxpy_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZcopy(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZcopy_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
        cublasStatus_t cuda_res;
        cuda_res = cublasZdgmm((cublasHandle_t)handle, cuda_side, m, n, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZdotc(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZdotc_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZdotu(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZdotu_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)result);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZdrot(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZdrot_v2((cublasHandle_t)handle, n, (cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy, c, s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZdscal(UPTKblasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZdscal_v2((cublasHandle_t)handle, n, alpha, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgbmv_v2((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)beta, (const cuDoubleComplex *)B, ldb, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda, const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, Aarray, lda, Barray, ldb, (const cuDoubleComplex *)beta, Carray, ldc, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (long long)strideA, (const cuDoubleComplex *)B, ldb, (long long)strideB, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc, (long long)strideC, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgemm_v2((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgemv_v2((cublasHandle_t)handle, cuda_trans, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgeqrfBatched(UPTKblasHandle_t handle, int m, int n, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const TauArray[], int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasZgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgerc(UPTKblasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZgerc_v2((cublasHandle_t)handle, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgeru(UPTKblasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZgeru_v2((cublasHandle_t)handle, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgetrfBatched(UPTKblasHandle_t handle, int n, cuDoubleComplex *const A[], int lda, int *P, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasZgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgetriBatched(UPTKblasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda, const int *P, cuDoubleComplex *const C[], int ldc, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasZgetriBatched((cublasHandle_t)handle, n, (cuDoubleComplex *const *)A, lda, (int *)P, C, ldc, info, batchSize));
    }

    // UNSUPPORTED
    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const cuDoubleComplex *const Aarray[], int lda, const int *devIpiv, cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasZgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (cuDoubleComplex *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZhbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZhbmv_v2((cublasHandle_t)handle, cuda_uplo, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZhemm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZhemm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZhemv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZhemv_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZher2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZher2_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZher2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZher2k_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZher(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZher_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZherk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZherk_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuDoubleComplex *)A, lda, beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZherkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZherkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZhpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZhpmv_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)AP, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZhpr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZhpr2_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZhpr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZhpr_v2((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZrot(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZrot_v2((cublasHandle_t)handle, n, (cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy, c, (const cuDoubleComplex *)s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZrotg(UPTKblasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZrotg_v2((cublasHandle_t)handle, (cuDoubleComplex *)a, (cuDoubleComplex *)b, c, (cuDoubleComplex *)s);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZscal(UPTKblasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZscal_v2((cublasHandle_t)handle, n, (const cuDoubleComplex *)alpha, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZswap(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZswap_v2((cublasHandle_t)handle, n, (cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsymm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsymv_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsyr2_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsyr2k_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsyr_v2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsyrk_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasZsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtbmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtbsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtpmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtpsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // FIXME: num of cuda paras not equal to cuda
    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrmm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc)
    {
        return cublasStatusToUPTKblasStatus(cublasZtrmm_v2(
            (cublasHandle_t)handle,
            UPTKblasSideModeTocublasSideMode(side),
            UPTKblasFillModeTocublasFillMode(uplo),
            UPTKblasOperationTocublasOperation(trans),
            UPTKblasDiagTypeTocublasDiagType(diag),
            m,
            n,
            alpha,
            A,
            lda,
            B,
            ldb,
            C,
            ldc));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtrmv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const B[], int ldb, int batchCount)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuDoubleComplex *)alpha, const_cast<cuDoubleComplex *const *>(A), lda, const_cast<cuDoubleComplex **>(B), ldb, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)
    {
        cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtrsm_v2((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuDoubleComplex *)alpha, (cuDoubleComplex *)A, lda, (cuDoubleComplex *)B, ldb);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtrsv_v2((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasGetProperty(libraryPropertyType type, int *value)
    {
        if (NULL == value)
        {
            return UPTKBLAS_STATUS_INVALID_VALUE;
        }

        switch (type)
        {
        case MAJOR_VERSION:
            *value = UPTKBLAS_VER_MAJOR;
            break;
        case MINOR_VERSION:
            *value = UPTKBLAS_VER_MINOR;
            break;
        case PATCH_LEVEL:
            *value = UPTKBLAS_VER_PATCH;
            break;
        default:
            return UPTKBLAS_STATUS_INVALID_VALUE;
        }

        return UPTKBLAS_STATUS_SUCCESS;
    }

    size_t UPTKBLASAPI UPTKblasGetCudartVersion(void)
    {
        return UPTKRT_VERSION;
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemm3m(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t transa,
                                                 UPTKblasOperation_t transb,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const cuComplex *alpha,
                                                 const cuComplex *A,
                                                 int lda,
                                                 const cuComplex *B,
                                                 int ldb,
                                                 const cuComplex *beta,
                                                 cuComplex *C,
                                                 int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemm3m((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemm3m(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t transa,
                                                 UPTKblasOperation_t transb,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const cuDoubleComplex *alpha,
                                                 const cuDoubleComplex *A,
                                                 int lda,
                                                 const cuDoubleComplex *B,
                                                 int ldb,
                                                 const cuDoubleComplex *beta,
                                                 cuDoubleComplex *C,
                                                 int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasZgemm3m((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemmEx(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t transa,
                                                 UPTKblasOperation_t transb,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const float *alpha,
                                                 const void *A,
                                                 UPTKDataType Atype,
                                                 int lda,
                                                 const void *B,
                                                 UPTKDataType Btype,
                                                 int ldb,
                                                 const float *beta,
                                                 void *C,
                                                 UPTKDataType Ctype,
                                                 int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cudaDataType_t cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType_t cuda_bType = UPTKDataTypeTocudaDataType(Btype);
        cudaDataType_t cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasStatus_t cuda_res;
        cuda_res = cublasSgemmEx((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, cuda_aType, lda, B, cuda_bType, ldb, beta, C, cuda_cType, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemmEx(UPTKblasHandle_t handle,
                                                 UPTKblasOperation_t transa,
                                                 UPTKblasOperation_t transb,
                                                 int m, int n, int k,
                                                 const cuComplex *alpha,
                                                 const void *A,
                                                 UPTKDataType Atype,
                                                 int lda,
                                                 const void *B,
                                                 UPTKDataType Btype,
                                                 int ldb,
                                                 const cuComplex *beta,
                                                 void *C,
                                                 UPTKDataType Ctype,
                                                 int ldc)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cudaDataType_t cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType_t cuda_bType = UPTKDataTypeTocudaDataType(Btype);
        cudaDataType_t cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemmEx((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, A, cuda_aType, lda, B, cuda_bType, ldb, beta, C, cuda_cType, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // diff para cuda : cuda
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrkEx(UPTKblasHandle_t handle,
                                                 UPTKblasFillMode_t uplo,
                                                 UPTKblasOperation_t trans,
                                                 int n,
                                                 int k,
                                                 const cuComplex *alpha,
                                                 const void *A,
                                                 UPTKDataType Atype,
                                                 int lda,
                                                 const cuComplex *beta,
                                                 void *C,
                                                 UPTKDataType Ctype,
                                                 int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cudaDataType_t cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType_t cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyrkEx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (cuComplex *)alpha, A, cuda_aType, lda, (cuComplex *)beta, (cuComplex *)C, cuda_cType, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    // diff para cuda : cuda
    UPTKblasStatus_t UPTKBLASAPI UPTKblasCsyrk3mEx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex *alpha,
                                                   const void *A,
                                                   UPTKDataType Atype,
                                                   int lda,
                                                   const cuComplex *beta,
                                                   void *C,
                                                   UPTKDataType Ctype,
                                                   int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cudaDataType_t cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType_t cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasStatus_t cuda_res;
        cuda_res = cublasCsyrk3mEx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (cuComplex *)alpha, A, cuda_aType, lda, (cuComplex *)beta, (cuComplex *)C, cuda_cType, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCherkEx(UPTKblasHandle_t handle,
                                                 UPTKblasFillMode_t uplo,
                                                 UPTKblasOperation_t trans,
                                                 int n,
                                                 int k,
                                                 const float *alpha,
                                                 const void *A,
                                                 UPTKDataType Atype,
                                                 int lda,
                                                 const float *beta,
                                                 void *C,
                                                 UPTKDataType Ctype,
                                                 int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cudaDataType_t cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType_t cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasStatus_t cuda_res;
        cuda_res = cublasCherkEx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuComplex *)A, cuda_aType, lda, beta, (cuComplex *)C, cuda_cType, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCherk3mEx(UPTKblasHandle_t handle,
                                                   UPTKblasFillMode_t uplo,
                                                   UPTKblasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const float *alpha,
                                                   const void *A, UPTKDataType Atype,
                                                   int lda,
                                                   const float *beta,
                                                   void *C,
                                                   UPTKDataType Ctype,
                                                   int ldc)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
        cudaDataType_t cuda_aType = UPTKDataTypeTocudaDataType(Atype);
        cudaDataType_t cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
        cublasStatus_t cuda_res;
        cuda_res = cublasCherk3mEx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuComplex *)A, cuda_aType, lda, beta, (cuComplex *)C, cuda_cType, ldc);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemm3mStridedBatched(UPTKblasHandle_t handle,
                                                               UPTKblasOperation_t transa,
                                                               UPTKblasOperation_t transb,
                                                               int m,
                                                               int n,
                                                               int k,
                                                               const cuComplex *alpha,
                                                               const cuComplex *A,
                                                               int lda,
                                                               long long int strideA,
                                                               const cuComplex *B,
                                                               int ldb,
                                                               long long int strideB,
                                                               const cuComplex *beta,
                                                               cuComplex *C,
                                                               int ldc,
                                                               long long int strideC,
                                                               int batchCount)
    {
        cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
        cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
        cublasStatus_t cuda_res;
        cuda_res = cublasCgemm3mStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (long long)strideA, (const cuComplex *)B, ldb, (long long)strideB, (const cuComplex *)beta, (cuComplex *)C, ldc, (long long)strideC, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSmatinvBatched(UPTKblasHandle_t handle,
                                                        int n,
                                                        const float *const A[],
                                                        int lda,
                                                        float *const Ainv[],
                                                        int lda_inv,
                                                        int *info,
                                                        int batchSize)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSmatinvBatched((cublasHandle_t)handle, n, A, lda, Ainv, lda_inv, info, batchSize);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDmatinvBatched(UPTKblasHandle_t handle,
                                                        int n,
                                                        const double *const A[],
                                                        int lda,
                                                        double *const Ainv[],
                                                        int lda_inv,
                                                        int *info,
                                                        int batchSize)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasDmatinvBatched((cublasHandle_t)handle, n, A, lda, Ainv, lda_inv, info, batchSize);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCmatinvBatched(UPTKblasHandle_t handle,
                                                        int n,
                                                        const cuComplex *const A[],
                                                        int lda,
                                                        cuComplex *const Ainv[],
                                                        int lda_inv,
                                                        int *info,
                                                        int batchSize)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasCmatinvBatched((cublasHandle_t)handle, n, (const cuComplex *const *)A, lda, (cuComplex *const *)Ainv, lda_inv, info, batchSize);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZmatinvBatched(UPTKblasHandle_t handle,
                                                        int n,
                                                        const cuDoubleComplex *const A[],
                                                        int lda,
                                                        cuDoubleComplex *const Ainv[],
                                                        int lda_inv,
                                                        int *info,
                                                        int batchSize)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasZmatinvBatched((cublasHandle_t)handle, n, (const cuDoubleComplex *const *)A, lda, (cuDoubleComplex *const *)Ainv, lda_inv, info, batchSize);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgelsBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int nrhs,
                                                      float *const Aarray[],
                                                      int lda,
                                                      float *const Carray[],
                                                      int ldc,
                                                      int *info,
                                                      int *devInfoArray,
                                                      int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasSgelsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgelsBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int nrhs,
                                                      double *const Aarray[],
                                                      int lda,
                                                      double *const Carray[],
                                                      int ldc,
                                                      int *info,
                                                      int *devInfoArray,
                                                      int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasDgelsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgelsBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int nrhs,
                                                      cuComplex *const Aarray[],
                                                      int lda,
                                                      cuComplex *const Carray[],
                                                      int ldc,
                                                      int *info,
                                                      int *devInfoArray,
                                                      int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasCgelsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgelsBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      int nrhs,
                                                      cuDoubleComplex *const Aarray[],
                                                      int lda,
                                                      cuDoubleComplex *const Carray[],
                                                      int ldc,
                                                      int *info,
                                                      int *devInfoArray,
                                                      int batchSize)
    {
        return cublasStatusToUPTKblasStatus(cublasZgelsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStpttr(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const float *AP,
                                                float *A,
                                                int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasStpttr((cublasHandle_t)handle, cuda_uplo, n, AP, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtpttr(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const double *AP,
                                                double *A,
                                                int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtpttr((cublasHandle_t)handle, cuda_uplo, n, AP, A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtpttr(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const cuComplex *AP,
                                                cuComplex *A,
                                                int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtpttr((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)AP, (cuComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtpttr(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const cuDoubleComplex *AP,
                                                cuDoubleComplex *A,
                                                int lda)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtpttr((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)A, lda);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasStrttp(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const float *A,
                                                int lda,
                                                float *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasStrttp((cublasHandle_t)handle, cuda_uplo, n, A, lda, AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDtrttp(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const double *A,
                                                int lda,
                                                double *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasDtrttp((cublasHandle_t)handle, cuda_uplo, n, A, lda, AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCtrttp(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const cuComplex *A,
                                                int lda,
                                                cuComplex *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasCtrttp((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)A, lda, (cuComplex *)AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZtrttp(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int n,
                                                const cuDoubleComplex *A,
                                                int lda,
                                                cuDoubleComplex *AP)
    {
        cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
        cublasStatus_t cuda_res;
        cuda_res = cublasZtrttp((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)AP);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    const char *UPTKBLASAPI UPTKblasGetStatusName(UPTKblasStatus_t status)
    {
        switch (status)
        {
        case UPTKBLAS_STATUS_SUCCESS:
            return "UPTKBLAS_STATUS_SUCCESS";
        case UPTKBLAS_STATUS_NOT_INITIALIZED:
            return "UPTKBLAS_STATUS_NOT_INITIALIZED";
        case UPTKBLAS_STATUS_ALLOC_FAILED:
            return "UPTKBLAS_STATUS_ALLOC_FAILED";
        case UPTKBLAS_STATUS_INVALID_VALUE:
            return "UPTKBLAS_STATUS_INVALID_VALUE";
        case UPTKBLAS_STATUS_ARCH_MISMATCH:
            return "UPTKBLAS_STATUS_ARCH_MISMATCH";
        case UPTKBLAS_STATUS_MAPPING_ERROR:
            return "UPTKBLAS_STATUS_MAPPING_ERROR";
        case UPTKBLAS_STATUS_EXECUTION_FAILED:
            return "UPTKBLAS_STATUS_EXECUTION_FAILED";
        case UPTKBLAS_STATUS_INTERNAL_ERROR:
            return "UPTKBLAS_STATUS_INTERNAL_ERROR";
        case UPTKBLAS_STATUS_NOT_SUPPORTED:
            return "UPTKBLAS_STATUS_NOT_SUPPORTED";
        case UPTKBLAS_STATUS_LICENSE_ERROR:
            return "UPTKBLAS_STATUS_LICENSE_ERROR";
        default:
            break;
        }

        return "unrecognized status";
    }

    const char *UPTKBLASAPI UPTKblasGetStatusString(UPTKblasStatus_t status)
    {
        switch (status)
        {
        case UPTKBLAS_STATUS_SUCCESS:
            return "success";
        case UPTKBLAS_STATUS_NOT_INITIALIZED:
            return "the library was not initialized";
        case UPTKBLAS_STATUS_ALLOC_FAILED:
            return "the resource allocation failed";
        case UPTKBLAS_STATUS_INVALID_VALUE:
            return "an unsupported value or parameter was passed to the function";
        case UPTKBLAS_STATUS_ARCH_MISMATCH:
            return "the function requires an architectural feature absent from the device";
        case UPTKBLAS_STATUS_MAPPING_ERROR:
            return "an access to GPU memory space failed";
        case UPTKBLAS_STATUS_EXECUTION_FAILED:
            return "the function failed to launch on the GPU";
        case UPTKBLAS_STATUS_INTERNAL_ERROR:
            return "an internal operation failed";
        case UPTKBLAS_STATUS_NOT_SUPPORTED:
            return "the requested functionality is not supported";
        case UPTKBLAS_STATUS_LICENSE_ERROR:
            return "the license check failed";
        default:
            break;
        }

        return "unrecognized status";
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemvBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const float *alpha,
                                                      const float *const Aarray[],
                                                      int lda,
                                                      const float *const xarray[],
                                                      int incx,
                                                      const float *beta,
                                                      float *const yarray[],
                                                      int incy,
                                                      int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasSgemvBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemvBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const double *alpha,
                                                      const double *const Aarray[],
                                                      int lda,
                                                      const double *const xarray[],
                                                      int incx,
                                                      const double *beta,
                                                      double *const yarray[],
                                                      int incy,
                                                      int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasDgemvBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemvBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *const Aarray[],
                                                      int lda,
                                                      const cuComplex *const xarray[],
                                                      int incx,
                                                      const cuComplex *beta,
                                                      cuComplex *const yarray[],
                                                      int incy,
                                                      int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasCgemvBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemvBatched(UPTKblasHandle_t handle,
                                                      UPTKblasOperation_t trans,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *const Aarray[],
                                                      int lda,
                                                      const cuDoubleComplex *const xarray[],
                                                      int incx,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *const yarray[],
                                                      int incy,
                                                      int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasZgemvBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                               m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHSHgemvBatched(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int m,
                                                        int n,
                                                        const float *alpha,
                                                        const __half *const Aarray[],
                                                        int lda,
                                                        const __half *const xarray[],
                                                        int incx,
                                                        const float *beta,
                                                        __half *const yarray[],
                                                        int incy,
                                                        int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasHSHgemvBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *const *)Aarray, lda, (const __half *const *)xarray, incx, beta, (__half *const *)yarray, incy, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHSSgemvBatched(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int m,
                                                        int n,
                                                        const float *alpha,
                                                        const __half *const Aarray[],
                                                        int lda,
                                                        const __half *const xarray[],
                                                        int incx,
                                                        const float *beta,
                                                        float *const yarray[],
                                                        int incy,
                                                        int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasHSSgemvBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *const *)Aarray, lda, (const __half *const *)xarray, incx, beta, yarray, incy, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasTSTgemvBatched(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int m,
                                                        int n,
                                                        const float *alpha,
                                                        const __nv_bfloat16 *const Aarray[],
                                                        int lda,
                                                        const __nv_bfloat16 *const xarray[],
                                                        int incx,
                                                        const float *beta,
                                                        __nv_bfloat16 *const yarray[],
                                                        int incy,
                                                        int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasTSTgemvBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *const *)Aarray, lda, (const __nv_bfloat16 *const *)xarray, incx, beta, (__nv_bfloat16 *const *)yarray, incy, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasTSSgemvBatched(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int m,
                                                        int n,
                                                        const float *alpha,
                                                        const __nv_bfloat16 *const Aarray[],
                                                        int lda,
                                                        const __nv_bfloat16 *const xarray[],
                                                        int incx,
                                                        const float *beta,
                                                        float *const yarray[],
                                                        int incy,
                                                        int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasTSSgemvBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *const *)Aarray, lda, (const __nv_bfloat16 *const *)xarray, incx, beta, yarray, incy, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSgemvStridedBatched(UPTKblasHandle_t handle,
                                                             UPTKblasOperation_t trans,
                                                             int m,
                                                             int n,
                                                             const float *alpha,
                                                             const float *A,
                                                             int lda,
                                                             long long int strideA,
                                                             const float *x,
                                                             int incx,
                                                             long long int stridex,
                                                             const float *beta,
                                                             float *y,
                                                             int incy,
                                                             long long int stridey,
                                                             int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasSgemvStridedBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                      m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasDgemvStridedBatched(UPTKblasHandle_t handle,
                                                             UPTKblasOperation_t trans,
                                                             int m,
                                                             int n,
                                                             const double *alpha,
                                                             const double *A,
                                                             int lda,
                                                             long long int strideA,
                                                             const double *x,
                                                             int incx,
                                                             long long int stridex,
                                                             const double *beta,
                                                             double *y,
                                                             int incy,
                                                             long long int stridey,
                                                             int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasDgemvStridedBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                      m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasCgemvStridedBatched(UPTKblasHandle_t handle,
                                                             UPTKblasOperation_t trans,
                                                             int m,
                                                             int n,
                                                             const cuComplex *alpha,
                                                             const cuComplex *A,
                                                             int lda,
                                                             long long int strideA,
                                                             const cuComplex *x,
                                                             int incx,
                                                             long long int stridex,
                                                             const cuComplex *beta,
                                                             cuComplex *y,
                                                             int incy,
                                                             long long int stridey,
                                                             int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasCgemvStridedBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                      m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasZgemvStridedBatched(UPTKblasHandle_t handle,
                                                             UPTKblasOperation_t trans,
                                                             int m,
                                                             int n,
                                                             const cuDoubleComplex *alpha,
                                                             const cuDoubleComplex *A,
                                                             int lda,
                                                             long long int strideA,
                                                             const cuDoubleComplex *x,
                                                             int incx,
                                                             long long int stridex,
                                                             const cuDoubleComplex *beta,
                                                             cuDoubleComplex *y,
                                                             int incy,
                                                             long long int stridey,
                                                             int batchCount)
    {
        return cublasStatusToUPTKblasStatus(cublasZgemvStridedBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                      m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHSHgemvStridedBatched(UPTKblasHandle_t handle,
                                                               UPTKblasOperation_t trans,
                                                               int m,
                                                               int n,
                                                               const float *alpha,
                                                               const __half *A,
                                                               int lda,
                                                               long long int strideA,
                                                               const __half *x,
                                                               int incx,
                                                               long long int stridex,
                                                               const float *beta,
                                                               __half *y,
                                                               int incy,
                                                               long long int stridey,
                                                               int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasHSHgemvStridedBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *)A, lda, strideA, (const __half *)x, incx, stridex, beta, (__half *)y, incy, stridey, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasHSSgemvStridedBatched(UPTKblasHandle_t handle,
                                                               UPTKblasOperation_t trans,
                                                               int m,
                                                               int n,
                                                               const float *alpha,
                                                               const __half *A,
                                                               int lda,
                                                               long long int strideA,
                                                               const __half *x,
                                                               int incx,
                                                               long long int stridex,
                                                               const float *beta,
                                                               float *y,
                                                               int incy,
                                                               long long int stridey,
                                                               int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasHSSgemvStridedBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *)A, lda, strideA, (const __half *)x, incx, stridex, beta, y, incy, stridey, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasTSTgemvStridedBatched(UPTKblasHandle_t handle,
                                                               UPTKblasOperation_t trans,
                                                               int m,
                                                               int n,
                                                               const float *alpha,
                                                               const __nv_bfloat16 *A,
                                                               int lda,
                                                               long long int strideA,
                                                               const __nv_bfloat16 *x,
                                                               int incx,
                                                               long long int stridex,
                                                               const float *beta,
                                                               __nv_bfloat16 *y,
                                                               int incy,
                                                               long long int stridey,
                                                               int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasTSTgemvStridedBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *)A, lda, strideA, (const __nv_bfloat16 *)x, incx, stridex, beta, (__nv_bfloat16 *)y, incy, stridey, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasTSSgemvStridedBatched(UPTKblasHandle_t handle,
                                                               UPTKblasOperation_t trans,
                                                               int m,
                                                               int n,
                                                               const float *alpha,
                                                               const __nv_bfloat16 *A,
                                                               int lda,
                                                               long long int strideA,
                                                               const __nv_bfloat16 *x,
                                                               int incx,
                                                               long long int stridex,
                                                               const float *beta,
                                                               float *y,
                                                               int incy,
                                                               long long int stridey,
                                                               int batchCount)
    {
        cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
        cublasStatus_t cuda_res;
        cuda_res = cublasTSSgemvStridedBatched((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *)A, lda, strideA, (const __nv_bfloat16 *)x, incx, stridex, beta, y, incy, stridey, batchCount);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

    UPTKblasStatus_t UPTKBLASAPI UPTKblasSetWorkspace(UPTKblasHandle_t handle,
                                                      void *workspace,
                                                      size_t workspaceSizeInBytes)
    {
        cublasStatus_t cuda_res;
        cuda_res = cublasSetWorkspace((cublasHandle_t)handle, workspace, workspaceSizeInBytes);
        return cublasStatusToUPTKblasStatus(cuda_res);
    }

#if defined(__cplusplus)
}
#endif /* __cplusplus */
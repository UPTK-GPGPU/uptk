#include "blas.hpp"

#if defined(__cplusplus)

extern "C"
{
#endif /* __cplusplus */

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVersion(UPTKblasHandle_t handle, int *version)
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasAxpyEx(UPTKblasHandle_t handle, int n, const void *alpha, UPTKDataType_t alphaType, const void *x, UPTKDataType_t xType, int incx, void *y, UPTKDataType_t yType, int incy, UPTKDataType_t executiontype)
{
    cudaDataType_t cuda_alphaType = UPTKDataTypeTocudaDataType(alphaType);
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executiontype);
    cublasStatus_t cuda_res;
    cuda_res = cublasAxpyEx((cublasHandle_t)handle, n, alpha, cuda_alphaType, x, cuda_xType, incx, y, cuda_yType, incy, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCaxpy(UPTKblasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCaxpy((cublasHandle_t)handle, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCcopy(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCcopy((cublasHandle_t)handle, n, (const cuComplex *)x, incx, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasCdgmm((cublasHandle_t)handle, cuda_side, m, n, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotc(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCdotc((cublasHandle_t)handle, n, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotu(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCdotu((cublasHandle_t)handle, n, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgbmv((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)beta, (const cuComplex *)B, ldb, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuComplex *alpha,
                                                    const cuComplex *const Aarray[], int lda, const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, Aarray, lda, Barray, ldb, (const cuComplex *)beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (long long)strideA, (const cuComplex *)B, ldb, (long long)strideB, (const cuComplex *)beta, (cuComplex *)C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemm((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemv((cublasHandle_t)handle, cuda_trans, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeqrfBatched(UPTKblasHandle_t handle, int m, int n, cuComplex *const Aarray[], int lda, cuComplex *const TauArray[], int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasCgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgerc(UPTKblasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCgerc((cublasHandle_t)handle, m, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeru(UPTKblasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCgeru((cublasHandle_t)handle, m, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasCgetrfBatched(UPTKblasHandle_t handle, int n, cuComplex *const A[], int lda, int *P, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasCgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasCgetriBatched(UPTKblasHandle_t handle, int n, const cuComplex *const A[], int lda, const int *P, cuComplex *const C[], int ldc, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasCgetriBatched((cublasHandle_t)handle, n, (cuComplex *const *)A, lda, (int *)P, C, ldc, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasCgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const cuComplex *const Aarray[], int lda, const int *devIpiv, cuComplex *const Barray[], int ldb, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasCgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (cuComplex *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChbmv((cublasHandle_t)handle, cuda_uplo, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChemm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChemm((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChemv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChemv((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCher2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCher2k((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCher(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCher((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuComplex *)x, incx, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCherk((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuComplex *)A, lda, beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCherkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChpmv((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)AP, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChpr2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChpr((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuComplex *)x, incx, (cuComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCreate(UPTKblasHandle_t *handle)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCreate((cublasHandle_t *)handle);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCrot(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCrot((cublasHandle_t)handle, n, (cuComplex *)x, incx, (cuComplex *)y, incy, c, (const cuComplex *)s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCrotg(UPTKblasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCrotg((cublasHandle_t)handle, (cuComplex *)a, (cuComplex *)b, c, (cuComplex *)s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCscal(UPTKblasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCscal((cublasHandle_t)handle, n, (const cuComplex *)alpha, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsrot(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCsrot((cublasHandle_t)handle, n, (cuComplex *)x, incx, (cuComplex *)y, incy, c, s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsscal(UPTKblasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCsscal((cublasHandle_t)handle, n, alpha, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCswap(UPTKblasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCswap((cublasHandle_t)handle, n, (cuComplex *)x, incx, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsymm((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsymv((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyr2((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyr2k((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyr((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyrk((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtbmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtbsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtpmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)AP, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtpsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)AP, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// FIXME: num of cuda paras not equal to cuda
UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCtrmm(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *const A[], int lda, cuComplex *const B[], int ldb, int batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuComplex *)alpha, const_cast<cuComplex *const *>(A), lda, const_cast<cuComplex **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrsm((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuComplex *)alpha, (cuComplex *)A, lda, (cuComplex *)B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDasum(UPTKblasHandle_t handle, int n, const double *x, int incx, double *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDasum((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDaxpy(UPTKblasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDcopy(UPTKblasHandle_t handle, int n, const double *x, int incx, double *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDcopy((cublasHandle_t)handle, n, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasDdgmm((cublasHandle_t)handle, cuda_side, m, n, A, lda, x, incx, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDdot(UPTKblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDdot((cublasHandle_t)handle, n, x, incx, y, incy, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDestroy(UPTKblasHandle_t handle)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDestroy((cublasHandle_t)handle);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgbmv((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, (long long)strideA, B, ldb, (long long)strideB, beta, C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemm((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemv((cublasHandle_t)handle, cuda_trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasDgeqrfBatched(UPTKblasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasDgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDger(UPTKblasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDger((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasDgetrfBatched(UPTKblasHandle_t handle, int n, double *const A[], int lda, int *P, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasDgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasDgetriBatched(UPTKblasHandle_t handle, int n, const double *const A[], int lda, const int *P, double *const C[], int ldc, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasDgetriBatched((cublasHandle_t)handle, n, (double *const *)A, lda, (int *)P, C, ldc, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasDgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasDgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (double *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDnrm2(UPTKblasHandle_t handle, int n, const double *x, int incx, double *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDnrm2((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDotEx(UPTKblasHandle_t handle, int n, const void *x, UPTKDataType_t xType, int incx, const void *y, UPTKDataType_t yType, int incy, void *result, UPTKDataType_t resultType, UPTKDataType_t executionType)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasDotEx((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, result, cuda_resultType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDotcEx(UPTKblasHandle_t handle, int n, const void *x, UPTKDataType_t xType, int incx, const void *y, UPTKDataType_t yType, int incy, void *result, UPTKDataType_t resultType, UPTKDataType_t executionType)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasDotcEx((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, result, cuda_resultType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDrot(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDrot((cublasHandle_t)handle, n, x, incx, y, incy, c, s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDrotg(UPTKblasHandle_t handle, double *a, double *b, double *c, double *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDrotg((cublasHandle_t)handle, a, b, c, s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDrotm(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *param)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDrotm((cublasHandle_t)handle, n, x, incx, y, incy, param);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDrotmg(UPTKblasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDrotmg((cublasHandle_t)handle, d1, d2, x1, y1, param);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsbmv((cublasHandle_t)handle, cuda_uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDscal(UPTKblasHandle_t handle, int n, const double *alpha, double *x, int incx)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDscal((cublasHandle_t)handle, n, alpha, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDspmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDspmv((cublasHandle_t)handle, cuda_uplo, n, alpha, AP, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDspr2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDspr((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDswap(UPTKblasHandle_t handle, int n, double *x, int incx, double *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDswap((cublasHandle_t)handle, n, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsymm((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsymv((cublasHandle_t)handle, cuda_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyr2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyr2k((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyr((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyrk((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtbmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtbsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *AP, double *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtpmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *AP, double *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtpsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// FIXME: num of cuda paras not equal to cuda
UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc)
{
    return cublasStatusToUPTKblasStatus(cublasDtrmm(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<double *const *>(A), lda, const_cast<double **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrsm((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<double *>(A), lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDzasum(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDzasum((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDznrm2(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDznrm2((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmBatchedEx(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], UPTKDataType_t Atype, int lda, const void *const Barray[], UPTKDataType_t Btype, int ldb, const void *beta, void *const Carray[], UPTKDataType_t Ctype, int ldc, int batchCount, UPTKblasComputeType_t computeType, UPTKblasGemmAlgo_t algo)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
    cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
    cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
    cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
    cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
    cublasStatus_t cuda_res;

    cuda_res = cublasGemmBatchedEx((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, (const void **)Aarray, cuda_aType, lda, (const void **)Barray, cuda_bType, ldb, beta, (void **)Carray, cuda_cType, ldc, batchCount, cuda_computeType, cuda_algo);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmEx(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, UPTKDataType_t Atype, int lda, const void *B, UPTKDataType_t Btype, int ldb, const void *beta, void *C, UPTKDataType_t Ctype, int ldc, UPTKblasComputeType_t computeType, UPTKblasGemmAlgo_t algo)
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmStridedBatchedEx(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, UPTKDataType_t Atype, int lda, long long int strideA, const void *B, UPTKDataType_t Btype, int ldb, long long int strideB, const void *beta, void *C, UPTKDataType_t Ctype, int ldc, long long int strideC, int batchCount, UPTKblasComputeType_t computeType, UPTKblasGemmAlgo_t algo)
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
UPTKBLASAPI UPTKblasStatus_t UPTKblasGetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t *mode)
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t *mode)
{
    if (nullptr == mode)
    {
        return UPTKBLAS_STATUS_INVALID_VALUE;
    }

    cublasPointerMode_t cuda_mode;
    cublasStatus_t cuda_res;
    cuda_res = cublasGetPointerMode((cublasHandle_t)handle, &cuda_mode);

    if (CUBLAS_STATUS_SUCCESS == cuda_res)
    {
        *mode = cublasPointerModeToUPTKblasPointerMode(cuda_mode);
    }
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetStream(UPTKblasHandle_t handle, cudaStream_t *streamId)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetStream((cublasHandle_t)handle, (cudaStream_t *)streamId);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetVector(n, elemSize, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasHgemm((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *)A, lda, (const __half *)B, ldb, (const __half *)beta, (__half *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *const Aarray[], int lda, const __half *const Barray[], int ldb, const __half *beta, __half *const Carray[], int ldc, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasHgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *const *)Aarray, lda, (const __half *const *)Barray, ldb, (const __half *)beta, (__half *const *)Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasHgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *)A, lda, (long long)strideA, (const __half *)B, ldb, (long long)strideB, (const __half *)beta, (__half *)C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIcamax(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIcamax((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIcamin(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIcamin((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIdamax(UPTKblasHandle_t handle, int n, const double *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIdamax((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIdamin(UPTKblasHandle_t handle, int n, const double *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIdamin((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIsamax(UPTKblasHandle_t handle, int n, const float *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIsamax((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIsamin(UPTKblasHandle_t handle, int n, const float *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIsamin((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIzamax(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIzamax((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIzamin(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasIzamin((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasNrm2Ex(UPTKblasHandle_t handle, int n, const void *x, UPTKDataType_t xType, int incx, void *result, UPTKDataType_t resultType, UPTKDataType_t executionType)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasNrm2Ex((cublasHandle_t)handle, n, x, cuda_xType, incx, result, cuda_resultType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasRotEx(UPTKblasHandle_t handle, int n, void *x, UPTKDataType_t xType, int incx, void *y, UPTKDataType_t yType, int incy, const void *c, const void *s, UPTKDataType_t csType, UPTKDataType_t executiontype)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_csType = UPTKDataTypeTocudaDataType(csType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executiontype);
    cublasStatus_t cuda_res;
    cuda_res = cublasRotEx((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, c, s, cuda_csType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSasum(UPTKblasHandle_t handle, int n, const float *x, int incx, float *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSasum((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSaxpy(UPTKblasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasScalEx(UPTKblasHandle_t handle, int n, const void *alpha, UPTKDataType_t alphaType, void *x, UPTKDataType_t xType, int incx, UPTKDataType_t executionType)
{
    cudaDataType_t cuda_alphaType = UPTKDataTypeTocudaDataType(alphaType);
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasScalEx((cublasHandle_t)handle, n, alpha, cuda_alphaType, x, cuda_xType, incx, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasScasum(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, float *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasScasum((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasScnrm2(UPTKblasHandle_t handle, int n, const cuComplex *x, int incx, float *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasScnrm2((cublasHandle_t)handle, n, (const cuComplex *)x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasScopy(UPTKblasHandle_t handle, int n, const float *x, int incx, float *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasScopy((cublasHandle_t)handle, n, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasSdgmm((cublasHandle_t)handle, cuda_side, m, n, A, lda, x, incx, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSdot(UPTKblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSdot((cublasHandle_t)handle, n, x, incx, y, incy, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasSetAtomicsMode(UPTKblasHandle_t handle, UPTKblasAtomicsMode_t mode)
{
    return cublasStatusToUPTKblasStatus(cublasSetAtomicsMode((cublasHandle_t)handle, UPTKblasAtomicsModeTocublasAtomicsMode(mode)));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t *mode)
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetMathMode(UPTKblasHandle_t handle, UPTKblasMath_t mode)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetMathMode((cublasHandle_t)handle, UPTKblasMathTocublasMath(mode));
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetPointerMode(UPTKblasHandle_t handle, UPTKblasPointerMode_t mode)
{
    cublasPointerMode_t cuda_mode = UPTKblasPointerModeTocublasPointerMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasSetPointerMode((cublasHandle_t)handle, cuda_mode);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetStream(UPTKblasHandle_t handle, cudaStream_t streamId)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetStream((cublasHandle_t)handle, (cudaStream_t)streamId);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetVector(n, elemSize, x, incx, devicePtr, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgbmv((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, (long long)strideA, B, ldb, (long long)strideB, beta, C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemm((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemv((cublasHandle_t)handle, cuda_trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasSgeqrfBatched(UPTKblasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasSgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSger(UPTKblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSger((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasSgetrfBatched(UPTKblasHandle_t handle, int n, float *const A[], int lda, int *P, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasSgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasSgetriBatched(UPTKblasHandle_t handle, int n, const float *const A[], int lda, const int *P, float *const C[], int ldc, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasSgetriBatched((cublasHandle_t)handle, n, (float *const *)A, lda, (int *)P, C, ldc, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasSgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasSgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (float *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSnrm2(UPTKblasHandle_t handle, int n, const float *x, int incx, float *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSnrm2((cublasHandle_t)handle, n, x, incx, result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSrot(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSrot((cublasHandle_t)handle, n, x, incx, y, incy, c, s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSrotg(UPTKblasHandle_t handle, float *a, float *b, float *c, float *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSrotg((cublasHandle_t)handle, a, b, c, s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSrotm(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSrotm((cublasHandle_t)handle, n, x, incx, y, incy, param);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSrotmg(UPTKblasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSrotmg((cublasHandle_t)handle, d1, d2, x1, y1, param);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsbmv((cublasHandle_t)handle, cuda_uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSscal(UPTKblasHandle_t handle, int n, const float *alpha, float *x, int incx)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSscal((cublasHandle_t)handle, n, alpha, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSspmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSspmv((cublasHandle_t)handle, cuda_uplo, n, alpha, AP, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSspr2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSspr((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSswap(UPTKblasHandle_t handle, int n, float *x, int incx, float *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSswap((cublasHandle_t)handle, n, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsymm((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsymv((cublasHandle_t)handle, cuda_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyr2((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyr2k((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyr((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyrk((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStbmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStbsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *AP, float *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStpmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *AP, float *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStpsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<float *const *>(A), lda, const_cast<float **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrsm((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<float *>(A), lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// FIXME: cuda have 14 arguments, but cuda have only 12 arguments
UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmm(UPTKblasHandle_t handle,
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
    return cublasStatusToUPTKblasStatus(cublasStrmm(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZaxpy(UPTKblasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZaxpy((cublasHandle_t)handle, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZcopy(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZcopy((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdgmm(UPTKblasHandle_t handle, UPTKblasSideMode_t mode, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasZdgmm((cublasHandle_t)handle, cuda_side, m, n, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotc(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZdotc((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotu(UPTKblasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZdotu((cublasHandle_t)handle, n, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)result);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdrot(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZdrot((cublasHandle_t)handle, n, (cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy, c, s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdscal(UPTKblasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZdscal((cublasHandle_t)handle, n, alpha, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgbmv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgbmv((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeam(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgeam((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)beta, (const cuDoubleComplex *)B, ldb, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda, const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemmBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, Aarray, lda, Barray, ldb, (const cuDoubleComplex *)beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmStridedBatched(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemmStridedBatched((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (long long)strideA, (const cuDoubleComplex *)B, ldb, (long long)strideB, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm(UPTKblasHandle_t handle, UPTKblasOperation_t transa, UPTKblasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemm((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemv(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemv((cublasHandle_t)handle, cuda_trans, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeqrfBatched(UPTKblasHandle_t handle, int m, int n, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const TauArray[], int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasZgeqrfBatched((cublasHandle_t)handle, m, n, Aarray, lda, TauArray, info, batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgerc(UPTKblasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZgerc((cublasHandle_t)handle, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeru(UPTKblasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZgeru((cublasHandle_t)handle, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasZgetrfBatched(UPTKblasHandle_t handle, int n, cuDoubleComplex *const A[], int lda, int *P, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasZgetrfBatched((cublasHandle_t)handle, n, A, lda, P, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasZgetriBatched(UPTKblasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda, const int *P, cuDoubleComplex *const C[], int ldc, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasZgetriBatched((cublasHandle_t)handle, n, (cuDoubleComplex *const *)A, lda, (int *)P, C, ldc, info, batchSize));
}

// UNSUPPORTED
UPTKBLASAPI UPTKblasStatus_t UPTKblasZgetrsBatched(UPTKblasHandle_t handle, UPTKblasOperation_t trans, int n, int nrhs, const cuDoubleComplex *const Aarray[], int lda, const int *devIpiv, cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize)
{
    return cublasStatusToUPTKblasStatus(cublasZgetrsBatched((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans), n, nrhs, (cuDoubleComplex *const *)Aarray, lda, devIpiv, Barray, ldb, info, (const int)batchSize));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhbmv((cublasHandle_t)handle, cuda_uplo, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhemm((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhemv((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZher2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZher2k((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZher(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZher((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZherk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZherk((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuDoubleComplex *)A, lda, beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZherkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZherkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhpmv((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)AP, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhpr2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhpr((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZrot(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZrot((cublasHandle_t)handle, n, (cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy, c, (const cuDoubleComplex *)s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZrotg(UPTKblasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZrotg((cublasHandle_t)handle, (cuDoubleComplex *)a, (cuDoubleComplex *)b, c, (cuDoubleComplex *)s);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZscal(UPTKblasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZscal((cublasHandle_t)handle, n, (const cuDoubleComplex *)alpha, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZswap(UPTKblasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZswap((cublasHandle_t)handle, n, (cuDoubleComplex *)x, incx, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsymm((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsymv((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyr2((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2k(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyr2k((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyr((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrk(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyrk((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrkx(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyrkx((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtbmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtbsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtpmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtpsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// FIXME: num of cuda paras not equal to cuda
UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc)
{
    return cublasStatusToUPTKblasStatus(cublasZtrmm(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsmBatched(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const B[], int ldb, int batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrsmBatched((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuDoubleComplex *)alpha, const_cast<cuDoubleComplex *const *>(A), lda, const_cast<cuDoubleComplex **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsm(UPTKblasHandle_t handle, UPTKblasSideMode_t side, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrsm((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuDoubleComplex *)alpha, (cuDoubleComplex *)A, lda, (cuDoubleComplex *)B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsv(UPTKblasHandle_t handle, UPTKblasFillMode_t uplo, UPTKblasOperation_t trans, UPTKblasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrsv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetProperty(libraryPropertyType type, int *value)
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

UPTKBLASAPI size_t UPTKblasGetCudartVersion(void)
{
    return UPTKRT_VERSION;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3m(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm3m(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmEx(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int m,
                                                int n,
                                                int k,
                                                const float *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int lda,
                                                const void *B,
                                                UPTKDataType_t Btype,
                                                int ldb,
                                                const float *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmEx(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int m, int n, int k,
                                                const cuComplex *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int lda,
                                                const void *B,
                                                UPTKDataType_t Btype,
                                                int ldb,
                                                const cuComplex *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
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
UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkEx(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int n,
                                                int k,
                                                const cuComplex *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int lda,
                                                const cuComplex *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
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
UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk3mEx(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int n,
                                                int k,
                                                const cuComplex *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int lda,
                                                const cuComplex *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkEx(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int n,
                                                int k,
                                                const float *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int lda,
                                                const float *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk3mEx(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int n,
                                                int k,
                                                const float *alpha,
                                                const void *A, UPTKDataType_t Atype,
                                                int lda,
                                                const float *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSmatinvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDmatinvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCmatinvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZmatinvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgelsBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgelsBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgelsBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgelsBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasStpttr(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpttr(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpttr(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpttr(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrttp(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrttp(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrttp(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrttp(UPTKblasHandle_t handle,
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

UPTKBLASAPI const char *UPTKblasGetStatusName(UPTKblasStatus_t status)
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

UPTKBLASAPI const char *UPTKblasGetStatusString(UPTKblasStatus_t status)
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvStridedBatched(UPTKblasHandle_t handle,
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetWorkspace(UPTKblasHandle_t handle,
                                                    void *workspace,
                                                    size_t workspaceSizeInBytes)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetWorkspace((cublasHandle_t)handle, workspace, workspaceSizeInBytes);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// ########################## << int64 api >> ##########################
UPTKblasStatus_t
UPTKblasSetVector_64(int64_t n, int64_t elemSize, const void *x, int64_t incx, void *devicePtr, int64_t incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetVector_64(n, elemSize, x, incx, devicePtr, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t
UPTKblasGetVector_64(int64_t n, int64_t elemSize, const void *x, int64_t incx, void *y, int64_t incy)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetVector_64(n, elemSize, x, incx, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t
UPTKblasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetMatrix_64(rows, cols, elemSize, A, lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t
UPTKblasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetMatrix_64(rows, cols, elemSize, A, lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t UPTKblasSetVectorAsync_64(
    int64_t n, int64_t elemSize, const void *hostPtr, int64_t incx, void *devicePtr, int64_t incy, cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t UPTKblasGetVectorAsync_64(
    int64_t n, int64_t elemSize, const void *devicePtr, int64_t incx, void *hostPtr, int64_t incy, cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetVectorAsync_64(n, elemSize, devicePtr, incx, hostPtr, incy, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t UPTKblasSetMatrixAsync_64(int64_t rows,
                                            int64_t cols,
                                            int64_t elemSize,
                                            const void *A,
                                            int64_t lda,
                                            void *B,
                                            int64_t ldb,
                                            cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKblasStatus_t UPTKblasGetMatrixAsync_64(int64_t rows,
                                            int64_t cols,
                                            int64_t elemSize,
                                            const void *A,
                                            int64_t lda,
                                            void *B,
                                            int64_t ldb,
                                            cudaStream_t stream)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasGetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, (cudaStream_t)stream);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasNrm2Ex_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *result,
                                                UPTKDataType_t resultType,
                                                UPTKDataType_t executionType)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasNrm2Ex_64((cublasHandle_t)handle, n, x, cuda_xType, incx, result, cuda_resultType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasSnrm2_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, float *result)
{
    return cublasStatusToUPTKblasStatus(cublasSnrm2_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDnrm2_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, double *result)
{
    return cublasStatusToUPTKblasStatus(cublasDnrm2_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasScnrm2_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, float *result)
{
    return cublasStatusToUPTKblasStatus(cublasScnrm2_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDznrm2_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, double *result)
{
    return cublasStatusToUPTKblasStatus(cublasDznrm2_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDotEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                const void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy,
                                                void *result,
                                                UPTKDataType_t resultType,
                                                UPTKDataType_t executionType)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasDotEx_64((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, result, cuda_resultType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDotcEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                const void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy,
                                                void *result,
                                                UPTKDataType_t resultType,
                                                UPTKDataType_t executionType)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_resultType = UPTKDataTypeTocudaDataType(resultType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasDotcEx_64((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, result, cuda_resultType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSdot_64(
    UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, const float *y, int64_t incy, float *result)
{
    return cublasStatusToUPTKblasStatus(cublasSdot_64((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDdot_64(
    UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, const double *y, int64_t incy, double *result)
{
    return cublasStatusToUPTKblasStatus(cublasDdot_64((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotu_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *result)
{
    return cublasStatusToUPTKblasStatus(cublasCdotu_64((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCdotc_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *result)
{
    return cublasStatusToUPTKblasStatus(cublasCdotc_64((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotu_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *result)
{
    return cublasStatusToUPTKblasStatus(cublasZdotu_64((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdotc_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *result)
{
    return cublasStatusToUPTKblasStatus(cublasZdotc_64((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasScalEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *alpha,
                                                UPTKDataType_t alphaType,
                                                void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                UPTKDataType_t executionType)
{
    cudaDataType_t cuda_alphaType = UPTKDataTypeTocudaDataType(alphaType);
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executionType);
    cublasStatus_t cuda_res;
    cuda_res = cublasScalEx_64((cublasHandle_t)handle, n, alpha, cuda_alphaType, x, cuda_xType, incx, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasSscal_64(UPTKblasHandle_t handle, int64_t n, const float *alpha, float *x, int64_t incx)
{
    return cublasStatusToUPTKblasStatus(cublasSscal_64((cublasHandle_t)handle, n, alpha, x, incx));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDscal_64(UPTKblasHandle_t handle, int64_t n, const double *alpha, double *x, int64_t incx)
{
    return cublasStatusToUPTKblasStatus(cublasDscal_64((cublasHandle_t)handle, n, alpha, x, incx));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasCscal_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *alpha, cuComplex *x, int64_t incx)
{
    return cublasStatusToUPTKblasStatus(cublasCscal_64((cublasHandle_t)handle, n, alpha, x, incx));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasCsscal_64(UPTKblasHandle_t handle, int64_t n, const float *alpha, cuComplex *x, int64_t incx)
{
    return cublasStatusToUPTKblasStatus(cublasCsscal_64((cublasHandle_t)handle, n, alpha, x, incx));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasZscal_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int64_t incx)
{
    return cublasStatusToUPTKblasStatus(cublasZscal_64((cublasHandle_t)handle, n, alpha, x, incx));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasZdscal_64(UPTKblasHandle_t handle, int64_t n, const double *alpha, cuDoubleComplex *x, int64_t incx)
{
    return cublasStatusToUPTKblasStatus(cublasZdscal_64((cublasHandle_t)handle, n, alpha, x, incx));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasAxpyEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *alpha,
                                                UPTKDataType_t alphaType,
                                                const void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy,
                                                UPTKDataType_t executiontype)
{
    cudaDataType_t cuda_alphaType = UPTKDataTypeTocudaDataType(alphaType);
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executiontype);
    cublasStatus_t cuda_res;
    cuda_res = cublasAxpyEx_64((cublasHandle_t)handle, n, alpha, cuda_alphaType, x, cuda_xType, incx, y, cuda_yType, incy, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
UPTKBLASAPI UPTKblasStatus_t UPTKblasSaxpy_64(
    UPTKblasHandle_t handle, int64_t n, const float *alpha, const float *x, int64_t incx, float *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasSaxpy_64((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDaxpy_64(
    UPTKblasHandle_t handle, int64_t n, const double *alpha, const double *x, int64_t incx, double *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasDaxpy_64((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCaxpy_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                cuComplex *y,
                                                int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasCaxpy_64((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZaxpy_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasZaxpy_64((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasScopy_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, float *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasScopy_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDcopy_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, double *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasDcopy_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasCcopy_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, cuComplex *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasCcopy_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZcopy_64(
    UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasZcopy_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasSswap_64(UPTKblasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasSswap_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDswap_64(UPTKblasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasDswap_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasCswap_64(UPTKblasHandle_t handle, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasCswap_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasZswap_64(UPTKblasHandle_t handle, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy)
{
    return cublasStatusToUPTKblasStatus(cublasZswap_64((cublasHandle_t)handle, n, x, incx, y, incy));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIsamax_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIsamax_64((cublasHandle_t)handle, n, x, incx, result));
}
UPTKBLASAPI UPTKblasStatus_t
UPTKblasIdamax_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIdamax_64((cublasHandle_t)handle, n, x, incx, result));
}
UPTKBLASAPI UPTKblasStatus_t
UPTKblasIcamax_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIcamax_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIzamax_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIzamax_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIsamin_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIsamin_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIdamin_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIdamin_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIcamin_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIcamin_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIzamin_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, int64_t *result)
{
    return cublasStatusToUPTKblasStatus(cublasIzamin_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasSasum_64(UPTKblasHandle_t handle, int64_t n, const float *x, int64_t incx, float *result)
{
    return cublasStatusToUPTKblasStatus(cublasSasum_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDasum_64(UPTKblasHandle_t handle, int64_t n, const double *x, int64_t incx, double *result)
{
    return cublasStatusToUPTKblasStatus(cublasDasum_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasScasum_64(UPTKblasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, float *result)
{
    return cublasStatusToUPTKblasStatus(cublasScasum_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasDzasum_64(UPTKblasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, double *result)
{
    return cublasStatusToUPTKblasStatus(cublasDzasum_64((cublasHandle_t)handle, n, x, incx, result));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSrot_64(
    UPTKblasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy, const float *c, const float *s)
{
    return cublasStatusToUPTKblasStatus(cublasSrot_64((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDrot_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                double *x,
                                                int64_t incx,
                                                double *y,
                                                int64_t incy,
                                                const double *c,
                                                const double *s)
{
    return cublasStatusToUPTKblasStatus(cublasDrot_64((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCrot_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                cuComplex *x,
                                                int64_t incx,
                                                cuComplex *y,
                                                int64_t incy,
                                                const float *c,
                                                const cuComplex *s)
{
    return cublasStatusToUPTKblasStatus(cublasCrot_64((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsrot_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                cuComplex *x,
                                                int64_t incx,
                                                cuComplex *y,
                                                int64_t incy,
                                                const float *c,
                                                const float *s)
{
    return cublasStatusToUPTKblasStatus(cublasCsrot_64((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZrot_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *y,
                                                int64_t incy,
                                                const double *c,
                                                const cuDoubleComplex *s)
{
    return cublasStatusToUPTKblasStatus(cublasZrot_64((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdrot_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *y,
                                                int64_t incy,
                                                const double *c,
                                                const double *s)
{
    return cublasStatusToUPTKblasStatus(cublasZdrot_64((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasRotEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy,
                                                const void *c,
                                                const void *s,
                                                UPTKDataType_t csType,
                                                UPTKDataType_t executiontype)
{
    cudaDataType_t cuda_xType = UPTKDataTypeTocudaDataType(xType);
    cudaDataType_t cuda_yType = UPTKDataTypeTocudaDataType(yType);
    cudaDataType_t cuda_csType = UPTKDataTypeTocudaDataType(csType);
    cudaDataType_t cuda_executionType = UPTKDataTypeTocudaDataType(executiontype);
    cublasStatus_t cuda_res;
    cuda_res = cublasRotEx_64((cublasHandle_t)handle, n, x, cuda_xType, incx, y, cuda_yType, incy, c, s, cuda_csType, cuda_executionType);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasSrotm_64(UPTKblasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy, const float *param)
{
    return cublasStatusToUPTKblasStatus(cublasSrotm_64((cublasHandle_t)handle, n, x, incx, y, incy, param));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDrotm_64(
    UPTKblasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy, const double *param)
{
    return cublasStatusToUPTKblasStatus(cublasDrotm_64((cublasHandle_t)handle, n, x, incx, y, incy, param));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *x,
                                                int64_t incx,
                                                const float *beta,
                                                float *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemv_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *x,
                                                int64_t incx,
                                                const double *beta,
                                                double *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemv_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *beta,
                                                cuComplex *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemv_64((cublasHandle_t)handle, cuda_trans, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemv_64((cublasHandle_t)handle, cuda_trans, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                int64_t kl,
                                                int64_t ku,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *x,
                                                int64_t incx,
                                                const float *beta,
                                                float *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgbmv_64((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                int64_t kl,
                                                int64_t ku,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *x,
                                                int64_t incx,
                                                const double *beta,
                                                double *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgbmv_64((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                int64_t kl,
                                                int64_t ku,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *beta,
                                                cuComplex *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgbmv_64((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t trans,
                                                int64_t m,
                                                int64_t n,
                                                int64_t kl,
                                                int64_t ku,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgbmv_64((cublasHandle_t)handle, cuda_trans, m, n, kl, ku, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const float *A,
                                                int64_t lda,
                                                float *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const double *A,
                                                int64_t lda,
                                                double *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuComplex *A,
                                                int64_t lda,
                                                cuComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrmv((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                cuDoubleComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const float *A,
                                                int64_t lda,
                                                float *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStbmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const double *A,
                                                int64_t lda,
                                                double *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtbmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *A,
                                                int64_t lda,
                                                cuComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtbmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                cuDoubleComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtbmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStpmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const float *AP,
                                                float *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStpmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const double *AP,
                                                double *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtpmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuComplex *AP,
                                                cuComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtpmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)AP, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuDoubleComplex *AP,
                                                cuDoubleComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtpmv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const float *A,
                                                int64_t lda,
                                                float *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const double *A,
                                                int64_t lda,
                                                double *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuComplex *A,
                                                int64_t lda,
                                                cuComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                cuDoubleComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStpsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const float *AP,
                                                float *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStpsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtpsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const double *AP,
                                                double *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtpsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, AP, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtpsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuComplex *AP,
                                                cuComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtpsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuComplex *)AP, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtpsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                const cuDoubleComplex *AP,
                                                cuDoubleComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtpsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, (const cuDoubleComplex *)AP, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStbsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const float *A,
                                                int64_t lda,
                                                float *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStbsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtbsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const double *A,
                                                int64_t lda,
                                                double *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtbsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, A, lda, x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtbsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *A,
                                                int64_t lda,
                                                cuComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtbsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuComplex *)A, lda, (cuComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtbsv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                cuDoubleComplex *x,
                                                int64_t incx)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtbsv_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, cuda_diag, n, k, (const cuDoubleComplex *)A, lda, (cuDoubleComplex *)x, incx);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *x,
                                                int64_t incx,
                                                const float *beta,
                                                float *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsymv_64((cublasHandle_t)handle, cuda_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *x,
                                                int64_t incx,
                                                const double *beta,
                                                double *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsymv_64((cublasHandle_t)handle, cuda_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *beta,
                                                cuComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsymv_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsymv_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChemv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *beta,
                                                cuComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChemv_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhemv_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *x,
                                                int64_t incx,
                                                const float *beta,
                                                float *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsbmv_64((cublasHandle_t)handle, cuda_uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                int64_t k,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *x,
                                                int64_t incx,
                                                const double *beta,
                                                double *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsbmv_64((cublasHandle_t)handle, cuda_uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *beta,
                                                cuComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChbmv_64((cublasHandle_t)handle, cuda_uplo, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhbmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhbmv_64((cublasHandle_t)handle, cuda_uplo, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSspmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const float *AP,
                                                const float *x,
                                                int64_t incx,
                                                const float *beta,
                                                float *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSspmv_64((cublasHandle_t)handle, cuda_uplo, n, alpha, AP, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDspmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const double *AP,
                                                const double *x,
                                                int64_t incx,
                                                const double *beta,
                                                double *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDspmv_64((cublasHandle_t)handle, cuda_uplo, n, alpha, AP, x, incx, beta, y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChpmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *AP,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *beta,
                                                cuComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChpmv_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)AP, (const cuComplex *)x, incx, (const cuComplex *)beta, (cuComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpmv_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *AP,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *y,
                                                int64_t incy)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhpmv_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)AP, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)beta, (cuDoubleComplex *)y, incy);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSger_64(UPTKblasHandle_t handle,
                                                int64_t m,
                                                int64_t n,
                                                const float *alpha,
                                                const float *x,
                                                int64_t incx,
                                                const float *y,
                                                int64_t incy,
                                                float *A,
                                                int64_t lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasSger_64((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDger_64(UPTKblasHandle_t handle,
                                                int64_t m,
                                                int64_t n,
                                                const double *alpha,
                                                const double *x,
                                                int64_t incx,
                                                const double *y,
                                                int64_t incy,
                                                double *A,
                                                int64_t lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasDger_64((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeru_64(UPTKblasHandle_t handle,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *A,
                                                int64_t lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCgeru_64((cublasHandle_t)handle, m, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgerc_64(UPTKblasHandle_t handle,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *A,
                                                int64_t lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasCgerc_64((cublasHandle_t)handle, m, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeru_64(UPTKblasHandle_t handle,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *A,
                                                int64_t lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZgeru_64((cublasHandle_t)handle, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgerc_64(UPTKblasHandle_t handle,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *A,
                                                int64_t lda)
{
    cublasStatus_t cuda_res;
    cuda_res = cublasZgerc_64((cublasHandle_t)handle, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const float *x,
                                                int64_t incx,
                                                float *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyr_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const double *x,
                                                int64_t incx,
                                                double *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyr_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                cuComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyr_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyr_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCher_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                cuComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCher_64((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuComplex *)x, incx, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZher_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZher_64((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const float *x,
                                                int64_t incx,
                                                float *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSspr_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const double *x,
                                                int64_t incx,
                                                double *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDspr_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                cuComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChpr_64((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuComplex *)x, incx, (cuComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhpr_64((cublasHandle_t)handle, cuda_uplo, n, alpha, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const float *x,
                                                int64_t incx,
                                                const float *y,
                                                int64_t incy,
                                                float *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyr2_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const double *x,
                                                int64_t incx,
                                                const double *y,
                                                int64_t incy,
                                                double *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyr2_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyr2_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyr2_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCher2_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *A,
                                                int64_t lda)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZher2_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)A, lda);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSspr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const float *alpha,
                                                const float *x,
                                                int64_t incx,
                                                const float *y,
                                                int64_t incy,
                                                float *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSspr2_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDspr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const double *alpha,
                                                const double *x,
                                                int64_t incx,
                                                const double *y,
                                                int64_t incy,
                                                double *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDspr2_64((cublasHandle_t)handle, cuda_uplo, n, alpha, x, incx, y, incy, AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChpr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *x,
                                                int64_t incx,
                                                const cuComplex *y,
                                                int64_t incy,
                                                cuComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChpr2_64((cublasHandle_t)handle, cuda_uplo, n, (const cuComplex *)alpha, (const cuComplex *)x, incx, (const cuComplex *)y, incy, (cuComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhpr2_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                const cuDoubleComplex *y,
                                                int64_t incy,
                                                cuDoubleComplex *AP)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhpr2_64((cublasHandle_t)handle, cuda_uplo, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)x, incx, (const cuDoubleComplex *)y, incy, (cuDoubleComplex *)AP);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const float *alpha,
                                                        const float *const Aarray[],
                                                        int64_t lda,
                                                        const float *const xarray[],
                                                        int64_t incx,
                                                        const float *beta,
                                                        float *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasSgemvBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const double *alpha,
                                                        const double *const Aarray[],
                                                        int64_t lda,
                                                        const double *const xarray[],
                                                        int64_t incx,
                                                        const double *beta,
                                                        double *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasDgemvBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const cuComplex *alpha,
                                                        const cuComplex *const Aarray[],
                                                        int64_t lda,
                                                        const cuComplex *const xarray[],
                                                        int64_t incx,
                                                        const cuComplex *beta,
                                                        cuComplex *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasCgemvBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const cuDoubleComplex *alpha,
                                                        const cuDoubleComplex *const Aarray[],
                                                        int64_t lda,
                                                        const cuDoubleComplex *const xarray[],
                                                        int64_t incx,
                                                        const cuDoubleComplex *beta,
                                                        cuDoubleComplex *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasZgemvBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const float *alpha,
                                                        const __half *const Aarray[],
                                                        int64_t lda,
                                                        const __half *const xarray[],
                                                        int64_t incx,
                                                        const float *beta,
                                                        __half *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasHSHgemvBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *const *)Aarray, lda, (const __half *const *)xarray, incx, beta, (__half *const *)yarray, incy, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const float *alpha,
                                                        const __half *const Aarray[],
                                                        int64_t lda,
                                                        const __half *const xarray[],
                                                        int64_t incx,
                                                        const float *beta,
                                                        float *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasHSSgemvBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *const *)Aarray, lda, (const __half *const *)xarray, incx, beta, yarray, incy, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const float *alpha,
                                                        const __nv_bfloat16 *const Aarray[],
                                                        int64_t lda,
                                                        const __nv_bfloat16 *const xarray[],
                                                        int64_t incx,
                                                        const float *beta,
                                                        __nv_bfloat16 *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasTSTgemvBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *const *)Aarray, lda, (const __nv_bfloat16 *const *)xarray, incx, beta, (__nv_bfloat16 *const *)yarray, incy, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t trans,
                                                        int64_t m,
                                                        int64_t n,
                                                        const float *alpha,
                                                        const __nv_bfloat16 *const Aarray[],
                                                        int64_t lda,
                                                        const __nv_bfloat16 *const xarray[],
                                                        int64_t incx,
                                                        const float *beta,
                                                        float *const yarray[],
                                                        int64_t incy,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasTSSgemvBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *const *)Aarray, lda, (const __nv_bfloat16 *const *)xarray, incx, beta, yarray, incy, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t trans,
                                                            int64_t m,
                                                            int64_t n,
                                                            const float *alpha,
                                                            const float *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const float *x,
                                                            int64_t incx,
                                                            long long int stridex,
                                                            const float *beta,
                                                            float *y,
                                                            int64_t incy,
                                                            long long int stridey,
                                                            int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasSgemvStridedBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                        m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t trans,
                                                            int64_t m,
                                                            int64_t n,
                                                            const double *alpha,
                                                            const double *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const double *x,
                                                            int64_t incx,
                                                            long long int stridex,
                                                            const double *beta,
                                                            double *y,
                                                            int64_t incy,
                                                            long long int stridey,
                                                            int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasDgemvStridedBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                        m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t trans,
                                                            int64_t m,
                                                            int64_t n,
                                                            const cuComplex *alpha,
                                                            const cuComplex *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const cuComplex *x,
                                                            int64_t incx,
                                                            long long int stridex,
                                                            const cuComplex *beta,
                                                            cuComplex *y,
                                                            int64_t incy,
                                                            long long int stridey,
                                                            int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasCgemvStridedBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                        m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t trans,
                                                            int64_t m,
                                                            int64_t n,
                                                            const cuDoubleComplex *alpha,
                                                            const cuDoubleComplex *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const cuDoubleComplex *x,
                                                            int64_t incx,
                                                            long long int stridex,
                                                            const cuDoubleComplex *beta,
                                                            cuDoubleComplex *y,
                                                            int64_t incy,
                                                            long long int stridey,
                                                            int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasZgemvStridedBatched_64((cublasHandle_t)handle, UPTKblasOperationTocublasOperation(trans),
                                                                        m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSHgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int64_t m,
                                                                int64_t n,
                                                                const float *alpha,
                                                                const __half *A,
                                                                int64_t lda,
                                                                long long int strideA,
                                                                const __half *x,
                                                                int64_t incx,
                                                                long long int stridex,
                                                                const float *beta,
                                                                __half *y,
                                                                int64_t incy,
                                                                long long int stridey,
                                                                int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasHSHgemvStridedBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *)A, lda, strideA, (const __half *)x, incx, stridex, beta, (__half *)y, incy, stridey, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHSSgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int64_t m,
                                                                int64_t n,
                                                                const float *alpha,
                                                                const __half *A,
                                                                int64_t lda,
                                                                long long int strideA,
                                                                const __half *x,
                                                                int64_t incx,
                                                                long long int stridex,
                                                                const float *beta,
                                                                float *y,
                                                                int64_t incy,
                                                                long long int stridey,
                                                                int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasHSSgemvStridedBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __half *)A, lda, strideA, (const __half *)x, incx, stridex, beta, y, incy, stridey, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSTgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int64_t m,
                                                                int64_t n,
                                                                const float *alpha,
                                                                const __nv_bfloat16 *A,
                                                                int64_t lda,
                                                                long long int strideA,
                                                                const __nv_bfloat16 *x,
                                                                int64_t incx,
                                                                long long int stridex,
                                                                const float *beta,
                                                                __nv_bfloat16 *y,
                                                                int64_t incy,
                                                                long long int stridey,
                                                                int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasTSTgemvStridedBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *)A, lda, strideA, (const __nv_bfloat16 *)x, incx, stridex, beta, (__nv_bfloat16 *)y, incy, stridey, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasTSSgemvStridedBatched_64(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t trans,
                                                                int64_t m,
                                                                int64_t n,
                                                                const float *alpha,
                                                                const __nv_bfloat16 *A,
                                                                int64_t lda,
                                                                long long int strideA,
                                                                const __nv_bfloat16 *x,
                                                                int64_t incx,
                                                                long long int stridex,
                                                                const float *beta,
                                                                float *y,
                                                                int64_t incy,
                                                                long long int stridey,
                                                                int64_t batchCount)
{
    cublasOperation_t cuda_trans = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasTSSgemvStridedBatched_64((cublasHandle_t)handle, cuda_trans, m, n, alpha, (const __nv_bfloat16 *)A, lda, strideA, (const __nv_bfloat16 *)x, incx, stridex, beta, y, incy, stridey, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemm_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *B,
                                                int64_t ldb,
                                                const float *beta,
                                                float *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemm_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemm_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *B,
                                                int64_t ldb,
                                                const double *beta,
                                                double *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemm_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemm_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3m_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCgemm3m_64((cublasHandle_t)handle,
                                                            UPTKblasOperationTocublasOperation(transa),
                                                            UPTKblasOperationTocublasOperation(transb),
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            lda,
                                                            B,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            ldc));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemm_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemm3m_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasZgemm3m_64((cublasHandle_t)handle,
                                                            UPTKblasOperationTocublasOperation(transa),
                                                            UPTKblasOperationTocublasOperation(transb),
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            lda,
                                                            B,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            ldc));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemm_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const __half *alpha,
                                                const __half *A,
                                                int64_t lda,
                                                const __half *B,
                                                int64_t ldb,
                                                const __half *beta,
                                                __half *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasHgemm_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *)A, lda, (const __half *)B, ldb, (const __half *)beta, (__half *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmEx_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int64_t lda,
                                                const void *B,
                                                UPTKDataType_t Btype,
                                                int64_t ldb,
                                                const float *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasSgemmEx_64((cublasHandle_t)handle,
                                                            UPTKblasOperationTocublasOperation(transa),
                                                            UPTKblasOperationTocublasOperation(transb),
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            UPTKDataTypeTocudaDataType(Atype),
                                                            lda,
                                                            B,
                                                            UPTKDataTypeTocudaDataType(Btype),
                                                            ldb,
                                                            beta,
                                                            C,
                                                            UPTKDataTypeTocudaDataType(Ctype),
                                                            ldc));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmEx_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const void *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int64_t lda,
                                                const void *B,
                                                UPTKDataType_t Btype,
                                                int64_t ldb,
                                                const void *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
                                                int64_t ldc,
                                                UPTKblasComputeType_t computeType,
                                                UPTKblasGemmAlgo_t algo)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
    cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
    cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
    cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
    cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
    cublasStatus_t cuda_res;

    cuda_res = cublasGemmEx_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, cuda_aType, lda, B, cuda_bType, ldb, beta, C, cuda_cType, ldc, cuda_computeType, cuda_algo);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmEx_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int64_t lda,
                                                const void *B,
                                                UPTKDataType_t Btype,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCgemmEx_64((cublasHandle_t)handle,
                                                            UPTKblasOperationTocublasOperation(transa),
                                                            UPTKblasOperationTocublasOperation(transb),
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            UPTKDataTypeTocudaDataType(Atype),
                                                            lda,
                                                            B,
                                                            UPTKDataTypeTocudaDataType(Btype),
                                                            ldb,
                                                            beta,
                                                            C,
                                                            UPTKDataTypeTocudaDataType(Ctype),
                                                            ldc));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrk_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *beta,
                                                float *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyrk_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrk_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *beta,
                                                double *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyrk_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyrk_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrk_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyrk_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
// diff
UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkEx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int64_t lda,
                                                const cuComplex *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCsyrkEx_64((cublasHandle_t)handle,
                                                            UPTKblasFillModeTocublasFillMode(uplo),
                                                            UPTKblasOperationTocublasOperation(trans),
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            UPTKDataTypeTocudaDataType(Atype),
                                                            lda,
                                                            beta,
                                                            (cuComplex *)C,
                                                            UPTKDataTypeTocudaDataType(Ctype),
                                                            ldc));
}
// diff
UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrk3mEx_64(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    UPTKblasOperation_t trans,
                                                    int64_t n,
                                                    int64_t k,
                                                    const cuComplex *alpha,
                                                    const void *A,
                                                    UPTKDataType_t Atype,
                                                    int64_t lda,
                                                    const cuComplex *beta,
                                                    void *C,
                                                    UPTKDataType_t Ctype,
                                                    int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCsyrk3mEx_64((cublasHandle_t)handle,
                                                            UPTKblasFillModeTocublasFillMode(uplo),
                                                            UPTKblasOperationTocublasOperation(trans),
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            UPTKDataTypeTocudaDataType(Atype),
                                                            lda,
                                                            beta,
                                                            (cuComplex *)C,
                                                            UPTKDataTypeTocudaDataType(Ctype),
                                                            ldc));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const float *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCherk_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuComplex *)A, lda, beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZherk_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const double *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const double *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZherk_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, (const cuDoubleComplex *)A, lda, beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}
// diff
UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkEx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int64_t lda,
                                                const float *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCherkEx_64((cublasHandle_t)handle,
                                                            UPTKblasFillModeTocublasFillMode(uplo),
                                                            UPTKblasOperationTocublasOperation(trans),
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            UPTKDataTypeTocudaDataType(Atype),
                                                            lda,
                                                            beta,
                                                            (cuComplex *)C,
                                                            UPTKDataTypeTocudaDataType(Ctype),
                                                            ldc));
}
// diff
UPTKBLASAPI UPTKblasStatus_t UPTKblasCherk3mEx_64(UPTKblasHandle_t handle,
                                                    UPTKblasFillMode_t uplo,
                                                    UPTKblasOperation_t trans,
                                                    int64_t n,
                                                    int64_t k,
                                                    const float *alpha,
                                                    const void *A,
                                                    UPTKDataType_t Atype,
                                                    int64_t lda,
                                                    const float *beta,
                                                    void *C,
                                                    UPTKDataType_t Ctype,
                                                    int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCherk3mEx_64((cublasHandle_t)handle,
                                                            UPTKblasFillModeTocublasFillMode(uplo),
                                                            UPTKblasOperationTocublasOperation(trans),
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            UPTKDataTypeTocudaDataType(Atype),
                                                            lda,
                                                            beta,
                                                            (cuComplex *)C,
                                                            UPTKDataTypeTocudaDataType(Ctype),
                                                            ldc));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyr2k_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *B,
                                                int64_t ldb,
                                                const float *beta,
                                                float *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyr2k_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyr2k_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *B,
                                                int64_t ldb,
                                                const double *beta,
                                                double *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyr2k_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyr2k_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyr2k_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyr2k_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyr2k_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCher2k_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const float *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCher2k_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZher2k_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const double *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZher2k_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsyrkx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *B,
                                                int64_t ldb,
                                                const float *beta,
                                                float *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsyrkx_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsyrkx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *B,
                                                int64_t ldb,
                                                const double *beta,
                                                double *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsyrkx_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsyrkx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsyrkx_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsyrkx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsyrkx_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCherkx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const float *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasCherkx_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZherkx_64(UPTKblasHandle_t handle,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                int64_t n,
                                                int64_t k,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const double *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasStatus_t cuda_res;
    cuda_res = cublasZherkx_64((cublasHandle_t)handle, cuda_uplo, cuda_transA, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSsymm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                int64_t m,
                                                int64_t n,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *B,
                                                int64_t ldb,
                                                const float *beta,
                                                float *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasSsymm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDsymm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                int64_t m,
                                                int64_t n,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *B,
                                                int64_t ldb,
                                                const double *beta,
                                                double *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasDsymm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCsymm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasCsymm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZsymm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZsymm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasChemm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                const cuComplex *beta,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasChemm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)B, ldb, (const cuComplex *)beta, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZhemm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                const cuDoubleComplex *beta,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasStatus_t cuda_res;
    cuda_res = cublasZhemm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)B, ldb, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                float *B,
                                                int64_t ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrsm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<float *>(A), lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                double *B,
                                                int64_t ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrsm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<double *>(A), lda, B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                cuComplex *B,
                                                int64_t ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrsm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuComplex *)alpha, (cuComplex *)A, lda, (cuComplex *)B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                cuDoubleComplex *B,
                                                int64_t ldb)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrsm_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuDoubleComplex *)alpha, (cuDoubleComplex *)A, lda, (cuDoubleComplex *)B, ldb);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *B,
                                                int64_t ldb,
                                                float *C,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasStrmm_64(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *B,
                                                int64_t ldb,
                                                double *C,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasDtrmm_64(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                cuComplex *C,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasCtrmm_64(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t side,
                                                UPTKblasFillMode_t uplo,
                                                UPTKblasOperation_t trans,
                                                UPTKblasDiagType_t diag,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    return cublasStatusToUPTKblasStatus(cublasZtrmm_64(
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

UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const __half *alpha,
                                                        const __half *const Aarray[],
                                                        int64_t lda,
                                                        const __half *const Barray[],
                                                        int64_t ldb,
                                                        const __half *beta,
                                                        __half *const Carray[],
                                                        int64_t ldc,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasHgemmBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *const *)Aarray, lda, (const __half *const *)Barray, ldb, (const __half *)beta, (__half *const *)Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const float *alpha,
                                                        const float *const Aarray[],
                                                        int64_t lda,
                                                        const float *const Barray[],
                                                        int64_t ldb,
                                                        const float *beta,
                                                        float *const Carray[],
                                                        int64_t ldc,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemmBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const double *alpha,
                                                        const double *const Aarray[],
                                                        int64_t lda,
                                                        const double *const Barray[],
                                                        int64_t ldb,
                                                        const double *beta,
                                                        double *const Carray[],
                                                        int64_t ldc,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemmBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const cuComplex *alpha,
                                                        const cuComplex *const Aarray[],
                                                        int64_t lda,
                                                        const cuComplex *const Barray[],
                                                        int64_t ldb,
                                                        const cuComplex *beta,
                                                        cuComplex *const Carray[],
                                                        int64_t ldc,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemmBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, Aarray, lda, Barray, ldb, (const cuComplex *)beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const cuDoubleComplex *alpha,
                                                        const cuDoubleComplex *const Aarray[],
                                                        int64_t lda,
                                                        const cuDoubleComplex *const Barray[],
                                                        int64_t ldb,
                                                        const cuDoubleComplex *beta,
                                                        cuDoubleComplex *const Carray[],
                                                        int64_t ldc,
                                                        int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemmBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, Aarray, lda, Barray, ldb, (const cuDoubleComplex *)beta, Carray, ldc, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasHgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t transa,
                                                            UPTKblasOperation_t transb,
                                                            int64_t m,
                                                            int64_t n,
                                                            int64_t k,
                                                            const __half *alpha,
                                                            const __half *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const __half *B,
                                                            int64_t ldb,
                                                            long long int strideB,
                                                            const __half *beta,
                                                            __half *C,
                                                            int64_t ldc,
                                                            long long int strideC,
                                                            int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasHgemmStridedBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const __half *)alpha, (const __half *)A, lda, (long long)strideA, (const __half *)B, ldb, (long long)strideB, (const __half *)beta, (__half *)C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t transa,
                                                            UPTKblasOperation_t transb,
                                                            int64_t m,
                                                            int64_t n,
                                                            int64_t k,
                                                            const float *alpha,
                                                            const float *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const float *B,
                                                            int64_t ldb,
                                                            long long int strideB,
                                                            const float *beta,
                                                            float *C,
                                                            int64_t ldc,
                                                            long long int strideC,
                                                            int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgemmStridedBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, (long long)strideA, B, ldb, (long long)strideB, beta, C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t transa,
                                                            UPTKblasOperation_t transb,
                                                            int64_t m,
                                                            int64_t n,
                                                            int64_t k,
                                                            const double *alpha,
                                                            const double *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const double *B,
                                                            int64_t ldb,
                                                            long long int strideB,
                                                            const double *beta,
                                                            double *C,
                                                            int64_t ldc,
                                                            long long int strideC,
                                                            int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgemmStridedBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, lda, (long long)strideA, B, ldb, (long long)strideB, beta, C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t transa,
                                                            UPTKblasOperation_t transb,
                                                            int64_t m,
                                                            int64_t n,
                                                            int64_t k,
                                                            const cuComplex *alpha,
                                                            const cuComplex *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const cuComplex *B,
                                                            int64_t ldb,
                                                            long long int strideB,
                                                            const cuComplex *beta,
                                                            cuComplex *C,
                                                            int64_t ldc,
                                                            long long int strideC,
                                                            int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgemmStridedBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuComplex *)alpha, (const cuComplex *)A, lda, (long long)strideA, (const cuComplex *)B, ldb, (long long)strideB, (const cuComplex *)beta, (cuComplex *)C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mStridedBatched_64(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t transa,
                                                                UPTKblasOperation_t transb,
                                                                int64_t m,
                                                                int64_t n,
                                                                int64_t k,
                                                                const cuComplex *alpha,
                                                                const cuComplex *A,
                                                                int64_t lda,
                                                                long long int strideA,
                                                                const cuComplex *B,
                                                                int64_t ldb,
                                                                long long int strideB,
                                                                const cuComplex *beta,
                                                                cuComplex *C,
                                                                int64_t ldc,
                                                                long long int strideC,
                                                                int64_t batchCount)
{
    return cublasStatusToUPTKblasStatus(cublasCgemm3mStridedBatched_64((cublasHandle_t)handle,
                                                                        UPTKblasOperationTocublasOperation(transa),
                                                                        UPTKblasOperationTocublasOperation(transb),
                                                                        m,
                                                                        n,
                                                                        k,
                                                                        alpha,
                                                                        A,
                                                                        lda,
                                                                        strideA,
                                                                        B,
                                                                        ldb,
                                                                        strideB,
                                                                        beta,
                                                                        C,
                                                                        ldc,
                                                                        strideC,
                                                                        batchCount));
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgemmStridedBatched_64(UPTKblasHandle_t handle,
                                                            UPTKblasOperation_t transa,
                                                            UPTKblasOperation_t transb,
                                                            int64_t m,
                                                            int64_t n,
                                                            int64_t k,
                                                            const cuDoubleComplex *alpha,
                                                            const cuDoubleComplex *A,
                                                            int64_t lda,
                                                            long long int strideA,
                                                            const cuDoubleComplex *B,
                                                            int64_t ldb,
                                                            long long int strideB,
                                                            const cuDoubleComplex *beta,
                                                            cuDoubleComplex *C,
                                                            int64_t ldc,
                                                            long long int strideC,
                                                            int64_t batchCount)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgemmStridedBatched_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (long long)strideA, (const cuDoubleComplex *)B, ldb, (long long)strideB, (const cuDoubleComplex *)beta, (cuDoubleComplex *)C, ldc, (long long)strideC, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmBatchedEx_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const void *alpha,
                                                        const void *const Aarray[],
                                                        UPTKDataType_t Atype,
                                                        int64_t lda,
                                                        const void *const Barray[],
                                                        UPTKDataType_t Btype,
                                                        int64_t ldb,
                                                        const void *beta,
                                                        void *const Carray[],
                                                        UPTKDataType_t Ctype,
                                                        int64_t ldc,
                                                        int64_t batchCount,
                                                        UPTKblasComputeType_t computeType,
                                                        UPTKblasGemmAlgo_t algo)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
    cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
    cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
    cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
    cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
    cublasStatus_t cuda_res;

    cuda_res = cublasGemmBatchedEx_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, (const void **)Aarray, cuda_aType, lda, (const void **)Barray, cuda_bType, ldb, beta, (void **)Carray, cuda_cType, ldc, batchCount, cuda_computeType, cuda_algo);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmStridedBatchedEx_64(UPTKblasHandle_t handle,
                                                                UPTKblasOperation_t transa,
                                                                UPTKblasOperation_t transb,
                                                                int64_t m,
                                                                int64_t n,
                                                                int64_t k,
                                                                const void *alpha,
                                                                const void *A,
                                                                UPTKDataType_t Atype,
                                                                int64_t lda,
                                                                long long int strideA,
                                                                const void *B,
                                                                UPTKDataType_t Btype,
                                                                int64_t ldb,
                                                                long long int strideB,
                                                                const void *beta,
                                                                void *C,
                                                                UPTKDataType_t Ctype,
                                                                int64_t ldc,
                                                                long long int strideC,
                                                                int64_t batchCount,
                                                                UPTKblasComputeType_t computeType,
                                                                UPTKblasGemmAlgo_t algo)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cudaDataType cuda_aType = UPTKDataTypeTocudaDataType(Atype);
    cudaDataType cuda_bType = UPTKDataTypeTocudaDataType(Btype);
    cudaDataType cuda_cType = UPTKDataTypeTocudaDataType(Ctype);
    cublasComputeType_t cuda_computeType = UPTKblasComputeTypeTocublasComputeType(computeType);
    cublasGemmAlgo_t cuda_algo = UPTKblasGemmAlgoTocublasGemmAlgo(algo);
    cublasStatus_t cuda_res;

    cuda_res = cublasGemmStridedBatchedEx_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, k, alpha, A, cuda_aType, lda, strideA, B, cuda_bType, ldb, strideB, beta, C, cuda_cType, ldc, strideC, batchCount, cuda_computeType, cuda_algo);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmGroupedBatched(UPTKblasHandle_t handle,
                                                            const UPTKblasOperation_t transa_array[],
                                                            const UPTKblasOperation_t transb_array[],
                                                            const int m_array[],
                                                            const int n_array[],
                                                            const int k_array[],
                                                            const float alpha_array[],
                                                            const float *const Aarray[],
                                                            const int lda_array[],
                                                            const float *const Barray[],
                                                            const int ldb_array[],
                                                            const float beta_array[],
                                                            float *const Carray[],
                                                            const int ldc_array[],
                                                            int group_count,
                                                            const int group_size[])
{
    cublasOperation_t *cuda_transA_array = new cublasOperation_t[group_count];
    cublasOperation_t *cuda_transB_array = new cublasOperation_t[group_count];

    for (int i = 0; i < group_count; i++)
    {
        cuda_transA_array[i] = UPTKblasOperationTocublasOperation(transa_array[i]);
        cuda_transB_array[i] = UPTKblasOperationTocublasOperation(transb_array[i]);
    }

    cublasStatus_t cuda_res;

    cuda_res = cublasSgemmGroupedBatched((cublasHandle_t)handle,
                                            cuda_transA_array,
                                            cuda_transB_array,
                                            m_array,
                                            n_array,
                                            k_array,
                                            alpha_array,
                                            Aarray,
                                            lda_array,
                                            Barray,
                                            ldb_array,
                                            beta_array,
                                            Carray,
                                            ldc_array,
                                            group_count,
                                            group_size);
    free(cuda_transA_array);
    free(cuda_transB_array);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgemmGroupedBatched_64(UPTKblasHandle_t handle,
                                                            const UPTKblasOperation_t transa_array[],
                                                            const UPTKblasOperation_t transb_array[],
                                                            const int64_t m_array[],
                                                            const int64_t n_array[],
                                                            const int64_t k_array[],
                                                            const float alpha_array[],
                                                            const float *const Aarray[],
                                                            const int64_t lda_array[],
                                                            const float *const Barray[],
                                                            const int64_t ldb_array[],
                                                            const float beta_array[],
                                                            float *const Carray[],
                                                            const int64_t ldc_array[],
                                                            int64_t group_count,
                                                            const int64_t group_size[])
{
    cublasOperation_t *cuda_transA_array = new cublasOperation_t[group_count];
    cublasOperation_t *cuda_transB_array = new cublasOperation_t[group_count];

    for (int i = 0; i < group_count; i++)
    {
        cuda_transA_array[i] = UPTKblasOperationTocublasOperation(transa_array[i]);
        cuda_transB_array[i] = UPTKblasOperationTocublasOperation(transb_array[i]);
    }

    cublasStatus_t cuda_res;

    cuda_res = cublasSgemmGroupedBatched_64((cublasHandle_t)handle,
                                            cuda_transA_array,
                                            cuda_transB_array,
                                            m_array,
                                            n_array,
                                            k_array,
                                            alpha_array,
                                            Aarray,
                                            lda_array,
                                            Barray,
                                            ldb_array,
                                            beta_array,
                                            Carray,
                                            ldc_array,
                                            group_count,
                                            group_size);
    free(cuda_transA_array);
    free(cuda_transB_array);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmGroupedBatched(UPTKblasHandle_t handle,
                                                            const UPTKblasOperation_t transa_array[],
                                                            const UPTKblasOperation_t transb_array[],
                                                            const int m_array[],
                                                            const int n_array[],
                                                            const int k_array[],
                                                            const double alpha_array[],
                                                            const double *const Aarray[],
                                                            const int lda_array[],
                                                            const double *const Barray[],
                                                            const int ldb_array[],
                                                            const double beta_array[],
                                                            double *const Carray[],
                                                            const int ldc_array[],
                                                            int group_count,
                                                            const int group_size[])
{
    cublasOperation_t *cuda_transA_array = new cublasOperation_t[group_count];
    cublasOperation_t *cuda_transB_array = new cublasOperation_t[group_count];

    for (int i = 0; i < group_count; i++)
    {
        cuda_transA_array[i] = UPTKblasOperationTocublasOperation(transa_array[i]);
        cuda_transB_array[i] = UPTKblasOperationTocublasOperation(transb_array[i]);
    }

    cublasStatus_t cuda_res;

    cuda_res = cublasDgemmGroupedBatched((cublasHandle_t)handle,
                                            cuda_transA_array,
                                            cuda_transB_array,
                                            m_array,
                                            n_array,
                                            k_array,
                                            alpha_array,
                                            Aarray,
                                            lda_array,
                                            Barray,
                                            ldb_array,
                                            beta_array,
                                            Carray,
                                            ldc_array,
                                            group_count,
                                            group_size);
    free(cuda_transA_array);
    free(cuda_transB_array);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgemmGroupedBatched_64(UPTKblasHandle_t handle,
                                                            const UPTKblasOperation_t transa_array[],
                                                            const UPTKblasOperation_t transb_array[],
                                                            const int64_t m_array[],
                                                            const int64_t n_array[],
                                                            const int64_t k_array[],
                                                            const double alpha_array[],
                                                            const double *const Aarray[],
                                                            const int64_t lda_array[],
                                                            const double *const Barray[],
                                                            const int64_t ldb_array[],
                                                            const double beta_array[],
                                                            double *const Carray[],
                                                            const int64_t ldc_array[],
                                                            int64_t group_count,
                                                            const int64_t group_size[])
{
    cublasOperation_t *cuda_transA_array = new cublasOperation_t[group_count];
    cublasOperation_t *cuda_transB_array = new cublasOperation_t[group_count];

    for (int i = 0; i < group_count; i++)
    {
        cuda_transA_array[i] = UPTKblasOperationTocublasOperation(transa_array[i]);
        cuda_transB_array[i] = UPTKblasOperationTocublasOperation(transb_array[i]);
    }

    cublasStatus_t cuda_res;

    cuda_res = cublasDgemmGroupedBatched_64((cublasHandle_t)handle,
                                            cuda_transA_array,
                                            cuda_transB_array,
                                            m_array,
                                            n_array,
                                            k_array,
                                            alpha_array,
                                            Aarray,
                                            lda_array,
                                            Barray,
                                            ldb_array,
                                            beta_array,
                                            Carray,
                                            ldc_array,
                                            group_count,
                                            group_size);
    free(cuda_transA_array);
    free(cuda_transB_array);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

// cublasStatus_t cublasComputeTypeTocublasDatatype(UPTKDataType_t Atype, UPTKDataType_t Btype,
//                                                     UPTKDataType_t Ctype, cublasComputeType_t computeType,
//                                                     UPTKDataType_t &cudaComputeData)
// {
//     if (Atype == CUDA_R_16F && Btype == CUDA_R_16F && Ctype == CUDA_R_16F && computeType == CUBLAS_COMPUTE_16F)
//     {
//         cudaComputeData = CUDA_R_16F;
//     }
//     else if (Atype == CUDA_R_16F && Btype == CUDA_R_16F && Ctype == CUDA_R_16F && computeType == CUBLAS_COMPUTE_32F)
//     {
//         cudaComputeData = CUDA_R_32F;
//     }
//     else if (Atype == CUDA_R_16F && Btype == CUDA_R_16F && Ctype == CUDA_R_32F && computeType ==CUBLAS_COMPUTE_32F)
//     {

//         cudaComputeData = CUDA_R_32F;
//     }
//     else if (Atype == CUDA_R_16B && Btype == CUDA_R_16B && Ctype == CUDA_R_16B && computeType == CUBLAS_COMPUTE_32F)
//     {

//         cudaComputeData = CUDA_R_32F;
//     }
//     else if (Atype == CUDA_R_16B && Btype == CUDA_R_16B && Ctype == CUDA_R_32F && computeType == CUBLAS_COMPUTE_32F)
//     {

//         cudaComputeData = CUDA_R_32F;
//     }
//     else if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F && computeType == CUBLAS_COMPUTE_32F)
//     {
//         cudaComputeData = CUDA_R_32F;
//     }
//     else if (Atype == CUDA_R_64F && Btype == CUDA_R_64F && Ctype == CUDA_R_64F && computeType == CUBLAS_COMPUTE_64F)
//     {
//         cudaComputeData = CUDA_R_64F;
//     }
//     else if (Atype == CUDA_R_8I && Btype == CUDA_R_8I && Ctype == CUDA_R_32I && computeType == CUBLAS_COMPUTE_32I)
//     {

//         cudaComputeData = CUDA_R_32I;
//     }
//     else if (Atype == CUDA_C_32F && Btype == CUDA_C_32F && Ctype == CUDA_C_32F && computeType == CUBLAS_COMPUTE_32F)
//     {
//         cudaComputeData = CUDA_C_32F;
//     }
//     else if (Atype == CUDA_C_64F && Btype == CUDA_C_64F && Ctype == CUDA_C_64F && computeType == CUBLAS_COMPUTE_64F)
//     {
//         cudaComputeData = CUDA_C_64F;
//     }
//     else
//     {
//         return  CUBLAS_STATUS_NOT_SUPPORTED;
//     }

//     return CUBLAS_STATUS_SUCCESS;
// }

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmGroupedBatchedEx(UPTKblasHandle_t handle,
                                                            const UPTKblasOperation_t transa_array[],
                                                            const UPTKblasOperation_t transb_array[],
                                                            const int m_array[],
                                                            const int n_array[],
                                                            const int k_array[],
                                                            const void *alpha_array,
                                                            const void *const Aarray[],
                                                            UPTKDataType_t Atype,
                                                            const int lda_array[],
                                                            const void *const Barray[],
                                                            UPTKDataType_t Btype,
                                                            const int ldb_array[],
                                                            const void *beta_array,
                                                            void *const Carray[],
                                                            UPTKDataType_t Ctype,
                                                            const int ldc_array[],
                                                            int group_count,
                                                            const int group_size[],
                                                            UPTKblasComputeType_t computeType)
{
    cublasOperation_t *cuda_transA_array = new cublasOperation_t[group_count];
    cublasOperation_t *cuda_transB_array = new cublasOperation_t[group_count];

    for (int i = 0; i < group_count; i++)
    {
        cuda_transA_array[i] = UPTKblasOperationTocublasOperation(transa_array[i]);
        cuda_transB_array[i] = UPTKblasOperationTocublasOperation(transb_array[i]);
    }

    cublasStatus_t cuda_res;
    cudaDataType_t cudaAtype = UPTKDataTypeTocudaDataType(Atype);
    cudaDataType_t cudaBtype = UPTKDataTypeTocudaDataType(Btype);
    cudaDataType_t cudaCtype = UPTKDataTypeTocudaDataType(Ctype);

    cublasComputeType_t cudaComputeType = UPTKblasComputeTypeTocublasComputeType(computeType);

    cuda_res = cublasGemmGroupedBatchedEx((cublasHandle_t)handle,
                                            cuda_transA_array,
                                            cuda_transB_array,
                                            m_array,
                                            n_array,
                                            k_array,
                                            alpha_array,
                                            Aarray,
                                            cudaAtype,
                                            lda_array,
                                            Barray,
                                            cudaBtype,
                                            ldb_array,
                                            beta_array,
                                            Carray,
                                            cudaCtype,
                                            ldc_array,
                                            group_count,
                                            group_size,
                                            cudaComputeType);

    free(cuda_transA_array);
    free(cuda_transB_array);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGemmGroupedBatchedEx_64(UPTKblasHandle_t handle,
                                                                const UPTKblasOperation_t transa_array[],
                                                                const UPTKblasOperation_t transb_array[],
                                                                const int64_t m_array[],
                                                                const int64_t n_array[],
                                                                const int64_t k_array[],
                                                                const void *alpha_array,
                                                                const void *const Aarray[],
                                                                UPTKDataType_t Atype,
                                                                const int64_t lda_array[],
                                                                const void *const Barray[],
                                                                UPTKDataType_t Btype,
                                                                const int64_t ldb_array[],
                                                                const void *beta_array,
                                                                void *const Carray[],
                                                                UPTKDataType_t Ctype,
                                                                const int64_t ldc_array[],
                                                                int64_t group_count,
                                                                const int64_t group_size[],
                                                                UPTKblasComputeType_t computeType)
{
    cublasOperation_t *cuda_transA_array = new cublasOperation_t[group_count];
    cublasOperation_t *cuda_transB_array = new cublasOperation_t[group_count];

    for (int i = 0; i < group_count; i++)
    {
        cuda_transA_array[i] = UPTKblasOperationTocublasOperation(transa_array[i]);
        cuda_transB_array[i] = UPTKblasOperationTocublasOperation(transb_array[i]);
    }

    cublasStatus_t cuda_res;
    cudaDataType_t cudaAtype = UPTKDataTypeTocudaDataType(Atype);
    cudaDataType_t cudaBtype = UPTKDataTypeTocudaDataType(Btype);
    cudaDataType_t cudaCtype = UPTKDataTypeTocudaDataType(Ctype);
    cublasComputeType_t cudaComputeType = UPTKblasComputeTypeTocublasComputeType(computeType);
    cuda_res = cublasGemmGroupedBatchedEx_64((cublasHandle_t)handle,
                                                cuda_transA_array,
                                                cuda_transB_array,
                                                m_array,
                                                n_array,
                                                k_array,
                                                alpha_array,
                                                Aarray,
                                                cudaAtype,
                                                lda_array,
                                                Barray,
                                                cudaBtype,
                                                ldb_array,
                                                beta_array,
                                                Carray,
                                                cudaCtype,
                                                ldc_array,
                                                group_count,
                                                group_size,
                                                cudaComputeType);

    free(cuda_transA_array);
    free(cuda_transB_array);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSgeam_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                const float *alpha,
                                                const float *A,
                                                int64_t lda,
                                                const float *beta,
                                                const float *B,
                                                int64_t ldb,
                                                float *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasSgeam_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDgeam_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                const double *alpha,
                                                const double *A,
                                                int64_t lda,
                                                const double *beta,
                                                const double *B,
                                                int64_t ldb,
                                                double *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasDgeam_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgeam_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *alpha,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *beta,
                                                const cuComplex *B,
                                                int64_t ldb,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasCgeam_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, (const cuComplex *)alpha, (const cuComplex *)A, lda, (const cuComplex *)beta, (const cuComplex *)B, ldb, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZgeam_64(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa,
                                                UPTKblasOperation_t transb,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *alpha,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *beta,
                                                const cuDoubleComplex *B,
                                                int64_t ldb,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(transa);
    cublasOperation_t cuda_transB = UPTKblasOperationTocublasOperation(transb);
    cublasStatus_t cuda_res;
    cuda_res = cublasZgeam_64((cublasHandle_t)handle, cuda_transA, cuda_transB, m, n, (const cuDoubleComplex *)alpha, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)beta, (const cuDoubleComplex *)B, ldb, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasStrsmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasSideMode_t side,
                                                        UPTKblasFillMode_t uplo,
                                                        UPTKblasOperation_t trans,
                                                        UPTKblasDiagType_t diag,
                                                        int64_t m,
                                                        int64_t n,
                                                        const float *alpha,
                                                        const float *const A[],
                                                        int64_t lda,
                                                        float *const B[],
                                                        int64_t ldb,
                                                        int64_t batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasStrsmBatched_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<float *const *>(A), lda, const_cast<float **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDtrsmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasSideMode_t side,
                                                        UPTKblasFillMode_t uplo,
                                                        UPTKblasOperation_t trans,
                                                        UPTKblasDiagType_t diag,
                                                        int64_t m,
                                                        int64_t n,
                                                        const double *alpha,
                                                        const double *const A[],
                                                        int64_t lda,
                                                        double *const B[],
                                                        int64_t ldb,
                                                        int64_t batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasDtrsmBatched_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, alpha, const_cast<double *const *>(A), lda, const_cast<double **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCtrsmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasSideMode_t side,
                                                        UPTKblasFillMode_t uplo,
                                                        UPTKblasOperation_t trans,
                                                        UPTKblasDiagType_t diag,
                                                        int64_t m,
                                                        int64_t n,
                                                        const cuComplex *alpha,
                                                        const cuComplex *const A[],
                                                        int64_t lda,
                                                        cuComplex *const B[],
                                                        int64_t ldb,
                                                        int64_t batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasCtrsmBatched_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuComplex *)alpha, const_cast<cuComplex *const *>(A), lda, const_cast<cuComplex **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZtrsmBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasSideMode_t side,
                                                        UPTKblasFillMode_t uplo,
                                                        UPTKblasOperation_t trans,
                                                        UPTKblasDiagType_t diag,
                                                        int64_t m,
                                                        int64_t n,
                                                        const cuDoubleComplex *alpha,
                                                        const cuDoubleComplex *const A[],
                                                        int64_t lda,
                                                        cuDoubleComplex *const B[],
                                                        int64_t ldb,
                                                        int64_t batchCount)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(side);
    cublasFillMode_t cuda_uplo = UPTKblasFillModeTocublasFillMode(uplo);
    cublasOperation_t cuda_transA = UPTKblasOperationTocublasOperation(trans);
    cublasDiagType_t cuda_diag = UPTKblasDiagTypeTocublasDiagType(diag);
    cublasStatus_t cuda_res;
    cuda_res = cublasZtrsmBatched_64((cublasHandle_t)handle, cuda_side, cuda_uplo, cuda_transA, cuda_diag, m, n, (const cuDoubleComplex *)alpha, const_cast<cuDoubleComplex *const *>(A), lda, const_cast<cuDoubleComplex **>(B), ldb, batchCount);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSdgmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t mode,
                                                int64_t m,
                                                int64_t n,
                                                const float *A,
                                                int64_t lda,
                                                const float *x,
                                                int64_t incx,
                                                float *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasSdgmm_64((cublasHandle_t)handle, cuda_side, m, n, A, lda, x, incx, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasDdgmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t mode,
                                                int64_t m,
                                                int64_t n,
                                                const double *A,
                                                int64_t lda,
                                                const double *x,
                                                int64_t incx,
                                                double *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasDdgmm_64((cublasHandle_t)handle, cuda_side, m, n, A, lda, x, incx, C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCdgmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t mode,
                                                int64_t m,
                                                int64_t n,
                                                const cuComplex *A,
                                                int64_t lda,
                                                const cuComplex *x,
                                                int64_t incx,
                                                cuComplex *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasCdgmm_64((cublasHandle_t)handle, cuda_side, m, n, (const cuComplex *)A, lda, (const cuComplex *)x, incx, (cuComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasZdgmm_64(UPTKblasHandle_t handle,
                                                UPTKblasSideMode_t mode,
                                                int64_t m,
                                                int64_t n,
                                                const cuDoubleComplex *A,
                                                int64_t lda,
                                                const cuDoubleComplex *x,
                                                int64_t incx,
                                                cuDoubleComplex *C,
                                                int64_t ldc)
{
    cublasSideMode_t cuda_side = UPTKblasSideModeTocublasSideMode(mode);
    cublasStatus_t cuda_res;
    cuda_res = cublasZdgmm_64((cublasHandle_t)handle, cuda_side, m, n, (const cuDoubleComplex *)A, lda, (const cuDoubleComplex *)x, incx, (cuDoubleComplex *)C, ldc);
    return cublasStatusToUPTKblasStatus(cuda_res);
}

#ifdef UPTK_NOT_SUPPORT
UPTKBLASAPI UPTKblasStatus_t UPTKblasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char *logFileName)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetLoggerCallback(UPTKblasLogCallback userCallback)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetLoggerCallback(UPTKblasLogCallback *userCallback)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI void UPTKblasXerbla(const char *srName, int info)
{
    Debug();
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCopyEx(UPTKblasHandle_t handle,
                                            int n,
                                            const void *x,
                                            UPTKDataType_t xType,
                                            int incx,
                                            void *y,
                                            UPTKDataType_t yType,
                                            int incy)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSwapEx(UPTKblasHandle_t handle,
                                            int n,
                                            void *x,
                                            UPTKDataType_t xType,
                                            int incx,
                                            void *y,
                                            UPTKDataType_t yType,
                                            int incy)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIamaxEx(UPTKblasHandle_t handle,
                                                int n,
                                                const void *x, UPTKDataType_t xType,
                                                int incx,
                                                int *result)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasIaminEx(UPTKblasHandle_t handle,
                                                int n,
                                                const void *x, UPTKDataType_t xType,
                                                int incx,
                                                int *result)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasAsumEx(UPTKblasHandle_t handle,
                                            int n,
                                            const void *x,
                                            UPTKDataType_t xType,
                                            int incx,
                                            void *result,
                                            UPTKDataType_t resultType,
                                            UPTKDataType_t executiontype)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasRotgEx(UPTKblasHandle_t handle,
                                            void *a,
                                            void *b,
                                            UPTKDataType_t abType,
                                            void *c,
                                            void *s,
                                            UPTKDataType_t csType,
                                            UPTKDataType_t executiontype)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasRotmEx(UPTKblasHandle_t handle,
                                            int n,
                                            void *x,
                                            UPTKDataType_t xType,
                                            int incx,
                                            void *y,
                                            UPTKDataType_t yType,
                                            int incy,
                                            const void *param,
                                            UPTKDataType_t paramType,
                                            UPTKDataType_t executiontype)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasRotmgEx(UPTKblasHandle_t handle,
                                                void *d1,
                                                UPTKDataType_t d1Type,
                                                void *d2,
                                                UPTKDataType_t d2Type,
                                                void *x1,
                                                UPTKDatUPTKblasRotmgExaType_t x1Type,
                                                const void *y1,
                                                UPTKDataType_t y1Type,
                                                void *param,
                                                UPTKDataType_t paramType,
                                                UPTKDataType_t executiontype)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mBatched_64(UPTKblasHandle_t handle,
                                                        UPTKblasOperation_t transa,
                                                        UPTKblasOperation_t transb,
                                                        int64_t m,
                                                        int64_t n,
                                                        int64_t k,
                                                        const cuComplex *alpha,
                                                        const cuComplex *const Aarray[],
                                                        int64_t lda,
                                                        const cuComplex *const Barray[],
                                                        int64_t ldb,
                                                        const cuComplex *beta,
                                                        cuComplex *const Carray[],
                                                        int64_t ldc,
                                                        int64_t batchCount)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasUint8gemmBias(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    UPTKblasOperation_t transc,
                                                    int m, int n, int k,
                                                    const unsigned char *A, int A_bias, int lda,
                                                    const unsigned char *B, int B_bias, int ldb,
                                                    unsigned char *C, int C_bias, int ldc,
                                                    int C_mult, int C_shift)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mBatched(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha,
                                                    const cuComplex *const Aarray[],
                                                    int lda,
                                                    const cuComplex *const Barray[],
                                                    int ldb,
                                                    const cuComplex *beta,
                                                    cuComplex *const Carray[],
                                                    int ldc,
                                                    int batchCount)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasGetSmCountTarget(UPTKblasHandle_t handle, int *smCountTarget)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSetSmCountTarget(UPTKblasHandle_t handle, int smCountTarget)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mEx_64(UPTKblasHandle_t handle,
                                                    UPTKblasOperation_t transa,
                                                    UPTKblasOperation_t transb,
                                                    int64_t m,
                                                    int64_t n,
                                                    int64_t k,
                                                    const cuComplex *alpha,
                                                    const void *A,
                                                    UPTKDataType_t Atype,
                                                    int64_t lda,
                                                    const void *B,
                                                    UPTKDataType_t Btype,
                                                    int64_t ldb,
                                                    const cuComplex *beta,
                                                    void *C,
                                                    UPTKDataType_t Ctype,
                                                    int64_t ldc)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCopyEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasSwapEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasRotmEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *y,
                                                UPTKDataType_t yType,
                                                int64_t incy,
                                                const void *param,
                                                UPTKDataType_t paramType,
                                                UPTKDataType_t executiontype)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIamaxEx_64(UPTKblasHandle_t handle, int64_t n, const void *x, UPTKDataType_t xType, int64_t incx, int64_t *result)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t
UPTKblasIaminEx_64(UPTKblasHandle_t handle, int64_t n, const void *x, UPTKDataType_t xType, int64_t incx, int64_t *result)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasAsumEx_64(UPTKblasHandle_t handle,
                                                int64_t n,
                                                const void *x,
                                                UPTKDataType_t xType,
                                                int64_t incx,
                                                void *result,
                                                UPTKDataType_t resultType,
                                                UPTKDataType_t executiontype)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

UPTKBLASAPI UPTKblasStatus_t UPTKblasCgemm3mEx(UPTKblasHandle_t handle,
                                                UPTKblasOperation_t transa, UPTKblasOperation_t transb,
                                                int m, int n, int k,
                                                const cuComplex *alpha,
                                                const void *A,
                                                UPTKDataType_t Atype,
                                                int lda,
                                                const void *B,
                                                UPTKDataType_t Btype,
                                                int ldb,
                                                const cuComplex *beta,
                                                void *C,
                                                UPTKDataType_t Ctype,
                                                int ldc)
{
    Debug();
    return UPTKBLAS_STATUS_NOT_SUPPORTED;
}

#endif

#if defined(__cplusplus)
}
#endif /* __cplusplus */
